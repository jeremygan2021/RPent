# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One on-policy PPO batch from completed context trajectories, on CPU."""

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .observation import FEATURE_VERSION, SCORING_VERSION
from .policy import ActorCritic, MLPController, feature_names, save_checkpoint


@dataclass(frozen=True)
class PPOConfig:
    """Cost weight, fixed reference token scale, and conservative update limits."""

    token_scale: float
    beta: float = 0.1
    learning_rate: float = 3e-4
    epochs: int = 4
    clip: float = 0.2
    value_weight: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    latency_beta: float = 0.0
    latency_scale: float = 1200.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
            if value < 0 or (name not in {"beta", "latency_beta", "value_weight"} and value == 0):
                raise ValueError(f"invalid {name}: {value}")
        if type(self.epochs) is not int or not 0 < self.clip < 1:
            raise ValueError("epochs must be an integer and clip must be in (0, 1)")


@dataclass(frozen=True)
class Batch:
    """Detached, validated samples with per-episode return boundaries."""

    observations: torch.Tensor
    actions: torch.Tensor
    old_log_prob: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    episode_lengths: tuple[int, ...]
    policy_version: str


def rewards_and_returns(
    costs: list[int], success: bool, config: PPOConfig, task_duration_s: float | None = None
) -> tuple[list[float], list[float]]:
    """Charge every attempt once, add terminal success once, use gamma=1."""
    if not costs or type(success) is not bool:
        raise ValueError("a completed episode with an environment verdict is required")
    if any(type(c) is not int or c < 0 for c in costs):
        raise ValueError("input token counts must be nonnegative integers")
    rewards = [-config.beta * c / config.token_scale for c in costs]
    if config.latency_beta:
        if isinstance(task_duration_s, bool) or not isinstance(task_duration_s, (int, float)) or not math.isfinite(task_duration_s) or task_duration_s < 0:
            raise ValueError("valid task_duration_s required for latency reward")
        rewards[-1] -= config.latency_beta * task_duration_s / config.latency_scale
    rewards[-1] += int(success)
    returns = [0.0] * len(rewards)
    total = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        total += rewards[i]
        returns[i] = total
    return rewards, returns


def prepare_batch(
    trajectories: list[dict], policy: MLPController, config: PPOConfig
) -> Batch:
    """Reject missing, stale, deterministic or inconsistent records before training."""
    if policy.legacy:
        raise ValueError(
            "four-dimensional PPO requires a new memory-enabled checkpoint"
        )
    xs, actions, log_probs, values, rewards, returns, lengths = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    episodes, responses = set(), set()
    for trajectory in trajectories:
        episode = trajectory.get("episode_id")
        if not isinstance(episode, str) or not episode or episode in episodes:
            raise ValueError("missing or duplicate episode_id")
        episodes.add(episode)
        if (
            trajectory.get("schema_version") != 1
            or trajectory.get("audit_complete") is not True
            or trajectory.get("issues")
        ):
            raise ValueError(f"{episode}: incomplete or unsupported trajectory audit")
        outcome = trajectory.get("outcome") or {}
        if type(outcome.get("environment_success")) is not bool or outcome.get(
            "agent_error"
        ):
            raise ValueError(
                f"{episode}: missing verdict or agent error requires review"
            )
        rows = trajectory.get("decisions", [])
        if not rows:
            raise ValueError(f"{episode}: no decisions")
        costs, requests = [], set()
        for index, row in enumerate(rows, 1):
            decision = row["decision"]
            observation = decision["observation"]
            if decision.get("id") != index:
                raise ValueError(f"{episode}: missing or out-of-order decisions")
            if (
                decision.get("policy_version") != policy.version
                or decision.get("scoring_version") != SCORING_VERSION
            ):
                raise ValueError(f"{episode}: policy/scoring version mismatch")
            if observation.get("version") != FEATURE_VERSION or observation.get(
                "names"
            ) != list(feature_names()):
                raise ValueError(f"{episode}: feature schema mismatch")
            if any(
                decision.get(k) is None for k in ("latent_action", "log_prob", "value")
            ):
                raise ValueError(f"{episode}: sampled policy metadata required")
            attempts = row.get("attempts", [])
            if not attempts:
                raise ValueError(f"{episode}: missing request feedback")
            cost = 0
            for attempt in attempts:
                rid, response_id = attempt.get("request_id"), attempt.get("response_id")
                if (
                    not rid
                    or rid in requests
                    or not response_id
                    or response_id in responses
                ):
                    raise ValueError(
                        f"{episode}: missing or duplicate request/response ID"
                    )
                requests.add(rid)
                responses.add(response_id)
                timing = attempt.get("timing") or {}
                if (
                    timing.get("transport_error")
                    or not 200 <= timing.get("status", 0) < 300
                ):
                    raise ValueError(f"{episode}: failed transport")
                count = (attempt.get("usage") or {}).get("input_tokens")
                if type(count) is not int or count < 0:
                    raise ValueError(f"{episode}: missing or invalid input tokens")
                cost += count
            costs.append(cost)
            xs.append(observation["values"])
            actions.append(decision["latent_action"])
            log_probs.append(decision["log_prob"])
            values.append(decision["value"])
            expected_rho = (
                torch.tensor(decision["latent_action"], dtype=torch.float64)
                .sigmoid()
                .clamp(1e-12, 1 - 1e-12)
            )
            recorded_rho = torch.tensor(
                [
                    decision["thresholds"][m]
                    for m in ("image", "text", "action", "memory")
                ],
                dtype=torch.float64,
            )
            if expected_rho.shape != (4,) or not torch.allclose(
                expected_rho, recorded_rho, atol=1e-9, rtol=1e-7
            ):
                raise ValueError(
                    f"{episode}: thresholds do not match sampled latent action"
                )
        reward, target = rewards_and_returns(
            costs, outcome["environment_success"], config,
            (outcome.get("timing") or {}).get("task_duration_s")
        )
        rewards.extend(reward)
        returns.extend(target)
        lengths.append(len(rows))
    if not xs:
        raise ValueError("empty training batch")
    x, action, old_log, old_value, target, reward = (
        torch.tensor(v, dtype=torch.float32)
        for v in (xs, actions, log_probs, values, returns, rewards)
    )
    n = len(xs)
    if (
        x.shape != (n, len(feature_names()))
        or action.shape != (n, 4)
        or old_log.shape != (n,)
        or old_value.shape != (n,)
    ):
        raise ValueError("invalid training tensor shapes")
    if any(
        not torch.isfinite(t).all()
        for t in (x, action, old_log, old_value, target, reward)
    ):
        raise ValueError("training data must be finite")
    with torch.no_grad():
        distribution, predicted_value = policy.model(x)
        predicted_log = distribution.log_prob(action).sum(-1)
    if not torch.allclose(
        predicted_log, old_log, atol=1e-4, rtol=1e-5
    ) or not torch.allclose(predicted_value, old_value, atol=1e-4, rtol=1e-5):
        raise ValueError(
            "stored log probabilities/values disagree with the collection checkpoint"
        )
    # No advantage normalization: singleton/small batches keep their learning signal.
    return Batch(
        x,
        action,
        old_log,
        target,
        target - old_value,
        reward,
        tuple(lengths),
        policy.version,
    )


def clipped_policy_loss(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    """PPO clipped objective; evaluate the stored Gaussian latent action."""
    ratio = (log_prob - old_log_prob).exp()
    return -torch.minimum(
        ratio * advantages,
        ratio.clamp(1 - clip, 1 + clip) * advantages,
    ).mean()


def update(
    policy: MLPController, batch: Batch, config: PPOConfig
) -> tuple[ActorCritic, dict]:
    """Update a copy using full-batch PPO epochs; leave collection policy frozen.

    This is one batch update, not a persistent training scheduler. Adam starts
    fresh for this invocation. Normalization buffers remain unchanged.
    """
    if batch.policy_version != policy.version:
        raise ValueError("batch belongs to a different collection policy")
    model = copy.deepcopy(policy.model).train().requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    metrics = {}
    steps = 0
    for _ in range(config.epochs):
        distribution, value = model(batch.observations)
        log_prob = distribution.log_prob(batch.actions).sum(-1)
        log_ratio = log_prob - batch.old_log_prob
        approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean()
        if not torch.isfinite(approx_kl):
            raise ValueError("non-finite PPO probability ratio")
        if float(approx_kl.detach()) > config.target_kl:
            break
        actor_loss = clipped_policy_loss(
            log_prob, batch.old_log_prob, batch.advantages, config.clip
        )
        value_loss = (value - batch.returns).square().mean()
        loss = actor_loss + config.value_weight * value_loss
        if not torch.isfinite(loss):
            raise ValueError("non-finite PPO loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        steps += 1
        metrics = {
            "loss_before_last_step": float(loss.detach()),
            "actor_loss_before_last_step": float(actor_loss.detach()),
            "value_loss_before_last_step": float(value_loss.detach()),
            "gradient_norm_before_clip": float(grad_norm),
        }
    with torch.no_grad():
        distribution, value = model(batch.observations)
        log_ratio = distribution.log_prob(batch.actions).sum(-1) - batch.old_log_prob
        kl = ((log_ratio.exp() - 1) - log_ratio).mean()
        if not torch.isfinite(kl) or not torch.isfinite(value).all():
            raise ValueError("non-finite PPO result")
        metrics.update(
            updates=steps,
            samples=len(batch.actions),
            episodes=len(batch.episode_lengths),
            reward_sum=float(batch.rewards.sum()),
            approx_kl=float(kl),
            clip_fraction=float(
                ((log_ratio.exp() - 1).abs() > config.clip).float().mean()
            ),
            value_mse=float((value - batch.returns).square().mean()),
        )
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("non-finite updated model weights")
    return model.eval(), metrics


def main() -> None:
    """Run one PPO update on explicit complete trajectories; never launch tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-scale", type=float, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new checkpoint path")
    try:
        config = PPOConfig(args.token_scale, args.beta, args.learning_rate, args.epochs)
        policy = MLPController(args.checkpoint)
        trajectories = [json.loads(p.read_text()) for p in args.trajectories]
        batch = prepare_batch(trajectories, policy, config)
        model, metrics = update(policy, batch, config)
        save_checkpoint(model, args.output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {"parent_policy": policy.version, "config": asdict(config), **metrics}
        )
    )


if __name__ == "__main__":
    main()
