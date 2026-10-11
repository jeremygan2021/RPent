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

"""Offline IQL over fixed context transitions; no online task collection.

Equations follow Kostrikov et al., https://arxiv.org/abs/2110.06169.
Q and V are separate from the deployable Actor to avoid shared-gradient coupling.
"""

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .observation import FEATURE_VERSION, SCORING_VERSION
from .policy import MLPController, feature_names, save_checkpoint
from .training import PPOConfig, rewards_and_returns


@dataclass(frozen=True)
class Transitions:
    """Offline tensors; actions are four rho values, not robot commands."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    done: torch.Tensor


def load_transitions(dataset: dict) -> Transitions:
    """Validate schemas, rewards and episode boundaries before tensor conversion."""
    if (
        dataset.get("schema_version") != 1
        or dataset.get("feature_version") != FEATURE_VERSION
        or dataset.get("scoring_version") != SCORING_VERSION
        or dataset.get("feature_names") != list(feature_names())
    ):
        raise ValueError("offline feature/scoring schema mismatch")
    reward = dataset["reward"]
    if reward.get("gamma") != 1.0 or reward.get("boundary") != "complete_agent_session":
        raise ValueError("unsupported reward/terminal boundary")
    config = PPOConfig(reward["token_scale"], reward["beta"])
    rows, seen = [], set()
    for episode in dataset["episodes"]:
        if not episode.get("episode_id") or episode["episode_id"] in seen:
            raise ValueError("missing or duplicate episode")
        seen.add(episode["episode_id"])
        transitions = episode["transitions"]
        expected, _ = rewards_and_returns(
            [t["input_tokens"] for t in transitions],
            episode["environment_success"],
            config,
        )
        for i, t in enumerate(transitions):
            terminal = i == len(transitions) - 1
            if type(t["done"]) is not bool or t["done"] != terminal:
                raise ValueError("invalid terminal boundary")
            if not math.isclose(t["reward"], expected[i], abs_tol=1e-9, rel_tol=1e-9):
                raise ValueError("reward disagrees with actual token cost and outcome")
            if terminal:
                if t["next_state"] != [0.0] * len(feature_names()):
                    raise ValueError(
                        "terminal next_state must be the absorbing zero vector"
                    )
            elif t["next_state"] != transitions[i + 1]["state"]:
                raise ValueError("next_state crosses or skips a decision boundary")
            rows.append(t)
    if not rows:
        raise ValueError("empty offline dataset")
    tensors = [
        torch.tensor([r[k] for r in rows], dtype=torch.float32)
        for k in ("state", "action", "reward", "next_state", "done")
    ]
    n, size = len(rows), len(feature_names())
    if [tuple(t.shape) for t in tensors] != [(n, size), (n, 4), (n,), (n, size), (n,)]:
        raise ValueError("invalid offline tensor shapes")
    if any(not torch.isfinite(t).all() for t in tensors):
        raise ValueError("offline data must be finite")
    if not ((tensors[1] > 0) & (tensors[1] < 1)).all():
        raise ValueError("IQL rho actions must be strictly between 0 and 1")
    return Transitions(*tensors)


def dataset_from_trajectories(trajectories: list[dict], config: PPOConfig) -> dict:
    """Convert complete modern audits, including genuine environment failures."""
    episodes, response_ids = [], set()
    for trajectory in trajectories:
        if (
            trajectory.get("schema_version") != 1
            or trajectory.get("audit_complete") is not True
            or trajectory.get("issues")
        ):
            raise ValueError("incomplete trajectory audit")
        outcome = trajectory.get("outcome") or {}
        success = outcome.get("environment_success")
        if type(success) is not bool or outcome.get("agent_error"):
            raise ValueError("missing environment verdict or agent error")
        transitions, decision_ids = [], set()
        for row in trajectory["decisions"]:
            decision = row["decision"]
            observation = decision["observation"]
            if (
                decision["id"] in decision_ids
                or decision.get("scoring_version") != SCORING_VERSION
                or observation.get("version") != FEATURE_VERSION
                or observation.get("names") != list(feature_names())
            ):
                raise ValueError("duplicate decision or feature/scoring mismatch")
            decision_ids.add(decision["id"])
            attempts = row["attempts"]
            if not attempts:
                raise ValueError("decision has no HTTP attempts")
            cost = 0
            for attempt in attempts:
                timing = attempt.get("timing") or {}
                count = (attempt.get("usage") or {}).get("input_tokens")
                response_id = attempt.get("response_id")
                if (
                    timing.get("status") != 200
                    or timing.get("transport_error")
                    or type(count) is not int
                    or count < 0
                    or not response_id
                    or response_id in response_ids
                ):
                    raise ValueError("invalid HTTP cost or duplicate response")
                response_ids.add(response_id)
                cost += count
            transitions.append(
                {
                    "state": observation["values"],
                    "action": [
                        decision["thresholds"][m]
                        for m in ("image", "text", "action", "memory")
                    ],
                    "input_tokens": cost,
                }
            )
        rewards, _ = rewards_and_returns(
            [r["input_tokens"] for r in transitions], success, config
        )
        for i, row in enumerate(transitions):
            terminal = i == len(transitions) - 1
            row.update(
                reward=rewards[i],
                done=terminal,
                next_state=[0.0] * len(feature_names())
                if terminal
                else transitions[i + 1]["state"],
            )
        episodes.append(
            {
                "episode_id": trajectory["episode_id"],
                "environment_success": success,
                "transitions": transitions,
            }
        )
    dataset = {
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "scoring_version": SCORING_VERSION,
        "feature_names": list(feature_names()),
        "reward": {
            "beta": config.beta,
            "token_scale": config.token_scale,
            "gamma": 1.0,
            "boundary": "complete_agent_session",
        },
        "episodes": episodes,
    }
    load_transitions(dataset)
    return dataset


@dataclass(frozen=True)
class IQLConfig:
    """Defaults for a small validation run, not tuned benchmark settings."""

    learning_rate: float = 3e-4
    expectile: float = 0.7
    inverse_temperature: float = 3.0
    max_weight: float = 100.0
    target_rate: float = 0.005

    def __post_init__(self) -> None:
        if any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in asdict(self).values()
        ):
            raise ValueError("IQL configuration must be finite")
        if not (
            self.learning_rate > 0
            and 0.5 < self.expectile < 1
            and self.inverse_temperature >= 0
            and self.max_weight >= 1
            and 0 < self.target_rate <= 1
        ):
            raise ValueError("invalid IQL configuration")


def expectile_loss(difference: torch.Tensor, expectile: float) -> torch.Tensor:
    """Asymmetric squared error for target Q minus V."""
    weight = torch.where(difference > 0, expectile, 1 - expectile)
    return (weight * difference.square()).mean()


def td_target(
    reward: torch.Tensor, done: torch.Tensor, next_value: torch.Tensor
) -> torch.Tensor:
    """Undiscounted finite-episode backup; never bootstrap terminal states."""
    return reward + (1 - done) * next_value


def value_network(inputs: int) -> nn.Sequential:
    """A small independent scalar MLP matching the Actor's hidden size."""
    return nn.Sequential(
        nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
    )


class IQLTrainer:
    """Own offline Q/V optimizers and a copy of the existing deployable Actor."""

    def __init__(self, policy: MLPController, config: IQLConfig, seed: int = 0) -> None:
        if policy.legacy:
            raise ValueError(
                "four-dimensional IQL requires a new memory-enabled checkpoint"
            )
        self.config = config
        self.actor = copy.deepcopy(policy.model).requires_grad_(True)
        # This shared-body PPO value head is not IQL's independent V network.
        # Export a zero baseline for future PPO, never mislabel it as trained V.
        with torch.no_grad():
            self.actor.critic.weight.zero_()
            self.actor.critic.bias.zero_()
        self.actor.critic.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.q = nn.ModuleList(
                [value_network(len(feature_names()) + 4) for _ in range(2)]
            )
            self.v = value_network(len(feature_names()))
        self.target_q = copy.deepcopy(self.q).requires_grad_(False)
        self.q_optimizer = torch.optim.Adam(
            self.q.parameters(), lr=config.learning_rate
        )
        self.v_optimizer = torch.optim.Adam(
            self.v.parameters(), lr=config.learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            [p for p in self.actor.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )
        self.generator = torch.Generator().manual_seed(seed)

    @staticmethod
    def _step(loss: torch.Tensor, optimizer: torch.optim.Optimizer, parameters) -> None:
        if not torch.isfinite(loss):
            raise ValueError("non-finite IQL loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True)
        optimizer.step()

    def update(self, data: Transitions, batch_size: int = 64) -> dict:
        """One minibatch update: expectile V, double Q, weighted Actor, target Q."""
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        indices = torch.randint(
            len(data.states), (batch_size,), generator=self.generator
        )
        state, action, reward, next_state, done = [
            t[indices]
            for t in (
                data.states,
                data.actions,
                data.rewards,
                data.next_states,
                data.done,
            )
        ]

        def normalize(x):
            return (x - self.actor.feature_mean) / self.actor.feature_scale

        x, nx = normalize(state), normalize(next_state)
        qa = torch.cat((x, action), dim=-1)
        with torch.no_grad():
            target_q = torch.minimum(*(q(qa).squeeze(-1) for q in self.target_q))
        v_loss = expectile_loss(target_q - self.v(x).squeeze(-1), self.config.expectile)
        self._step(v_loss, self.v_optimizer, self.v.parameters())
        with torch.no_grad():
            target = td_target(reward, done, self.v(nx).squeeze(-1))
            advantage = target_q - self.v(x).squeeze(-1)
            weights = (
                (self.config.inverse_temperature * advantage)
                .clamp(max=math.log(self.config.max_weight))
                .exp()
            )
        q_loss = sum((q(qa).squeeze(-1) - target).square().mean() for q in self.q)
        self._step(q_loss, self.q_optimizer, self.q.parameters())
        distribution, _ = self.actor(state)
        # The logit transform's Jacobian is constant with respect to Actor weights.
        actor_loss = -(
            weights * distribution.log_prob(torch.logit(action)).sum(-1)
        ).mean()
        self._step(
            actor_loss,
            self.actor_optimizer,
            [p for p in self.actor.parameters() if p.requires_grad],
        )
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_q.parameters(), self.q.parameters()
            ):
                target_parameter.lerp_(parameter, self.config.target_rate)
        return {
            "value_loss": float(v_loss.detach()),
            "q_loss": float(q_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "mean_weight": float(weights.mean()),
            "max_weight": float(weights.max()),
        }


def main() -> None:
    """Run an explicit bounded offline validation and export the Actor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or args.updates <= 0 or args.batch_size <= 0:
        parser.error("choose a fresh output and positive updates/batch size")
    try:
        torch.set_num_threads(1)
        raw = args.dataset.read_bytes()
        dataset = json.loads(raw)
        data = load_transitions(dataset)
        policy = MLPController(args.checkpoint)
        config = IQLConfig()
        trainer = IQLTrainer(policy, config, args.seed)
        for _ in range(args.updates):
            metrics = trainer.update(data, args.batch_size)
        if any(
            not torch.isfinite(t).all() for t in trainer.actor.state_dict().values()
        ):
            raise ValueError("non-finite Actor weights")
        save_checkpoint(trainer.actor, args.output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                "parent_policy": policy.version,
                "config": asdict(config),
                "updates": args.updates,
                "transitions": len(data.states),
                "episodes": len(dataset["episodes"]),
                "metrics": metrics,
                "scope": "offline update validation only; no task-performance evaluation",
                "exported_ppo_value_head": "zero baseline; independent IQL V is not exported",
            }
        )
    )


if __name__ == "__main__":
    main()
