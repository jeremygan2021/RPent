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

"""SAC for four context-retention thresholds; robot actions are unchanged.

Twin-Q and entropy-regularized backups follow Haarnoja et al. (2018).
The existing latent Gaussian Actor is squashed with sigmoid into NC in (0,1).
Entropy and Q operate in NC space, not latent space.
"""

import copy
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch.nn import functional as F

from .iql import Transitions, dataset_from_trajectories, load_transitions, value_network
from .observation import FEATURE_VERSION, SCORING_VERSION
from .policy import MLPController, feature_names, save_checkpoint
from .reward import ComponentRewardConfig, account, component_reward
from .training import PPOConfig, rewards_and_returns


@dataclass(frozen=True)
class SACConfig:
    learning_rate: float = 3e-4
    alpha: float = 0.05
    gamma: float = 1.0
    target_rate: float = 0.005
    batch_size: int = 64
    capacity: int = 100000
    critic_warmup_steps: int = 0
    actor_update_interval: int = 1

    def __post_init__(self):
        vals = asdict(self)
        if any(
            type(v) not in (int, float) or not math.isfinite(v) for v in vals.values()
        ):
            raise ValueError("SAC configuration must be finite")
        if not (
            self.learning_rate > 0
            and self.alpha >= 0
            and 0 <= self.gamma <= 1
            and 0 < self.target_rate <= 1
        ):
            raise ValueError("invalid SAC configuration")
        if (
            type(self.batch_size) is not int
            or type(self.capacity) is not int
            or min(self.batch_size, self.capacity) < 1
        ):
            raise ValueError("batch size and capacity must be positive integers")
        if (
            type(self.critic_warmup_steps) is not int
            or self.critic_warmup_steps < 0
            or type(self.actor_update_interval) is not int
            or self.actor_update_interval < 1
        ):
            raise ValueError(
                "warmup must be a nonnegative integer and Actor interval positive"
            )


def sample_nc(actor, states):
    """Reparameterized sample and sigmoid-Jacobian corrected log density."""
    distribution, _ = actor(states)
    latent = distribution.rsample()
    action = latent.sigmoid()
    log_jacobian = F.logsigmoid(latent) + F.logsigmoid(-latent)
    log_prob = (distribution.log_prob(latent) - log_jacobian).sum(-1)
    return action, log_prob


def soft_target(reward, done, next_q, next_log_prob, config):
    """No value or entropy bootstrap at terminal session boundaries."""
    return reward + config.gamma * (1 - done) * (next_q - config.alpha * next_log_prob)


def transitions_from_trajectory(
    trajectory: dict,
    reward_config: PPOConfig | ComponentRewardConfig,
    *,
    boundary: str = "agent",
    requests: dict[str, dict] | None = None,
) -> Transitions:
    """Validate the full audit, optionally terminate training at the native verdict.

    Task mode charges whole requests started no later than task_end_at. It keeps
    raw audit records intact and never accepts incomplete post-task evidence.
    """
    if isinstance(reward_config, ComponentRewardConfig):
        return component_transitions(
            trajectory, reward_config, requests, boundary=boundary
        )
    if boundary not in {"agent", "task"}:
        raise ValueError("boundary must be agent or task")
    token_config = PPOConfig(
        token_scale=reward_config.token_scale, beta=reward_config.beta
    )
    data = dataset_from_trajectories([trajectory], token_config)
    transitions = load_transitions(data)
    if boundary == "task":
        timing = trajectory["outcome"].get("timing") or {}
        start, end, duration = (
            timing.get(k) for k in ("started_at", "task_end_at", "task_duration_s")
        )
        if (
            any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (start, end, duration)
            )
            or duration < 0
            or end < start
            or not math.isclose(end - start, duration, abs_tol=0.05)
        ):
            raise ValueError("consistent native task timestamps required")
        costs = []
        previous = start
        task_finished = False
        for row in trajectory["decisions"]:
            cost = 0
            active = False
            for attempt in row["attempts"]:
                began = (attempt.get("timing") or {}).get("started_at")
                if (
                    type(began) not in (int, float)
                    or not math.isfinite(began)
                    or began < previous
                ):
                    raise ValueError("ordered request start timestamps required")
                previous = began
                if began <= end:
                    if task_finished:
                        raise ValueError("task decisions must form a prefix")
                    cost += attempt["usage"]["input_tokens"]
                    active = True
                else:
                    task_finished = True
            if active:
                costs.append(cost)
        if not costs:
            raise ValueError("no decisions before the native task verdict")
        rewards, _ = rewards_and_returns(
            costs, trajectory["outcome"]["environment_success"], token_config
        )
        n = len(costs)
        next_states = transitions.next_states[:n].clone()
        next_states[-1].zero_()
        done = torch.zeros(n, dtype=transitions.done.dtype)
        done[-1] = 1
        transitions = Transitions(
            transitions.states[:n],
            transitions.actions[:n],
            torch.tensor(rewards, dtype=transitions.rewards.dtype),
            next_states,
            done,
        )
    if reward_config.latency_beta:
        duration = (trajectory["outcome"].get("timing") or {}).get("task_duration_s")
        if (
            type(duration) not in (int, float)
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("valid task_duration_s required")
        transitions.rewards[-1] -= (
            reward_config.latency_beta * duration / reward_config.latency_scale
        )
    return transitions


def component_transitions(trajectory, config, requests, *, boundary):
    """Use audited executed NC and outgoing payloads; no post-verdict costs.

    All requests, including post-task attempts, require attribution evidence.
    Reuse native audit/state/action/boundary validation before replacing rewards.
    """
    if boundary != "task":
        raise ValueError("success component reward requires task boundary")
    if requests is None:
        raise ValueError("outgoing request payloads required for component reward")
    data = transitions_from_trajectory(
        trajectory, PPOConfig(1000000, beta=0), boundary="task"
    )
    end = trajectory["outcome"]["timing"]["task_end_at"]
    duration = trajectory["outcome"]["timing"]["task_duration_s"]
    rewards = []
    totals = {}
    request_ids = set()
    for row in trajectory["decisions"]:
        row_cost = 0.0
        active = False
        for attempt in row["attempts"]:
            rid = attempt["request_id"]
            if rid in request_ids or rid not in requests:
                raise ValueError("missing or duplicate outgoing request payload")
            request_ids.add(rid)
            groups = account(requests[rid], attempt["usage"])
            if attempt["timing"]["started_at"] <= end:
                active = True
                row_cost += config.token_cost(groups)
                for name, value in groups.items():
                    totals[name] = totals.get(name, 0.0) + value
        if active:
            rewards.append(-row_cost)
    success = trajectory["outcome"]["environment_success"]
    rewards[-1] += config.terminal(duration, success)
    expected = component_reward(totals, duration, config, success=success)["reward"]
    if len(rewards) != len(data.rewards) or not math.isclose(
        sum(rewards), expected, abs_tol=1e-8
    ):
        raise ValueError("component reward disagrees with episode total")
    return Transitions(
        data.states,
        data.actions,
        torch.tensor(rewards, dtype=data.rewards.dtype),
        data.next_states,
        data.done,
    )


class SACTrainer:
    """Persistent Actor/Q optimizers, target networks, and bounded replay."""

    def __init__(
        self,
        policy: MLPController,
        config: SACConfig = SACConfig(),
        seed: int = 0,
        *,
        boundary: str = "agent",
    ) -> None:
        if policy.legacy:
            raise ValueError("SAC requires a four-dimensional policy")
        if boundary not in {"agent", "task"}:
            raise ValueError("boundary must be agent or task")
        self.boundary = boundary
        self.config = config
        self.actor = copy.deepcopy(policy.model).train().requires_grad_(True)
        # Existing deployment artifact includes a V head; SAC does not train it.
        self.actor.critic.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.q1 = value_network(len(feature_names()) + 4)
            self.q2 = value_network(len(feature_names()) + 4)
        self.target1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target2 = copy.deepcopy(self.q2).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            [p for p in self.actor.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=config.learning_rate,
        )
        self.generator = torch.Generator().manual_seed(seed)
        self.replay = None
        self.episode_ids = set()
        self.steps = 0
        self.reward_spec = None

    def add(self, trajectory, reward_config=None, *, requests=None):
        """Add an audited episode using success component reward by default.

        New callers use boundary="task" and supply actual outgoing payloads.
        Passing PPOConfig explicitly retains legacy total-input behavior.
        Replay never mixes reward definitions or fixed reference scales.
        """
        reward_config = reward_config or ComponentRewardConfig()
        if isinstance(reward_config, ComponentRewardConfig) and self.config.gamma != 1:
            raise ValueError("component episode reward requires gamma=1")
        spec = (
            reward_config.spec()
            if isinstance(reward_config, ComponentRewardConfig)
            else {"version": "legacy-total-input-v1", **asdict(reward_config)}
        )
        if self.reward_spec is not None and self.reward_spec != spec:
            raise ValueError("SAC replay reward configuration mismatch")
        episode_id = trajectory.get("episode_id")
        if episode_id in self.episode_ids:
            raise ValueError("duplicate replay episode")
        t = transitions_from_trajectory(
            trajectory, reward_config, boundary=self.boundary, requests=requests
        )
        if self.replay is None:
            self.replay = t
        else:
            self.replay = Transitions(
                *(
                    torch.cat((getattr(self.replay, k.name), getattr(t, k.name)))[
                        -self.config.capacity :
                    ]
                    for k in fields(Transitions)
                )
            )
        if len(self.replay.rewards) > self.config.capacity:
            self.replay = Transitions(
                *(
                    getattr(self.replay, k.name)[-self.config.capacity :]
                    for k in fields(Transitions)
                )
            )
        self.episode_ids.add(episode_id)
        self.reward_spec = spec

    def add_episode(self, output: Path, reward_config=None):
        """Load the completed worker audit and its actual outgoing requests."""
        import json

        output = Path(output)
        trajectory = json.loads((output / "trajectory.json").read_text())
        requests = {
            a["request_id"]: json.loads(
                (output / "requests" / (a["request_id"] + ".after.json")).read_text()
            )
            for row in trajectory["decisions"]
            for a in row["attempts"]
        }
        self.add(trajectory, reward_config, requests=requests)

    def update(self):
        if self.replay is None:
            raise ValueError("empty replay")
        ids = torch.randint(
            len(self.replay.rewards),
            (self.config.batch_size,),
            generator=self.generator,
        )
        state, action, reward, next_state, done = (
            getattr(self.replay, k.name)[ids] for k in fields(Transitions)
        )
        with torch.no_grad():
            next_action, next_log = sample_nc(self.actor, next_state)
            next_input = torch.cat((next_state, next_action), -1)
            next_q = torch.minimum(
                self.target1(next_input), self.target2(next_input)
            ).squeeze(-1)
            target = soft_target(reward, done, next_q, next_log, self.config)
        inputs = torch.cat((state, action), -1)
        q_loss = F.mse_loss(self.q1(inputs).squeeze(-1), target) + F.mse_loss(
            self.q2(inputs).squeeze(-1), target
        )
        if not torch.isfinite(q_loss):
            raise ValueError("non-finite Q loss")
        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_grad = torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            10,
            error_if_nonfinite=True,
        )
        self.q_optimizer.step()
        actor_updated = (
            self.steps >= self.config.critic_warmup_steps
            and (self.steps - self.config.critic_warmup_steps + 1)
            % self.config.actor_update_interval
            == 0
        )
        actor_grad = 0.0
        for q in (self.q1, self.q2):
            q.requires_grad_(False)
        try:
            with torch.set_grad_enabled(actor_updated):
                sampled, log_prob = sample_nc(self.actor, state)
                inputs = torch.cat((state, sampled), -1)
                q = torch.minimum(self.q1(inputs), self.q2(inputs)).squeeze(-1)
                actor_loss = (self.config.alpha * log_prob - q).mean()
                if not torch.isfinite(actor_loss):
                    raise ValueError("non-finite Actor loss")
            if actor_updated:
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.actor.parameters() if p.requires_grad],
                    10,
                    error_if_nonfinite=True,
                )
                self.actor_optimizer.step()
        finally:
            for q in (self.q1, self.q2):
                q.requires_grad_(True)
        with torch.no_grad():
            for target_q, online_q in [
                (self.target1, self.q1),
                (self.target2, self.q2),
            ]:
                for target_p, online_p in zip(
                    target_q.parameters(), online_q.parameters()
                ):
                    target_p.lerp_(online_p, self.config.target_rate)
        self.steps += 1
        return {
            "step": self.steps,
            "actor_updated": actor_updated,
            "q_loss": float(q_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "q_grad": float(q_grad),
            "actor_grad": float(actor_grad),
            "entropy": float(-log_prob.detach().mean()),
            "replay_size": len(self.replay.rewards),
        }

    def save(self, path):
        """Training checkpoint includes optimizers and replay; Actor export is separate."""
        payload = {
            "config": asdict(self.config),
            "boundary": self.boundary,
            "steps": self.steps,
            "reward_spec": self.reward_spec,
            "scoring_version": SCORING_VERSION,
            "feature_version": FEATURE_VERSION,
            "episode_ids": sorted(self.episode_ids),
            "replay": None
            if self.replay is None
            else {k.name: getattr(self.replay, k.name) for k in fields(Transitions)},
            "generator": self.generator.get_state(),
            "rng": torch.get_rng_state(),
        }
        for name in (
            "actor",
            "q1",
            "q2",
            "target1",
            "target2",
            "actor_optimizer",
            "q_optimizer",
        ):
            payload[name] = getattr(self, name).state_dict()
        torch.save(payload, path)

    def restore(self, path):
        """Resume only a training state with matching hyperparameters."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("scoring_version") != SCORING_VERSION
            or payload.get("feature_version") != FEATURE_VERSION
        ):
            raise ValueError("SAC rule/feature version mismatch")
        if payload.get("boundary", "agent") != self.boundary:
            raise ValueError("SAC replay boundary mismatch")
        if SACConfig(**payload["config"]) != self.config:
            raise ValueError("SAC configuration mismatch")
        spec = payload.get("reward_spec")
        if payload.get("replay") is not None and spec is None:
            raise ValueError("SAC replay lacks reward metadata; cannot safely restore")
        if self.reward_spec is not None and self.reward_spec != spec:
            raise ValueError("SAC replay reward configuration mismatch")
        for name in (
            "actor",
            "q1",
            "q2",
            "target1",
            "target2",
            "actor_optimizer",
            "q_optimizer",
        ):
            getattr(self, name).load_state_dict(payload[name])
        self.replay = (
            None if payload["replay"] is None else Transitions(**payload["replay"])
        )
        self.reward_spec = spec
        self.episode_ids = set(payload["episode_ids"])
        self.steps = payload["steps"]
        self.generator.set_state(payload["generator"])
        torch.set_rng_state(payload["rng"])

    def export(self, path: Path):
        save_checkpoint(self.actor, path)
