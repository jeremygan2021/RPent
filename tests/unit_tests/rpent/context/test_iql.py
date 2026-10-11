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

"""Offline IQL math and deployment contracts, with controlled trajectories."""

import copy
import dataclasses

import pytest

from rpent.context.observation import FEATURE_VERSION, SCORING_VERSION
from tests.unit_tests.rpent.context.test_policy import checkpoint, observation

torch = pytest.importorskip("torch")
from rpent.context.iql import (  # noqa: E402
    IQLConfig,
    IQLTrainer,
    expectile_loss,
    load_transitions,
    td_target,
)
from rpent.context.policy import (  # noqa: E402
    MLPController,
    feature_names,
    save_checkpoint,
)


def dataset():
    state = list(observation().values)
    next_state = state.copy()
    next_state[0] = 1.0
    return {
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "scoring_version": SCORING_VERSION,
        "feature_names": list(feature_names()),
        "reward": {
            "beta": 0.1,
            "token_scale": 1000,
            "gamma": 1.0,
            "boundary": "complete_agent_session",
        },
        "episodes": [
            {
                "episode_id": "controlled",
                "environment_success": True,
                "transitions": [
                    {
                        "state": state,
                        "action": [0.2, 0.1, 0.1, 0.5],
                        "input_tokens": 100,
                        "reward": -0.01,
                        "next_state": next_state.copy(),
                        "done": False,
                    },
                    {
                        "state": next_state,
                        "action": [0.4, 0.2, 0.2, 0.7],
                        "input_tokens": 200,
                        "reward": 0.98,
                        "next_state": [0.0] * len(state),
                        "done": True,
                    },
                ],
            }
        ],
    }


def test_expectile_and_terminal_backup_match_hand_calculation():
    diff = torch.tensor([-2.0, 1.0], requires_grad=True)
    loss = expectile_loss(diff, 0.7)
    assert loss.item() == pytest.approx((0.3 * 4 + 0.7) / 2)
    loss.backward()
    assert diff.grad.tolist() == pytest.approx([-0.6, 0.7])
    target = td_target(
        torch.tensor([-0.1, 1.0]), torch.tensor([0.0, 1.0]), torch.tensor([0.8, 99.0])
    )
    assert target.tolist() == pytest.approx([0.7, 1.0])


@pytest.mark.parametrize(
    "fault", ["boundary", "next", "reward", "action", "schema", "duplicate"]
)
def test_invalid_offline_dataset_is_rejected(fault):
    data = dataset()
    row = data["episodes"][0]["transitions"][0]
    if fault == "boundary":
        row["done"] = True
    elif fault == "next":
        row["next_state"][0] = 8
    elif fault == "reward":
        row["reward"] = 1
    elif fault == "action":
        row["action"][0] = 1
    elif fault == "schema":
        data["feature_names"].reverse()
    else:
        data["episodes"].append(copy.deepcopy(data["episodes"][0]))
    with pytest.raises(ValueError):
        load_transitions(data)


def test_iql_updates_independent_networks_and_exports_actor(tmp_path):
    policy = MLPController(checkpoint(tmp_path))
    rng = torch.random.get_rng_state().clone()
    trainer = IQLTrainer(policy, IQLConfig(), seed=4)
    assert torch.equal(rng, torch.random.get_rng_state())
    old_q = [x.clone() for x in trainer.q.parameters()]
    old_v = [x.clone() for x in trainer.v.parameters()]
    old_actor = [x.clone() for x in trainer.actor.parameters()]
    old_target = [x.clone() for x in trainer.target_q.parameters()]
    metrics = trainer.update(load_transitions(dataset()), 16)
    for old, module in (
        (old_q, trainer.q),
        (old_v, trainer.v),
        (old_actor, trainer.actor),
    ):
        assert any(not torch.equal(a, b) for a, b in zip(old, module.parameters()))
    for old, current, target in zip(
        old_target, trainer.q.parameters(), trainer.target_q.parameters()
    ):
        assert torch.allclose(target, old * 0.995 + current * 0.005)
        assert not target.requires_grad
    assert metrics["max_weight"] <= 100
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert policy.decide(observation()).thresholds.image == pytest.approx(0.2)
    path = tmp_path / "offline.pt"
    save_checkpoint(trainer.actor, path)
    restored = MLPController(path)
    assert restored.decide(observation()).value == 0
    assert all(
        0 < v < 1
        for v in dataclasses.astuple(restored.decide(observation()).thresholds)
    )


def test_modern_audit_conversion_accepts_verified_failure_and_rejects_bad_costs():
    from rpent.context.iql import dataset_from_trajectories
    from rpent.context.training import PPOConfig

    observation_data = {
        "version": FEATURE_VERSION,
        "names": list(feature_names()),
        "values": list(observation().values),
    }
    trajectory = {
        "schema_version": 1,
        "audit_complete": True,
        "issues": [],
        "episode_id": "failed-task",
        "outcome": {"environment_success": False, "agent_error": None},
        "decisions": [
            {
                "decision": {
                    "id": 1,
                    "observation": observation_data,
                    "scoring_version": SCORING_VERSION,
                    "thresholds": {
                        "image": 0.2,
                        "text": 0.1,
                        "action": 0.1,
                        "memory": 0.5,
                    },
                },
                "attempts": [
                    {
                        "response_id": "r1",
                        "timing": {"status": 200},
                        "usage": {"input_tokens": 100},
                    }
                ],
            }
        ],
    }
    converted = dataset_from_trajectories([trajectory], PPOConfig(1000, 0.1))
    row = converted["episodes"][0]["transitions"][0]
    assert row["reward"] == pytest.approx(-0.01)
    assert row["done"]
    assert row["next_state"] == [0.0] * len(feature_names())
    with pytest.raises(ValueError):
        dataset_from_trajectories([trajectory, trajectory], PPOConfig(1000))
    trajectory["decisions"][0]["attempts"][0]["usage"] = {}
    with pytest.raises(ValueError):
        dataset_from_trajectories([trajectory], PPOConfig(1000))


def test_iql_then_ppo_updates_memory_and_exports_four_thresholds(tmp_path):
    from rpent.context.training import PPOConfig, prepare_batch, update
    from tests.unit_tests.rpent.context.test_training import episode

    base = MLPController(checkpoint(tmp_path))
    trainer = IQLTrainer(base, IQLConfig(), seed=7)
    before = trainer.actor.actor.bias[3].detach().clone()
    for _ in range(5):
        trainer.update(load_transitions(dataset()), 16)
    assert trainer.actor.actor.bias[3] != before
    offline = tmp_path / "iql.pt"
    save_checkpoint(trainer.actor, offline)
    policy = MLPController(offline, sample=True, seed=9)
    trajectories = [
        episode(policy, episode_id=f"e{i}", costs=(100,), success=i % 2 == 0)
        for i in range(4)
    ]
    config = PPOConfig(1000, epochs=2)
    batch = prepare_batch(trajectories, policy, config)
    assert batch.actions.shape == (4, 4)
    model, metrics = update(policy, batch, config)
    assert model.actor.bias[3] != policy.model.actor.bias[3]
    final = tmp_path / "iql-ppo.pt"
    save_checkpoint(model, final)
    decision = MLPController(final).decide(observation())
    assert 0 < decision.thresholds.memory < 1
    assert metrics["updates"] > 0
