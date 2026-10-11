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

"""Native SAC consumes audited relative component costs without reward mixing."""

import copy
import json
import math
from dataclasses import replace

import pytest
import torch

from rpent.context.reward import ComponentRewardConfig, account, component_reward
from rpent.context.sac import SACConfig, SACTrainer, transitions_from_trajectory
from rpent.context.training import PPOConfig
from tests.unit_tests.rpent.context.test_sac import setup_policy, task_episode
from tests.unit_tests.rpent.context.test_toolrules_pose import history, usage_for


def audited(policy, *, success=True):
    trajectory = task_episode(policy, success=success)
    requests = {}
    for row in trajectory["decisions"]:
        attempt = row["attempts"][0]
        request = history()
        request["input"].extend(
            [
                {
                    "id": "rules",
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "robot rules"}],
                },
                {"id": "thought", "type": "reasoning", "encrypted_content": "opaque"},
                {
                    "id": "plan",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "move to basket"}],
                },
                {
                    "id": "photo",
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AA==",
                        }
                    ],
                },
                {
                    "id": "memory",
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "task [RPENT_MEMORY]remember basket[/RPENT_MEMORY] rule",
                        }
                    ],
                },
            ]
        )
        usage = usage_for(request)
        usage["output_tokens"] = 10
        usage["input_tokens_details"] = {"cached_tokens": usage["input_tokens"] // 2}
        attempt["usage"] = usage
        requests[attempt["request_id"]] = request
    return trajectory, requests


def task_costs(trajectory, requests):
    totals = {}
    end = trajectory["outcome"]["timing"]["task_end_at"]
    for row in trajectory["decisions"]:
        for attempt in row["attempts"]:
            if attempt["timing"]["started_at"] <= end:
                for k, v in account(
                    requests[attempt["request_id"]], attempt["usage"]
                ).items():
                    totals[k] = totals.get(k, 0) + v
    return totals


@pytest.mark.parametrize("success", [True, False])
def test_native_add_matches_episode_formula_and_executed_nc(tmp_path, success):
    policy = setup_policy(tmp_path)
    trajectory, requests = audited(policy, success=success)
    config = ComponentRewardConfig()
    trajectory["decisions"][0]["decision"]["behavior_policy"] = (
        "lower-relative-0.2-0.6-v1"
    )
    trajectory["decisions"][0]["decision"]["thresholds"] = {
        "image": 0.18,
        "text": 0.2,
        "action": 0.3,
        "memory": 0.4,
    }
    original = copy.deepcopy(trajectory)
    trainer = SACTrainer(policy, SACConfig(batch_size=4), boundary="task")
    trainer.add(trajectory, config, requests=requests)
    expected = component_reward(
        task_costs(trajectory, requests), 15.0, config, success=success
    )["reward"]
    assert trainer.replay.rewards.sum().item() == pytest.approx(expected, abs=2e-6)
    assert trainer.replay.actions[0].tolist() == pytest.approx([0.18, 0.2, 0.3, 0.4])
    assert trainer.replay.done.tolist() == [0.0, 1.0]
    assert trainer.replay.next_states[-1].eq(0).all()
    assert trajectory == original
    assert math.isfinite(trainer.update()["q_loss"])


def test_default_add_uses_components_and_fixed_costs_are_excluded(tmp_path):
    policy = setup_policy(tmp_path)
    trajectory, requests = audited(policy)
    trainer = SACTrainer(policy, boundary="task")
    trainer.add(trajectory, requests=requests)
    assert trainer.reward_spec["version"] == "success-components-task-v1"
    original = trainer.replay.rewards.clone()
    changed = copy.deepcopy(trajectory)
    changed["episode_id"] = "changed"
    for row in changed["decisions"]:
        usage = row["attempts"][0]["usage"]
        for key in ["rules", "thought"]:
            usage["attribution"]["items"][key]["input_tokens"] += 10000
            if key == "rules":
                usage["attribution"]["items"][key]["content"][0]["input_tokens"] += (
                    10000
                )
            usage["input_tokens"] += 10000
    trainer.add(changed, requests=requests)
    assert torch.equal(original, trainer.replay.rewards[2:])


def test_constant_is_paid_once_not_per_request(tmp_path):
    trajectory, requests = audited(setup_policy(tmp_path))
    cfg = ComponentRewardConfig()
    data = transitions_from_trajectory(
        trajectory, cfg, boundary="task", requests=requests
    )
    first = account(requests["1"], trajectory["decisions"][0]["attempts"][0]["usage"])
    assert data.rewards[0].item() == pytest.approx(-cfg.token_cost(first))
    assert data.rewards[-1].item() == pytest.approx(
        -cfg.token_cost(first) + cfg.terminal(15.0, True)
    )


def test_retry_cost_is_counted_once_and_post_verdict_cost_is_excluded(tmp_path):
    trajectory, requests = audited(setup_policy(tmp_path))
    retry = copy.deepcopy(trajectory["decisions"][1]["attempts"][0])
    retry.update(request_id="retry", response_id="retry")
    retry["timing"]["started_at"] = 114.0
    trajectory["decisions"][1]["attempts"].append(retry)
    requests["retry"] = requests["2"]
    data = transitions_from_trajectory(
        trajectory, ComponentRewardConfig(), boundary="task", requests=requests
    )
    expected = component_reward(task_costs(trajectory, requests), 15.0)["reward"]
    assert len(data.rewards) == 2
    assert data.rewards.sum().item() == pytest.approx(expected, abs=2e-6)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_payload",
        "missing_attribution",
        "post_task_attribution",
        "duplicate_request",
        "transport",
        "fallback",
    ],
)
def test_incomplete_evidence_does_not_mutate_replay(tmp_path, fault):
    trajectory, requests = audited(setup_policy(tmp_path))
    trainer = SACTrainer(setup_policy(tmp_path), boundary="task")
    if fault == "missing_payload":
        requests.pop("1")
    elif fault == "missing_attribution":
        trajectory["decisions"][0]["attempts"][0]["usage"].pop("attribution")
    elif fault == "post_task_attribution":
        trajectory["decisions"][-1]["attempts"][0]["usage"].pop("attribution")
    elif fault == "duplicate_request":
        trajectory["decisions"][1]["attempts"][0]["request_id"] = "1"
    elif fault == "transport":
        trajectory["decisions"][-1]["attempts"][0]["timing"]["transport_error"] = (
            "disconnect"
        )
    else:
        trajectory.update(audit_complete=False, issues=["compression_fallback"])
    with pytest.raises((ValueError, KeyError)):
        trainer.add(trajectory, requests=requests)
    assert (
        trainer.replay is None
        and not trainer.episode_ids
        and trainer.reward_spec is None
    )


def test_reward_spec_survives_restore_and_cannot_be_mixed(tmp_path):
    policy = setup_policy(tmp_path)
    trajectory, requests = audited(policy)
    trainer = SACTrainer(policy, boundary="task")
    trainer.add(trajectory, requests=requests)
    path = tmp_path / "training.pt"
    trainer.save(path)
    resumed = SACTrainer(policy, boundary="task")
    resumed.restore(path)
    assert resumed.reward_spec == trainer.reward_spec
    assert torch.equal(resumed.replay.rewards, trainer.replay.rewards)
    trajectory["episode_id"] = "second"
    for config in [
        PPOConfig(1000000),
        replace(ComponentRewardConfig(), weights=(0.6, 0.6, 0.3, 0.5, 0.1)),
    ]:
        with pytest.raises(ValueError, match="reward configuration mismatch"):
            resumed.add(trajectory, config, requests=requests)
    payload = torch.load(path, weights_only=True)
    payload.pop("reward_spec")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="lacks reward metadata"):
        resumed.restore(path)


def test_worker_output_loads_via_native_add_episode(tmp_path):
    policy = setup_policy(tmp_path)
    trajectory, requests = audited(policy)
    out = tmp_path / "output"
    (out / "requests").mkdir(parents=True)
    (out / "trajectory.json").write_text(json.dumps(trajectory))
    for rid, request in requests.items():
        (out / "requests" / (rid + ".after.json")).write_text(json.dumps(request))
    trainer = SACTrainer(policy, boundary="task")
    trainer.add_episode(out)
    assert len(trainer.replay.rewards) == 2
    assert trainer.reward_spec == ComponentRewardConfig().spec()
    from component_reward import episode

    diagnostic = episode(out)
    assert diagnostic["reward"] == pytest.approx(
        trainer.replay.rewards.sum().item(), abs=2e-6
    )
    assert diagnostic["reward_spec"] == trainer.reward_spec


def test_components_require_native_task_boundary(tmp_path):
    policy = setup_policy(tmp_path)
    trajectory, requests = audited(policy)
    with pytest.raises(ValueError, match="requires task boundary"):
        SACTrainer(policy).add(trajectory, requests=requests)


@pytest.mark.parametrize(
    "change",
    [
        {"weights": (1, 2)},
        {"references": (1, 1, 1, 1, 0)},
        {"weights": (0, 0, 0, 0, 0)},
        {"weights": (1, 1, float("nan"), 1, 1)},
    ],
)
def test_invalid_reward_config_is_rejected(change):
    with pytest.raises(ValueError):
        ComponentRewardConfig(**change)


def test_failure_has_no_positive_savings_bonus(tmp_path):
    policy = setup_policy(tmp_path)
    success, requests = audited(policy)
    failure = copy.deepcopy(success)
    failure["outcome"]["environment_success"] = False
    cfg = ComponentRewardConfig()
    a = transitions_from_trajectory(success, cfg, boundary="task", requests=requests)
    b = transitions_from_trajectory(failure, cfg, boundary="task", requests=requests)
    assert a.rewards.sum().item() - b.rewards.sum().item() == pytest.approx(
        1.0, abs=2e-6
    )
    assert b.rewards.sum().item() < 0
    assert component_reward({}, 0.0, success=False)["reward"] == 0
    assert component_reward({}, 0.0, success=True)["reward"] == 1


def test_discounted_config_cannot_change_episode_reward_semantics(tmp_path):
    policy = setup_policy(tmp_path)
    trajectory, requests = audited(policy)
    with pytest.raises(ValueError, match="requires gamma=1"):
        SACTrainer(policy, SACConfig(gamma=0.99), boundary="task").add(
            trajectory, requests=requests
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True])
def test_invalid_excluded_token_counts_are_not_silently_ignored(tmp_path, bad):
    policy = setup_policy(tmp_path)
    trajectory, requests = audited(policy)
    trajectory["decisions"][0]["attempts"][0]["usage"]["attribution"]["items"][
        "thought"
    ]["input_tokens"] = bad
    with pytest.raises(ValueError, match="token counts"):
        SACTrainer(policy, boundary="task").add(trajectory, requests=requests)
