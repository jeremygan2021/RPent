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

"""Offline learned-controller contracts, without model or simulator calls."""

import dataclasses
import subprocess
import sys

import pytest

from rpent.context.codex import CodexAdapter
from rpent.context.controller import Thresholds
from rpent.context.observation import Observation, observe
from rpent.context.session import ContextSession
from tests.unit_tests.rpent.context.test_images import FakeFeatures, request

torch = pytest.importorskip("torch")
from rpent.context.policy import (  # noqa: E402 -- Torch is optional in the base test environment.
    MLPController,
    feature_names,
    initialize,
    save_checkpoint,
)


def observation():
    modalities = ("image", "text", "action")
    return observe(
        {m: [] for m in modalities},
        {m: (set(), set()) for m in modalities},
        set(modalities),
        {},
        Thresholds(0.2, 0.1, 0.1),
        0,
        False,
    )


def checkpoint(tmp_path):
    path = tmp_path / "policy.pt"
    save_checkpoint(initialize(Thresholds(0.2, 0.1, 0.1)), path)
    return path


def test_roundtrip_deterministic_initial_policy_and_global_rng(tmp_path):
    rng = torch.random.get_rng_state().clone()
    path = checkpoint(tmp_path)
    first = MLPController(path)
    second = MLPController(path)
    assert torch.equal(rng, torch.random.get_rng_state())
    decision = first.decide(observation())
    assert decision == second.decide(observation()) == first.decide(observation())
    assert dataclasses.astuple(decision.thresholds) == pytest.approx(
        (0.2, 0.1, 0.1, 0.5)
    )
    assert decision.latent_action is None and decision.log_prob is None
    assert decision.value == 0
    assert all(not p.requires_grad for p in first.model.parameters())


def test_sampling_is_seeded_log_prob_matches_latent_and_retry_reuses(tmp_path):
    path = checkpoint(tmp_path)
    first = MLPController(path, sample=True, seed=31)
    second = MLPController(path, sample=True, seed=31)
    obs = observation()
    d = first.decide(obs)
    assert d == second.decide(obs)
    distribution, _ = first.model(torch.tensor(obs.values))
    assert d.log_prob == pytest.approx(
        float(distribution.log_prob(torch.tensor(d.latent_action)).sum())
    )
    assert first.decide(obs).thresholds != d.thresholds
    session = ContextSession(
        CodexAdapter(("front",)), FakeFeatures(), mode="all", controller=second
    )
    source = request()
    output = session.process(source)
    rng = second.generator.get_state().clone()
    assert session.process(source) == output
    assert torch.equal(rng, second.generator.get_state())
    assert session.stats["decision"]["reused"]


def test_network_can_use_features_and_preserves_normalization(tmp_path):
    model = initialize(Thresholds(0.2, 0.1, 0.1))
    with torch.no_grad():
        for layer in (model.body[0], model.body[2], model.actor):
            layer.weight.zero_()
            layer.bias.zero_()
            layer.weight[0, 0] = 1
        model.feature_mean[0] = 1
        model.feature_scale[0] = 2
    path = tmp_path / "policy.pt"
    save_checkpoint(model, path)
    policy = MLPController(path)
    zero = Observation(feature_names(), (0.0,) * len(feature_names()))
    changed = dataclasses.replace(zero, values=(3.0,) + zero.values[1:])
    assert (
        policy.decide(changed).thresholds.image > policy.decide(zero).thresholds.image
    )
    assert policy.model.feature_scale[0] == 2
    assert policy.model.feature_mean[0] == 1


@pytest.mark.parametrize("kind", ["version", "order", "scale", "nan"])
def test_invalid_checkpoints_fail_before_running(tmp_path, kind):
    path = checkpoint(tmp_path)
    data = torch.load(path, weights_only=True)
    if kind == "version":
        data["scoring_version"] = "other"
    elif kind == "order":
        data["feature_names"].reverse()
    elif kind == "scale":
        data["state_dict"]["feature_scale"][0] = 0
    else:
        data["state_dict"]["actor.bias"][0] = float("nan")
    torch.save(data, path)
    with pytest.raises(ValueError):
        MLPController(path)


def test_invalid_observation_is_rejected(tmp_path):
    policy = MLPController(checkpoint(tmp_path))
    obs = observation()
    for changed in (
        dataclasses.replace(obs, names=obs.names[::-1]),
        dataclasses.replace(obs, values=obs.values[:-1]),
        dataclasses.replace(obs, values=(float("nan"),) + obs.values[1:]),
    ):
        with pytest.raises(ValueError):
            policy.decide(changed)


def test_fixed_mode_import_does_not_load_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rpent.context.proxy; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


def test_cli_init_and_proxy_invalid_mode(tmp_path):
    path = tmp_path / "initialized.pt"
    subprocess.run(
        [sys.executable, "-m", "rpent.context.policy", "--output", str(path)],
        check=True,
    )
    assert MLPController(path).decide(observation()).thresholds.image == pytest.approx(
        0.2
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rpent.context.proxy",
            "--log",
            str(tmp_path / "selection.jsonl"),
            "--controller-checkpoint",
            str(path),
            "--mode",
            "full",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Full mode" in result.stderr


def test_mlp_thresholds_drive_existing_image_selector(tmp_path):
    counts = []
    for rho in (0.2, 0.8):
        path = tmp_path / f"policy-{rho}.pt"
        save_checkpoint(initialize(Thresholds(rho, 0.1, 0.1)), path)
        session = ContextSession(
            CodexAdapter(("front",)),
            FakeFeatures(),
            mode="image",
            controller=MLPController(path),
        )
        session.process(request())
        assert session.stats["rho"] == pytest.approx(rho)
        counts.append(session.stats["images_after"])
    assert counts[0] < counts[1]


def test_legacy_checkpoint_remains_deployable_but_cannot_train(tmp_path):
    from rpent.context.iql import IQLConfig, IQLTrainer
    from rpent.context.observation import SCORING_VERSION
    from rpent.context.policy import ActorCritic, legacy_feature_names
    from rpent.context.training import PPOConfig, prepare_batch

    model = ActorCritic(legacy=True)
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.zero_()
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "format_version": 1,
            "feature_version": "counts-and-lagged-scores-v1",
            "scoring_version": SCORING_VERSION,
            "feature_names": list(legacy_feature_names()),
            "state_dict": model.state_dict(),
        },
        path,
    )
    policy = MLPController(path)
    assert policy.legacy
    assert policy.decide(observation()).thresholds.image == pytest.approx(0.5)
    with pytest.raises(ValueError, match="new memory-enabled"):
        prepare_batch([], policy, PPOConfig(1000))
    with pytest.raises(ValueError, match="new memory-enabled"):
        IQLTrainer(policy, IQLConfig())


@pytest.mark.parametrize("behavior,low,high", [
    ("lower", .2, .6),
    ("lower-moderate", .3, .6),
    ("lower-strong", .05, .25),
    ("lower-strong-safe", .2, .4),
])
@pytest.mark.parametrize("initial", [.99, .5])
def test_relative_exploration_ranges_and_retry(tmp_path, behavior, low, high, initial):
    path = tmp_path / "explore.pt"
    save_checkpoint(initialize(Thresholds(initial, initial, initial, initial)), path)
    first = MLPController(path, sample=True, seed=31, behavior=behavior)
    second = MLPController(path, sample=True, seed=31, behavior=behavior)
    weights = {k: v.clone() for k, v in first.model.state_dict().items()}
    for _ in range(16):
        decision = first.decide(observation())
        assert decision == second.decide(observation())
        assert decision.behavior_policy == f"lower-relative-{low:g}-{high:g}-v1"
        assert decision.log_prob is None and decision.latent_action is None
        values = dataclasses.astuple(decision.thresholds)
        assert all(initial * low <= v <= initial * high for v in values)
    assert all(torch.equal(weights[k], v) for k, v in first.model.state_dict().items())
    session = ContextSession(CodexAdapter(("front",)), FakeFeatures(),
                             mode="all", controller=first)
    source = request()
    output = session.process(source)
    rng = first.generator.get_state().clone()
    assert session.process(source) == output
    assert torch.equal(rng, first.generator.get_state())
    assert session.stats["decision"]["reused"]
    with pytest.raises(ValueError, match="stochastic"):
        MLPController(path, behavior=behavior)
