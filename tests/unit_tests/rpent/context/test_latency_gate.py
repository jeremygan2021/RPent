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

"""Fixed latency admission excludes fast/slow conditions without altering rewards."""

import copy

import pytest

from rpent.context.network_baseline import latency_gate, run_gated_baseline


@pytest.fixture
def config():
    return {
        "version": "test",
        "model": "gpt-5.6-sol",
        "reasoning": "xhigh",
        "upstream": "http://proxy",
        "median_relative_tolerance": 0.3,
        "peak_relative_tolerance": 0.3,
        "allow_faster": False,
        "reference": {
            "link": {"median_s": 0.8, "maximum_s": 1.2},
            "fixed_request": {"median_s": 2.3, "maximum_s": 3.6},
        },
    }


def baseline(link=(0.8,) * 5, post=(2.3,) * 3):
    return {
        "model": "gpt-5.6-sol",
        "reasoning": "xhigh",
        "upstream": "http://proxy",
        "outside_task": True,
        "link": [{"ok": True, "status": 200, "latency_s": t} for t in link],
        "fixed_request": [
            {
                "ok": True,
                "completed": True,
                "status": 200,
                "latency_s": t,
                "usage": {"input_tokens": 18},
            }
            for t in post
        ],
    }


def test_accepts_reference_without_mutating_measurements(config):
    data = baseline()
    original = copy.deepcopy(data)
    gate = latency_gate(data, config)
    assert gate["accepted"] and not gate["reasons"]
    assert data == original


@pytest.mark.parametrize("post", [(7.0, 6.0, 8.0), (1.0, 1.2, 1.3), (2.3, 2.3, 5.0)])
def test_slow_fast_and_single_spike_are_blocked(config, post):
    result = latency_gate(baseline(post=post), config)
    assert not result["accepted"]
    assert result["reasons"]


def test_get_delay_also_blocks(config):
    assert not latency_gate(baseline(link=(1.5,) * 5), config)["accepted"]


@pytest.mark.parametrize(
    "mutation",
    [
        "503",
        "missing_usage",
        "incomplete",
        "nan",
        "missing_sample",
        "wrong_model",
        "setup_error",
    ],
)
def test_missing_failed_or_mismatched_baseline_fails_closed(config, mutation):
    data = baseline()
    if mutation == "503":
        data["fixed_request"][0].update(ok=False, status=503)
    if mutation == "missing_usage":
        data["fixed_request"][0].pop("usage")
    if mutation == "incomplete":
        data["fixed_request"][0]["completed"] = False
    if mutation == "nan":
        data["link"][0]["latency_s"] = float("nan")
    if mutation == "missing_sample":
        data["link"].pop()
    if mutation == "wrong_model":
        data["model"] = "different"
    if mutation == "setup_error":
        data["setup_error"] = "FileNotFoundError"
    assert not latency_gate(data, config)["accepted"]


def test_optional_fast_conditions_policy(config):
    config["allow_faster"] = True
    assert latency_gate(baseline(link=(0.1,) * 5, post=(1.0,) * 3), config)["accepted"]
    assert not latency_gate(baseline(post=(7.0,) * 3), config)["accepted"]


def test_invalid_reference_is_configuration_error(config):
    config["reference"]["link"]["median_s"] = 0
    with pytest.raises(ValueError, match="reference"):
        latency_gate(baseline(), config)


@pytest.mark.parametrize(
    "outcomes", [(True,), (False, True), (False, False, True), (False, False, False)]
)
def test_gate_rechecks_stop_at_first_pass_or_after_three(config, monkeypatch, outcomes):
    from rpent.context import network_baseline

    measurements = [
        baseline() if passed else baseline(post=(7.0,) * 3) for passed in outcomes
    ]
    iterator = iter(measurements)
    calls, recorded = [], []
    original = copy.deepcopy(config)

    def measure(*args, **kwargs):
        calls.append((args, kwargs))
        return next(iterator)

    monkeypatch.setattr(network_baseline, "run_baseline", measure)
    result = run_gated_baseline(
        "http://proxy",
        config["model"],
        config["reasoning"],
        config,
        on_check=recorded.append,
    )
    assert len(calls) == len(outcomes)
    assert [row["gate_check"] for row in recorded] == list(range(1, len(outcomes) + 1))
    assert [row["latency_gate"]["accepted"] for row in result["gate_checks"]] == list(
        outcomes
    )
    assert result["latency_gate"]["accepted"] is outcomes[-1]
    assert result["outside_task"]
    assert all(kwargs == {"link_samples": 5, "model_samples": 3} for _, kwargs in calls)
    assert config == original
