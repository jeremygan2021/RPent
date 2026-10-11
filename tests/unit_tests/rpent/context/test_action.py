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

"""Action thinning preserves executable protocol and explicit outcomes."""

import copy
import json

import pytest

from rpent.context.action import ActionSelector, difference
from rpent.context.codex import CodexAdapter
from rpent.context.session import ContextSession


class Features:
    def similarity(self, a, b):
        return float(a == b)


def history():
    items = []
    for step in range(7):
        items.extend(
            [
                {
                    "type": "custom_tool_call",
                    "call_id": str(step),
                    "input": "original invocation",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": str(step),
                    "output": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {
                                    "step": step,
                                    "state": {"position": step},
                                    "log": {
                                        "command": {
                                            "action": "move",
                                            "prompt": "move cup",
                                        },
                                        "result": {
                                            "success": step != 2,
                                            "details": "x" * 300,
                                        },
                                    },
                                    "terminated": False,
                                }
                            ),
                        }
                    ],
                },
            ]
        )
    return {"input": items}


def test_ranges_and_missing_are_distinct():
    assert difference({"x": 4}, {"x": 2}, {}) is None
    assert difference({"x": 4}, {"x": 2}, {"x": (0, 10)}) == 0.2
    assert difference({"x": 4}, {"x": 2}, {"x": (0, 0)}) is None


@pytest.mark.parametrize("style", ["custom_tool_call", "function_call"])
def test_action_preserves_protocol_outcomes_and_latest(style):
    request = history()
    for item in request["input"]:
        if item["type"] == "custom_tool_call":
            item["type"] = style
            if style == "function_call":
                item["arguments"] = "{}"
                item.pop("input")
        else:
            item["type"] = style + "_output"
    original = copy.deepcopy(request)
    session = ContextSession(CodexAdapter(), Features(), mode="action")
    result = session.process(request)
    assert request == original
    assert session.stats["action"]["records_after"] < 7
    for i in range(0, len(request["input"]), 2):
        assert result["input"][i] == request["input"][i]
        before = json.loads(request["input"][i + 1]["output"][0]["text"])
        after = json.loads(result["input"][i + 1]["output"][0]["text"])
        assert before["log"]["command"] == after["log"]["command"]
        assert before["log"]["result"]["success"] == after["log"]["result"]["success"]
    assert result["input"][-4:] == request["input"][-4:]
    assert session.process(request) == result


def test_unknown_and_retained_image_dependency():
    request = history()
    for item in request["input"]:
        if item["type"] == "custom_tool_call_output":
            item["output"].append(
                {"type": "input_image", "image_url": "unknown-format"}
            )
    assert (
        ContextSession(CodexAdapter(), Features(), mode="action").process(request)
        == request
    )
    opaque = {
        "input": [
            {
                "type": "custom_tool_call_output",
                "call_id": "unknown",
                "output": [{"type": "input_text", "text": "opaque"}],
            }
        ]
    }
    assert (
        ContextSession(CodexAdapter(), Features(), mode="action").process(opaque)
        == opaque
    )


def test_numeric_ranges_change_selection_without_touching_commands():
    source = history()
    for item in source["input"]:
        if item["type"] != "custom_tool_call_output":
            continue
        value = json.loads(item["output"][0]["text"])
        value["state"] = {
            "robot0_gripper_qpos": [0.04, -0.04] if value["step"] == 0 else [0, 0],
            "robot0_eef_pos": [0.1, 0.2, 0.3],
        }
        value["log"]["result"]["success"] = True
        item["output"][0]["text"] = json.dumps(value)
    original = copy.deepcopy(source)
    from pathlib import Path

    ranges = json.loads(
        (
            Path(__file__).resolve().parents[4] / "robots/libero/context_ranges.json"
        ).read_text()
    )
    session = ContextSession(
        CodexAdapter(), Features(), mode="action", action_ranges=ranges
    )
    result = session.process(source)
    records = session.stats["action"]["records"]
    assert records[0]["state_change"] is None
    assert records[1]["state_change"] == 1.0
    assert records[1]["selected"]
    assert not records[2]["selected"]
    assert records[2]["state_change"] == 0.0
    assert (
        "state.robot0_eef_pos.0" in session.stats["action"]["unscaled_numeric_fields"]
    )
    assert (
        "state.robot0_gripper_qpos.0"
        not in session.stats["action"]["unscaled_numeric_fields"]
    )
    assert source == original
    assert session.process(source) == result
    for before, after in zip(
        CodexAdapter().extract_action(source), CodexAdapter().extract_action(result)
    ):
        assert before.snapshot["log"]["command"] == after.snapshot["log"]["command"]
        assert (
            before.snapshot["log"]["result"]["success"]
            == after.snapshot["log"]["result"]["success"]
        )


@pytest.mark.parametrize(
    "ranges",
    [
        [],
        {"state.x": [0, 0]},
        {"state.x": [1, 0]},
        {"state.x": [0, float("nan")]},
        {"state.x": [0, float("inf")]},
        {"state.x": [False, 1]},
        {"x": [0, 1]},
        {"state.": [0, 1]},
        {"state.x": [0]},
        {"state.x": ["0", 1]},
    ],
)
def test_invalid_action_ranges_fail_before_requests(ranges):
    with pytest.raises(ValueError, match="action range"):
        ActionSelector(ranges)


def test_proxy_loads_explicit_numeric_ranges(monkeypatch, tmp_path):
    import sys

    from rpent.context import proxy

    config = tmp_path / "ranges.json"
    config.write_text(json.dumps({"state.position": [0, 10]}))
    captured = {}
    monkeypatch.setattr(proxy, "serve", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "proxy",
            "--mode",
            "all",
            "--rho-image",
            "0.2",
            "--rho-text",
            "0.1",
            "--rho-action",
            "0.1",
            "--action-ranges",
            str(config),
            "--log",
            str(tmp_path / "selection.jsonl"),
        ],
    )
    proxy.main()
    session = captured["session"]
    assert (session.rho_image, session.rho_text, session.rho_action) == (0.2, 0.1, 0.1)
    assert session._action.ranges == {"state.position": (0, 10)}
    assert session._action.ranges != {}


@pytest.mark.parametrize("content", ["{", "null", "[]", '{"state.x": [0, 0]}'])
def test_proxy_rejects_bad_ranges_at_startup(monkeypatch, tmp_path, content):
    import sys

    from rpent.context import proxy

    config = tmp_path / "ranges.json"
    config.write_text(content)

    def unexpected_serve(**kwargs):
        pytest.fail("invalid configuration must not start a proxy")

    monkeypatch.setattr(proxy, "serve", unexpected_serve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "proxy",
            "--action-ranges",
            str(config),
            "--log",
            str(tmp_path / "selection.jsonl"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        proxy.main()
    assert error.value.code == 2


@pytest.mark.parametrize("with_image", [False, True])
def test_robocasa_omitted_action_retains_execution_evidence(with_image):
    from rpent.context.anchor import AnchorAdapter

    request = history()
    output = request["input"][1]
    snapshot = json.loads(output["output"][0]["text"])
    snapshot["task_progress"] = {"water_on": False}
    snapshot["success"] = False
    snapshot["log"]["result"] = {
        "status": "cap", "grasp_detected": False, "details": "long diagnostics"
    }
    output["output"][0]["text"] = json.dumps(snapshot)
    if with_image:
        output["output"].append({"type": "input_image", "image_url": "opaque"})
    original = copy.deepcopy(request)
    adapter = AnchorAdapter()
    records = adapter.extract_action(request)
    keep = {record.id for record in records if record.snapshot["step"] != 0}
    result = adapter.render_action(request, records, keep)
    value = json.loads(result["input"][1]["output"][0]["text"])
    summary = value["log"] if with_image else value
    assert summary["command"]["action"] == "move"
    assert summary["result"] == {"status": "cap", "grasp_detected": False}
    assert value["task_progress"] == {"water_on": False}
    assert value["success"] is False
    assert request == original
    assert result["input"][-4:] == request["input"][-4:]
