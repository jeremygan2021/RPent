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

"""Regressions for rule ownership, pose scoring, and transparent cost accounting."""

import copy
import json
import math
from pathlib import Path

import pytest

from rpent.context.action import ActionSelector, orientation_difference
from rpent.context.anchor import AnchorAdapter
from rpent.context.controller import Thresholds
from rpent.context.policy import MLPController, initialize, save_checkpoint
from rpent.context.reward import account
from rpent.context.reward import component_reward as relative_reward
from rpent.context.scope import schema
from rpent.context.session import ContextSession

TOOL = """mcp__rpent__back_project: Look up world XYZ from a pixel.

exec tool declaration:
```ts
declare const tools: { mcp__rpent__back_project(args: {
  row?: number; col?: number; row_range?: Array<number>;
}): Promise<CallToolResult>; };
```"""
PLAIN_TOOL = """Tool name: back_project
Parameters: row (number), col (number). Returns world_xyz."""


class Features:
    def similarity(self, a, b):
        return float(a == b)


def history():
    items = [
        {
            "id": "task",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "put cream cheese in basket"}],
        }
    ]
    for step in range(5):
        value = {
            "step": step,
            "state": {
                "robot0_eef_pos": [0.0, step * 0.1, 0.2],
                "robot0_eef_quat": [
                    0.0,
                    0.0,
                    math.sin(step * 0.1),
                    math.cos(step * 0.1),
                ],
            },
            "task_language": "put cream cheese in basket",
            "log": {
                "command": {
                    "action": "move_to",
                    "xyz": [0.0, step * 0.1, 0.2],
                    "gripper": 1,
                },
                "result": {"name": "move_to"},
            },
        }
        items.extend(
            [
                {
                    "id": f"call{step}",
                    "type": "custom_tool_call",
                    "call_id": str(step),
                    "input": "robot invocation",
                },
                {
                    "id": f"out{step}",
                    "type": "custom_tool_call_output",
                    "call_id": str(step),
                    "output": [{"type": "input_text", "text": json.dumps(value)}],
                },
            ]
        )
    return {"input": items}


@pytest.mark.parametrize(
    "text",
    [
        TOOL,
        PLAIN_TOOL,
        json.dumps({"name": "back_project", "inputSchema": {"type": "object"}}),
    ],
)
def test_tool_definitions_are_rules(text):
    assert schema(text)


@pytest.mark.parametrize(
    "text",
    [
        '{"world_xyz": [0.1, 0.2, 0.3], "step": 0}',
        "Use back_project with row and col; tool description was loaded earlier.",
        "declare const tools: { this is not a function declaration }",
        "The log says Tool name: back_project. Parameters: row and col.",
        "Tool name: cream_cheese\nDescription: blue package near basket",
        json.dumps(
            {
                "name": "cream_cheese",
                "description": "blue package",
                "world_xyz": [0.1, 0.2, 0.3],
            }
        ),
        json.dumps([{"name": "cream_cheese", "description": "blue package"}]),
        json.dumps({"name": "back_project", "inputSchema": "not a schema"}),
        json.dumps({"name": "cream_cheese", "description": TOOL}),
    ],
)
def test_query_results_and_mentions_are_not_rules(text):
    assert not schema(text)


@pytest.mark.parametrize("description", [TOOL, PLAIN_TOOL])
def test_old_tool_definition_survives_text_omission_but_query_result_is_selectable(
    description,
):
    request = history()
    request["input"][1:1] = [
        {
            "id": "query",
            "type": "custom_tool_call",
            "call_id": "query",
            "input": "lookup()",
        },
        {
            "id": "queryout",
            "type": "custom_tool_call_output",
            "call_id": "query",
            "output": [
                {"type": "input_text", "text": description},
                {"type": "input_text", "text": "cream_center"},
                {"type": "input_text", "text": '{"world_xyz": [0.1,0.2,0.3]}'},
            ],
        },
    ]
    before = copy.deepcopy(request)
    adapter = AnchorAdapter()
    records, _ = adapter.extract_text(request)
    assert not any(description in r.text for r in records)
    result_record = next(r for r in records if "world_xyz" in r.text)
    assert result_record.text.startswith("cream_center\n")
    assert not result_record.protected
    rendered = adapter.render_text(
        request, records, {r.id for r in records if r.protected}
    )
    assert request == before
    assert rendered["input"][2]["output"] == [
        {"type": "input_text", "text": description}
    ]


def test_reference_definitions_with_parameter_comments_are_rules():
    path = Path(__file__).resolve().parents[4] / "tool_reference.json"
    definitions = json.loads(path.read_text())
    assert schema(json.dumps(definitions))
    for definition in definitions:
        assert schema(json.dumps(definition))
        assert schema(definition["description"])


@pytest.mark.parametrize("recent", [False, True])
def test_named_query_result_is_text_and_only_recent_results_are_protected(recent):
    request = history()
    result_text = json.dumps(
        {
            "name": "cream_cheese",
            "description": "blue package near basket",
            "world_xyz": [0.1, 0.2, 0.3],
            "error": None,
        }
    )
    query = [
        {
            "id": "query",
            "type": "custom_tool_call",
            "call_id": "query",
            "input": "lookup()",
        },
        {
            "id": "queryout",
            "type": "custom_tool_call_output",
            "call_id": "query",
            "output": [
                {"type": "input_text", "text": result_text},
            ],
        },
    ]
    if recent:
        request["input"].extend(query)
    else:
        request["input"][1:1] = query
    adapter = AnchorAdapter()
    records, _ = adapter.extract_text(request)
    record = next(r for r in records if r.text == result_text)
    assert record.protected is recent
    rendered = adapter.render_text(
        request, records, {r.id for r in records if r.protected}
    )
    result = next(i for i in rendered["input"] if i.get("id") == "queryout")
    assert (result["output"][0]["text"] == result_text) is recent
    costs = account(request, usage_for(request))
    assert costs["text"] == 20  # query code and its result
    assert costs.get("tool_rules", 0) == 0
    assert sum(costs.values()) == usage_for(request)["input_tokens"]


def test_low_nc_preserves_plain_interfaces_and_active_errors():
    request = history()
    request["input"][1:1] = [
        {
            "id": "lookup",
            "type": "custom_tool_call",
            "call_id": "lookup",
            "input": "lookup()",
        },
        {
            "id": "lookupout",
            "type": "custom_tool_call_output",
            "call_id": "lookup",
            "output": [
                {"type": "input_text", "text": PLAIN_TOOL},
                {"type": "input_text", "text": '{"error":"coordinate lookup failed"}'},
            ],
        },
    ]
    session = ContextSession(AnchorAdapter(), Features(), mode="text", rho_text=0.1)
    result = session.process(request)
    assert result["input"][2]["output"] == request["input"][2]["output"]
    costs = account(result, usage_for(result))
    assert costs["tool_rules"] == 10
    assert costs["text"] >= 10  # error result remains a protected Text cost


def test_recency_tie_keeps_later_atomic_result_instead_of_earlier_call_code():
    request = history()
    queries = []
    for n in range(3):
        queries.extend(
            [
                {
                    "id": f"query{n}",
                    "type": "custom_tool_call",
                    "call_id": f"query{n}",
                    "input": f"lookup({n})",
                },
                {
                    "id": f"queryout{n}",
                    "type": "custom_tool_call_output",
                    "call_id": f"query{n}",
                    "output": [
                        {"type": "input_text", "text": f"point{n}"},
                        {
                            "type": "input_text",
                            "text": json.dumps({"world_xyz": [n, 0, 0]}),
                        },
                    ],
                },
            ]
        )
    request["input"][1:1] = queries

    class EqualFeatures:
        def similarity(self, a, b):
            return 0.5

    before = copy.deepcopy(request)
    session = ContextSession(
        AnchorAdapter(), EqualFeatures(), mode="text", rho_text=0.1
    )
    result = session.process(request)
    assert request == before
    assert [i["id"] for i in result["input"]] == [i["id"] for i in request["input"]]
    latest = next(i for i in result["input"] if i["id"] == "queryout2")
    assert latest["output"] == queries[-1]["output"]  # label and JSON stay together
    selected = [r["id"] for r in session.stats["text"]["records"] if r["selected"]]
    assert selected == ["queryout2:text:0"]
    assert session.process(request) == result  # exact retries reuse the decision
    assert session.stats["decision"]["reused"]


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([0, 0, 0, 1], [0, 0, 0, -1], 0.0),
        ([0, 0, 1, 1], [0, 0, 0, 1], 0.5),
        ([1, 0, 0, 0], [0, 0, 0, 1], 1.0),
        ([0, 0, 0, 0], [0, 0, 0, 1], None),
        ([0, 0, 0, float("nan")], [0, 0, 0, 1], None),
    ],
)
def test_quaternion_rotation_angle(a, b, expected):
    actual = orientation_difference({"robot0_eef_quat": a}, {"robot0_eef_quat": b})
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)


def test_same_move_type_gets_position_and_orientation_scores_and_keeps_boundaries():
    request = history()
    ranges = {
        "command.xyz.1": [0.0, 0.5],
        "state.robot0_eef_pos.1": [0.0, 0.5],
        "command.gripper": [-1.0, 1.0],
    }
    selector = ActionSelector(ranges)
    adapter = AnchorAdapter()
    before = copy.deepcopy(request)
    selector.process(request, adapter, Features(), rho=0.2)
    row = selector.stats["records"][1]
    assert row["action_change"] == pytest.approx(
        0.2 / 2
    )  # position plus unchanged gripper
    assert row["orientation_change"] == pytest.approx(0.2 / math.pi)
    assert row["state_change"] > 0
    assert selector.stats["records"][-1]["selected"]
    assert request == before
    assert selector.process(request, adapter, Features(), rho=1) == request


def usage_for(request, residual=2):
    attrs = {}
    for item in request["input"]:
        blocks = item.get("output", item.get("content"))
        parts = (
            [{"input_tokens": 10} for _ in blocks] if isinstance(blocks, list) else None
        )
        attrs[item["id"]] = {
            "input_tokens": 10 * len(parts) if parts is not None else 10
        }
        if parts is not None:
            attrs[item["id"]]["content"] = parts
    return {
        "input_tokens": sum(a["input_tokens"] for a in attrs.values()) + residual,
        "attribution": {"items": attrs},
    }


def test_rule_cost_is_separate_and_unknown_residual_is_explicit():
    request = history()
    request["input"].extend(
        [
            {
                "id": "lookup",
                "type": "custom_tool_call",
                "call_id": "lookup",
                "input": "lookup()",
            },
            {
                "id": "lookupout",
                "type": "custom_tool_call_output",
                "call_id": "lookup",
                "output": [
                    {"type": "input_text", "text": TOOL},
                    {"type": "input_text", "text": '{"world_xyz": [0.1,0.2,0.3]}'},
                ],
            },
        ]
    )
    usage = usage_for(request)
    costs = account(request, usage)
    assert costs["tool_rules"] == 10
    assert costs["text"] == 20  # one query call and its coordinate result
    assert costs["unattributed"] == 2
    assert sum(costs.values()) == usage["input_tokens"]


def test_missing_attribution_is_rejected():
    request = history()
    usage = usage_for(request)
    del usage["attribution"]["items"]["out0"]
    with pytest.raises(KeyError):
        account(request, usage)


def test_rule_and_reasoning_costs_do_not_enter_relative_reward():
    costs = {"image": 100.0, "text": 200.0, "action": 300.0, "memory": 20.0}
    assert relative_reward(costs, 100.0) == relative_reward(
        dict(costs, tool_rules=100000, reasoning=100000), 100.0
    )
    assert (
        relative_reward(dict(costs, text=100.0), 100.0)["reward"]
        > relative_reward(costs, 100.0)["reward"]
    )


@pytest.mark.parametrize(
    "old_scoring",
    [
        "score-mass-full-history-extended-v1",
        "score-mass-full-history-toolrules-pose-v2",
    ],
)
def test_checkpoint_roundtrip_uses_new_scoring_version(tmp_path, old_scoring):
    path = tmp_path / "actor.pt"
    save_checkpoint(initialize(Thresholds(0.9, 0.9, 0.9, 0.9)), path)
    assert MLPController(path).model.actor.out_features == 4
    import torch

    payload = torch.load(path, weights_only=True)
    payload["scoring_version"] = old_scoring
    torch.save(payload, path)
    with pytest.raises(ValueError, match="version mismatch"):
        MLPController(path)
