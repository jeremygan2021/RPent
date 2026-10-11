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

import gzip
import json
from copy import deepcopy

import pytest

from rpent.context.cache_prefix import OMITTED, REFERENCE_ID, ToolReference


@pytest.fixture
def reference(tmp_path):
    tool = {
        "name": "mcp__rpent__back_project",
        "description": "Use row and col",
        "parameters": {"row": "int", "col": "int"},
    }
    path = tmp_path / "tools.json"
    path.write_text(json.dumps([tool]))
    return ToolReference(path), tool


def test_fixed_prefix_and_pairs_preserved(reference):
    ref, tool = reference
    items = [
        {
            "type": "message",
            "role": "developer",
            "id": "rules",
            "content": [{"type": "input_text", "text": "rules"}],
        },
        {
            "type": "message",
            "role": "user",
            "id": "task",
            "content": [{"type": "input_text", "text": "task"}],
        },
        {
            "type": "custom_tool_call",
            "id": "call",
            "call_id": "a",
            "input": "query tools",
        },
        {
            "type": "custom_tool_call_output",
            "id": "result",
            "call_id": "a",
            "output": [{"type": "input_text", "text": json.dumps([tool])}],
        },
    ]
    request = {"input": items}
    before = deepcopy(request)
    out, stats = ref.apply(request)
    assert request == before
    assert out["input"][1]["id"] == REFERENCE_ID
    assert out["input"][-2] == items[-2]
    assert out["input"][-1]["call_id"] == "a"
    assert out["input"][-1]["output"][0]["text"] == OMITTED
    assert stats["duplicates_removed"] == 1
    changed = deepcopy(request)
    changed["input"][1]["content"][0]["text"] = "new task observation"
    assert ref.apply(changed)[0]["input"][:2] == out["input"][:2]


def test_unknown_and_non_schema_results_preserved(reference):
    ref, tool = reference
    unknown = dict(tool, name="other")
    request = {
        "input": [
            {
                "type": "custom_tool_call_output",
                "output": [
                    {"type": "input_text", "text": json.dumps([tool, unknown])},
                    {"type": "input_text", "text": "result row=50 col=60"},
                ],
            }
        ]
    }
    out, _ = ref.apply(request)
    assert json.loads(out["input"][-1]["output"][0]["text"]) == [unknown]
    assert out["input"][-1]["output"][1] == request["input"][0]["output"][1]


def test_changed_definition_rejected(reference):
    ref, tool = reference
    request = {
        "input": [
            {
                "type": "custom_tool_call_output",
                "output": [
                    {
                        "type": "input_text",
                        "text": json.dumps(dict(tool, description="different")),
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="changed"):
        ref.apply(request)


def test_named_query_result_is_not_deduplicated_or_rejected(reference):
    ref, tool = reference
    result = {
        "name": tool["name"],
        "description": "returned point",
        "world_xyz": [0.1, 0.2, 0.3],
    }
    request = {
        "input": [
            {
                "type": "custom_tool_call_output",
                "id": "queryout",
                "call_id": "query",
                "output": [{"type": "input_text", "text": json.dumps(result)}],
            }
        ]
    }
    out, stats = ref.apply(request)
    assert out["input"][-1] == request["input"][0]
    assert stats["duplicates_removed"] == 0


def test_reference_rejects_result_metadata_without_an_interface(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps([{"name": "cream_cheese", "description": "blue package"}])
    )
    with pytest.raises(ValueError, match="explicit interface"):
        ToolReference(path)


def test_reference_and_duplicate_markers_are_rules_not_reward_cost(reference):
    from rpent.context.reward import account
    from tests.unit_tests.rpent.context.test_toolrules_pose import usage_for

    ref, tool = reference
    request = {
        "input": [
            {
                "type": "custom_tool_call_output",
                "id": "out",
                "call_id": "a",
                "output": [{"type": "input_text", "text": json.dumps([tool])}],
            }
        ]
    }
    out, _ = ref.apply(request)
    usage = usage_for(out)
    costs = account(out, usage)
    assert costs["tool_rules"] == 20
    assert costs.get("text", 0) == 0
    assert costs.get("memory", 0) == 0
    assert sum(costs.values()) == usage["input_tokens"]


def test_transport_uses_supported_message_id_and_keeps_cache_key(
    reference, monkeypatch
):
    from types import SimpleNamespace

    from rpent.context.proxy import rewrite_body

    ref, _ = reference
    session = SimpleNamespace(
        process=lambda req: req, stats={}, _fixed_tool_reference=ref
    )
    monkeypatch.setenv("RPENT_FIXED_TOOL_REFERENCE", "already-loaded")
    request = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "id": "msg_task",
                "content": [{"type": "input_text", "text": "task"}],
            }
        ],
        "prompt_cache_key": "same-episode",
    }
    data, changed = rewrite_body(json.dumps(request).encode(), "", session)
    result = json.loads(data)
    assert changed
    assert result["input"][0]["id"].startswith("msg")
    assert result["prompt_cache_key"] == request["prompt_cache_key"]


def test_fixed_reference_enabled_without_environment(monkeypatch):
    from types import SimpleNamespace

    from rpent.context.proxy import rewrite_body

    monkeypatch.delenv("RPENT_FIXED_TOOL_REFERENCE", raising=False)
    session = SimpleNamespace(process=lambda req: req, stats={}, mode="all")
    request = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "id": "msg_task",
                "content": [{"type": "input_text", "text": "task"}],
            }
        ]
    }
    data, changed = rewrite_body(json.dumps(request).encode(), "", session)
    out = json.loads(data)
    assert changed
    assert out["input"][0]["id"] == REFERENCE_ID
    assert len(json.loads(out["input"][0]["content"][0]["text"])) == 12
    assert out["input"][1] == request["input"][0]
    assert session.stats["cache_prefix"]["enabled"]


def test_full_forwarding_does_not_add_reference(monkeypatch):
    from types import SimpleNamespace

    from rpent.context.proxy import rewrite_body

    monkeypatch.delenv("RPENT_FIXED_TOOL_REFERENCE", raising=False)
    session = SimpleNamespace(process=lambda req: req, stats={}, mode="full")
    raw = b'{ "input": [{"type":"message", "role":"user", "content": []}] }'
    assert rewrite_body(raw, "", session) == (raw, False)


@pytest.mark.parametrize("encoding", ["", "gzip"])
@pytest.mark.parametrize("primed", [False, True])
def test_incremental_transport_preserves_original_bytes(monkeypatch, encoding, primed):
    from rpent.context.anchor import AnchorAdapter
    from rpent.context.proxy import rewrite_body
    from rpent.context.session import ContextSession
    from tests.unit_tests.rpent.context.test_toolrules_pose import Features

    session = ContextSession(AnchorAdapter(), Features(), mode="all")
    monkeypatch.delenv("RPENT_FIXED_TOOL_REFERENCE", raising=False)
    if primed:
        full = b'{"input": [{"type": "message", "role": "user", "content": []}]}'
        result, changed = rewrite_body(full, "", session)
        assert changed
        assert json.loads(result)["input"][0]["id"] == REFERENCE_ID
    else:
        monkeypatch.setenv("RPENT_FIXED_TOOL_REFERENCE", "/nonexistent/tools.json")
    raw = (
        b'{ "previous_response_id": "resp_previous", "prompt_cache_key": "same-episode",'
        b' "input": [{"type": "custom_tool_call_output", "call_id": "call_previous",'
        b' "output": [{"type": "input_text", "text": "new observation"}]}] }'
    )
    if encoding == "gzip":
        raw = gzip.compress(raw)
    assert rewrite_body(raw, encoding, session) == (raw, False)
    assert rewrite_body(raw, encoding, session) == (raw, False)
    assert "cache_prefix" not in session.stats
    assert session.stats["decision"]["reused"]


def test_reference_still_rejects_incremental_requests(reference):
    ref, _ = reference
    with pytest.raises(ValueError, match="requires full-history requests"):
        ref.apply({"input": [], "previous_response_id": "resp_previous"})


@pytest.mark.parametrize("selected", ["old experience", "different experience", ""])
def test_memory_tail_keeps_fixed_prefix_and_history(reference, selected):
    ref, _ = reference

    def request(memory):
        return {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "id": "msg_task",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "task\n[RPENT_MEMORY]\n"
                            + memory
                            + "\n[/RPENT_MEMORY]\nfixed final rule",
                        }
                    ],
                },
                {
                    "type": "custom_tool_call",
                    "id": "call",
                    "call_id": "x",
                    "input": "observe",
                },
                {
                    "type": "custom_tool_call_output",
                    "id": "out",
                    "call_id": "x",
                    "output": [{"type": "input_text", "text": "observation"}],
                },
            ],
            "prompt_cache_key": "episode",
        }

    original = request(selected)
    snapshot = deepcopy(original)
    out, stats = ref.apply(original)
    other, _ = ref.apply(request("another selection"))
    assert original == snapshot
    assert out["input"][:-1] == other["input"][:-1]
    assert out["input"][2:4] == original["input"][1:]
    assert out["input"][-1]["content"][0]["text"].endswith(
        "[RPENT_MEMORY]\n" + selected + "\n[/RPENT_MEMORY]"
    )
    assert "fixed final rule" in out["input"][1]["content"][0]["text"]
    assert out["prompt_cache_key"] == "episode"
    assert ref.apply(original)[0] == out
    assert stats["memory_moved"]
    from rpent.context.reward import account
    from tests.unit_tests.rpent.context.test_toolrules_pose import usage_for

    usage = usage_for(out)
    costs = account(out, usage)
    assert costs["memory"] >= 0
    assert abs(sum(costs.values()) - usage["input_tokens"]) < 1e-6


@pytest.mark.parametrize("mode", ["full", "all"])
def test_native_tools_preserved_without_duplicate_reference_and_memory_at_tail(
    reference, mode
):
    from types import SimpleNamespace

    from rpent.context.proxy import rewrite_body

    ref, tool = reference
    native = {
        "type": "namespace",
        "name": "mcp__rpent",
        "tools": [
            {
                "type": "function",
                "name": "back_project",
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
        ],
    }
    request = {
        "tools": [native],
        "input": [
            {
                "type": "message",
                "role": "user",
                "id": "msg_task",
                "content": [
                    {
                        "type": "input_text",
                        "text": "task\n[RPENT_MEMORY]\nexperience\n[/RPENT_MEMORY]\nrule",
                    }
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "id": "msg_history",
                "content": [{"type": "output_text", "text": "history"}],
            },
        ],
        "prompt_cache_key": "episode",
    }
    before = deepcopy(request)
    session = SimpleNamespace(
        process=lambda req: req, stats={}, mode=mode, _fixed_tool_reference=ref
    )

    raw, changed = rewrite_body(json.dumps(request).encode(), "", session)
    out = json.loads(raw)

    assert changed
    assert request == before
    assert out["tools"] == request["tools"]
    assert not any(item.get("id") == REFERENCE_ID for item in out["input"])
    assert out["input"][1] == request["input"][1]
    assert out["input"][-1]["id"] == "msg_rpent_dynamic_memory_v1"
    assert (
        "[RPENT_MEMORY]\nexperience\n[/RPENT_MEMORY]"
        in (out["input"][-1]["content"][0]["text"])
    )
    assert out["prompt_cache_key"] == "episode"
    assert not session.stats["cache_prefix"]["enabled"]
    forwarded, changed_again = rewrite_body(raw, "", session)
    assert not changed_again
    assert forwarded == raw


def test_partial_native_interface_keeps_legacy_reference(reference):
    from types import SimpleNamespace

    from rpent.context.proxy import rewrite_body

    ref, tool = reference
    request = {
        "tools": [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
            }
        ],
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "task"}]}
        ],
    }
    session = SimpleNamespace(
        process=lambda req: req, stats={}, mode="all", _fixed_tool_reference=ref
    )
    raw, _ = rewrite_body(json.dumps(request).encode(), "", session)
    assert json.loads(raw)["input"][0]["id"] == REFERENCE_ID
