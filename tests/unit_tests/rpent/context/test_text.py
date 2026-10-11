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

"""Offline contracts for public text selection and untouched native protocol."""

import copy
import json

import pytest

from rpent.context.anchor import AnchorAdapter
from rpent.context.codex import CodexAdapter
from rpent.context.session import ContextSession


class Features:
    def similarity(self, text, query):
        return 0.6


def history():
    items = [
        {
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": "Never discard rules"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Place the cup"}],
        },
        {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
    ]
    for i in range(8):
        items.extend(
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "id": f"m{i}",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"Plan {i}: I will inspect the cup and destination before the next movement.",
                        }
                    ],
                },
                {
                    "type": "custom_tool_call",
                    "call_id": f"call-{i}",
                    "input": "observe",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": f"call-{i}",
                    "output": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {"task_language": "Place the cup", "success": False}
                            ),
                        }
                    ],
                },
            ]
        )
    return {"input": items, "model": "test"}


def test_text_only_preserves_tools_roles_reasoning_and_recent_plans():
    source = history()
    original = copy.deepcopy(source)
    session = ContextSession(CodexAdapter(), Features(), mode="text")
    result = session.process(source)
    assert source == original
    assert result != source
    kept_ids = {x.get("id") for x in result["input"] if x.get("role") == "assistant"}
    expected = [
        x
        for x in source["input"]
        if x.get("role") != "assistant" or x.get("id") in kept_ids
    ]
    assert result["input"] == expected
    assert len(result["input"]) < len(source["input"])
    assert all("Earlier assistant text omitted" not in str(x) for x in result["input"])
    records = session.stats["text"]["records"]
    assert all(r["selected"] and r["protected"] for r in records[-2:])
    assert records[0]["novelty"] is None
    assert abs(records[0]["score"] - 0.2) < 1e-9
    assert abs(records[1]["score"] - 0.28) < 1e-9
    assert records[1]["novelty_contribution"] == 0.24
    assert session.stats["text"]["nc"] >= 0.5
    assert session.process(source) == result


def test_event_requires_explicit_reference_to_observed_tool_outcome():
    source = history()
    source["input"][3]["content"][0]["text"] += " success failed error"
    records, _ = CodexAdapter().extract_text(source)
    assert not records[0].event_refs
    source["input"][6]["content"][0]["text"] += (
        " The failure reported by call-0 requires a retry."
    )
    records, _ = CodexAdapter().extract_text(source)
    assert records[1].event_refs == ("call-0",)


def test_incremental_text_does_not_resurrect_omitted_messages():
    source = history()
    session = ContextSession(CodexAdapter(), Features(), mode="text")
    session.process(source)
    dropped = {r["id"] for r in session.stats["text"]["records"] if not r["selected"]}
    assert dropped
    source["input"].append(
        {
            "type": "message",
            "role": "assistant",
            "id": "new",
            "content": [
                {
                    "type": "output_text",
                    "text": "A new sufficiently long public plan to inspect the cup and verify the next movement.",
                }
            ],
        }
    )
    session.process(source)
    assert all(
        not r["selected"] and r["score"] is None
        for r in session.stats["text"]["records"]
        if r["id"] in dropped
    )


def test_unknown_and_incremental_formats_are_preserved():
    source = history()
    source["previous_response_id"] = "remote"
    assert (
        ContextSession(CodexAdapter(), Features(), mode="text").process(source)
        is source
    )
    source.pop("previous_response_id")
    for item in source["input"]:
        if item.get("role") == "assistant":
            item["content"].append({"type": "unknown", "data": "must retain"})
    assert (
        ContextSession(CodexAdapter(), Features(), mode="text").process(source)
        is source
    )


def test_explicit_user_update_is_in_query_and_never_selected():
    source = history()
    source["input"].append(
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Use the tray instead of the basket"}
            ],
        }
    )
    records, query = CodexAdapter().extract_text(source)
    assert "Use the tray instead of the basket" in query
    assert len(records) == 8


def test_task_query_fallback_keeps_user_correction():
    source = history()
    source["input"][1]["content"][0]["text"] = "Begin the configured task."
    source["input"].append(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Avoid the left edge"}],
        }
    )
    _, query = CodexAdapter().extract_text(source)
    assert "Avoid the left edge" in query


def test_annotated_output_text_is_not_rewritten():
    source = history()
    for item in source["input"]:
        if item.get("role") == "assistant":
            item["content"][0]["annotations"] = [{"type": "citation", "ref": "keep"}]
    result = ContextSession(CodexAdapter(), Features(), mode="text").process(source)
    assert result is source


def test_unrelated_novel_text_does_not_displace_relevant_duplicate():
    class SemanticFeatures:
        def similarity(self, left, right):
            if right.strip() == "Place the cup":
                return 0.0 if "weather" in left else 0.9
            return 0.1 if "weather" in left or "weather" in right else 1.0

    source = history()
    messages = [x for x in source["input"] if x.get("role") == "assistant"]
    for message in messages:
        message["content"][0]["text"] = "Grasp the cup."
    messages[1]["content"][0]["text"] = (
        "Grasp the cup. This repeats the existing plan to approach and grasp the cup."
    )
    messages[2]["content"][0]["text"] = (
        "The weather is sunny today. This unrelated sentence adds no task information."
    )
    session = ContextSession(CodexAdapter(), SemanticFeatures(), mode="text")
    result = session.process(source)
    kept = {x.get("id") for x in result["input"]}
    assert "m1" in kept
    assert "m2" not in kept


def test_zero_relevance_does_not_remove_explicit_event_credit():
    class UnrelatedFeatures:
        def similarity(self, text, query):
            return 0.0

    source = history()
    source["input"][6]["content"][0]["text"] += " Check the outcome of call-0."
    session = ContextSession(CodexAdapter(), UnrelatedFeatures(), mode="text")
    session.process(source)
    record = session.stats["text"]["records"][1]
    assert record["event"] == 1
    assert record["novelty_contribution"] == 0.0
    assert record["score"] == 1 / 3
    assert record["selected"]


def test_task_restatement_yields_to_execution_update_after_recent_protection():
    task = (
        "The task is to place the cup in the tray. The cup and destination "
        "are identified in the initial observation."
    )
    offset = (
        "The held cup is seven centimetres ahead of the gripper; compensate "
        "for this offset before lowering it into the tray."
    )

    class SemanticFeatures:
        def similarity(self, text, query):
            if query.strip() == "Place the cup":
                return {task: 0.6991, offset: 0.6005}.get(text, 0.1)
            if {text, query} == {task, offset}:
                return 1 - 0.5153
            return 0.2

    source = history()
    messages = [x for x in source["input"] if x.get("role") == "assistant"]
    messages[0]["content"][0]["text"] = task
    messages[1]["content"][0]["text"] = offset
    session = ContextSession(CodexAdapter(), SemanticFeatures(), mode="text", rho=0.1)
    # Both messages begin protected. The task then becomes the oldest optional
    # record; when the offset also ages out, it must be able to replace it.
    for count in (2, 3, 4):
        request = dict(source, input=source["input"][: 3 + 3 * count])
        result = session.process(request)

    records = {r["id"]: r for r in session.stats["text"]["records"]}
    assert records["m0"]["novelty"] is None
    assert not records["m0"]["protected"]
    assert not records["m1"]["protected"]
    assert not records["m0"]["selected"]
    assert records["m1"]["selected"]
    assert records["m2"]["selected"] and records["m3"]["selected"]
    assert result["input"][:3] == source["input"][:3]
    assert session.process(request) == result

    # The omitted restatement is not resurrected when the history grows.
    session.process(dict(source, input=source["input"][: 3 + 3 * 5]))
    records = {r["id"]: r for r in session.stats["text"]["records"]}
    assert records["m0"]["reason"] == "previously_omitted"
    assert not records["m0"]["selected"]


def test_missing_reference_does_not_outrank_equally_relevant_duplicate():
    class DuplicateFeatures:
        def similarity(self, text, query):
            return 0.6 if query.strip() == "Place the cup" else 1.0

    source = history()
    source["input"] = source["input"][: 3 + 3 * 4]
    for item in source["input"]:
        if item.get("role") == "assistant":
            item["content"][0]["text"] = (
                "The cup remains in the gripper. Continue holding it closed "
                "while moving toward the tray."
            )
    session = ContextSession(CodexAdapter(), DuplicateFeatures(), mode="text", rho=0.1)
    session.process(source)
    records = {r["id"]: r for r in session.stats["text"]["records"]}
    assert records["m0"]["novelty"] is None
    assert records["m1"]["novelty"] == 0.0
    # Neither has evidence of new information; the existing recency tie-break
    # should select the newer occurrence, not reward the missing reference.
    assert not records["m0"]["selected"]
    assert records["m1"]["selected"]


@pytest.mark.parametrize("adapter_type", [CodexAdapter, AnchorAdapter])
@pytest.mark.parametrize(
    "coordinates",
    [
        {"world_xyz": [-0.11, -0.23, 0.018]},
        {"center_xyz": [-0.045, 0.284, 0.097], "median_xyz": [-0.045, 0.284, 0.097]},
        {
            "results": [
                {"world_xyz": [1.0, 2.0, 3.0], "valid": True, "error": None},
                {"world_xyz": None, "valid": False, "error": "invalid depth"},
            ],
            "summary": {"median_xyz": [1.0, 2.0, 3.0], "valid_count": 1, "total_count": 2},
        },
        {
            "results": [
                {"world_xyz": None, "valid": False, "error": "invalid depth"},
            ],
            "summary": {"median_xyz": None, "valid_count": 0, "total_count": 1},
        },
    ],
)
def test_old_localization_results_survive_text_selection(adapter_type, coordinates):
    source = history()
    output = next(
        item
        for item in source["input"]
        if item.get("type") == "custom_tool_call_output"
    )
    text = json.dumps(
        {
            "camera": "agentview",
            "resolution": "high",
            "step": 0,
            "source_artifact": "agentview_world_high.npz",
            **coordinates,
        }
    )
    output["output"][0]["text"] = text
    original = copy.deepcopy(source)
    session = ContextSession(adapter_type(), Features(), mode="text", rho_text=0.1)

    result = session.process(source)

    assert source == original
    assert any(not record["selected"] for record in session.stats["text"]["records"])
    assert all(record["text"] != text for record in session.stats["text"]["records"])
    retained = next(
        item
        for item in result["input"]
        if item.get("call_id") == output["call_id"]
        and item.get("type") == "custom_tool_call_output"
    )
    assert retained == output
