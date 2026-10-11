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

"""Memory parent restoration, re-entry and prompt isolation contracts."""

import copy
import json
from types import SimpleNamespace

import pytest

from rpent.context.memory import END, START, MemorySelector
from rpent.context.session import ContextSession


class Features:
    def __init__(self):
        self.scores = {}

    def similarity(self, text, query):
        return self.scores.get(text, 0.1)


class Adapter:
    task = "Pick object"

    def extract_action(self, request):
        return [SimpleNamespace(snapshot={"task_language": self.task})]


def setup(tmp_path):
    blocks = [
        {"id": "identify", "text": "Object is blue; distinguish it."},
        {"id": "place", "text": "Place inside basket."},
    ]
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(blocks))
    memory = "\n" + "\n".join(block["text"] for block in blocks) + "\n"
    request = {
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": "Rules"}]},
            {
                "role": "user",
                "id": "u1",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Goal\n{START}{memory}{END}\nFixed rules",
                    }
                ],
            },
            {"type": "reasoning", "encrypted_content": "opaque"},
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Found the basket. Pick object."}
                ],
            },
            {
                "type": "custom_tool_call_output",
                "output": [{"type": "input_text", "text": "DO NOT USE THIS AS QUERY"}],
            },
        ],
        "tools": [{"name": "move"}],
    }
    return MemorySelector(path), request, Features(), Adapter()


def test_parent_completion_and_observed_task_with_public_sentence_query(tmp_path):
    selector, request, features, adapter = setup(tmp_path)
    features.scores = {"Object is blue; distinguish it.": 0.9}
    original = copy.deepcopy(request)
    result, stats = selector.process(request, adapter, features)
    assert stats["query"] == "Found the basket. Pick object."
    assert stats["selected_blocks"] == ["identify"]
    assert stats["selection_nc"] >= 0.5
    assert "Object is blue; distinguish it." in result["input"][1]["content"][0]["text"]
    assert "Place inside basket." not in result["input"][1]["content"][0]["text"]
    assert request == original
    restored = copy.deepcopy(result)
    restored["input"][1]["content"][0]["text"] = original["input"][1]["content"][0][
        "text"
    ]
    assert restored == original


def test_deselected_experience_can_return(tmp_path):
    selector, request, features, adapter = setup(tmp_path)
    features.scores = {"Object is blue; distinguish it.": 0.9}
    first, stats = selector.process(request, adapter, features)
    assert stats["omitted_blocks"] == ["place"]
    features.scores = {"Place inside basket.": 0.9}
    second, stats = selector.process(first, adapter, features)
    assert stats["selected_blocks"] == ["place"]
    features.scores = {"Object is blue; distinguish it.": 0.9}
    third, stats = selector.process(second, adapter, features)
    assert stats["selected_blocks"] == ["identify"]
    assert "distinguish it." in third["input"][1]["content"][0]["text"]


def test_unknown_task_keeps_all_and_missing_plan_uses_observed_task(tmp_path):
    selector, request, features, adapter = setup(tmp_path)
    adapter.task = ""
    result, stats = selector.process(request, adapter, features)
    assert result is request
    assert stats["selected_blocks"] == ["identify", "place"]
    adapter.task = "Pick object"
    request["input"][3]["content"] = []
    _, stats = selector.process(request, adapter, features)
    assert stats["selected_blocks"]
    assert stats["query"] == "Pick object"


def test_full_memory_threshold_and_missing_region(tmp_path):
    selector, request, features, adapter = setup(tmp_path)
    selector.rho = 1.0
    result, stats = selector.process(request, adapter, features)
    assert result is request
    assert stats["selection_nc"] == 1.0
    request["input"][1]["content"][0]["text"] = "Missing markers"
    with pytest.raises(ValueError, match="managed user region"):
        selector.process(request, adapter, features)


def test_full_session_preserves_memory_stats_and_retries(tmp_path):
    selector, request, features, _ = setup(tmp_path)
    from rpent.context.codex import CodexAdapter

    session = ContextSession(CodexAdapter(), features, memory_selector=selector)
    first = session.process(request)
    stats = copy.deepcopy(session.stats["memory"])
    assert session.process(request) == first
    assert session.stats["memory"] == stats
    assert session.stats["decision"]["reused"]


@pytest.mark.parametrize(
    "content",
    [
        "【抓取】\n夹住目标并抬起。\n\n【搬运】\n保持物体高于障碍物。\n",
        "【Grasp】\nGrasp the package.\n\n【Transfer】\nKeep clearance above groceries.\n",
    ],
)
def test_titled_demonstration_loads_complete_blocks(tmp_path, content):
    from rpent.context.memory import append_memory, load_memory_blocks

    path = tmp_path / "memory.txt"
    path.write_text(content, encoding="utf-8")
    blocks = load_memory_blocks(path)
    assert [b["id"] for b in blocks] == ["block_001", "block_002"]
    assert blocks[0]["text"] == content.split("\n\n")[0]
    prompt = append_memory("Fixed rules", blocks)
    assert prompt.startswith("Fixed rules")
    assert prompt.count(START) == prompt.count(END) == 1
    assert blocks[0]["text"] in prompt
    assert blocks[1]["text"] in prompt
    assert append_memory("Fixed rules", []) == "Fixed rules"


@pytest.mark.parametrize(
    "content",
    [
        "Unstructured memory",
        "【Empty】\n",
        "Preface\n【Topic】\nBody",
        "【Topic】\n[RPENT_MEMORY] bad marker",
    ],
)
def test_invalid_demonstration_rejected(tmp_path, content):
    from rpent.context.memory import load_memory_blocks

    path = tmp_path / "memory.txt"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_memory_blocks(path)


def test_initial_prompt_must_match_selector_library(tmp_path):
    selector, request, features, adapter = setup(tmp_path)
    adapter.task = ""
    request["input"][1]["content"][0]["text"] = f"{START}Old experience{END}"
    with pytest.raises(ValueError, match="differs from selector library"):
        selector.process(request, adapter, features)


def test_demo_region_survives_request_rewrite_with_protocol_unchanged(tmp_path):
    from rpent.context.codex import CodexAdapter
    from rpent.context.memory import append_memory, load_memory_blocks
    from rpent.context.proxy import rewrite_body

    path = tmp_path / "demo.txt"
    path.write_text(
        "【Identify】\nIdentify the package.\n\n【Place】\nLower inside basket.\n"
    )
    blocks = load_memory_blocks(path)
    request = {
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": "Fixed protocol"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": append_memory("Goal", blocks)}
                ],
            },
            {"type": "reasoning", "encrypted_content": "opaque"},
        ],
        "tools": [{"name": "move"}],
    }
    session = ContextSession(
        CodexAdapter(), Features(), memory_selector=MemorySelector(path)
    )
    raw = json.dumps(request).encode()
    processed, changed = rewrite_body(raw, "", session)
    initial = json.loads(processed)
    assert initial["tools"] == request["tools"]
    assert initial["input"][0] == request["input"][0]
    initial_text = "\n".join(
        part.get("text", "")
        for item in initial["input"]
        for part in item.get("content", [])
    )
    assert all(block["text"] in initial_text for block in blocks)
    request["input"] += [
        {
            "type": "custom_tool_call",
            "call_id": "c1",
            "name": "exec",
            "input": "observe",
        },
        {
            "type": "custom_tool_call_output",
            "call_id": "c1",
            "output": [
                {
                    "type": "input_text",
                    "text": json.dumps(
                        {
                            "step": 0,
                            "state": {},
                            "log": {"command": None, "result": None},
                            "task_language": "Pick package",
                        }
                    ),
                }
            ],
        },
    ]
    # Task fallback scores whole blocks; only the memory region may change.
    original = copy.deepcopy(request)
    processed, changed = rewrite_body(json.dumps(request).encode(), "", session)
    assert changed
    result = json.loads(processed)
    assert session.stats["memory"]["selected_blocks"] == ["block_001", "block_002"]
    full_session = ContextSession(
        CodexAdapter(), Features(), memory_selector=MemorySelector(path, rho=1)
    )
    full, _ = rewrite_body(json.dumps(original).encode(), "", full_session)
    expected = json.loads(full)
    for index, item in enumerate(result["input"]):
        if item.get("id") == "msg_rpent_dynamic_memory_v1":
            result["input"][index] = expected["input"][index]
    assert result == expected


def test_learned_memory_threshold_controls_selection_and_retry(tmp_path):
    pytest.importorskip("torch")
    from rpent.context.controller import Thresholds
    from rpent.context.policy import MLPController, initialize, save_checkpoint

    class SessionAdapter(Adapter):
        def extract(self, request):
            return []

        def extract_text(self, request):
            return [], ""

        def extract_action(self, request):
            return [
                SimpleNamespace(
                    id="a", protected=True, snapshot={"task_language": self.task}
                )
            ]

        def render(self, request, records, kept):
            return request

    counts = []
    for rho in (0.2, 0.9):
        selector, source, features, _ = setup(tmp_path)
        path = tmp_path / f"memory-{rho}.pt"
        save_checkpoint(initialize(Thresholds(0.2, 0.1, 0.1, rho)), path)
        policy = MLPController(path)
        session = ContextSession(
            SessionAdapter(),
            features,
            mode="image",
            controller=policy,
            memory_selector=selector,
        )
        result = session.process(source)
        assert session.stats["memory"]["rho"] == pytest.approx(rho)
        assert session.stats["decision"]["thresholds"]["memory"] == pytest.approx(rho)
        obs = session.stats["decision"]["observation"]
        assert dict(zip(obs["names"], obs["values"]))["memory.enabled"] == 1
        assert session.process(source) == result
        assert session.stats["decision"]["reused"]
        counts.append(len(session.stats["memory"]["selected_blocks"]))
    assert counts == [1, 2]


def test_reasoning_summary_drives_whole_block_selection_and_reentry(tmp_path):
    selector, request, features, adapter = setup(tmp_path)
    request["input"].append(
        {
            "type": "reasoning",
            "encrypted_content": "must not be used",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "Lower the held object into the basket.",
                }
            ],
        }
    )
    features.scores = {"Place inside basket.": 0.9}
    result, stats = selector.process(request, adapter, features)
    assert stats["query"] == "Lower the held object into the basket."
    assert stats["query_source"] == "reasoning_summary"
    assert stats["score_rule"] == "whole_block_cosine"
    assert stats["selected_blocks"] == ["place"]
    assert len(stats["records"]) == 2
    result["input"].append(
        {
            "type": "reasoning",
            "summary": [
                {"type": "summary_text", "text": "Identify the blue object again."}
            ],
        }
    )
    features.scores = {"Object is blue; distinguish it.": 0.9}
    _, stats = selector.process(result, adapter, features)
    assert stats["selected_blocks"] == ["identify"]


def test_empty_summary_and_tool_text_do_not_override_public_query(tmp_path):
    selector, request, features, adapter = setup(tmp_path)
    request["input"].append(
        {"type": "reasoning", "summary": [], "encrypted_content": "opaque"}
    )
    _, stats = selector.process(request, adapter, features)
    assert stats["query"] == "Found the basket. Pick object."
    assert stats["query_source"] == "public_text"


def _append_recovery_result(request, call_id, action, outcome, *, step=1, **flags):
    request["input"].append(
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": [
                {
                    "type": "input_text",
                    "text": json.dumps(
                        {
                            "step": step,
                            "state": {},
                            "task_language": "Pick object",
                            "log": {"command": {"action": action}, "result": outcome},
                            "terminated": False,
                            "truncated": False,
                            **flags,
                        }
                    ),
                }
            ],
        }
    )


@pytest.mark.parametrize("action", ["move_to", "navigate_to"])
def test_execution_error_restores_all_memory_until_action_success(tmp_path, action):
    selector, request, features, adapter = setup(tmp_path)
    features.scores = {"Object is blue; distinguish it.": 0.9}
    original = copy.deepcopy(request)
    _, normal = selector.process(request, adapter, features)
    assert normal["selected_blocks"] == ["identify"]
    assert request == original

    _append_recovery_result(request, "failed", action, {"ok": False})
    _, restored = selector.process(request, adapter, features)
    assert restored["reason"] == "execution_error_keep_all"
    assert restored["selected_blocks"] == ["identify", "place"]

    _append_recovery_result(request, "observed", "view_env_state", {"ok": True}, step=2)
    _, observed = selector.process(request, adapter, features)
    assert observed["recovery"]["active"]
    _append_recovery_result(request, "unknown", "release", {}, step=3)
    _, unknown = selector.process(request, adapter, features)
    assert unknown["recovery"]["active"]

    _append_recovery_result(request, "recovered", action, {"ok": True}, step=4)
    _, recovered = selector.process(request, adapter, features)
    assert not recovered["recovery"]["active"]
    assert recovered["selected_blocks"] == normal["selected_blocks"]

    # Viewing the old failed snapshot must not resurrect an old error.
    _append_recovery_result(request, "old-view", action, {"ok": False})
    _, repeated = selector.process(request, adapter, features)
    assert not repeated["recovery"]["active"]
    assert repeated == selector.process(request, adapter, features)[1]


@pytest.mark.parametrize("outcome", [{"success": False}, {"grasped": False}, {"ok": True}])
def test_unconfirmed_grasp_does_not_trigger_memory_recovery(tmp_path, outcome):
    selector, request, features, adapter = setup(tmp_path)
    features.scores = {"Object is blue; distinguish it.": 0.9}
    _append_recovery_result(request, "pick", "pi0_pick", outcome)
    _, stats = selector.process(request, adapter, features)
    assert not stats["recovery"]["active"]
    assert stats["selected_blocks"] == ["identify"]


@pytest.mark.parametrize(
    "output",
    [
        json.dumps({"error": "RPC execution failed"}),
        json.dumps({"isError": True, "content": []}),
        "Wall time: 1 seconds\nOutput:\nTraceback (most recent call last):\nRPC failed",
        "Uncaught Error: RPC failed",
    ],
)
def test_tool_exceptions_restore_memory_without_action_snapshot(tmp_path, output):
    selector, request, features, adapter = setup(tmp_path)
    features.scores = {"Object is blue; distinguish it.": 0.9}
    request["input"].append(
        {"type": "custom_tool_call_output", "call_id": "error", "output": output}
    )
    _, stats = selector.process(request, adapter, features)
    assert stats["selected_blocks"] == ["identify", "place"]
    assert stats["recovery"]["trigger_call_id"] == "error"


@pytest.mark.parametrize("flag", ["terminated", "truncated"])
def test_episode_end_clears_memory_recovery(tmp_path, flag):
    selector, request, features, adapter = setup(tmp_path)
    _append_recovery_result(request, "failed", "move_to", {"ok": False})
    _append_recovery_result(request, "done", "move_to", {}, step=2, **{flag: True})
    _, stats = selector.process(request, adapter, features)
    assert not stats["recovery"]["active"]
