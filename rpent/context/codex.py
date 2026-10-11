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

"""Conservative adapter for full-history Codex Responses requests."""

import base64
import copy
import hashlib
import json
from dataclasses import dataclass

_OUTPUT_TYPES = {
    "custom_tool_call_output": "custom_tool_call",
    "function_call_output": "function_call",
}
_CALL_TYPES = set(_OUTPUT_TYPES.values())


def action_summary(snapshot):
    """Retain decision evidence while omitting detailed robot diagnostics."""
    log = snapshot["log"]
    command = log.get("command") or {}
    outcome = log.get("result") or {}
    return {
        "command": {k: command[k] for k in (
            "action", "name", "instruction", "prompt", "xyz", "xy", "dxyz", "gripper"
        ) if k in command},
        "result": {k: outcome[k] for k in (
            "name", "ok", "success", "error", "status", "terminated", "truncated",
            "grasped", "grasp_detected", "grasp_contact", "held_apart", "grasp_obj",
            "chunks", "steps_applied", "pos_error", "distance", "target_distance"
        ) if k in outcome},
    }


def _paired_outputs(items):
    """Recognize results only when the matching native call type is present."""
    calls = {(x.get("type"), x.get("call_id")) for x in items if x.get("call_id")}
    return {
        i
        for i, x in enumerate(items)
        if x.get("call_id")
        and (_OUTPUT_TYPES.get(x.get("type")), x.get("call_id")) in calls
    }


@dataclass(frozen=True)
class ImageRecord:
    """One image occurrence; content hashes are not occurrence identifiers."""

    id: str
    item: int
    block: int
    camera: str
    metadata: str
    task: str
    digest: str
    data: bytes
    protected: bool
    initial: bool = False


class CodexAdapter:
    """Extract only recognized snapshot images; preserve all other content."""

    def __init__(self, cameras: tuple[str, ...] = (), *, extended_history=True):
        self.cameras = cameras
        self.extended_history = extended_history

    def extract(self, request: dict) -> list[ImageRecord]:
        records = []
        items = request.get("input", [])
        if not isinstance(items, list) or request.get("previous_response_id"):
            return records
        paired = _paired_outputs(items)
        completed = [x.get("call_id") for i, x in enumerate(items) if i in paired]
        recent = set(completed[-2:])
        calls = {x.get("call_id") for x in items if x.get("type") in _CALL_TYPES}
        for mi, item in enumerate(items):
            if mi not in paired or item.get("call_id") not in calls:
                continue
            snapshot, camera_index = None, 0
            output = item.get("output")
            if not isinstance(output, list):
                continue
            for bi, block in enumerate(output):
                if block.get("type") == "input_text":
                    try:
                        value = json.loads(block.get("text", ""))
                    except ValueError:
                        value = None
                    snapshot = (
                        value
                        if isinstance(value, dict)
                        and "step" in value
                        and "state" in value
                        else None
                    )
                    camera_index = 0
                elif block.get("type") == "input_image" and snapshot is not None:
                    url = block.get("image_url")
                    if (
                        not isinstance(url, str)
                        or not url.startswith("data:image/")
                        or ";base64," not in url
                    ):
                        continue
                    data = base64.b64decode(url.split(",", 1)[1], validate=True)
                    camera = (
                        self.cameras[camera_index]
                        if camera_index < len(self.cameras)
                        else ""
                    )
                    camera_index += 1
                    command = (snapshot.get("log") or {}).get("command")
                    metadata = (
                        json.dumps(command, ensure_ascii=False) if command else ""
                    )
                    records.append(
                        ImageRecord(
                            f"{item.get('id', item['call_id'])}:{bi}",
                            mi,
                            bi,
                            camera,
                            metadata,
                            snapshot.get("task_language", ""),
                            hashlib.sha256(data).hexdigest(),
                            data,
                            not camera or item["call_id"] in recent,
                            snapshot.get("step") == 0,
                        )
                    )
        # The latest observation stays protected even after several read-only calls.
        if records:
            from dataclasses import replace

            latest = records[-1].item
            records = [
                replace(r, protected=True) if r.item == latest else r for r in records
            ]
        return records

    def render(self, request: dict, records: list[ImageRecord], keep: set[str]) -> dict:
        """Replace only omitted image blocks, leaving every tool pair intact."""
        dropped = [r for r in records if r.id not in keep]
        if not dropped:
            return request
        result = copy.deepcopy(request)
        for record in dropped:
            result["input"][record.item]["output"][record.block] = {
                "type": "input_text",
                "text": "[Earlier image omitted by context selection.]",
            }
        return result

    def extract_text(self, request: dict):
        """Extract only identified assistant messages consisting of output_text."""
        from .text import TEXT_OMITTED, TextRecord

        items = request.get("input", [])
        if not isinstance(items, list) or request.get("previous_response_id"):
            return [], ""
        users, task, events, outputs = [], "", set(), []
        updates = []
        first_assistant = next(
            (
                i
                for i, x in enumerate(items)
                if x.get("type") == "message" and x.get("role") == "assistant"
            ),
            len(items),
        )
        calls = {
            x.get("call_id"): i
            for i, x in enumerate(items)
            if x.get("type") in _CALL_TYPES
        }
        paired = _paired_outputs(items)
        for position, item in enumerate(items):
            if item.get("type") == "message" and item.get("role") == "user":
                if not isinstance(item.get("content"), list):
                    continue
                users.append(
                    "\n".join(
                        b.get("text", "")
                        for b in item.get("content", [])
                        if isinstance(b, dict)
                    )
                )
                if position > first_assistant:
                    updates.append(users[-1])
            if position not in paired:
                continue
            if item.get("call_id") in calls:
                outputs.append(item["call_id"])
            blocks = item.get("output", [])
            for block in blocks if isinstance(blocks, list) else []:
                if not isinstance(block, dict) or block.get("type") != "input_text":
                    continue
                try:
                    value = json.loads(block.get("text", ""))
                except ValueError:
                    continue
                if not isinstance(value, dict):
                    continue
                if value.get("task_language"):
                    task = value["task_language"]
                outcome = (value.get("log") or {}).get("result", value)
                if isinstance(outcome, dict) and (
                    isinstance(outcome.get("success"), bool) or outcome.get("error")
                ):
                    if item.get("call_id") in calls:
                        events.add(item["call_id"])
        # System/developer instructions are never used as disposable text.
        # Before a task-bearing observation, use the latest user prompt as fallback.
        query = task or (users[-1] if users else "")
        if task:
            query += "\n" + "\n".join(updates)
        candidates = []
        for i, item in enumerate(items):
            content = item.get("content")
            if (
                item.get("type") != "message"
                or item.get("role") != "assistant"
                or not item.get("id")
                or not isinstance(content, list)
                or not content
                or any(
                    not isinstance(b, dict)
                    or b.get("type") != "output_text"
                    or set(b) - {"type", "text"}
                    or not isinstance(b.get("text"), str)
                    for b in content
                )
            ):
                continue
            text = "\n".join(b["text"] for b in content)
            candidates.append((i, item["id"], text))
        protected = {i for i, _, _ in candidates[-2:]}
        # Retain the public plan preceding each recent or outstanding tool call.
        recent_calls = set(outputs[-2:]) | (set(calls) - set(outputs))
        for call_id in recent_calls:
            preceding = [i for i, _, _ in candidates if i < calls[call_id]]
            if preceding:
                protected.add(preceding[-1])
        records = [
            TextRecord(
                key,
                i,
                text,
                i in protected or len(text) <= len(TEXT_OMITTED),
                tuple(sorted(call for call in events if call in text)),
            )
            for i, key, text in candidates
        ]
        if self.extended_history:
            from .scope import active_error, schema

            action_blocks = {(r.item, r.block) for r in self.extract_action(request)}
            recent = set(outputs[-2:])
            for i in sorted(paired):
                item = items[i]
                blocks = item.get("output")
                if not isinstance(blocks, list):
                    continue
                for j, block in enumerate(blocks):
                    if block.get("type") != "input_text" or (i, j) in action_blocks:
                        continue
                    text = block.get("text", "")
                    if (
                        not text
                        or text == "[Earlier image omitted by context selection.]"
                    ):
                        continue
                    try:
                        payload = json.loads(text)
                    except ValueError:
                        payload = None
                    if (
                        isinstance(payload, dict)
                        and type(payload.get("step")) is int
                        and payload.get("context_omitted") is True
                    ):
                        continue
                    # Localization results remain available outside ordinary Text NC.
                    if (
                        isinstance(payload, dict)
                        and isinstance(payload.get("camera"), str)
                        and type(payload.get("step")) is int
                        and (
                            any(
                                isinstance(payload.get(key), list)
                                and len(payload[key]) == 3
                                for key in ("world_xyz", "center_xyz", "median_xyz")
                            )
                            or (
                                isinstance(payload.get("results"), list)
                                and isinstance(payload.get("summary"), dict)
                                and "median_xyz" in payload["summary"]
                                and all(
                                    isinstance(result, dict) and "world_xyz" in result
                                    for result in payload["results"]
                                )
                            )
                        )
                    ):
                        continue
                    # Tool schemas and errors remain available to the agent.
                    records.append(
                        TextRecord(
                            f"{item.get('id', item['call_id'])}:text:{j}",
                            i,
                            text,
                            item["call_id"] in recent
                            or schema(text)
                            or active_error(text),
                            (),
                            j,
                        )
                    )
        return records, query

    def render_text(self, request: dict, records, keep: set[str]) -> dict:
        """Remove selected plain assistant messages; preserve tools and reasoning."""
        dropped = [r for r in records if r.id not in keep]
        if not dropped:
            return request
        result = copy.deepcopy(request)
        messages = {r.item for r in dropped if r.block is None}
        blocks = {(r.item, r.block) for r in dropped if r.block is not None}
        for i, item in enumerate(result["input"]):
            if isinstance(item.get("output"), list):
                item["output"] = [
                    block
                    for j, block in enumerate(item["output"])
                    if (i, j) not in blocks
                ]
                if not item["output"]:
                    item["output"] = [
                        {"type": "input_text", "text": "[Earlier result omitted.]"}
                    ]
        result["input"] = [
            item for i, item in enumerate(result["input"]) if i not in messages
        ]
        return result

    def extract_action(self, request):
        """Only known RPent snapshots; preserve unknown tools unchanged."""
        from dataclasses import replace

        from .action import ActionRecord

        if request.get("previous_response_id") or not isinstance(
            request.get("input"), list
        ):
            return []
        items = request["input"]
        paired = _paired_outputs(items)
        calls = {x.get("call_id") for x in items if x.get("type") in _CALL_TYPES}
        outputs = [x.get("call_id") for i, x in enumerate(items) if i in paired]
        records = []
        for i, item in enumerate(items):
            if (
                i not in paired
                or item.get("call_id") not in calls
                or not isinstance(item.get("output"), list)
            ):
                continue
            blocks = item["output"]
            for j, block in enumerate(blocks):
                if block.get("type") != "input_text":
                    continue
                try:
                    value = json.loads(block.get("text", ""))
                except ValueError:
                    continue
                if (
                    not isinstance(value, dict)
                    or type(value.get("step")) is not int
                    or not isinstance(value.get("state"), dict)
                    or not isinstance(value.get("log"), dict)
                ):
                    continue
                if any(
                    value["log"].get(k) is not None
                    and not isinstance(value["log"][k], dict)
                    for k in ("command", "result")
                ):
                    continue
                # Retained images depend on their snapshot's explanation.
                protected = item["call_id"] in outputs[-2:] or any(
                    b.get("type") == "input_image" for b in blocks[j + 1 :]
                )
                records.append(
                    ActionRecord(
                        f"{item.get('id', item['call_id'])}:{j}", i, j, value, protected
                    )
                )
        if records:
            latest = max(r.snapshot["step"] for r in records)
            records = [
                replace(r, protected=True) if r.snapshot["step"] == latest else r
                for r in records
            ]
        return records

    def render_action(self, request, records, keep):
        """Thin snapshot details, retaining commands, outcomes and native call pairs."""
        dropped = [r for r in records if r.id not in keep]
        if not dropped:
            return request
        result = copy.deepcopy(request)
        for record in dropped:
            if self.extended_history:
                snapshot = record.snapshot
                outcome = snapshot["log"].get("result") or {}
                value = {
                    "step": snapshot["step"],
                    "context_omitted": True,
                    **{
                        k: snapshot[k]
                        for k in ("terminated", "truncated")
                        if k in snapshot
                    },
                    **{
                        k: snapshot[k]
                        for k in ("task_progress", "success", "vla_desync")
                        if k in snapshot
                    },
                    **action_summary(snapshot),
                }
                result["input"][record.item]["output"][record.block]["text"] = (
                    json.dumps(value, ensure_ascii=False)
                )
                continue
            value = copy.deepcopy(record.snapshot)
            value["state"] = {"context_omitted": True}
            original = value["log"].get("result")
            outcome = {
                k: v
                for k, v in (original or {}).items()
                if k in {"name", "success", "error", "terminated", "truncated"}
            }
            value["log"] = dict(
                value["log"], result=dict(outcome, context_omitted=True)
            )
            result["input"][record.item]["output"][record.block]["text"] = json.dumps(
                value, ensure_ascii=False
            )
        if self.extended_history:
            # Only known robot calls with all snapshots omitted are shortened.
            # Preserve native identifiers, selected-image dependencies and unknown calls.
            grouped = {}
            for record in records:
                grouped.setdefault(record.item, []).append(record)
            for position, group in grouped.items():
                if any(r.id in keep for r in group):
                    continue
                output = result["input"][position]
                for call in result["input"]:
                    if call.get("type") == "custom_tool_call" and call.get(
                        "call_id"
                    ) == output.get("call_id"):
                        call["input"] = (
                            "// Earlier robot call; outcome retained in paired result."
                        )
        return result
