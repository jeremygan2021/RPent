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

"""Select reusable experiences without modifying fixed prompt instructions."""

import copy
import json
import re
from pathlib import Path
from typing import Any

from .parsing import parse_payload
from .selection import select_by_nc

START = "[RPENT_MEMORY]"
END = "[/RPENT_MEMORY]"
_STAGE_INTENT = re.compile(
    r"^Stage:[ \t]*(?P<stage>[^;\n]+);[ \t]*Next:[ \t]*(?P<intent>[^\n]+)$",
    re.MULTILINE,
)


def load_memory_blocks(path: Path) -> list[dict[str, str]]:
    """Read a titled demonstration text or the existing block-list JSON format."""
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        blocks = json.loads(text)
    else:
        headings = list(re.finditer(r"^【([^\n【】]+)】[ \t]*$", text, re.MULTILINE))
        if not headings or text[: headings[0].start()].strip():
            raise ValueError(
                "memory text must start with an independent 【title】 line"
            )
        blocks = []
        for index, heading in enumerate(headings):
            end = (
                headings[index + 1].start() if index + 1 < len(headings) else len(text)
            )
            body = text[heading.end() : end].strip()
            if not heading.group(1).strip() or not body:
                raise ValueError("memory blocks require a title and nonempty body")
            blocks.append(
                {
                    "id": f"block_{index + 1:03d}",
                    "text": text[heading.start() : end].strip(),
                }
            )
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("memory library must be a nonempty list")
    ids = set()
    for block in blocks:
        if (
            not isinstance(block, dict)
            or not isinstance(block.get("id"), str)
            or not block["id"]
            or block["id"] in ids
            or not isinstance(block.get("text"), str)
            or not block["text"].strip()
        ):
            raise ValueError("memory blocks require unique ids and nonempty text")
        if START in block["text"] or END in block["text"]:
            raise ValueError("memory text must not contain reserved prompt markers")
        ids.add(block["id"])
    return blocks


def append_memory(prompt: str, blocks: list[dict[str, str]]) -> str:
    """Append one managed reference region after prompt template rendering."""
    if not blocks:
        return prompt
    if START in prompt or END in prompt:
        raise ValueError("prompt already contains a managed memory region")
    region = "\n" + "\n".join(block["text"] for block in blocks) + "\n"
    return (
        prompt.rstrip()
        + "\n\nDemonstration experience references:\n"
        + START
        + region
        + END
        + "\nUse these as references, not current observations or proof of success. "
        "Re-localize from current images and use the current environment verdict.\n"
    )


_OBSERVATION_ACTIONS = {
    "view_env_state",
    "back_project",
    "back_project_batch",
    "query_world_map",
    "segment",
}


def _memory_recovery(items: list[dict]) -> dict:
    """Derive recovery from ordered tool results, not task or grasp verdicts.

    Full-history replay makes retries idempotent and requires no mutable
    selector state. Repeated views of a recorded snapshot are not new actions.
    """
    active, trigger, resolved_by = False, None, None
    seen_outputs, seen_snapshots = set(), set()
    for item in items:
        if item.get("type") not in ("function_call_output", "custom_tool_call_output"):
            continue
        call_id = item.get("call_id") or item.get("id")
        if call_id and call_id in seen_outputs:
            continue
        if call_id:
            seen_outputs.add(call_id)
        output = item.get("output", [])
        blocks = [{"text": output}] if isinstance(output, str) else output
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict) or not isinstance(block.get("text"), str):
                continue
            text = block["text"]
            try:
                value = parse_payload(text)
            except ValueError:
                if re.search(
                    r"(?:^|\n)\s*(?:Uncaught )?(?:Error:|Exception:|"
                    r"Traceback \(most recent call last\):|Error executing tool)",
                    text,
                ):
                    active, trigger, resolved_by = True, call_id, None
                continue
            if not isinstance(value, dict):
                continue
            log = value.get("log")
            log = log if isinstance(log, dict) else {}
            command, outcome = log.get("command"), log.get("result")
            command = command if isinstance(command, dict) else {}
            outcome = outcome if isinstance(outcome, dict) else {}
            snapshot = "step" in value and "state" in value and "log" in value
            if snapshot:
                key = (value["step"], json.dumps(log, sort_keys=True))
                if key in seen_snapshots:
                    continue
                seen_snapshots.add(key)
                if value.get("terminated") is True or value.get("truncated") is True:
                    active, resolved_by = False, call_id
                    continue
            failed = (
                value.get("ok") is False
                or bool(value.get("error"))
                or value.get("isError") is True
                or outcome.get("ok") is False
                or bool(outcome.get("error"))
            )
            if failed:
                active, trigger, resolved_by = True, call_id, None
            elif (
                snapshot
                and command.get("action")
                and command["action"] not in _OBSERVATION_ACTIONS
                and (outcome.get("ok") is True or outcome.get("success") is True)
            ):
                active, resolved_by = False, call_id
    return {"active": active, "trigger_call_id": trigger, "resolved_by": resolved_by}


class MemorySelector:
    """Reconsider complete operation blocks using the latest readable agent intent."""

    def __init__(self, path: Path, rho: float = 0.6):
        """Load a pre-separated experience library and validate its NC threshold."""
        blocks = load_memory_blocks(path)
        if not 0 < rho <= 1:
            raise ValueError("memory rho must be in (0, 1]")
        self.blocks, self.rho = blocks, rho
        # Keep the existing observation interface; each scoring record is now a block.
        self.fragments = [{**block, "parent": block["id"]} for block in blocks]

    def process(self, request: dict, adapter: Any, features: Any) -> tuple[dict, dict]:
        """Select whole blocks using readable reasoning summaries or public intent."""
        items = request.get("input")
        if not isinstance(items, list) or request.get("previous_response_id"):
            raise ValueError("memory selection requires explicit full input history")
        regions = []
        plans = []
        stage_intents = []
        for i, item in enumerate(items):
            if item.get("type") == "reasoning":
                summary = item.get("summary", [])
                if isinstance(summary, list):
                    text = "\n".join(
                        part["text"].strip()
                        for part in summary
                        if isinstance(part, dict)
                        and part.get("type") == "summary_text"
                        and isinstance(part.get("text"), str)
                        and part["text"].strip()
                    )
                    if text:
                        plans.append((text, "reasoning_summary"))
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for j, block in enumerate(content):
                text = block.get("text", "")
                if not isinstance(text, str):
                    continue
                if (
                    item.get("role") == "assistant"
                    and block.get("type") == "output_text"
                    and item.get("phase") != "final_answer"
                    and text.strip()
                ):
                    plans.append((text.strip(), "public_text"))
                    for match in _STAGE_INTENT.finditer(text.strip()):
                        stage, intent = match["stage"].strip(), match["intent"].strip()
                        if stage and intent:
                            stage_intents.append((stage, intent))
                if item.get("role") != "user":
                    continue
                if START in text or END in text:
                    if text.count(START) != 1 or text.count(END) != 1:
                        raise ValueError(
                            "memory prompt requires exactly one marker pair"
                        )
                    a, b = text.index(START) + len(START), text.index(END)
                    if a > b:
                        raise ValueError("memory prompt markers out of order")
                    regions.append((i, j, a, b))
        if len(regions) != 1:
            raise ValueError(
                "memory selection requires exactly one managed user region"
            )
        actions = adapter.extract_action(request)
        task = actions[-1].snapshot.get("task_language", "") if actions else ""
        query, query_source = plans[-1] if plans else (task.strip(), "task")
        stage, intent = stage_intents[-1] if stage_intents else (None, None)
        if stage_intents:
            query, query_source = f"Stage: {stage}; Next: {intent}", "stage_intent"
        scores = (
            {
                fragment["id"]: features.similarity(fragment["text"], query)
                for fragment in self.fragments
            }
            if task and query
            else {}
        )
        recovery = _memory_recovery(items)
        if self.rho == 1:
            kept = {fragment["id"] for fragment in self.fragments}
            reason = "full_memory"
        elif recovery["active"]:
            kept = {fragment["id"] for fragment in self.fragments}
            reason = "execution_error_keep_all"
        elif not task:
            i, j, a, b = regions[0]
            expected = "\n" + "\n".join(block["text"] for block in self.blocks) + "\n"
            if items[i]["content"][j]["text"][a:b] != expected:
                raise ValueError("initial prompt memory differs from selector library")
            kept = {fragment["id"] for fragment in self.fragments}
            reason = "task_not_observed_keep_all"
        elif not query or sum(scores.values()) <= 0:
            kept, reason = set(), "no_positive_relevance"
        else:
            kept, reason = select_by_nc(scores, self.rho), "nc_selection"
        parents = {
            fragment["parent"] for fragment in self.fragments if fragment["id"] in kept
        }
        selected = [block for block in self.blocks if block["id"] in parents]
        replacement = "\n" + "\n".join(block["text"] for block in selected) + "\n"
        i, j, a, b = regions[0]
        original = items[i]["content"][j]["text"]
        rewritten = original[:a] + replacement + original[b:]
        result = request
        if rewritten != original:
            result = copy.deepcopy(request)
            result["input"][i]["content"][j]["text"] = rewritten
        total = sum(scores.values())
        stats = {
            "recovery": recovery,
            "rho": self.rho,
            "query": query,
            "query_source": query_source,
            "stage": stage,
            "next_intent": intent,
            "score_rule": "whole_block_cosine",
            "reason": reason,
            "selection_nc": sum(scores[k] for k in kept) / total if total else None,
            "selected_blocks": [block["id"] for block in selected],
            "omitted_blocks": [
                block["id"] for block in self.blocks if block["id"] not in parents
            ],
            "selected_text": [block["text"] for block in selected],
            "chars_before": b - a,
            "chars_after": len(replacement),
            "records": [
                {
                    **fragment,
                    "relevance": scores.get(fragment["id"]),
                    "selected": fragment["id"] in kept,
                    "parent_sent": fragment["parent"] in parents,
                }
                for fragment in self.fragments
            ],
        }
        return result, stats
