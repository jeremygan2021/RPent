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

"""Incremental selection of public assistant messages, not tool payloads."""

import copy
import re
from dataclasses import dataclass

from .selection import select_by_nc

TEXT_OMITTED = "[Earlier assistant text omitted by context selection.]"


_COMPLETED_WRAPPER = re.compile(
    r"\A(?:Script completed\r?\n)?Wall time:?[ \t]+\d+(?:\.\d+)? seconds"
    r"\r?\nOutput:[ \t]*(?:\r?\n|\Z)"
)


def execution_status(text: str) -> bool:
    """Keep transport failures and pending execution handles out of selection."""
    return bool(
        re.search(
            r"(?mi)^(?:Script running\b|Process (?:still running|running)\b"
            r"|.*\b(?:cell|session) ID\b|Process exited with code [1-9]\d*\b"
            r"|(?:Error|[A-Za-z]+Error|Exception):|Traceback \(most recent call last\))",
            text,
        )
    )


def normalize_tool_wrappers(request: dict) -> dict:
    """Strip known completed-output headers without changing payload block IDs."""
    if request.get("previous_response_id") or not isinstance(
        request.get("input"), list
    ):
        return request
    result = request
    for i, item in enumerate(request["input"]):
        if item.get("type") not in {"function_call_output", "custom_tool_call_output"}:
            continue
        output = item.get("output")
        blocks = output if isinstance(output, list) else []
        for j, block in enumerate(blocks):
            if not isinstance(block, dict) or block.get("type") != "input_text":
                continue
            text = block.get("text")
            if not isinstance(text, str) or execution_status(text):
                continue
            match = _COMPLETED_WRAPPER.match(text)
            if match is None:
                continue
            if result is request:
                result = copy.deepcopy(request)
            # Leave an empty text block for a header-only result, preserving the
            # paired call and stable block indices used by all selectors.
            result["input"][i]["output"][j]["text"] = text[match.end() :]
    return result


@dataclass(frozen=True)
class TextRecord:
    """One complete, plain-text assistant message in the native request."""

    id: str
    item: int
    text: str
    protected: bool
    event_refs: tuple[str, ...] = ()
    block: int | None = None


class TextSelector:
    """Reuse feature caching and NC while keeping text selection state separate."""

    def __init__(self):
        self._seen: set[str] = set()
        self._kept: set[str] = set()
        self._signature = None
        self.stats = {}

    def process(self, request, adapter, features, *, rho, stage):
        """Select eligible public text; preserve original request and message IDs."""
        records, query = adapter.extract_text(request)
        if stage != "Unknown":
            query += "; " + stage
        signature = (query, rho, tuple(records))
        if signature == self._signature:
            return adapter.render_text(request, records, self._kept)
        previous, scores, details = [], {}, []
        for record in records:
            eligible = True
            relevance = novelty = score = None
            event = int(bool(record.event_refs))
            if eligible:
                relevance = features.similarity(record.text, query)
                novelty = (
                    (
                        1
                        - max(
                            features.similarity(record.text, text)
                            for text in previous[-8:]
                        )
                    )
                    if previous
                    else None
                )
                parts = [relevance, event]
                if novelty is not None:
                    parts.append(relevance * novelty)
                # Missing novelty earns no bonus; keep the same scale as records
                # with a predecessor so the oldest survivor gets no extra weight.
                score = sum(parts) / 3
                previous.append(record.text)
                if not record.protected:
                    scores[record.id] = score
            details.append(
                {
                    "id": record.id,
                    "text": record.text,
                    "protected": record.protected,
                    "relevance": relevance,
                    "novelty": novelty,
                    "novelty_contribution": relevance * novelty
                    if novelty is not None
                    else None,
                    "event": event,
                    "event_refs": record.event_refs,
                    "score": score,
                    "reason": "protected"
                    if record.protected
                    else "candidate"
                    if eligible
                    else "previously_omitted",
                }
            )
        keep = select_by_nc(scores, rho) | {r.id for r in records if r.protected}
        result = adapter.render_text(request, records, keep)
        total = sum(scores.values())
        self.stats = {
            "score_rule": "relevance_gated_novelty_fixed_scale",
            "records_before": len(records),
            "records_after": len(keep),
            "chars_before": sum(len(r.text) for r in records),
            "chars_after": sum(len(r.text) if r.id in keep else 0 for r in records),
            "nc": sum(s for key, s in scores.items() if key in keep) / total
            if total
            else None,
            "records": [dict(r, selected=r["id"] in keep) for r in details],
        }
        self._seen.update(r.id for r in records)
        self._kept, self._signature = keep, signature
        return result
