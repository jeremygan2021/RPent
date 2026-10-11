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

"""Assemble decision feedback from existing audit records, without rewards."""

import gzip
import json
from pathlib import Path

from rpent.evaluation.result import write_json_atomic


def read_response(
    path: Path, encoding: str = ""
) -> tuple[dict | None, list, str | None]:
    """Read provider-reported usage once per response, never per SSE event."""
    if not path.exists():
        return None, [], None
    raw = path.read_bytes()
    if encoding == "gzip":
        raw = gzip.decompress(raw)
    elif encoding == "zstd":
        import io

        import zstandard

        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)) as reader:
            raw = reader.read()
    elif encoding:
        raise ValueError(f"unsupported response encoding: {encoding}")
    decoded = raw.decode(errors="replace")
    try:
        events = [json.loads(decoded)]
    except ValueError:
        events = []
        for line in decoded.splitlines():
            if line.startswith("data:"):
                try:
                    events.append(json.loads(line[5:].strip()))
                except ValueError:
                    continue
    usage, outputs, response_id = None, [], None
    for event in events:
        if not isinstance(event, dict):
            continue
        response = event.get(
            "response", event if event.get("object") == "response" else {}
        )
        if not isinstance(response, dict):
            continue
        if response.get("usage") is not None:
            usage = response["usage"]
        response_id = response.get("id", response_id)
        if event.get("type") == "response.output_item.done":
            outputs.append(event.get("item"))
        if response.get("output"):
            outputs = response["output"]
    return usage, outputs, response_id


def assemble(records: list[dict], outcome: dict | None, *, episode_id: str) -> dict:
    """Join HTTP attempts to controller decisions, retaining incomplete evidence.

    Readiness means complete fixed-controller audit data, NOT PPO eligibility.
    Provider costs for ambiguous duplicate response IDs are not guessed.
    """
    decisions = {}
    issues = set()
    response_ids = set()
    for record in records:
        selection = record.get("selection", {})
        decision = selection.get("decision")
        if selection.get("fallback"):
            issues.add("compression_fallback")
        if not decision:
            issues.add("missing_decision")
            continue
        key = decision["id"]
        row = decisions.setdefault(
            key,
            {
                "decision": {k: v for k, v in decision.items() if k != "reused"},
                "attempts": [],
            },
        )
        if row["decision"] != {k: v for k, v in decision.items() if k != "reused"}:
            issues.add("inconsistent_retry_decision")
        usage = record.get("usage")
        if not isinstance(usage, dict) or any(
            usage.get(k) is None for k in ("input_tokens", "output_tokens")
        ):
            issues.add("missing_usage")
        timing = record.get("timing", {})
        if timing.get("transport_error") or timing.get("status", 0) >= 400:
            issues.add("transport_failure")
        if timing.get("latency_s") is None:
            issues.add("missing_latency")
        if timing.get("status") is None:
            issues.add("missing_http_status")
        response_id = record.get("response_id")
        if response_id:
            if response_id in response_ids:
                issues.add("duplicate_response_id")
            response_ids.add(response_id)
        selection_result = {}
        for modality in ("image", "text", "action"):
            stats = selection if modality == "image" else selection.get(modality, {})
            selection_result[modality] = {
                "nc": stats.get("nc"),
                "kept_ids": [
                    r["id"] for r in stats.get("records", []) if r["selected"]
                ],
                "omitted_ids": [
                    r["id"] for r in stats.get("records", []) if not r["selected"]
                ],
            }
        if "memory" in selection:
            selection_result["memory"] = selection["memory"]
        row["attempts"].append(
            {
                "request_id": record["request_id"],
                "response_id": response_id,
                "usage": usage,
                "timing": timing,
                "processing_s": selection.get("processing_s"),
                "selection": selection_result,
            }
        )
    if not decisions:
        issues.add("no_decisions")
    if outcome is None or type(outcome.get("environment_success")) is not bool:
        issues.add("missing_environment_verdict")
    if outcome and outcome.get("agent_error"):
        issues.add("agent_error_requires_review")
    return {
        "schema_version": 1,
        "episode_id": episode_id,
        "decisions": list(decisions.values()),
        "outcome": outcome,
        "audit_complete": not issues,
        "issues": sorted(issues),
    }


def write_trajectory(output: Path, records: list[dict], summary: dict) -> Path:
    """Write an atomic snapshot beside existing per-request and summary files."""
    task = summary.get("task_result")
    outcome = None
    if task is not None:
        outcome = {
            "environment_success": task.get("environment_success"),
            "agent_error": task.get("agent_error"),
            "runner_elapsed_s": task.get("elapsed_s"),
            "timing": summary.get("timing", {}),
            "action_steps": summary.get("action_steps"),
        }
    return write_json_atomic(
        output / "trajectory.json",
        assemble(records, outcome, episode_id=str(output.resolve())),
    )
