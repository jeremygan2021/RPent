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

"""Incremental selection of snapshot state and execution details."""

from dataclasses import dataclass
from math import acos, isfinite, pi, sqrt

from .selection import select_by_nc


@dataclass(frozen=True)
class ActionRecord:
    id: str
    item: int
    block: int
    snapshot: dict
    protected: bool


def flatten(value, prefix=""):
    """Stable paths for interface-provided numeric ranges and discrete values."""
    if isinstance(value, dict):
        return {
            k: v
            for key, child in value.items()
            for k, v in flatten(child, f"{prefix}.{key}" if prefix else key).items()
        }
    if isinstance(value, list):
        return {
            k: v
            for i, child in enumerate(value)
            for k, v in flatten(child, f"{prefix}.{i}").items()
        }
    return {prefix: value}


def difference(current, previous, ranges, features=None):
    values = []
    for key in current.keys() & previous.keys():
        a, b = current[key], previous[key]
        if isinstance(a, bool) and isinstance(b, bool):
            values.append(float(a != b))
        elif isinstance(a, str) and isinstance(b, str):
            values.append(
                1 - features.similarity(a, b)
                if features and key in {"prompt", "instruction"}
                else float(a != b)
            )
        elif key in ranges and type(a) in (int, float) and type(b) in (int, float):
            low, high = ranges[key]
            if all(isfinite(x) for x in (a, b, low, high)) and high > low:
                values.append(min(1.0, abs(a - b) / (high - low)))
    return sum(values) / len(values) if values else None


def orientation_difference(current: dict, previous: dict) -> float | None:
    """Compare EEF quaternion states by rotation angle, independent of q/-q sign.

    Snapshot fields ending in ``_eef_quat`` carry four-component quaternions.
    Unknown, zero-length, or nonfinite representations contribute no comparison.
    Their scalar components must not also enter numeric field averaging.
    """
    changes = []
    for key in current.keys() & previous.keys():
        if not key.endswith("_eef_quat"):
            continue
        a, b = current[key], previous[key]
        if not all(
            isinstance(q, (list, tuple))
            and len(q) == 4
            and all(type(v) in (int, float) and isfinite(v) for v in q)
            for q in (a, b)
        ):
            continue
        na, nb = sqrt(sum(v * v for v in a)), sqrt(sum(v * v for v in b))
        if not all(isfinite(n) and n > 1e-12 for n in (na, nb)):
            continue
        dot = abs(sum((x / na) * (y / nb) for x, y in zip(a, b)))
        changes.append(2 * acos(min(1.0, dot)) / pi)
    return sum(changes) / len(changes) if changes else None


class ActionSelector:
    def __init__(self, ranges=None):
        # Keys are state.<field> or command.<field>; no inferred numeric scales.
        if ranges is None:
            ranges = {}
        if not isinstance(ranges, dict):
            raise ValueError("action ranges must be a field-to-bounds object")
        self.ranges = {}
        for field, bounds in ranges.items():
            if (
                not isinstance(field, str)
                or not field.startswith(("state.", "command."))
                or not field.split(".", 1)[1]
                or not isinstance(bounds, (list, tuple))
                or len(bounds) != 2
                or any(type(x) not in (int, float) or not isfinite(x) for x in bounds)
                or bounds[1] <= bounds[0]
            ):
                raise ValueError(f"invalid action range for {field!r}: {bounds!r}")
            self.ranges[field] = tuple(bounds)
        self._seen, self._kept = set(), set()
        self.stats = {}
        self._signature = None

    def process(self, request, adapter, features, *, rho):
        records = adapter.extract_action(request)
        signature = (rho, repr(records), repr(self.ranges))
        if signature == self._signature:
            return adapter.render_action(request, records, self._kept)
        previous, scores, details = None, {}, []
        unscaled = set()
        for record in records:
            value = record.snapshot
            eligible = True
            command = value["log"].get("command") or {}
            result = value["log"].get("result") or {}
            event = int(
                result.get("success") is False
                or bool(result.get("error"))
                or value.get("terminated") is True
                or value.get("truncated") is True
            )
            action = state = orientation = score = None
            reference = None
            if eligible:
                numeric_fields = dict(
                    flatten(command, "command"), **flatten(value["state"], "state")
                )
                unscaled.update(
                    key
                    for key, number in numeric_fields.items()
                    if type(number) in (int, float) and key not in self.ranges
                )
                if previous:
                    reference = previous.id
                    old = previous.snapshot
                    old_command = old["log"].get("command") or {}
                    if command and old_command:
                        if command.get("action") != old_command.get("action"):
                            action = 1.0
                        else:
                            a = {
                                k: v
                                for k, v in flatten(command).items()
                                if k != "action"
                            }
                            b = {
                                k: v
                                for k, v in flatten(old_command).items()
                                if k != "action"
                            }
                            ranges = {
                                k.removeprefix("command."): v
                                for k, v in self.ranges.items()
                                if k.startswith("command.")
                            }
                            action = difference(a, b, ranges, features)
                    # State descriptors are not physical changes. Quaternions
                    # use angular distance rather than four scalar differences.
                    current_state, old_state = value["state"], old["state"]
                    quat_prefixes = tuple(
                        "state." + key + "."
                        for key in current_state.keys() | old_state.keys()
                        if key.endswith("_eef_quat")
                    )
                    a = {
                        k: v
                        for k, v in flatten(current_state, "state").items()
                        if not isinstance(v, str) and not k.startswith(quat_prefixes)
                    }
                    b = {
                        k: v
                        for k, v in flatten(old_state, "state").items()
                        if not isinstance(v, str) and not k.startswith(quat_prefixes)
                    }
                    state = difference(a, b, self.ranges)
                    orientation = orientation_difference(current_state, old_state)
                    state_parts = [v for v in (state, orientation) if v is not None]
                    state = sum(state_parts) / len(state_parts) if state_parts else None
                    old_result = old["log"].get("result") or {}
                    if (
                        isinstance(result.get("success"), bool)
                        and isinstance(old_result.get("success"), bool)
                        and result["success"] != old_result["success"]
                    ):
                        event = 1
                    for key in ("gripper_closed", "holding"):
                        if (
                            isinstance(value["state"].get(key), bool)
                            and isinstance(old["state"].get(key), bool)
                            and value["state"][key] != old["state"][key]
                        ):
                            event = 1
                parts = [x for x in (action, state) if x is not None]
                delta = sum(parts) / len(parts) if parts else 0.0
                score = (delta + event) / 2
                if not record.protected:
                    scores[record.id] = score
                previous = record
            details.append(
                {
                    "id": record.id,
                    "step": value["step"],
                    "command": command,
                    "protected": record.protected,
                    "reference": reference,
                    "action_change": action,
                    "state_change": state,
                    "orientation_change": orientation,
                    "event": event,
                    "score": score,
                    "reason": "protected"
                    if record.protected
                    else "candidate"
                    if eligible
                    else "previously_omitted",
                }
            )
        keep = select_by_nc(scores, rho) | {r.id for r in records if r.protected}
        result = adapter.render_action(request, records, keep)
        total = sum(scores.values())
        self.stats = {
            "numeric_ranges": self.ranges,
            "unscaled_numeric_fields": sorted(unscaled),
            "records_before": len(records),
            "records_after": len(keep),
            "nc": sum(v for k, v in scores.items() if k in keep) / total
            if total
            else None,
            "records": [dict(d, selected=d["id"] in keep) for d in details],
            "chars_before": sum(
                len(request["input"][r.item]["output"][r.block]["text"])
                for r in records
            ),
            "chars_after": sum(
                len(result["input"][r.item]["output"][r.block]["text"]) for r in records
            ),
        }
        self._seen.update(r.id for r in records)
        self._kept, self._signature = keep, signature
        return result
