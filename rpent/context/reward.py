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

"""Disjoint provider-input accounting and fixed-reference SAC savings reward."""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite

from .memory import END, START
from .scope import schema


def robot_calls(req):
    result = set()
    for item in req["input"]:
        for b in item.get("output", []) if isinstance(item.get("output"), list) else []:
            if b.get("type") != "input_text":
                continue
            try:
                x = json.loads(b.get("text", ""))
            except (ValueError, TypeError):
                continue
            if (
                isinstance(x, dict)
                and "step" in x
                and ("log" in x or x.get("context_omitted"))
            ):
                result.add(item.get("call_id"))
    return result


def split_memory(text):
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    if START not in text or END not in text:
        return 0.0
    region = text.split(START, 1)[1].split(END, 1)[0]
    return min(1.0, len(encoding.encode(region)) / max(1, len(encoding.encode(text))))


def account(req, usage):
    """Partition attributed input; never disguise unknown costs as protocol.

    Memory allocation within a shared user-text block remains an estimate.
    The exact server input total includes cached input; output tokens are separate.
    """
    from rpent.context.anchor import AnchorAdapter

    def count(value):
        if type(value) not in (int, float) or not isfinite(value) or value < 0:
            raise ValueError("finite nonnegative attributed token counts required")
        return value

    count(usage["input_tokens"])
    groups = Counter()
    attrs = usage["attribution"]["items"]
    calls = robot_calls(req)
    adapter = AnchorAdapter()
    candidates, _ = adapter.extract_text(req)
    text_blocks, text_messages, text_calls = set(), set(), set()
    for record in candidates:
        if record.block is None:
            text_messages.add(record.item)
        elif record.block == -1:
            text_calls.add(record.item)
        else:
            members = getattr(record, "members", ()) or (record,)
            text_blocks.update((r.item, r.block) for r in members)
    for position, item in enumerate(req["input"]):
        a = attrs[item["id"]]
        n = count(a["input_tokens"])
        kind, role = item.get("type"), item.get("role")
        if kind == "reasoning":
            groups["reasoning"] += n
            continue
        if kind == "additional_tools":
            groups["tool_rules"] += n
            continue
        if role in ("system", "developer"):
            groups["rules"] += n
            continue
        if kind in ("custom_tool_call", "function_call"):
            group = (
                "action"
                if item.get("call_id") in calls
                else "text"
                if position in text_calls
                else "unmanaged"
            )
            groups[group] += n
            continue
        blocks = item.get("output", item.get("content", []))
        if not isinstance(blocks, list):
            groups["unmanaged"] += n
            continue
        parts = a.get("content", [])
        if len(parts) != len(blocks):
            raise ValueError("Attribution block mismatch")
        overhead = n - sum(count(x["input_tokens"]) for x in parts)
        if overhead < 0:
            raise ValueError("Attribution block costs exceed item cost")
        groups["protocol"] += overhead
        for index, (block, part) in enumerate(zip(blocks, parts)):
            token = count(part["input_tokens"])
            text = block.get("text", "")
            if (
                item.get("id") == "msg_rpent_fixed_tool_reference_v1"
                or text
                == "[Tool definitions are available in the fixed tool reference.]"
            ):
                groups["tool_rules"] += token
                continue
            if role == "user" and block.get("type") == "input_text":
                fraction = split_memory(text)
                groups["memory"] += token * fraction
                groups["user_fixed"] += token * (1 - fraction)
                continue
            if (
                block.get("type") == "input_image"
                or text == "[Earlier image omitted by context selection.]"
            ):
                group = "image"
            elif role != "assistant" and schema(text):
                group = "tool_rules"
            else:
                try:
                    value = json.loads(text)
                except (ValueError, TypeError):
                    value = None
                if (
                    isinstance(value, dict)
                    and "step" in value
                    and ("log" in value or value.get("context_omitted"))
                ):
                    group = "action"
                elif position in text_messages or (position, index) in text_blocks:
                    group = "text"
                else:
                    group = "unmanaged"
            groups[group] += token
    # A request-level residual may be framing or unreported fields; do not guess.
    residual = usage["input_tokens"] - sum(groups.values())
    if residual < -1e-5:
        raise ValueError("Attributed costs exceed total input usage")
    groups["unattributed"] = max(0.0, residual)
    if abs(sum(groups.values()) - usage["input_tokens"]) > 1e-5:
        raise ValueError("Input ledger does not conserve server usage")
    if min(groups.values()) < -1e-5:
        raise ValueError("Negative accounting category")
    return dict(groups)


RELATIVE_REFERENCE = {
    "image": 69777.0,
    "text": 11192.0,
    "action": 20368.0,
    "memory": 1961.5228577657122,
    "time": 141.16386097297072,
}
RELATIVE_WEIGHTS = {
    "image": 0.03,
    "text": 0.6,
    "action": 0.03,
    "memory": 0.03,
    "time": 0.03,
}


@dataclass(frozen=True)
class ComponentRewardConfig:
    """Frozen episode reference costs; constants are paid once at task end.

    Reward is native success (0 or 1) minus weighted reference-scaled costs.
    Reference scales and coefficients stay fixed for a training chain.
    """

    references: tuple[float, ...] = tuple(RELATIVE_REFERENCE.values())
    weights: tuple[float, ...] = tuple(RELATIVE_WEIGHTS.values())

    def __post_init__(self):
        if len(self.references) != 5 or len(self.weights) != 5:
            raise ValueError("five references and weights required")
        for values, positive in ((self.references, True), (self.weights, False)):
            if any(
                type(v) not in (int, float)
                or not isfinite(v)
                or (v <= 0 if positive else v < 0)
                for v in values
            ):
                raise ValueError(
                    "finite positive references and nonnegative weights required"
                )
        if not any(self.weights):
            raise ValueError("positive total weight required")

    def spec(self):
        return {"version": "success-components-task-v1", **asdict(self)}

    def token_cost(self, groups):
        costs = [groups.get(k, 0.0) for k in ("image", "text", "action", "memory")]
        if any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in costs):
            raise ValueError("finite nonnegative component costs required")
        return sum(
            w * c / ref for w, c, ref in zip(self.weights, costs, self.references)
        )

    def terminal(self, duration, success):
        if (
            type(duration) not in (int, float)
            or not isfinite(duration)
            or duration < 0
            or type(success) is not bool
        ):
            raise ValueError(
                "finite nonnegative task duration and native verdict required"
            )
        return float(success) - self.weights[-1] * duration / self.references[-1]


def component_reward(groups, task_duration_s, config=None, *, success=True):
    """Evaluate complete episode savings with the same config used by SAC.add()."""
    config = config or ComponentRewardConfig()
    config.token_cost(groups)
    config.terminal(task_duration_s, success)
    costs = [groups.get(k, 0.0) for k in ("image", "text", "action", "memory")]
    costs.append(task_duration_s)
    terms = {
        k: -w * c / ref
        for k, w, c, ref in zip(
            RELATIVE_REFERENCE, config.weights, costs, config.references
        )
    }
    terms["success"] = float(success)
    return {"reward": sum(terms.values()), "contributions": terms}
