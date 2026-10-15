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

"""Text ownership and shared tool-interface recognition for selection/accounting."""

import copy
import re
from dataclasses import replace

from rpent.context.codex import CodexAdapter
from rpent.context.text import TextRecord

from .parsing import parse_payload


def _declared_tools(text: str) -> set[str]:
    """Read tool names from explicit TS or line-based interface declarations."""
    names = set()
    if re.search(r"(?m)^\s*declare\s+const\s+tools\s*:\s*\{", text):
        # Standalone parameter comments may contain parentheses, e.g. defaults.
        declaration = re.sub(r"(?m)^\s*//[^\n]*", "", text)
        names.update(
            re.findall(
                r"\b([A-Za-z_]\w*)\s*\([^)]*\)\s*:\s*Promise\s*<",
                declaration,
            )
        )
    if re.search(r"(?mi)^\s*(?:Parameters?|Arguments?|Input schema)\s*:", text):
        names.update(re.findall(r"(?mi)^\s*Tool name\s*:\s*([A-Za-z_]\w*)\s*$", text))
    return names


def is_tool_definition(value: object) -> bool:
    """Recognize executable interfaces, rather than named result descriptions."""
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        return False
    name = value["name"]
    if not name.strip():
        return False
    if any(isinstance(value.get(key), dict) for key in ("inputSchema", "parameters")):
        return True
    description = value.get("description")
    return isinstance(description, str) and name in _declared_tools(description)


def schema(text: str) -> bool:
    """Identify tool interfaces using the same rule as reference deduplication."""
    if not isinstance(text, str):
        return False
    try:
        value = parse_payload(text)
    except ValueError:
        value = None

    if value is not None:
        return is_tool_definition(value) or (
            isinstance(value, list)
            and bool(value)
            and all(is_tool_definition(item) for item in value)
        )
    return bool(_declared_tools(text))


def active_error(text):
    from .text import execution_status

    if execution_status(text):
        return True
    try:
        value = parse_payload(text)
    except ValueError:
        return "Traceback (most recent call last)" in text
    if not isinstance(value, dict):
        return False
    return bool(value.get("error")) or value.get("success") is False


class ScopeAdapter(CodexAdapter):
    def extract_text(self, request):
        records, query = super().extract_text(request)
        items = request.get("input", [])
        if request.get("previous_response_id"):
            return [], query
        outputs = [
            i
            for i in items
            if i.get("type") in ("custom_tool_call_output", "function_call_output")
        ]
        recent = {i.get("call_id") for i in outputs[-2:]}
        result = []
        for r in records:
            if r.block is None:
                result.append(r)
                continue
            if schema(r.text):
                continue
            protected = (
                r.protected
                or items[r.item].get("call_id") in recent
                or active_error(r.text)
            )
            result.append(replace(r, protected=protected))
        paired = {
            i.get("call_id")
            for i in outputs
            if i.get("type") == "custom_tool_call_output"
        }
        robot = {items[r.item].get("call_id") for r in self.extract_action(request)}
        for i, item in enumerate(items):
            if (
                item.get("type") != "custom_tool_call"
                or item.get("call_id") not in paired
                or item.get("call_id") in robot
            ):
                continue
            text = item.get("input")
            if not isinstance(text, str) or not text:
                continue
            result.append(
                TextRecord(
                    f"{item.get('id', item['call_id'])}:call_input",
                    i,
                    text,
                    item["call_id"] in recent,
                    (),
                    -1,
                )
            )
        return result, query

    def render_text(self, request, records, keep):
        calls = [r for r in records if r.block == -1 and r.id not in keep]
        result = copy.deepcopy(request) if calls else request
        for r in calls:
            result["input"][r.item]["input"] = (
                "// Earlier completed query; see its paired result."
            )
        return super().render_text(result, [r for r in records if r.block != -1], keep)
