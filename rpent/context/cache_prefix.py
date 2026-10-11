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

"""Stable episode tool reference without reordering calls or reasoning."""

import copy
import json
from pathlib import Path

from .memory import END, START
from .scope import is_tool_definition

REFERENCE_ID = "msg_rpent_fixed_tool_reference_v1"
OMITTED = "[Tool definitions are available in the fixed tool reference.]"


class ToolReference:
    def __init__(self, path):
        definitions = json.loads(Path(path).read_text())
        if not isinstance(definitions, list) or not definitions:
            raise ValueError("nonempty tool reference required")
        if not all(is_tool_definition(item) for item in definitions):
            raise ValueError("tool reference requires explicit interface definitions")
        self.tools = {x["name"]: x for x in definitions}
        if len(self.tools) != len(definitions):
            raise ValueError("duplicate tool definition")
        self.text = json.dumps(definitions, ensure_ascii=False, separators=(",", ":"))

    def native_tools_cover_reference(self, request: dict) -> bool:
        """Require complete native interfaces for every referenced robot tool."""
        names = set()
        for tool in request.get("tools", []):
            if tool.get("type") == "namespace":
                prefix = tool.get("name", "") + "__"
                functions = tool.get("tools", [])
            else:
                prefix = ""
                functions = [tool]
            for function in functions:
                if (
                    function.get("type") == "function"
                    and isinstance(function.get("name"), str)
                    and isinstance(function.get("description"), str)
                    and isinstance(function.get("parameters"), dict)
                ):
                    names.add(prefix + function["name"])
        return set(self.tools).issubset(names)

    def apply(self, request):
        if request.get("previous_response_id"):
            raise ValueError("fixed reference requires full-history requests")
        result = copy.deepcopy(request)
        items = result["input"]
        if any(x.get("id") == REFERENCE_ID for x in items):
            raise ValueError("reference already present")
        removed = 0
        for item in items:
            if item.get("type") not in (
                "custom_tool_call_output",
                "function_call_output",
            ):
                continue
            blocks = item.get("output")
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if block.get("type") != "input_text":
                    continue
                try:
                    value = json.loads(block.get("text", ""))
                except (ValueError, TypeError):
                    continue
                definitions = value if isinstance(value, list) else [value]
                if not definitions or not all(
                    is_tool_definition(x) for x in definitions
                ):
                    continue
                retained = []
                for definition in definitions:
                    expected = self.tools.get(definition["name"])
                    if expected is None:
                        retained.append(definition)
                    elif expected != definition:
                        raise ValueError(
                            "tool reference changed: " + definition["name"]
                        )
                    else:
                        removed += 1
                if len(retained) != len(definitions):
                    block["text"] = (
                        json.dumps(retained, ensure_ascii=False)
                        if retained
                        else OMITTED
                    )
        position = 0
        while position < len(items) and (
            items[position].get("role") in ("system", "developer")
            or items[position].get("type") == "additional_tools"
        ):
            position += 1
        items.insert(
            position,
            {
                "type": "message",
                "role": "user",
                "id": REFERENCE_ID,
                "content": [{"type": "input_text", "text": self.text}],
            },
        )
        memory_moved = move_memory_to_tail(result)
        return result, {
            "enabled": True,
            "definitions": len(self.tools),
            "memory_layout": "dynamic-tail-v1",
            "memory_moved": memory_moved,
            "duplicates_removed": removed,
        }


def move_memory_to_tail(request: dict) -> bool:
    """Move selected references after history without changing selection or call order."""
    if (
        request["input"]
        and request["input"][-1].get("id") == "msg_rpent_dynamic_memory_v1"
    ):
        return False
    regions = []
    for item in request["input"]:
        if item.get("role") != "user":
            continue
        for block in item.get("content", []):
            text = block.get("text", "")
            if not isinstance(text, str) or (START not in text and END not in text):
                continue
            if text.count(START) != 1 or text.count(END) != 1:
                raise ValueError("memory layout requires exactly one marker pair")
            start, end = text.index(START), text.index(END) + len(END)
            if start >= end - len(END):
                raise ValueError("memory layout markers out of order")
            regions.append((block, text, start, end))
    if not regions:
        return False
    if len(regions) != 1:
        raise ValueError("memory layout requires exactly one managed region")
    block, text, start, end = regions[0]
    block["text"] = (
        text[:start]
        + "[Selected experience references follow the history.]"
        + text[end:]
    )
    request["input"].append(
        {
            "type": "message",
            "role": "user",
            "id": "msg_rpent_dynamic_memory_v1",
            "content": [
                {
                    "type": "input_text",
                    "text": "Demonstration experience references; not current observations or proof of success. "
                    "Re-localize from current images and use the current environment verdict.\n"
                    + text[start:end],
                }
            ],
        }
    )
    return True
