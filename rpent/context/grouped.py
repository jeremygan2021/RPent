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

"""Atomic label/result candidates layered on the isolated Text prototype."""

import json
import re
from dataclasses import dataclass, replace

from rpent.context.text import TextRecord

from .scope import ScopeAdapter


@dataclass(frozen=True)
class GroupedRecord(TextRecord):
    members: tuple = ()


class GroupedAdapter(ScopeAdapter):
    def extract_text(self, request):
        records, query = super().extract_text(request)
        by_position = {
            (r.item, r.block): r
            for r in records
            if r.block is not None and r.block >= 0
        }
        consumed = set()
        grouped = []
        for record in records:
            position = (record.item, record.block)
            if position in consumed:
                continue
            following = (
                by_position.get((record.item, record.block + 1))
                if isinstance(record.block, int) and record.block >= 0
                else None
            )
            label = bool(
                re.fullmatch(r"[A-Za-z][A-Za-z0-9_. -]{0,79}", record.text.strip())
            )
            try:
                payload = json.loads(following.text) if following else None
            except ValueError:
                payload = None
            if label and isinstance(payload, (dict, list)):
                grouped.append(
                    GroupedRecord(
                        record.id,
                        record.item,
                        record.text + "\n" + following.text,
                        record.protected or following.protected,
                        tuple(sorted(set(record.event_refs + following.event_refs))),
                        record.block,
                        (record, following),
                    )
                )
                consumed.add((following.item, following.block))
            else:
                grouped.append(record)
        # Scoring windows and recency tie breaks follow the native history,
        # including call inputs before their paired result blocks.
        grouped.sort(
            key=lambda record: (
                record.item,
                -1 if record.block is None else record.block,
            )
        )
        return grouped, query

    def render_text(self, request, records, keep):
        expanded = []
        for record in records:
            if isinstance(record, GroupedRecord):
                expanded.extend(
                    replace(member, id=record.id) for member in record.members
                )
            else:
                expanded.append(record)
        return super().render_text(request, expanded, keep)
