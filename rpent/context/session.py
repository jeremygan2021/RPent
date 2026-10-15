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

"""Episode-scoped selection state, independent of network transport."""

import copy
import hashlib
import json
from dataclasses import asdict, replace

from .controller import Controller, FixedController, Thresholds
from .observation import SCORING_VERSION, observe
from .parsing import parsed_payloads
from .selection import select_by_nc


class ContextSession:
    """Compose independently enabled image, text and action selectors."""

    def __init__(
        self,
        adapter,
        features,
        *,
        mode="full",
        rho=0.5,
        rho_image=None,
        rho_text=None,
        rho_action=None,
        stage="Unknown",
        action_ranges=None,
        controller: Controller | None = None,
        memory_selector=None,
    ):
        if (
            mode not in {"full", "image", "text", "image_text", "action", "all"}
            or not 0 < rho <= 1
        ):
            raise ValueError("invalid context mode or rho")
        self.adapter, self.features = adapter, features
        self.memory_selector = memory_selector
        self.rho_memory = memory_selector.rho if memory_selector else 0.5
        self.mode, self.rho, self.stage = mode, rho, stage
        self.rho_image = rho if rho_image is None else rho_image
        self.rho_text = rho if rho_text is None else rho_text
        self.rho_action = rho if rho_action is None else rho_action
        if any(
            not 0 < v <= 1 for v in (self.rho_image, self.rho_text, self.rho_action)
        ):
            raise ValueError("modality rho must be in (0, 1]")
        self.controller = controller or FixedController(
            Thresholds(self.rho_image, self.rho_text, self.rho_action, self.rho_memory)
        )
        self._decision_count = 0
        self._last_request = None
        self._last_result = None
        self._last_stats = {}
        self.stats = {}
        self._seen = set()
        self._kept = set()
        self._signature = None
        from .text import TextSelector

        self._text = TextSelector()
        from .action import ActionSelector

        self._action = ActionSelector(action_ranges)

    def process(self, request: dict) -> dict:
        """Apply independently enabled modalities using the same adapter and features."""
        with parsed_payloads():
            return self._process(request)

    def _process(self, request: dict) -> dict:
        signature = hashlib.sha256(
            json.dumps(
                (self.mode, self.stage, request), sort_keys=True, ensure_ascii=False
            ).encode()
        ).hexdigest()
        if signature == self._last_request:
            self.stats = copy.deepcopy(self._last_stats)
            self.stats["decision"]["reused"] = True
            return (
                request
                if self._last_result is None
                else copy.deepcopy(self._last_result)
            )
        original_request = request
        if self.mode in {"text", "image_text", "all"}:
            from .text import normalize_tool_wrappers

            request = normalize_tool_wrappers(request)
        records = {
            "image": self.adapter.extract(request),
            "text": self.adapter.extract_text(request)[0],
            "action": self.adapter.extract_action(request),
        }
        enabled = {
            "full": set(),
            "image": {"image"},
            "text": {"text"},
            "image_text": {"image", "text"},
            "action": {"action"},
            "all": {"image", "text", "action"},
        }[self.mode]
        observation = observe(
            records,
            {
                "image": (self._seen, self._kept),
                "text": (self._text._seen, self._text._kept),
                "action": (self._action._seen, self._action._kept),
            },
            enabled,
            self._last_stats,
            Thresholds(self.rho_image, self.rho_text, self.rho_action, self.rho_memory),
            self._decision_count,
            self.stage != "Unknown",
            self.memory_selector,
        )
        decision = self.controller.decide(observation)
        thresholds = Thresholds(**asdict(decision.thresholds))
        saved = copy.deepcopy(
            (
                self._seen,
                self._kept,
                self._signature,
                self._text,
                self._action,
                self.rho_image,
                self.rho_text,
                self.rho_action,
                self.rho_memory,
            )
        )
        self.rho_image, self.rho_text, self.rho_action = (
            thresholds.image,
            thresholds.text,
            thresholds.action,
        )
        if not getattr(self.controller, "legacy", False):
            self.rho_memory = thresholds.memory
        else:
            decision = replace(
                decision, thresholds=replace(thresholds, memory=self.rho_memory)
            )
        try:
            result = self._select(request, records["image"])
        except Exception:
            (
                self._seen,
                self._kept,
                self._signature,
                self._text,
                self._action,
                self.rho_image,
                self.rho_text,
                self.rho_action,
                self.rho_memory,
            ) = saved
            if self.memory_selector is not None:
                self.memory_selector.rho = self.rho_memory
            self.stats = copy.deepcopy(self._last_stats)
            raise
        self._decision_count += 1
        self.stats["decision"] = {
            "id": self._decision_count,
            "request_digest": signature,
            "reused": False,
            "scoring_version": SCORING_VERSION,
            "observation": asdict(observation),
            **asdict(decision),
        }
        self._last_request = signature
        self._last_result = (
            None if result is original_request else copy.deepcopy(result)
        )
        self._last_stats = copy.deepcopy(self.stats)
        return result

    def _select(self, request: dict, image_records=None) -> dict:
        """Run selectors once in their existing dependency order."""
        memory_stats = None
        if self.memory_selector is not None:
            self.memory_selector.rho = self.rho_memory
            request, memory_stats = self.memory_selector.process(
                request, self.adapter, self.features
            )
        # Memory selection replaces user text only; image positions stay unchanged.
        result = self._process_images(request, image_records)
        if self.mode in {"text", "image_text", "all"}:
            result = self._text.process(
                result, self.adapter, self.features, rho=self.rho_text, stage=self.stage
            )
            self.stats["text"] = self._text.stats
        if self.mode in {"action", "all"}:
            result = self._action.process(
                result, self.adapter, self.features, rho=self.rho_action
            )
            self.stats["action"] = self._action.stats
        if memory_stats is not None:
            self.stats["memory"] = memory_stats
        return result

    def _process_images(self, request: dict, records=None) -> dict:
        """Select from current native history; never inject stale archived items."""
        if records is None:
            records = self.adapter.extract(request)
        if self.mode in {"full", "text", "action"}:
            self.stats = {
                "mode": self.mode,
                "images_before": len(records),
                "images_after": len(records),
                "bytes_before": sum(len(r.data) for r in records),
                "bytes_after": sum(len(r.data) for r in records),
            }
            return request
        signature = (
            self.stage,
            self.rho_image,
            tuple(
                (r.id, r.digest, r.protected, r.metadata, r.task, r.initial)
                for r in records
            ),
        )
        if signature == self._signature:
            return self.adapter.render(request, records, self._kept)
        eligible = {r.id for r in records}
        previous, scores, details = {}, {}, []
        for record in records:
            if record.id not in eligible:
                details.append(
                    {
                        "id": record.id,
                        "camera": record.camera,
                        "protected": record.protected,
                        "novelty": None,
                        "relevance": None,
                        "score": None,
                        "reason": "previously_omitted",
                    }
                )
                continue
            h = self.features.dhash(record.digest, record.data)
            window = previous.setdefault(record.camera, [])
            novelty = min(
                ((h ^ old).bit_count() / 64 for old in window[-8:]), default=None
            )
            window.append(h)
            query = record.task
            if self.stage != "Unknown":
                query += "; " + self.stage
            relevance = self.features.similarity(record.metadata, query)
            # Missing comparison evidence contributes no novelty bonus.
            novelty_bonus = 0.0 if novelty is None else novelty
            score = (relevance + novelty_bonus) / 2
            if not record.protected:
                scores[record.id] = score
            details.append(
                {
                    "id": record.id,
                    "camera": record.camera,
                    "novelty": novelty,
                    "relevance": relevance,
                    "score": score,
                    "protected": record.protected,
                }
            )
        keep = select_by_nc(scores, self.rho_image) | {
            r.id for r in records if r.protected
        }
        total = sum(scores.values())
        self.stats = {
            "mode": self.mode,
            "stage": self.stage,
            "rho": self.rho_image,
            "images_before": len(records),
            "images_after": len(keep),
            "bytes_before": sum(len(r.data) for r in records),
            "bytes_after": sum(len(r.data) for r in records if r.id in keep),
            "nc": sum(v for k, v in scores.items() if k in keep) / total
            if total
            else None,
            "records": [dict(d, selected=d["id"] in keep) for d in details],
        }
        result = self.adapter.render(request, records, keep)
        self._seen.update(r.id for r in records)
        self._kept = keep
        self._signature = signature
        return result
