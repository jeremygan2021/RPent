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

"""Threshold decisions independent of model protocols and robot environments."""

from dataclasses import dataclass
from typing import Protocol

from .observation import Observation


@dataclass(frozen=True)
class Thresholds:
    """Requested score-mass coverage, in image/text/action/memory order."""

    image: float
    text: float
    action: float
    memory: float = 0.5

    def __post_init__(self) -> None:
        for value in (self.image, self.text, self.action, self.memory):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < value <= 1
            ):
                raise ValueError("thresholds must be finite numbers in (0, 1]")


@dataclass(frozen=True)
class Decision:
    """Threshold choice with optional learned-policy sampling metadata."""

    thresholds: Thresholds
    policy_version: str = "fixed-v1"
    latent_action: tuple[float, ...] | None = None
    log_prob: float | None = None
    value: float | None = None
    behavior_policy: str = "actor"


class Controller(Protocol):
    """Choose thresholds without changing selector state."""

    def decide(self, observation: Observation) -> Decision:
        """Return one decision for a new logical request."""
        ...


@dataclass(frozen=True)
class FixedController:
    """Preserve the existing configured thresholds."""

    thresholds: Thresholds

    def decide(self, observation: Observation) -> Decision:
        """Return configured thresholds without training dependencies."""
        return Decision(self.thresholds)
