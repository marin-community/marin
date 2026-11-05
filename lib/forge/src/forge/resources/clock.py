# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recorded wall-clock reads, with an injectable source for tests."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from forge.effects.events import Event, Fingerprint, ReplayPolicy


@dataclass(frozen=True, slots=True)
class Now(Event[float]):
    """A deterministic call-site key; repeated emits of the same key replay the same value."""

    result_type: ClassVar[object] = float
    key: str


@dataclass(frozen=True, slots=True)
class Clock:
    source: Callable[[], float] = time.time

    def fingerprint(self) -> Fingerprint:
        return Fingerprint("clock", "wall-time-v1", ReplayPolicy.CACHE)

    async def now(self, event: Now) -> float:
        return self.source()
