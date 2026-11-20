# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Allocate a reserved GPU window across concurrent task executions."""

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class GpuAccounting:
    """Share each reserved interval equally among the tasks active in it."""

    gpu_count: int
    clock: Callable[[], float]
    active: dict[str, float] = field(default_factory=dict)
    idle_hours: float = 0.0
    total_hours: float = 0.0
    last: float | None = None
    lock: Lock = field(default_factory=Lock)

    def advance(self) -> None:
        now = self.clock()
        if self.last is not None:
            hours = (now - self.last) * self.gpu_count / 3600
            self.total_hours += hours
            if self.active:
                for key in self.active:
                    self.active[key] += hours / len(self.active)
            else:
                self.idle_hours += hours
        self.last = now

    def start(self, key: str) -> None:
        with self.lock:
            self.advance()
            if key in self.active:
                raise ValueError(f"Duplicate active task: {key}")
            self.active[key] = 0.0

    def finish(self, key: str) -> float:
        with self.lock:
            self.advance()
            return self.active.pop(key)

    def idle_cost(self) -> float:
        with self.lock:
            self.advance()
            if self.active:
                raise ValueError("Cannot seal GPU accounting while tasks are active")
            return self.idle_hours
