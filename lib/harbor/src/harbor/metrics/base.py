# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseMetric(ABC, Generic[T]):
    @abstractmethod
    def compute(self, rewards: list[T | None]) -> dict[str, float | int]:
        pass
