"""Tensor reductions for local and distributed training metrics."""

from collections.abc import Callable
import math
from typing import Literal, Protocol

import torch


class MetricReduction(Protocol):
    """Reduce diagnostic tensors at one local or distributed ownership scope."""

    def mean(self, values: torch.Tensor) -> float: ...
    def sum(self, values: torch.Tensor) -> torch.Tensor: ...
    def combine(self, values: torch.Tensor, operation: Literal["sum", "max"]) -> torch.Tensor: ...


class LocalReduction:
    """Preserve local tensor arithmetic for driver diagnostics."""

    def mean(self, values: torch.Tensor) -> float:
        return values.mean().item()

    def sum(self, values: torch.Tensor) -> torch.Tensor:
        return values.sum()

    def combine(self, values: torch.Tensor, operation: Literal["sum", "max"]) -> torch.Tensor:
        return values


class WorldReduction:
    """Count one contribution per ordinary DP slice in world-wide diagnostics."""

    def __init__(self, all_reduce: Callable, contributes: bool):
        self.all_reduce = all_reduce
        self.contributes = contributes

    def combine(self, values: torch.Tensor, operation: Literal["sum", "max"]) -> torch.Tensor:
        values = values.clone()
        if not self.contributes:
            minimum = -math.inf if values.is_floating_point() else torch.iinfo(values.dtype).min
            values.fill_(0 if operation == "sum" else minimum)
        return self.all_reduce(values, operation)

    def sum(self, values: torch.Tensor) -> torch.Tensor:
        return self.combine(values.double().sum(), "sum")

    def mean(self, values: torch.Tensor) -> float:
        totals = torch.stack([values.new_tensor(values.numel(), dtype=torch.float64), values.double().sum()])
        count, total = self.combine(totals, "sum").tolist()
        return total / count if count else math.nan
