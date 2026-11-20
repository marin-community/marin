"""Tensor reductions for local and distributed training metrics."""

from collections.abc import Callable
import math
from typing import Literal, Protocol

import torch


def linear_quantiles(values: torch.Tensor, probabilities: tuple[float, ...]) -> list[float]:
    """Linear-interpolated quantiles of a finite 1-D tensor; torch.quantile refuses more than 2**24 values."""
    count = values.numel()
    positions = torch.tensor(probabilities, dtype=torch.float64) * (count - 1)
    lowest = int(positions.min().floor())
    threshold = values.kthvalue(lowest + 1).values
    greater = values[values > threshold].sort().values
    tail = torch.cat([threshold.expand(count - lowest - greater.numel()), greater]).double()
    lower = positions.floor()
    below = tail[(lower - lowest).long()]
    above = tail[(positions.ceil() - lowest).long()]
    return (below + (above - below) * (positions - lower)).tolist()


class MetricReduction(Protocol):
    """Reduce diagnostic tensors at one local or distributed ownership scope."""

    def mean(self, values: torch.Tensor) -> float: ...
    def std(self, values: torch.Tensor) -> float: ...
    def quantiles(self, values: torch.Tensor, probabilities: tuple[float, ...]) -> list[float]: ...
    def sum(self, values: torch.Tensor) -> torch.Tensor: ...
    def combine(self, values: torch.Tensor, operation: Literal["sum", "max"]) -> torch.Tensor: ...


class LocalReduction:
    """Preserve local tensor arithmetic for driver diagnostics."""

    def mean(self, values: torch.Tensor) -> float:
        return values.mean().item()

    def std(self, values: torch.Tensor) -> float:
        return values.std().item()

    def quantiles(self, values: torch.Tensor, probabilities: tuple[float, ...]) -> list[float]:
        return linear_quantiles(values, probabilities)

    def sum(self, values: torch.Tensor) -> torch.Tensor:
        return values.sum()

    def combine(self, values: torch.Tensor, operation: Literal["sum", "max"]) -> torch.Tensor:
        return values


class WorldReduction:
    """Count one contribution per ordinary DP slice in world-wide diagnostics."""

    def __init__(self, all_reduce: Callable, gather: Callable[[torch.Tensor], torch.Tensor], contributes: bool):
        self.all_reduce = all_reduce
        self.contributes = contributes
        self.gather = gather

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

    def std(self, values: torch.Tensor) -> float:
        centered = values.double() - self.mean(values)
        count, square_sum = self.combine(
            torch.stack([centered.new_tensor(centered.numel()), centered.square().sum()]), "sum"
        ).tolist()
        return (square_sum / (count - 1)) ** 0.5 if count > 1 else math.nan

    def quantiles(self, values: torch.Tensor, probabilities: tuple[float, ...]) -> list[float]:
        received = self.gather(values if self.contributes else values[:0])
        result = received.new_zeros(len(probabilities))
        if received.numel():
            result = result.new_tensor(linear_quantiles(received.cpu(), probabilities))
        return self.all_reduce(result, "sum").tolist()
