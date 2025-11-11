# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Metric abstraction for correct aggregation across microbatches.

See docs/metrics.md for design rationale.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import jax
import jax.numpy as jnp


class ReductionType(Enum):
    """Reduction strategy for metric aggregation."""

    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    LAST = "last"


@dataclass
class Metric:
    """
    A foldable metric that uses internal state for reduction.

    We use a fixed-size internal representation (value + count) instead of a
    variable-length tuple to maintain constant pytree structure across fold operations.
    JAX's scan requires the carry to have the same structure in each iteration,
    so we can't use a growing list of samples.

    Internal representation by reduction type:
    - MEAN: (_value=sum, _count=count)
    - SUM: (_value=sum, _count=0)
    - MAX: (_value=max_value, _count=0)
    - MIN: (_value=min_value, _count=0)
    - LAST: (_value=last_value, _count=0)

    Forms a monoid: fold is associative.
    """

    _value: float | jax.Array = 0.0
    _count: float | jax.Array = 0.0
    reduction: ReductionType = ReductionType.MEAN

    def value(self):
        """Extract the scalar value by applying reduction."""
        # Extract underlying array from NamedArray if needed
        value = self._value.array if hasattr(self._value, "array") else self._value
        count = self._count.array if hasattr(self._count, "array") else self._count

        match self.reduction:
            case ReductionType.MEAN:
                # Use jnp.where for JIT compatibility
                return jnp.where(count > 0, value / count, 0.0)
            case ReductionType.SUM:
                return value
            case ReductionType.MAX:
                return value
            case ReductionType.MIN:
                return value
            case ReductionType.LAST:
                return value

    def __float__(self) -> float:
        """Coerce Metric to float outside of a JIT context."""
        return float(self.value())

    @classmethod
    def from_value(cls, value: float | jax.Array, reduction: ReductionType) -> "Metric":
        """Create metric from single observation."""
        match reduction:
            case ReductionType.MEAN:
                return cls(_value=value, _count=1, reduction=reduction)
            case ReductionType.SUM:
                return cls(_value=value, _count=0, reduction=reduction)
            case ReductionType.MAX:
                return cls(_value=value, _count=0, reduction=reduction)
            case ReductionType.MIN:
                return cls(_value=value, _count=0, reduction=reduction)
            case ReductionType.LAST:
                return cls(_value=value, _count=0, reduction=reduction)


def _metric_flatten(m: Metric):
    """Flatten Metric for JAX - reduction is aux_data, value/count are children."""
    return (m._value, m._count), m.reduction


def _metric_unflatten(reduction: ReductionType, children):
    """Unflatten Metric for JAX."""
    value, count = children
    return Metric(_value=value, _count=count, reduction=reduction)


jax.tree_util.register_pytree_node(Metric, _metric_flatten, _metric_unflatten)


def fold(m1: Metric, m2: Metric) -> Metric:
    """Combine two Metrics according to their reduction type."""
    reduction = m1.reduction

    match reduction:
        case ReductionType.MEAN:
            # Combine sum and count
            new_value = m1._value + m2._value
            new_count = m1._count + m2._count
        case ReductionType.SUM:
            # Sum the values
            new_value = m1._value + m2._value
            new_count = 0.0
        case ReductionType.MAX:
            # Take maximum
            new_value = jnp.maximum(m1._value, m2._value)
            new_count = 0.0
        case ReductionType.MIN:
            # Take minimum
            new_value = jnp.minimum(m1._value, m2._value)
            new_count = 0.0
        case ReductionType.LAST:
            # Keep most recent (second argument)
            new_value = m2._value
            new_count = 0.0

    return Metric(_value=new_value, _count=new_count, reduction=reduction)


def auto_metric_from_name(name: str, value: float | jax.Array) -> Metric:
    """
    Infer metric type from name and create Metric with appropriate reduction.

    Naming conventions:
    - num_*, total_*, *_count, *_total, *_sum → SUM
    - *_max, max_* → MAX
    - *_min, min_* → MIN
    - learning_rate, *_rate (but not accuracy_rate) → LAST
    - Default: MEAN (accuracy, loss, perplexity, etc.)
    """
    name_lower = name.lower()

    sum_indicators = (
        "num_",
        "total_",
        "_count",
        "_total",
        "_sum",
    )

    max_indicators = ("_max", "max_")
    min_indicators = ("_min", "min_")
    last_indicators = ("learning_rate", "_rate")
    mean_indicators = ("accuracy", "loss", "perplexity", "error", "precision", "recall")

    # Check more specific patterns (max/min/sum) before general patterns (mean)
    if any(ind in name_lower for ind in sum_indicators):
        reduction = ReductionType.SUM
    elif any(ind in name_lower for ind in max_indicators):
        reduction = ReductionType.MAX
    elif any(ind in name_lower for ind in min_indicators):
        reduction = ReductionType.MIN
    elif any(ind in name_lower for ind in last_indicators):
        reduction = ReductionType.LAST
    elif any(ind in name_lower for ind in mean_indicators):
        reduction = ReductionType.MEAN
    else:
        jax.debug.print(
            f"Ambiguous metric name: {name}, defaulting to MEAN. Return an explicit Metric to avoid this message."
        )
        reduction = ReductionType.MEAN

    return Metric.from_value(value, reduction)


def unwrap_metrics(pytree):
    """
    Walk a pytree and extract .value() from all Metric objects.
    """

    def _unwrap(x):
        if isinstance(x, Metric):
            return x.value()
        return x

    return jax.tree_util.tree_map(_unwrap, pytree, is_leaf=lambda x: isinstance(x, Metric))


class LossFunctionWithMetrics(Protocol):
    """
    Loss function protocol for internal use after wrapping.

    Returns (scalar_loss, metrics_dict) where metrics are Metric objects.
    User code returns plain floats/arrays which WrappedLossFunction converts to Metrics.
    """

    def __call__(
        self, model: Any, batch: Any, **batch_kwargs: dict[str, Any]
    ) -> tuple[jax.Array, dict[str, Metric]]: ...
