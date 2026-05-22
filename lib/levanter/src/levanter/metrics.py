# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Metric abstraction for correct aggregation across microbatches.

See docs/metrics.md for design rationale.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import jax
import jax.numpy as jnp


logger = logging.getLogger(__name__)


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
        value = self._value.array if hasattr(self._value, "array") else self._value
        if self.reduction is ReductionType.MEAN:
            count = self._count.array if hasattr(self._count, "array") else self._count
            # jnp.where keeps this JIT-safe and avoids divide-by-zero on empty folds.
            return jnp.where(count > 0, value / count, 0.0)
        return value

    def __float__(self) -> float:
        """Coerce Metric to float outside of a JIT context."""
        return float(self.value())

    @classmethod
    def from_value(cls, value: float | jax.Array, reduction: ReductionType) -> "Metric":
        """Create metric from single observation."""
        # Float literals (not int) keep dtype consistent across scan iterations.
        count = 1.0 if reduction is ReductionType.MEAN else 0.0
        return cls(_value=value, _count=count, reduction=reduction)


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
            return Metric(
                _value=m1._value + m2._value,
                _count=m1._count + m2._count,
                reduction=reduction,
            )
        case ReductionType.SUM:
            new_value = m1._value + m2._value
        case ReductionType.MAX:
            new_value = jnp.maximum(m1._value, m2._value)
        case ReductionType.MIN:
            new_value = jnp.minimum(m1._value, m2._value)
        case ReductionType.LAST:
            new_value = m2._value

    return Metric(_value=new_value, _count=0.0, reduction=reduction)


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
        # `name` is a Python string decided at trace time, so a plain logger.warning
        # fires once per trace rather than once per step (as `jax.debug.print` would).
        logger.warning(
            "Ambiguous metric name: %s, defaulting to MEAN. Return an explicit Metric to avoid this message.",
            name,
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
