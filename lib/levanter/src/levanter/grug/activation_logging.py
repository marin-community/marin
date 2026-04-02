# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp

from levanter.tracker.histogram import Histogram, SummaryStats


MetricValue: TypeAlias = SummaryStats | jax.Array
LayerLogging: TypeAlias = dict[str, MetricValue]

_COSINE_EPS = 1e-6
_HISTOGRAM_NUM_BINS = 31


def block_activation_logging(
    *,
    attn_in: jax.Array,
    attn_out: jax.Array,
    mlp_in: jax.Array,
    mlp_out: jax.Array,
    block_out: jax.Array,
    max_abs_attn_logit: jax.Array,
) -> LayerLogging:
    out: LayerLogging = {
        "attn_in": SummaryStats.from_sharded_array(attn_in, num_bins=_HISTOGRAM_NUM_BINS, include_histogram=False),
        "attn_out": SummaryStats.from_sharded_array(attn_out, num_bins=_HISTOGRAM_NUM_BINS, include_histogram=False),
        "mlp_out": SummaryStats.from_sharded_array(mlp_out, num_bins=_HISTOGRAM_NUM_BINS, include_histogram=False),
        "block_out": SummaryStats.from_sharded_array(block_out, num_bins=_HISTOGRAM_NUM_BINS, include_histogram=False),
        "attn_out_cosine_with_in": _cosine_similarity_summary(attn_out, attn_in),
        "mlp_out_cosine_with_in": _cosine_similarity_summary(mlp_out, mlp_in),
        "max_abs_attn_logit": max_abs_attn_logit.astype(jnp.float32),
    }
    return out


def flatten_layer_logging(
    layer_logging: Sequence[Mapping[str, MetricValue]] | Mapping[str, MetricValue],
    *,
    prefix: str = "train/activations",
) -> dict[str, MetricValue]:
    if isinstance(layer_logging, Mapping):
        per_layer_logging = unstack_layer_logging(layer_logging)
    else:
        per_layer_logging = tuple(dict(layer_metrics) for layer_metrics in layer_logging)

    flattened: dict[str, MetricValue] = {}
    for layer_index, layer_metrics in enumerate(per_layer_logging):
        for name, value in layer_metrics.items():
            flattened[f"{prefix}/layer_{layer_index}/{name}"] = value

    return flattened


def unstack_layer_logging(stacked_logging: Mapping[str, MetricValue]) -> tuple[LayerLogging, ...]:
    if not stacked_logging:
        return ()

    num_layers = _leading_axis_size(next(iter(stacked_logging.values())))
    per_layer: list[LayerLogging] = []
    for layer_index in range(num_layers):
        per_layer.append({name: _slice_metric(value, layer_index) for name, value in stacked_logging.items()})

    return tuple(per_layer)


def _cosine_similarity_summary(lhs: jax.Array, rhs: jax.Array) -> SummaryStats:
    lhs32 = lhs.astype(jnp.float32)
    rhs32 = rhs.astype(jnp.float32)
    lhs_norm = jnp.linalg.norm(lhs32, axis=-1)
    rhs_norm = jnp.linalg.norm(rhs32, axis=-1)
    denom = jnp.maximum(lhs_norm * rhs_norm, _COSINE_EPS)
    cosine = jnp.sum(lhs32 * rhs32, axis=-1) / denom
    return SummaryStats.from_sharded_array(
        jnp.clip(cosine, -1.0, 1.0),
        num_bins=_HISTOGRAM_NUM_BINS,
        include_histogram=False,
    )


def _leading_axis_size(value: MetricValue) -> int:
    if isinstance(value, SummaryStats):
        return _leading_axis_size(value.min)
    if isinstance(value, Histogram):
        return int(value.bucket_counts.shape[0])
    return int(value.shape[0])


def _slice_metric(value: MetricValue, index: int) -> MetricValue:
    if isinstance(value, SummaryStats):
        histogram = value.histogram
        if histogram is not None:
            histogram = _slice_histogram(histogram, index)
        return SummaryStats(
            min=value.min[index],
            max=value.max[index],
            num=value.num[index],
            nonzero_count=value.nonzero_count[index],
            sum=value.sum[index],
            sum_squares=value.sum_squares[index],
            histogram=histogram,
        )
    if isinstance(value, Histogram):
        return _slice_histogram(value, index)
    return value[index]


def _slice_histogram(value: Histogram, index: int) -> Histogram:
    return Histogram(bucket_limits=value.bucket_limits[index], bucket_counts=value.bucket_counts[index])


__all__ = [
    "LayerLogging",
    "MetricValue",
    "block_activation_logging",
    "flatten_layer_logging",
    "unstack_layer_logging",
]
