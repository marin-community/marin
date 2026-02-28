# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass

import tensorstore as ts

import levanter.tracker
from levanter.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MetricSpec:
    name: str
    tensorstore_path: str
    value_key: str = "value"


_METRIC_SPECS = (
    _MetricSpec("cache_hit_count", "/tensorstore/cache/hit_count"),
    _MetricSpec("cache_miss_count", "/tensorstore/cache/miss_count"),
    _MetricSpec("cache_evict_count", "/tensorstore/cache/evict_count"),
    _MetricSpec("cache_chunk_reads", "/tensorstore/cache/chunk_cache/reads"),
    _MetricSpec("cache_chunk_writes", "/tensorstore/cache/chunk_cache/writes"),
    _MetricSpec("gcs_read_count", "/tensorstore/kvstore/gcs/read"),
    _MetricSpec("gcs_bytes_read", "/tensorstore/kvstore/gcs/bytes_read"),
    _MetricSpec("gcs_grpc_read_count", "/tensorstore/kvstore/gcs_grpc/read"),
    _MetricSpec("gcs_grpc_bytes_read", "/tensorstore/kvstore/gcs_grpc/bytes_read"),
)


def _metric_total(metric_path: str, value_key: str) -> tuple[bool, float]:
    matched = ts.experimental_collect_matching_metrics(metric_path, include_zero_metrics=True)
    found = False
    total = 0.0
    for metric in matched:
        if metric.get("name") != metric_path:
            continue
        found = True
        for value in metric.get("values", []):
            metric_value = value.get(value_key)
            if metric_value is not None:
                total += float(metric_value)
    return found, total


def tensorstore_metrics_interval_from_env() -> int | None:
    """Return TensorStore metrics logging cadence from env, if enabled."""
    every_env = os.environ.get("LEVANTER_LOG_TENSORSTORE_METRICS_EVERY")
    if every_env is None:
        return None

    every = int(every_env)
    if every <= 0:
        return None

    return every


def _collect_tensorstore_metrics(
    previous_totals: dict[str, float], active_metric_names: set[str]
) -> tuple[dict[str, float], dict[str, float]]:
    metrics: dict[str, float] = {}
    current: dict[str, float] = {}

    for spec in _METRIC_SPECS:
        found, total = _metric_total(spec.tensorstore_path, spec.value_key)
        if not found and spec.name not in active_metric_names:
            continue

        active_metric_names.add(spec.name)
        prev = previous_totals.get(spec.name, total)

        current[spec.name] = total
        metrics[f"data/tensorstore/{spec.name}_total"] = total
        metrics[f"data/tensorstore/{spec.name}_delta"] = total - prev

    hits_total = current.get("cache_hit_count")
    misses_total = current.get("cache_miss_count")
    if hits_total is not None and misses_total is not None:
        total_accesses = hits_total + misses_total
        if total_accesses > 0:
            metrics["data/tensorstore/cache_hit_rate_total"] = hits_total / total_accesses

        hits_delta = metrics.get("data/tensorstore/cache_hit_count_delta", 0.0)
        misses_delta = metrics.get("data/tensorstore/cache_miss_count_delta", 0.0)
        access_delta = hits_delta + misses_delta
        if access_delta > 0:
            metrics["data/tensorstore/cache_hit_rate_delta"] = hits_delta / access_delta

    return metrics, current


def build_tensorstore_metrics_logger(every: int):
    """Build a step-based TensorStore metrics logger.

    Args:
        every: Logging cadence in steps. Must be > 0.

    Returns:
        Callable taking a `step` int and logging TensorStore metrics to tracker.
    """
    if every <= 0:
        raise ValueError(f"every must be positive, got {every}")

    previous_totals: dict[str, float] = {}
    active_metric_names: set[str] = set()
    logger.info("Enabling TensorStore metrics logging every %s step(s).", every)

    def _log_tensorstore_metrics(step: int) -> None:
        metrics, current = _collect_tensorstore_metrics(previous_totals, active_metric_names)
        if not metrics:
            return
        previous_totals.update(current)
        levanter.tracker.log(metrics, step=step)

    return _log_tensorstore_metrics


def install_tensorstore_metrics_hook_if_enabled(trainer: Trainer) -> None:
    """Install a hook that logs TensorStore metrics if configured via environment."""
    every = tensorstore_metrics_interval_from_env()
    if every is None:
        return

    log_tensorstore_metrics = build_tensorstore_metrics_logger(every)

    def _log_tensorstore_metrics(step_info):
        log_tensorstore_metrics(step_info.step)

    trainer.add_hook(_log_tensorstore_metrics, every=every)
