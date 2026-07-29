# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed bridge projections for the custom infrastructure status view."""

from dataclasses import dataclass
from datetime import datetime

CANARY_METRICS_NAMESPACE = "infra.canary.metrics"
PROVISIONING_LOOKBACK_HOURS = 6

_FLEET_SCOPE = "fleet"
_METRIC_READY = "provision_ready"
_METRIC_STOCKOUT = "provision_stockout"
_METRIC_ERROR = "provision_error"
_METRIC_PREEMPTED = "provision_preempted"
_METRIC_OUTCOMES = "provision_outcomes"
_METRIC_SUCCESS_RATIO = "provision_success_ratio"
_METRIC_POOLS_PLACING = "provision_pools_placing"
_METRIC_POOLS_NO_READY_OUTCOME = "provision_pools_stockout_dead"
_METRIC_WINDOW_HOURS = "provision_window_hours"
_METRIC_LATENCY_SECONDS = "provision_latency_seconds"


@dataclass(frozen=True)
class ProvisioningRow:
    """One fleet or resource-pool summary from the latest provisioning cycle."""

    scope: str
    collected_at: int
    resource_type: str
    scale_group: str
    zone: str
    ready: float
    stockout: float
    error: float
    preempted: float
    outcomes: float
    success_ratio: float | None
    pools_placing: float
    pools_no_ready_outcome: float
    latency_p50_seconds: float | None
    latency_p95_seconds: float | None
    window_hours: float | None


@dataclass
class _Counts:
    ready: float = 0
    stockout: float = 0
    error: float = 0
    preempted: float = 0
    outcomes: float = 0
    latency_p50_seconds: float | None = None
    latency_p95_seconds: float | None = None


@dataclass
class _Fleet(_Counts):
    success_ratio: float | None = None
    pools_placing: float = 0
    pools_no_ready_outcome: float = 0
    window_hours: float | None = None


def provisioning_query(cutoff: datetime) -> str:
    """Return the bounded query for the latest complete provisioning cycle."""
    cutoff_text = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    return f"""
SELECT metric, value, labels, collected_at
FROM "{CANARY_METRICS_NAMESPACE}"
WHERE metric LIKE 'provision_%'
  AND collected_at >= TIMESTAMP '{cutoff_text}'
  AND collected_at = (
    SELECT MAX(collected_at)
    FROM "{CANARY_METRICS_NAMESPACE}"
    WHERE metric LIKE 'provision_%'
      AND collected_at >= TIMESTAMP '{cutoff_text}'
  )
""".strip()


def _apply_counts(target: _Counts, metric: str, value: float, quantile: str) -> None:
    if metric == _METRIC_READY:
        target.ready = value
    elif metric == _METRIC_STOCKOUT:
        target.stockout = value
    elif metric == _METRIC_ERROR:
        target.error = value
    elif metric == _METRIC_PREEMPTED:
        target.preempted = value
    elif metric == _METRIC_OUTCOMES:
        target.outcomes = value
    elif metric == _METRIC_LATENCY_SECONDS and quantile == "p50":
        target.latency_p50_seconds = value
    elif metric == _METRIC_LATENCY_SECONDS and quantile == "p95":
        target.latency_p95_seconds = value


def _apply_fleet(target: _Fleet, metric: str, value: float, quantile: str) -> None:
    _apply_counts(target, metric, value, quantile)
    if metric == _METRIC_SUCCESS_RATIO:
        target.success_ratio = value
    elif metric == _METRIC_POOLS_PLACING:
        target.pools_placing = value
    elif metric == _METRIC_POOLS_NO_READY_OUTCOME:
        target.pools_no_ready_outcome = value
    elif metric == _METRIC_WINDOW_HOURS:
        target.window_hours = value


def _row(
    scope: str,
    collected_at: int,
    counts: _Counts,
    *,
    resource_type: str = "",
    scale_group: str = "",
    zone: str = "",
    fleet: _Fleet | None = None,
) -> ProvisioningRow:
    success_ratio = counts.ready / counts.outcomes if counts.outcomes else None
    return ProvisioningRow(
        scope=scope,
        collected_at=collected_at,
        resource_type=resource_type,
        scale_group=scale_group,
        zone=zone,
        ready=counts.ready,
        stockout=counts.stockout,
        error=counts.error,
        preempted=counts.preempted,
        outcomes=counts.outcomes,
        success_ratio=fleet.success_ratio if fleet is not None else success_ratio,
        pools_placing=fleet.pools_placing if fleet is not None else 0,
        pools_no_ready_outcome=fleet.pools_no_ready_outcome if fleet is not None else 0,
        latency_p50_seconds=counts.latency_p50_seconds,
        latency_p95_seconds=counts.latency_p95_seconds,
        window_hours=fleet.window_hours if fleet is not None else None,
    )


def provisioning_rows(rows: list[dict[str, object]]) -> list[ProvisioningRow]:
    """Convert the latest EAV cycle to one fleet row and ordered pool rows."""
    if not rows:
        return []

    collected_at = int(rows[0]["collected_at"])
    fleet = _Fleet()
    pools: dict[tuple[str, str, str], _Counts] = {}

    for source in rows:
        metric = str(source["metric"])
        value = float(source["value"])
        quantile = str(source.get("label_quantile") or "")
        if source.get("label_scope") == _FLEET_SCOPE:
            _apply_fleet(fleet, metric, value, quantile)
            continue

        key = (
            str(source.get("label_resource_type") or ""),
            str(source.get("label_scale_group") or ""),
            str(source.get("label_zone") or ""),
        )
        if not all(key):
            continue
        _apply_counts(pools.setdefault(key, _Counts()), metric, value, quantile)

    result = [_row(_FLEET_SCOPE, collected_at, fleet, fleet=fleet)]
    result.extend(
        _row("pool", collected_at, counts, resource_type=key[0], scale_group=key[1], zone=key[2])
        for key, counts in sorted(pools.items())
    )
    return result
