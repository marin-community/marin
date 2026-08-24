# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded finelog query and alert projection for hero-run training-loss spikes.

A run alerts when the lowest loss of the last few minutes clears its own trailing
mean plus six standard deviations, the band Levanter's skip-step optimizer
applies to a single step. Reducing the window to its floor means every step in it
has to be above the band, so a single excursion, the case skip-step already
discards, does not fire.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

import pyarrow as pa
from hero_runs import HeroRun, run_id_predicate, sql_timestamp

_LOSS_METRIC = "train_loss"

# The baseline window is [now - _BASELINE_LOOKBACK, now - _RECENT_WINDOW); the
# recent window is the remainder. One scan covers both.
_BASELINE_LOOKBACK = timedelta(minutes=60)
_RECENT_WINDOW = timedelta(minutes=5)

# Levanter's SkipStepConfig.sigma_factor, over a window of samples.
_SIGMA_FACTOR = 6.0
# A rise below this is inside the noise nobody would act on, however tight the
# trailing standard deviation has become.
_MIN_RISE = 0.05

# A run publishes train_loss once per step, so these clear within seconds on a
# fast run and within an hour on a slow one. Below them the run is warming up.
_MIN_BASELINE_SAMPLES = 20
_MIN_RECENT_SAMPLES = 5


@dataclass(frozen=True)
class LossWindows:
    """One run's loss statistics over the baseline and recent windows."""

    baseline_samples: int
    baseline_loss: float | None
    baseline_stddev: float | None
    recent_samples: int
    recent_loss: float | None
    recent_floor: float | None
    recent_peak: float | None


def loss_window_query(now: datetime, runs: tuple[HeroRun, ...]) -> str:
    """Return baseline and recent loss statistics for exact active hero run IDs."""
    run_predicate = run_id_predicate(runs)
    start = sql_timestamp(now - _BASELINE_LOOKBACK)
    recent = sql_timestamp(now - _RECENT_WINDOW)
    end = sql_timestamp(now)
    recent_ms = f"CAST(EXTRACT(EPOCH FROM TIMESTAMP '{recent}') * 1000 AS BIGINT)"
    return (
        'WITH telemetry AS (SELECT * FROM "telemetry_v1" '
        'UNION ALL SELECT * FROM "telemetry_v1.levanter.priority"), samples AS ('
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS origin_cluster, run_id, value, timestamp_ms "
        "FROM telemetry "
        f"WHERE service = 'levanter' AND name = '{_LOSS_METRIC}' "
        f"AND {run_predicate} "
        f"AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM TIMESTAMP '{start}') * 1000 AS BIGINT) "
        f"AND timestamp_ms < CAST(EXTRACT(EPOCH FROM TIMESTAMP '{end}') * 1000 AS BIGINT)"
        ") "
        "SELECT origin_cluster AS cluster, run_id, "
        f"SUM(CASE WHEN timestamp_ms < {recent_ms} THEN 1 ELSE 0 END) AS baseline_samples, "
        f"AVG(CASE WHEN timestamp_ms < {recent_ms} THEN value END) AS baseline_loss, "
        f"STDDEV(CASE WHEN timestamp_ms < {recent_ms} THEN value END) AS baseline_stddev, "
        f"SUM(CASE WHEN timestamp_ms >= {recent_ms} THEN 1 ELSE 0 END) AS recent_samples, "
        f"AVG(CASE WHEN timestamp_ms >= {recent_ms} THEN value END) AS recent_loss, "
        f"MIN(CASE WHEN timestamp_ms >= {recent_ms} THEN value END) AS recent_floor, "
        f"MAX(CASE WHEN timestamp_ms >= {recent_ms} THEN value END) AS recent_peak "
        "FROM samples GROUP BY 1, 2"
    )


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _diverged(value: float | None) -> bool:
    """True when a reduction came back NaN or infinite. A missing one has not diverged."""
    return value is not None and not isfinite(value)


def _windows_by_run(loss_windows: pa.Table) -> dict[tuple[str, str], LossWindows]:
    windows = {}
    for row in loss_windows.to_pylist():
        key = (str(row["cluster"]), str(row["run_id"]))
        windows[key] = LossWindows(
            baseline_samples=int(row["baseline_samples"] or 0),
            baseline_loss=_number(row["baseline_loss"]),
            baseline_stddev=_number(row["baseline_stddev"]),
            recent_samples=int(row["recent_samples"] or 0),
            recent_loss=_number(row["recent_loss"]),
            recent_floor=_number(row["recent_floor"]),
            recent_peak=_number(row["recent_peak"]),
        )
    return windows


def _classify(windows: LossWindows | None) -> tuple[str, int]:
    """Return the alert reason and firing value for one run's loss windows."""
    if windows is None:
        return "warming_up", 0
    if windows.recent_samples < _MIN_RECENT_SAMPLES or windows.baseline_samples < _MIN_BASELINE_SAMPLES:
        return "warming_up", 0

    # A NaN loss averages to NaN and compares false against every threshold, so
    # divergence into NaN or infinity needs its own case. Both reductions are
    # checked because engines order NaN differently.
    if _diverged(windows.recent_loss) or _diverged(windows.recent_peak):
        return "not_finite", 1

    floor, baseline = windows.recent_floor, windows.baseline_loss
    if floor is None or baseline is None or not isfinite(baseline):
        return "warming_up", 0
    # A flat baseline has zero spread, and an engine that cannot compute one
    # returns NULL. The absolute rise carries the band in both.
    stddev = windows.baseline_stddev
    spread = stddev if stddev is not None and isfinite(stddev) else 0.0
    band = baseline + max(_MIN_RISE, _SIGMA_FACTOR * spread)
    return ("spiking", 1) if floor > band else ("healthy", 0)


def _row(cluster: str, job: str, run: str, reason: str, value: int) -> dict:
    return {"cluster": cluster, "job": job, "run": run, "reason": reason, "value": value}


def loss_spike_alert_rows(runs: tuple[HeroRun, ...], loss_windows: pa.Table) -> list[dict]:
    """Project each enrolled hero run's loss windows into alert rows."""
    windows = _windows_by_run(loss_windows)
    rows: list[dict] = []
    for run in runs:
        reason, value = _classify(windows.get((run.cluster, run.run_id)))
        rows.append(_row(run.cluster, run.root_job, run.run_id, reason, value))
    if rows:
        return rows
    return [_row("fleet", "", "", "healthy", 0)]
