# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded finelog queries and alert projection for stalled Levanter jobs."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa

_TASK_STATE_FRESHNESS = timedelta(seconds=90)
_TRAINING_STALL_AGE = timedelta(minutes=15)
_INITIALIZING_STALL_AGE = timedelta(minutes=45)
_TASK_STATE_LOOKBACK = timedelta(hours=1)
_TELLTALE_LOOKBACK = timedelta(days=1)

_STEP_METRIC = "levanter_step"
_PROGRESS_TIME_METRIC = "levanter_progress_time_seconds"
_PHASE_METRIC = "levanter_phase"

_INITIALIZING_PHASE = 0
_TRAINING_PHASE = 1
_FINISHED_PHASE = 2


def _sql_timestamp(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def task_state_query(now: datetime) -> str:
    """Return the bounded query for recently running root jobs."""
    start = _sql_timestamp(now - _TASK_STATE_LOOKBACK)
    fresh = _sql_timestamp(now - _TASK_STATE_FRESHNESS)
    return (
        "WITH history AS ("
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS cluster, root_job_id, ts, running, "
        "MIN(CASE WHEN running > 0 THEN ts END) OVER ("
        "PARTITION BY COALESCE(NULLIF(cluster,''),'unknown'), root_job_id"
        ") AS running_since, "
        "ROW_NUMBER() OVER ("
        "PARTITION BY COALESCE(NULLIF(cluster,''),'unknown'), root_job_id ORDER BY ts DESC"
        ") AS rn "
        'FROM "iris.task_state" '
        f"WHERE root_job_id <> '' AND ts >= TIMESTAMP '{start}'"
        ") "
        "SELECT cluster, root_job_id AS job, ts AS state_at, running_since "
        "FROM history "
        f"WHERE rn = 1 AND running > 0 AND ts >= TIMESTAMP '{fresh}'"
    )


def telltale_query(now: datetime) -> str:
    """Return the bounded query for the latest progress metrics per root job."""
    start = _sql_timestamp(now - _TELLTALE_LOOKBACK)
    names = f"'{_STEP_METRIC}', '{_PROGRESS_TIME_METRIC}', '{_PHASE_METRIC}'"
    return (
        "WITH recent AS ("
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS cluster, job_id AS job, "
        "name, value, ts, "
        "ROW_NUMBER() OVER ("
        "PARTITION BY COALESCE(NULLIF(cluster,''),'unknown'), job_id, name ORDER BY ts DESC"
        ") AS rn "
        'FROM "telltale" '
        f"WHERE job_id IS NOT NULL AND job_id <> '' AND name IN ({names}) "
        f"AND ts >= TIMESTAMP '{start}'"
        ") "
        "SELECT cluster, job, name, value, ts FROM recent WHERE rn = 1"
    )


def _as_utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"expected timestamp, got {value!r}")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _phase_name(phase: int | None) -> str:
    return {
        _INITIALIZING_PHASE: "initializing",
        _TRAINING_PHASE: "training",
        _FINISHED_PHASE: "finished",
    }.get(phase, "unknown")


def training_stall_alert_rows(task_states: pa.Table, telltale_metrics: pa.Table, now: datetime) -> list[dict]:
    """Project active jobs and progress metrics into Grafana warning rows.

    Each row has string labels and exactly one numeric value. Missing Telltale
    data is itself evidence after the initialization grace period.
    """
    metrics_by_job: dict[tuple[str, str], dict[str, float]] = {}
    for row in telltale_metrics.to_pylist():
        key = (str(row["cluster"]), str(row["job"]))
        metrics_by_job.setdefault(key, {})[str(row["name"])] = float(row["value"])

    now = _as_utc(now)
    rows: list[dict] = []
    for state in task_states.to_pylist():
        cluster = str(state["cluster"])
        job = str(state["job"])
        running_age = now - _as_utc(state["running_since"])
        metrics = metrics_by_job.get((cluster, job), {})
        if not metrics:
            continue

        raw_phase = metrics.get(_PHASE_METRIC)
        phase = int(raw_phase) if raw_phase is not None else None
        step = metrics.get(_STEP_METRIC, 0.0)
        progress_time = metrics.get(_PROGRESS_TIME_METRIC, 0.0)

        reason = "healthy"
        value = 0
        is_training = phase == _TRAINING_PHASE or step > 0
        if phase == _FINISHED_PHASE:
            reason = "finished"
        elif is_training and progress_time > 0:
            progress_age = now - datetime.fromtimestamp(progress_time, tz=UTC)
            if progress_age >= _TRAINING_STALL_AGE:
                reason = "optimizer_progress_stale"
                value = 1
        elif is_training and running_age >= _TRAINING_STALL_AGE:
            reason = "optimizer_progress_missing"
            value = 1
        elif running_age >= _INITIALIZING_STALL_AGE:
            reason = "initializing_stale"
            value = 1
        else:
            reason = "training" if is_training else "initializing"

        rows.append(
            {
                "cluster": cluster,
                "job": job,
                "phase": _phase_name(phase),
                "reason": reason,
                "value": value,
            }
        )

    if rows:
        return rows
    return [{"cluster": "fleet", "job": "", "phase": "idle", "reason": "healthy", "value": 0}]
