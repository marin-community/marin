# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded finelog queries and alert projection for stalled Levanter jobs."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa

_TASK_STATE_FRESHNESS = timedelta(seconds=90)
_TRAINING_STALL_AGE = timedelta(minutes=15)
_INITIALIZING_STALL_AGE = timedelta(minutes=45)
_TASK_STATE_LOOKBACK = timedelta(hours=1)
_PROGRESS_LOOKBACK = 2 * _TRAINING_STALL_AGE
# Levanter republishes `phase` every 60s, so enrollment is always recent and this
# scan can be bounded. Unbounded, it read every telemetry_v1 row once a minute and
# saturated the finelog hub. Keep this many multiples above that heartbeat:
# telemetry is best-effort, and a few dropped batches must not un-enroll a live job.
_ENROLLMENT_LOOKBACK = timedelta(minutes=15)

_STEP_METRIC = "step"
_PROGRESS_TIME_METRIC = "progress_time_seconds"
_PHASE_METRIC = "phase"

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


def telemetry_query(now: datetime) -> str:
    """Return latest retained enrollment and bounded progress per root job."""
    progress_since = _sql_timestamp(now - _PROGRESS_LOOKBACK)
    enrolled_since = _sql_timestamp(now - _ENROLLMENT_LOOKBACK)
    end = _sql_timestamp(now)
    metric_names = f"'{_PHASE_METRIC}', '{_STEP_METRIC}', '{_PROGRESS_TIME_METRIC}'"
    return (
        "WITH filtered AS ("
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS origin_cluster, "
        "json_get(resource_attributes_json, 'job_id') AS job, name, value, "
        "timestamp_ms, seq, to_timestamp_millis(timestamp_ms) AS ts "
        'FROM "telemetry_v1" '
        f"WHERE service = 'levanter' AND name IN ({metric_names}) "
        f"AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM TIMESTAMP '{progress_since}') * 1000 AS BIGINT) "
        f"AND timestamp_ms < CAST(EXTRACT(EPOCH FROM TIMESTAMP '{end}') * 1000 AS BIGINT) "
        f"AND (name <> '{_PHASE_METRIC}' OR timestamp_ms >= "
        f"CAST(EXTRACT(EPOCH FROM TIMESTAMP '{enrolled_since}') * 1000 AS BIGINT))"
        "), recent AS ("
        "SELECT origin_cluster, job, name, value, ts, "
        "ROW_NUMBER() OVER ("
        "PARTITION BY origin_cluster, job, name ORDER BY timestamp_ms DESC, seq DESC"
        ") AS rn "
        "FROM filtered WHERE job IS NOT NULL AND job <> ''"
        ") "
        "SELECT origin_cluster AS cluster, job, name, value, ts FROM recent WHERE rn = 1"
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


def _row(cluster: str, job: str, phase: str, reason: str, value: int) -> dict:
    return {"cluster": cluster, "job": job, "phase": phase, "reason": reason, "value": value}


def training_stall_alert_rows(task_states: pa.Table, telemetry_metrics: pa.Table, now: datetime) -> list[dict]:
    """Project active jobs and progress metrics into Grafana warning rows.

    Each row has string labels and exactly one numeric value. A job enrolls on
    `phase`; without it the row reports `producer_missing` at zero
    rather than a stall. Absent progress is itself evidence once a job is
    enrolled.
    """
    metrics_by_job: dict[tuple[str, str], dict[str, float]] = {}
    for row in telemetry_metrics.to_pylist():
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
        if raw_phase is None:
            # A producer old enough to predate the phase and progress metrics
            # still emits step. Judging it against progress it cannot
            # report would call every healthy job stalled, so report the gap.
            rows.append(_row(cluster, job, _phase_name(None), "producer_missing", 0))
            continue

        phase = int(raw_phase)
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

        rows.append(_row(cluster, job, _phase_name(phase), reason, value))

    if rows:
        return rows
    return [_row("fleet", "", "idle", "healthy", 0)]
