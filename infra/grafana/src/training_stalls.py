# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded finelog queries and alert projection for stalled Levanter jobs."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
from hero_runs import TASK_STATE_LOOKBACK, HeroRun, as_utc, run_id_predicate, sql_timestamp

_TRAINING_STALL_AGE = timedelta(minutes=15)
_INITIALIZING_STALL_AGE = timedelta(minutes=45)
_PROGRESS_LOOKBACK = 2 * _TRAINING_STALL_AGE
_EXECUTION_LOOKBACK = TASK_STATE_LOOKBACK
# Keep the selected execution visible after its last heartbeat crosses the stall
# threshold. The exact run predicate and training-status projection keep this
# bounded scan below Finelog's deadline.
_ENROLLMENT_LOOKBACK = _EXECUTION_LOOKBACK

_STEP_METRIC = "step"
_PROGRESS_TIME_METRIC = "progress_time_seconds"
_PHASE_METRIC = "phase"

_INITIALIZING_PHASE = 0
_TRAINING_PHASE = 1
_FINISHED_PHASE = 2


def telemetry_query(now: datetime, runs: tuple[HeroRun, ...]) -> str:
    """Return current execution metrics for exact active hero run IDs."""
    run_predicate = run_id_predicate(runs)
    execution_since = sql_timestamp(now - _EXECUTION_LOOKBACK)
    progress_since = sql_timestamp(now - _PROGRESS_LOOKBACK)
    enrolled_since = sql_timestamp(now - _ENROLLMENT_LOOKBACK)
    end = sql_timestamp(now)
    metric_names = f"'{_PHASE_METRIC}', '{_STEP_METRIC}', '{_PROGRESS_TIME_METRIC}'"
    return (
        "WITH filtered AS ("
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS origin_cluster, "
        "run_id, job_id AS telemetry_job, execution_uid, name, value, "
        "timestamp_ms, seq, to_timestamp_millis(timestamp_ms) AS ts "
        'FROM "telemetry_v1" '
        f"WHERE service = 'levanter' AND name IN ({metric_names}) "
        f"AND {run_predicate} "
        "AND job_id IS NOT NULL AND execution_uid IS NOT NULL "
        f"AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM TIMESTAMP '{execution_since}') * 1000 AS BIGINT) "
        f"AND timestamp_ms < CAST(EXTRACT(EPOCH FROM TIMESTAMP '{end}') * 1000 AS BIGINT) "
        f"AND (name = '{_PHASE_METRIC}' OR timestamp_ms >= "
        f"CAST(EXTRACT(EPOCH FROM TIMESTAMP '{progress_since}') * 1000 AS BIGINT))"
        "), phase_history AS ("
        "SELECT origin_cluster, run_id, telemetry_job, execution_uid, ts, "
        "ROW_NUMBER() OVER ("
        "PARTITION BY origin_cluster, run_id, telemetry_job ORDER BY timestamp_ms DESC, seq DESC"
        ") AS rn FROM filtered "
        f"WHERE name = '{_PHASE_METRIC}' AND ts >= TIMESTAMP '{enrolled_since}'"
        "), enrolled AS ("
        "SELECT origin_cluster, run_id, telemetry_job, execution_uid FROM phase_history WHERE rn = 1"
        "), execution AS ("
        "SELECT enrolled.origin_cluster, enrolled.run_id, enrolled.telemetry_job, enrolled.execution_uid, "
        "MIN(filtered.ts) AS execution_started_at FROM enrolled JOIN filtered "
        "ON filtered.origin_cluster = enrolled.origin_cluster AND filtered.run_id = enrolled.run_id "
        "AND filtered.telemetry_job = enrolled.telemetry_job "
        "AND filtered.execution_uid = enrolled.execution_uid "
        f"WHERE filtered.name = '{_PHASE_METRIC}' "
        "GROUP BY enrolled.origin_cluster, enrolled.run_id, enrolled.telemetry_job, enrolled.execution_uid"
        "), recent AS ("
        "SELECT filtered.origin_cluster, filtered.run_id, filtered.telemetry_job, "
        "filtered.name, filtered.value, filtered.ts, execution.execution_started_at, "
        "ROW_NUMBER() OVER ("
        "PARTITION BY filtered.origin_cluster, filtered.run_id, filtered.telemetry_job, filtered.name "
        "ORDER BY filtered.timestamp_ms DESC, filtered.seq DESC"
        ") AS rn "
        "FROM filtered JOIN execution ON filtered.origin_cluster = execution.origin_cluster "
        "AND filtered.run_id = execution.run_id AND filtered.telemetry_job = execution.telemetry_job "
        "AND filtered.execution_uid = execution.execution_uid"
        ") "
        "SELECT origin_cluster AS cluster, run_id, telemetry_job, name, value, ts, execution_started_at "
        "FROM recent WHERE rn = 1"
    )


def _phase_name(phase: int | None) -> str:
    return {
        _INITIALIZING_PHASE: "initializing",
        _TRAINING_PHASE: "training",
        _FINISHED_PHASE: "finished",
    }.get(phase, "unknown")


def _row(cluster: str, job: str, phase: str, reason: str, value: int) -> dict:
    return {"cluster": cluster, "job": job, "phase": phase, "reason": reason, "value": value}


def training_stall_alert_rows(runs: tuple[HeroRun, ...], telemetry_metrics: pa.Table, now: datetime) -> list[dict]:
    """Project enrolled hero roots and their latest execution metrics into alert rows."""
    metrics_by_job: dict[tuple[str, str], dict[str, tuple[datetime, float]]] = {}
    execution_started_by_job: dict[tuple[str, str], datetime] = {}
    for row in telemetry_metrics.to_pylist():
        cluster = str(row["cluster"])
        run_id = str(row["run_id"])
        telemetry_job = str(row["telemetry_job"])
        for run in runs:
            if cluster != run.cluster or run_id != run.run_id:
                continue
            if telemetry_job != run.root_job and not telemetry_job.startswith(run.root_job + "/"):
                continue
            key = (run.cluster, run.root_job)
            execution_started = as_utc(row["execution_started_at"])
            current_execution = execution_started_by_job.get(key)
            if current_execution is None or execution_started > current_execution:
                execution_started_by_job[key] = execution_started
            metric = str(row["name"])
            observed = as_utc(row["ts"])
            current = metrics_by_job.setdefault(key, {}).get(metric)
            if current is None or observed > current[0]:
                metrics_by_job[key][metric] = (observed, float(row["value"]))

    now = as_utc(now)
    rows: list[dict] = []
    for run in runs:
        metrics = {name: value for name, (_, value) in metrics_by_job.get((run.cluster, run.root_job), {}).items()}
        raw_phase = metrics.get(_PHASE_METRIC)
        phase = int(raw_phase) if raw_phase is not None else None
        step = metrics.get(_STEP_METRIC, 0.0)
        progress_time = metrics.get(_PROGRESS_TIME_METRIC, 0.0)
        execution_started = execution_started_by_job.get((run.cluster, run.root_job), run.running_since)
        attempt_age = now - max(run.running_since, execution_started)

        reason = "healthy"
        value = 0
        is_training = phase == _TRAINING_PHASE or step > 0
        if phase == _FINISHED_PHASE:
            reason = "finished"
        elif is_training and progress_time > 0:
            progress_age = now - datetime.fromtimestamp(progress_time, tz=UTC)
            if progress_age >= _TRAINING_STALL_AGE:
                reason = "training_stalled"
                value = 1
        elif is_training and attempt_age >= _TRAINING_STALL_AGE:
            reason = "training_stalled"
            value = 1
        elif attempt_age >= _INITIALIZING_STALL_AGE:
            reason = "initializing_stale"
            value = 1
        else:
            reason = "training" if is_training else "initializing"

        phase_name = _phase_name(phase)
        if phase is None:
            phase_name = "training" if is_training else "initializing"
        rows.append(_row(run.cluster, run.root_job, phase_name, reason, value))

    if rows:
        return rows
    return [_row("fleet", "", "idle", "healthy", 0)]
