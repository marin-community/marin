# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded finelog queries and alert projection for stalled Levanter jobs."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pyarrow as pa
from vllm_observability import sql_string

_TASK_STATE_FRESHNESS = timedelta(seconds=90)
_TRAINING_STALL_AGE = timedelta(minutes=15)
_INITIALIZING_STALL_AGE = timedelta(minutes=45)
_TASK_STATE_LOOKBACK = timedelta(hours=1)
_PROGRESS_LOOKBACK = 2 * _TRAINING_STALL_AGE
_EXECUTION_LOOKBACK = _TASK_STATE_LOOKBACK
# Keep the selected execution visible after its last heartbeat crosses the stall
# threshold. The exact run predicate and training-status projection keep this
# bounded scan below Finelog's deadline.
_ENROLLMENT_LOOKBACK = _EXECUTION_LOOKBACK

_STEP_METRIC = "step"
_PROGRESS_TIME_METRIC = "progress_time_seconds"
_PHASE_METRIC = "phase"
_HERO_RUN_PREFIX = "hero-"
_COORDINATOR_SUFFIX = "-coord"
_HERO_ROOT_PATTERN = f"%/{_HERO_RUN_PREFIX}%{_COORDINATOR_SUFFIX}"

_INITIALIZING_PHASE = 0
_TRAINING_PHASE = 1
_FINISHED_PHASE = 2


@dataclass(frozen=True)
class HeroRun:
    cluster: str
    root_job: str
    run_id: str
    running_since: datetime


def _sql_timestamp(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def task_state_query(now: datetime) -> str:
    """Return fresh active root jobs whose run name opts into hero alerts."""
    start = _sql_timestamp(now - _TASK_STATE_LOOKBACK)
    fresh = _sql_timestamp(now - _TASK_STATE_FRESHNESS)
    end = _sql_timestamp(now)
    return (
        "WITH samples AS ("
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS cluster, root_job_id, ts, running "
        'FROM "iris.task_state" '
        f"WHERE root_job_id LIKE '{_HERO_ROOT_PATTERN}' AND ts >= TIMESTAMP '{start}' "
        f"AND ts < TIMESTAMP '{end}'"
        "), segmented AS ("
        "SELECT cluster, root_job_id, ts, running, "
        "SUM(CASE WHEN running <= 0 THEN 1 ELSE 0 END) OVER ("
        "PARTITION BY cluster, root_job_id ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        ") AS running_segment FROM samples"
        "), history AS ("
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS cluster, root_job_id, ts, running, "
        "MIN(CASE WHEN running > 0 THEN ts END) OVER ("
        "PARTITION BY cluster, root_job_id, running_segment"
        ") AS running_since, "
        "ROW_NUMBER() OVER ("
        "PARTITION BY cluster, root_job_id ORDER BY ts DESC"
        ") AS rn "
        "FROM segmented"
        ") "
        "SELECT cluster, root_job_id AS job, ts AS state_at, running_since "
        "FROM history "
        f"WHERE rn = 1 AND running > 0 AND ts >= TIMESTAMP '{fresh}'"
    )


def _run_id(root_job: str) -> str | None:
    root_name = root_job.rsplit("/", 1)[-1]
    if not root_name.startswith(_HERO_RUN_PREFIX) or not root_name.endswith(_COORDINATOR_SUFFIX):
        return None
    run_id = root_name.removesuffix(_COORDINATOR_SUFFIX)
    return run_id if run_id != _HERO_RUN_PREFIX else None


def active_hero_runs(task_states: pa.Table) -> tuple[HeroRun, ...]:
    """Return structured run identities from task-state selector rows."""
    runs = []
    for row in task_states.to_pylist():
        root_job = str(row["job"])
        run_id = _run_id(root_job)
        if run_id is None:
            continue
        runs.append(
            HeroRun(
                cluster=str(row["cluster"]),
                root_job=root_job,
                run_id=run_id,
                running_since=_as_utc(row["running_since"]),
            )
        )
    return tuple(runs)


def telemetry_query(now: datetime, runs: tuple[HeroRun, ...]) -> str:
    """Return current execution metrics for exact active hero run IDs."""
    run_ids = sorted({run.run_id for run in runs})
    if not run_ids:
        raise ValueError("at least one active hero run is required")
    if len(run_ids) == 1:
        run_predicate = f"run_id = {sql_string(run_ids[0])}"
    else:
        run_predicate = f"run_id IN ({', '.join(sql_string(run_id) for run_id in run_ids)})"

    execution_since = _sql_timestamp(now - _EXECUTION_LOOKBACK)
    progress_since = _sql_timestamp(now - _PROGRESS_LOOKBACK)
    enrolled_since = _sql_timestamp(now - _ENROLLMENT_LOOKBACK)
    end = _sql_timestamp(now)
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
    """Project active hero roots and their latest execution metrics into alert rows."""
    runs = active_hero_runs(task_states)
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
            execution_started = _as_utc(row["execution_started_at"])
            current_execution = execution_started_by_job.get(key)
            if current_execution is None or execution_started > current_execution:
                execution_started_by_job[key] = execution_started
            metric = str(row["name"])
            observed = _as_utc(row["ts"])
            current = metrics_by_job.setdefault(key, {}).get(metric)
            if current is None or observed > current[0]:
                metrics_by_job[key][metric] = (observed, float(row["value"]))

    now = _as_utc(now)
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
