# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded finelog queries and alert projection for stalled Levanter jobs."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pyarrow as pa
from hero_runs import (
    FINISHED_PHASE,
    INITIALIZING_PHASE,
    PHASE_METRIC,
    TASK_STATE_LOOKBACK,
    TELEMETRY_GONE_AGE,
    TRAINING_PHASE,
    HeroRun,
    as_utc,
    run_id_predicate,
    sql_epoch_ms,
    sql_timestamp,
)

_TRAINING_STALL_AGE = timedelta(minutes=15)
_INITIALIZING_STALL_AGE = timedelta(minutes=45)
# Every rank publishes `step` and `phase`, so an unfiltered scan of one hour of
# the hero run reads about 1.4M rows a minute against the hub for the ~2k that
# carry the tracker's own view. Process zero is that view, and the same replica
# the training dashboard selects.
_PROGRESS_LOOKBACK = 2 * _TRAINING_STALL_AGE
_EXECUTION_LOOKBACK = TASK_STATE_LOOKBACK
# The phase heartbeat reaches back a day so a run that went silent hours ago is
# still recognisable as one that stopped publishing, which is TrainingTelemetryGone's
# case. Without the reach this rule calls a silent training run `initializing_stale`
# and pages beside it. Phase is one row a minute for one process.
_PHASE_LOOKBACK = timedelta(hours=24)

_STEP_METRIC = "step"
_PROGRESS_TIME_METRIC = "progress_time_seconds"


def telemetry_query(now: datetime, runs: tuple[HeroRun, ...]) -> str:
    """Return current execution metrics for exact active hero run IDs."""
    run_predicate = run_id_predicate(runs)
    phase_since = sql_epoch_ms(now - _PHASE_LOOKBACK)
    progress_since = sql_epoch_ms(now - _PROGRESS_LOOKBACK)
    enrolled_since = sql_timestamp(now - _PHASE_LOOKBACK)
    end = sql_epoch_ms(now)
    metric_names = f"'{PHASE_METRIC}', '{_STEP_METRIC}', '{_PROGRESS_TIME_METRIC}'"
    return (
        'WITH telemetry AS (SELECT * FROM "telemetry_v1" '
        'UNION ALL SELECT * FROM "telemetry_v1.levanter"), filtered AS ('
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS origin_cluster, "
        "run_id, job_id AS telemetry_job, execution_uid, name, value, "
        "timestamp_ms, seq, to_timestamp_millis(timestamp_ms) AS ts "
        "FROM telemetry "
        f"WHERE service = 'levanter' AND name IN ({metric_names}) "
        f"AND {run_predicate} AND process_index = '0' "
        "AND job_id IS NOT NULL AND execution_uid IS NOT NULL "
        f"AND timestamp_ms >= {phase_since} AND timestamp_ms < {end} "
        f"AND (name = '{PHASE_METRIC}' OR timestamp_ms >= {progress_since})"
        "), phase_history AS ("
        "SELECT origin_cluster, run_id, telemetry_job, execution_uid, ts, "
        "ROW_NUMBER() OVER ("
        "PARTITION BY origin_cluster, run_id, telemetry_job ORDER BY timestamp_ms DESC, seq DESC"
        ") AS rn FROM filtered "
        f"WHERE name = '{PHASE_METRIC}' AND ts >= TIMESTAMP '{enrolled_since}'"
        "), enrolled AS ("
        "SELECT origin_cluster, run_id, telemetry_job, execution_uid FROM phase_history WHERE rn = 1"
        "), execution AS ("
        "SELECT enrolled.origin_cluster, enrolled.run_id, enrolled.telemetry_job, enrolled.execution_uid, "
        "MIN(filtered.ts) AS execution_started_at FROM enrolled JOIN filtered "
        "ON filtered.origin_cluster = enrolled.origin_cluster AND filtered.run_id = enrolled.run_id "
        "AND filtered.telemetry_job = enrolled.telemetry_job "
        "AND filtered.execution_uid = enrolled.execution_uid "
        f"WHERE filtered.name = '{PHASE_METRIC}' "
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
        INITIALIZING_PHASE: "initializing",
        TRAINING_PHASE: "training",
        FINISHED_PHASE: "finished",
    }.get(phase, "unknown")


def _row(cluster: str, job: str, run: str, phase: str, reason: str, value: int) -> dict:
    return {"cluster": cluster, "job": job, "run": run, "phase": phase, "reason": reason, "value": value}


@dataclass(frozen=True)
class ExecutionMetrics:
    """One root job's newest value per metric, and when its execution started."""

    metrics: dict[str, float]
    execution_started: datetime | None
    observed_at: datetime


def _metrics_by_job(runs: tuple[HeroRun, ...], telemetry_metrics: pa.Table) -> dict[tuple[str, str], ExecutionMetrics]:
    """Fold telemetry rows onto the root job that owns them, keeping the newest of each metric."""
    newest: dict[tuple[str, str], dict[str, tuple[datetime, float]]] = {}
    started: dict[tuple[str, str], datetime] = {}
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
            if key not in started or execution_started > started[key]:
                started[key] = execution_started
            metric = str(row["name"])
            observed = as_utc(row["ts"])
            current = newest.setdefault(key, {}).get(metric)
            if current is None or observed > current[0]:
                newest[key][metric] = (observed, float(row["value"]))

    return {
        key: ExecutionMetrics(
            metrics={name: value for name, (_, value) in metrics.items()},
            execution_started=started.get(key),
            observed_at=max(observed for observed, _ in metrics.values()),
        )
        for key, metrics in newest.items()
    }


def _classify(run: HeroRun, observed: ExecutionMetrics | None, now: datetime) -> tuple[str, str, int]:
    """Return the phase name, alert reason, and firing value for one enrolled root."""
    metrics = observed.metrics if observed is not None else {}
    raw_phase = metrics.get(PHASE_METRIC)
    phase = int(raw_phase) if raw_phase is not None else None
    step = metrics.get(_STEP_METRIC, 0.0)
    progress_time = metrics.get(_PROGRESS_TIME_METRIC, 0.0)
    execution_started = observed.execution_started if observed is not None else None
    attempt_age = now - max(run.running_since, execution_started or run.running_since)

    reason = "healthy"
    value = 0
    is_training = phase == TRAINING_PHASE or step > 0
    if observed is not None and now - observed.observed_at > TELEMETRY_GONE_AGE:
        # A run that stopped publishing is TrainingTelemetryGone's, which names the
        # failure precisely. Reporting a stall as well would page twice for it.
        reason = "telemetry_gone"
    elif phase == FINISHED_PHASE:
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

    if phase is None:
        return ("training" if is_training else "initializing"), reason, value
    return _phase_name(phase), reason, value


def training_stall_alert_rows(runs: tuple[HeroRun, ...], telemetry_metrics: pa.Table, now: datetime) -> list[dict]:
    """Project enrolled hero roots and their latest execution metrics into alert rows."""
    observed = _metrics_by_job(runs, telemetry_metrics)
    now = as_utc(now)
    rows = [
        _row(run.cluster, run.root_job, run.run_id, *_classify(run, observed.get((run.cluster, run.root_job)), now))
        for run in runs
    ]
    if rows:
        return rows
    return [_row("fleet", "", "", "idle", "healthy", 0)]
