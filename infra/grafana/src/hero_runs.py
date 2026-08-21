# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hero-run enrollment shared by the training alert projections.

A root job is a hero run while its latest `iris.task_state` row is fresh and
reports a running task. Its last path component is `<run-id>-coord` or
`<run-id>-coord-<retry>`, and `<run-id>` begins with `hero-`. This naming
contract also gives the exact `run_id` in its Levanter telemetry. See
docs/ops/training-stall-alert-contract.md.

`phase_enrollment_query` is the second path: a run that still publishes Levanter
`phase` telemetry. The run-health projections watch the union, so an outage on
one side leaves the other watching. See docs/ops/hero-run-health-alerts.md.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

import pyarrow as pa
from vllm_observability import sql_string

TASK_STATE_FRESHNESS = timedelta(seconds=90)
TASK_STATE_LOOKBACK = timedelta(hours=1)
# Long enough that a run which goes silent stays watched while
# `TrainingTelemetryGone` counts out its threshold and pending period.
PHASE_ENROLLMENT_LOOKBACK = timedelta(minutes=60)
# Silence this long is the telemetry path or the process. `TrainingTelemetryGone`
# owns that case; the other projections defer rather than page for it again.
TELEMETRY_GONE_AGE = timedelta(minutes=10)

HERO_RUN_PREFIX = "hero-"
_COORDINATOR_MARKER = "-coord"
HERO_ROOT_PATTERNS = (
    f"%/{HERO_RUN_PREFIX}%{_COORDINATOR_MARKER}",
    f"%/{HERO_RUN_PREFIX}%{_COORDINATOR_MARKER}-%",
)
# Levanter's tracker phase, republished every minute.
PHASE_METRIC = "phase"
INITIALIZING_PHASE = 0
TRAINING_PHASE = 1
FINISHED_PHASE = 2


class RunIdentity(Protocol):
    """The cluster, root job, and run ID that name one enrolled hero run."""

    cluster: str
    root_job: str
    run_id: str


@dataclass(frozen=True)
class HeroRun:
    cluster: str
    root_job: str
    run_id: str
    running_since: datetime


def sql_timestamp(at: datetime) -> str:
    """Format at as the tz-naive UTC literal finelog compares timestamps against."""
    return at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def sql_epoch_ms(at: datetime) -> str:
    """Return the epoch-millisecond expression finelog compares `timestamp_ms` against."""
    return f"CAST(EXTRACT(EPOCH FROM TIMESTAMP '{sql_timestamp(at)}') * 1000 AS BIGINT)"


def as_utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"expected timestamp, got {value!r}")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def as_number(value: object) -> float | None:
    """Return a numeric cell as a float, or None when the reduction came back NULL."""
    return float(value) if isinstance(value, int | float) else None


def task_state_query(now: datetime) -> str:
    """Return the newest root-job state row for every run name that opts into hero alerts.

    `active_hero_runs` applies the freshness and running-task filters, so the
    run-health projections can also read the age of a row that went stale.
    """
    start = sql_timestamp(now - TASK_STATE_LOOKBACK)
    end = sql_timestamp(now)
    root_predicate = " OR ".join(f"root_job_id LIKE '{pattern}'" for pattern in HERO_ROOT_PATTERNS)
    return (
        "WITH samples AS ("
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS cluster, root_job_id, ts, running "
        'FROM "iris.task_state" '
        f"WHERE ({root_predicate}) AND ts >= TIMESTAMP '{start}' "
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
        "SELECT cluster, root_job_id AS job, ts AS state_at, running_since, running "
        "FROM history "
        "WHERE rn = 1"
    )


def phase_enrollment_query(now: datetime) -> str:
    """Return each hero run that still publishes Levanter phase telemetry."""
    start = sql_epoch_ms(now - PHASE_ENROLLMENT_LOOKBACK)
    end = sql_epoch_ms(now)
    return (
        "WITH samples AS ("
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS origin_cluster, "
        "run_id, job_id, timestamp_ms, seq "
        'FROM "telemetry_v1" '
        f"WHERE service = 'levanter' AND name = '{PHASE_METRIC}' "
        f"AND run_id LIKE '{HERO_RUN_PREFIX}%' AND job_id IS NOT NULL "
        f"AND timestamp_ms >= {start} AND timestamp_ms < {end}"
        "), ranked AS ("
        "SELECT origin_cluster, run_id, job_id, "
        "ROW_NUMBER() OVER ("
        "PARTITION BY origin_cluster, run_id ORDER BY timestamp_ms DESC, seq DESC"
        ") AS rn FROM samples"
        ") "
        "SELECT origin_cluster AS cluster, run_id, job_id AS telemetry_job "
        "FROM ranked WHERE rn = 1"
    )


def hero_run_id(root_job: str) -> str | None:
    """Return the run ID a hero coordinator root job names, or None."""
    root_name = root_job.rsplit("/", 1)[-1]
    run_id, marker, retry = root_name.rpartition(_COORDINATOR_MARKER)
    if not marker or not run_id.startswith(HERO_RUN_PREFIX):
        return None
    if retry and not retry.startswith("-"):
        return None
    return run_id if run_id != HERO_RUN_PREFIX else None


def root_job_for(telemetry_job: str) -> str | None:
    """Return the hero coordinator root that owns a Levanter telemetry job ID.

    Levanter reports the leaf task, so the root is the job's longest prefix that
    still names a hero run.
    """
    parts = telemetry_job.split("/")
    for depth in range(len(parts), 0, -1):
        candidate = "/".join(parts[:depth])
        if hero_run_id(candidate) is not None:
            return candidate
    return None


def active_hero_runs(task_states: pa.Table, now: datetime) -> tuple[HeroRun, ...]:
    """Return the run identities whose state row is fresh and reports a running task."""
    fresh = as_utc(now) - TASK_STATE_FRESHNESS
    runs = []
    for row in task_states.to_pylist():
        root_job = str(row["job"])
        run_id = hero_run_id(root_job)
        if run_id is None or int(row["running"] or 0) <= 0 or as_utc(row["state_at"]) < fresh:
            continue
        runs.append(
            HeroRun(
                cluster=str(row["cluster"]),
                root_job=root_job,
                run_id=run_id,
                running_since=as_utc(row["running_since"]),
            )
        )
    return tuple(runs)


def run_id_predicate(runs: Sequence[RunIdentity]) -> str:
    """Return an exact-match `run_id` predicate over the enrolled runs."""
    run_ids = sorted({run.run_id for run in runs})
    if not run_ids:
        raise ValueError("at least one active hero run is required")
    if len(run_ids) == 1:
        return f"run_id = {sql_string(run_ids[0])}"
    return f"run_id IN ({', '.join(sql_string(run_id) for run_id in run_ids)})"
