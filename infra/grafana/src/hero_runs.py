# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hero-run enrollment shared by the training alert projections.

A root job is a hero run while its latest `iris.task_state` row is fresh, reports
a running task, and its last path component is `<run-id>-coord` with `<run-id>`
beginning with `hero-`. That naming contract also gives the exact `run_id` its
Levanter telemetry carries, so an alert reads structured columns and never scans
for a pattern. See docs/ops/training-stall-alert-contract.md.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pyarrow as pa
from vllm_observability import sql_string

_TASK_STATE_FRESHNESS = timedelta(seconds=90)
TASK_STATE_LOOKBACK = timedelta(hours=1)

HERO_RUN_PREFIX = "hero-"
_COORDINATOR_SUFFIX = "-coord"
_HERO_ROOT_PATTERN = f"%/{HERO_RUN_PREFIX}%{_COORDINATOR_SUFFIX}"


@dataclass(frozen=True)
class HeroRun:
    cluster: str
    root_job: str
    run_id: str
    running_since: datetime


def sql_timestamp(at: datetime) -> str:
    """Format at as the tz-naive UTC literal finelog compares timestamps against."""
    return at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def as_utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"expected timestamp, got {value!r}")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def task_state_query(now: datetime) -> str:
    """Return fresh active root jobs whose run name opts into hero alerts."""
    start = sql_timestamp(now - TASK_STATE_LOOKBACK)
    fresh = sql_timestamp(now - _TASK_STATE_FRESHNESS)
    end = sql_timestamp(now)
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
    if not root_name.startswith(HERO_RUN_PREFIX) or not root_name.endswith(_COORDINATOR_SUFFIX):
        return None
    run_id = root_name.removesuffix(_COORDINATOR_SUFFIX)
    return run_id if run_id != HERO_RUN_PREFIX else None


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
                running_since=as_utc(row["running_since"]),
            )
        )
    return tuple(runs)


def run_id_predicate(runs: tuple[HeroRun, ...]) -> str:
    """Return an exact-match `run_id` predicate over the enrolled runs."""
    run_ids = sorted({run.run_id for run in runs})
    if not run_ids:
        raise ValueError("at least one active hero run is required")
    if len(run_ids) == 1:
        return f"run_id = {sql_string(run_ids[0])}"
    return f"run_id IN ({', '.join(sql_string(run_id) for run_id in run_ids)})"
