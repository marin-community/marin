# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded finelog query and alert rows for stalled reinforcement-learning runs.

Stall detection here is per-producer and opt-in: the hero rules cover Levanter runs whose
root job opts in by being named `hero-*-coord`, and `zephyr_stalls.py` covers Zephyr. Nothing
covered RL runs, and nothing would have even if MarinSkyRL adopted Levanter's whole vocabulary,
because `hero_runs.task_state_query` selects candidates from `iris.task_state` by job name
before a metric is read. This module is the RL producer's opt-in, modelled on the Zephyr one.

It reads only `progress_time_seconds`, which MarinSkyRL already emits from `record_policy_step`
and `record_generated_work`. An RL step is long — a rollout phase alone can run several
minutes — so the stall threshold is well above the Levanter reader's.
"""

from datetime import UTC, datetime, timedelta

import pyarrow as pa

# A producer that has stopped reporting must stop alerting rather than alert forever: a stale
# row would otherwise pin the alert on indefinitely after the job is gone, and an alert nobody
# can clear is one people learn to ignore.
_PRODUCER_FRESHNESS = timedelta(seconds=90)
# Generation dominates an RL step and a single rollout phase can legitimately run for minutes,
# so this sits above the Levanter reader's fifteen. Long enough not to fire on a slow step,
# short enough to catch a wedged engine within a step or two.
_STALL_AGE = timedelta(minutes=30)
_TELEMETRY_LOOKBACK = timedelta(minutes=5)

_PROGRESS_TIME_METRIC = "progress_time_seconds"
_SERVICE = "marinskyrl"


def _sql_timestamp(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def rl_progress_query(now: datetime) -> str:
    """Return the latest RL progress metric for each active run and task attempt.

    Keyed on the `run_id` column rather than on a JSON attribute: MarinSkyRL stamps the run
    identity as a promoted resource attribute, so it arrives as a typed, sorted column.
    """
    start = _sql_timestamp(now - _TELEMETRY_LOOKBACK)
    end = _sql_timestamp(now)
    return (
        "WITH filtered AS ("
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS origin_cluster, "
        "run_id AS run, execution_uid AS execution, value AS progress_time, "
        "timestamp_ms, seq, to_timestamp_millis(timestamp_ms) AS producer_at "
        'FROM "telemetry_v1" '
        f"WHERE service = '{_SERVICE}' AND name = '{_PROGRESS_TIME_METRIC}' "
        f"AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM TIMESTAMP '{start}') * 1000 AS BIGINT) "
        f"AND timestamp_ms < CAST(EXTRACT(EPOCH FROM TIMESTAMP '{end}') * 1000 AS BIGINT)"
        "), recent AS ("
        "SELECT origin_cluster, run, execution, progress_time, producer_at, "
        "ROW_NUMBER() OVER ("
        "PARTITION BY origin_cluster, run, execution ORDER BY timestamp_ms DESC, seq DESC"
        ") AS rn "
        "FROM filtered WHERE run IS NOT NULL AND run <> '' AND execution IS NOT NULL AND execution <> ''"
        ") "
        "SELECT origin_cluster AS cluster, run, execution, progress_time, producer_at "
        "FROM recent WHERE rn = 1 ORDER BY cluster, run, execution"
    )


def _as_utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"expected timestamp, got {value!r}")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _row(cluster: str, run: str, execution: str, reason: str, value: int) -> dict:
    return {
        "cluster": cluster,
        "run": run,
        "execution": execution,
        "reason": reason,
        "value": value,
    }


def rl_stall_alert_rows(progress_metrics: pa.Table, now: datetime) -> list[dict]:
    """Return warning rows for reporting RL runs whose last progress is too old."""
    now = _as_utc(now)
    rows: list[dict] = []
    for metric in progress_metrics.to_pylist():
        producer_at = _as_utc(metric["producer_at"])
        if now - producer_at > _PRODUCER_FRESHNESS:
            continue

        progress_at = datetime.fromtimestamp(float(metric["progress_time"]), tz=UTC)
        stale = now - progress_at >= _STALL_AGE
        rows.append(
            _row(
                str(metric["cluster"]),
                str(metric["run"]),
                str(metric["execution"]),
                "rollout_progress_stale" if stale else "healthy",
                int(stale),
            )
        )

    if rows:
        return rows
    return [_row("fleet", "", "", "healthy", 0)]
