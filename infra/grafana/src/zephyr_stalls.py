# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded finelog query and alert rows for stale Zephyr pipeline progress."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa

_PRODUCER_FRESHNESS = timedelta(seconds=90)
_STALL_AGE = timedelta(minutes=45)
_TELEMETRY_LOOKBACK = timedelta(minutes=5)

_PROGRESS_TIME_METRIC = "progress_time_seconds"


def _sql_timestamp(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def zephyr_progress_query(now: datetime) -> str:
    """Return the latest Zephyr progress metric for each active execution."""
    start = _sql_timestamp(now - _TELEMETRY_LOOKBACK)
    end = _sql_timestamp(now)
    return (
        "WITH filtered AS ("
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS origin_cluster, "
        "json_get(resource_attributes_json, 'job_id') AS job, "
        "json_get(attributes_json, 'run') AS execution, value AS progress_time, "
        "timestamp_ms, seq, to_timestamp_millis(timestamp_ms) AS producer_at "
        'FROM "telemetry_v1" '
        f"WHERE service = 'zephyr' AND name = '{_PROGRESS_TIME_METRIC}' "
        f"AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM TIMESTAMP '{start}') * 1000 AS BIGINT) "
        f"AND timestamp_ms < CAST(EXTRACT(EPOCH FROM TIMESTAMP '{end}') * 1000 AS BIGINT)"
        "), recent AS ("
        "SELECT origin_cluster, job, execution, progress_time, producer_at, "
        "ROW_NUMBER() OVER ("
        "PARTITION BY origin_cluster, job, execution ORDER BY timestamp_ms DESC, seq DESC"
        ") AS rn "
        "FROM filtered WHERE job IS NOT NULL AND job <> '' AND execution IS NOT NULL AND execution <> ''"
        ") "
        "SELECT origin_cluster AS cluster, job, execution, progress_time, producer_at "
        "FROM recent WHERE rn = 1 ORDER BY cluster, job, execution"
    )


def _as_utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"expected timestamp, got {value!r}")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _row(cluster: str, job: str, execution: str, reason: str, value: int) -> dict:
    return {
        "cluster": cluster,
        "job": job,
        "execution": execution,
        "reason": reason,
        "value": value,
    }


def zephyr_stall_alert_rows(progress_metrics: pa.Table, now: datetime) -> list[dict]:
    """Return warning rows for active executions without a recent shard completion."""
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
                str(metric["job"]),
                str(metric["execution"]),
                "shard_progress_stale" if stale else "healthy",
                int(stale),
            )
        )

    if rows:
        return rows
    return [_row("fleet", "", "", "healthy", 0)]
