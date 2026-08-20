# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The RL producer's stall reader.

Stall detection in this repo is per-producer and opt-in. `training_stalls.py` selects its
candidates from `iris.task_state` by job name before it reads a metric, so an RL run is never
one no matter what it emits; this reader is MarinSkyRL's own opt-in. It reads only
`progress_time_seconds`, which MarinSkyRL already emits, and keys on the promoted `run_id`
column rather than on a JSON attribute.
"""

from datetime import UTC, datetime, timedelta

import duckdb
import pyarrow as pa
import pytest
from rl_stalls import rl_progress_query, rl_stall_alert_rows

NOW = datetime(2026, 8, 20, 12, tzinfo=UTC)


def _result(**columns: list) -> pa.Table:
    return pa.table(dict(columns))


def test_a_run_without_recent_progress_alerts_and_a_slow_one_does_not() -> None:
    progress = _result(
        cluster=["cw-rno2a", "cw-rno2a"],
        run=["snowball-e6-muonh-0", "iceball-micro-rl"],
        execution=["iris:/atqamar/e6/0:attempt:0", "iris:/atqamar/iceball/0:attempt:0"],
        # An RL step is long. 31 minutes is a stall; 29 is a slow rollout phase.
        progress_time=[NOW.timestamp() - 31 * 60, NOW.timestamp() - 29 * 60],
        producer_at=[NOW - timedelta(seconds=20), NOW - timedelta(seconds=20)],
    )

    assert rl_stall_alert_rows(progress, NOW) == [
        {
            "cluster": "cw-rno2a",
            "run": "snowball-e6-muonh-0",
            "execution": "iris:/atqamar/e6/0:attempt:0",
            "reason": "rollout_progress_stale",
            "value": 1,
        },
        {
            "cluster": "cw-rno2a",
            "run": "iceball-micro-rl",
            "execution": "iris:/atqamar/iceball/0:attempt:0",
            "reason": "healthy",
            "value": 0,
        },
    ]


def test_a_producer_that_stopped_reporting_stops_alerting() -> None:
    # The failure this guards is an alert nobody can clear: a finished run's last row would
    # otherwise stay stale forever and pin the alert on, which teaches people to ignore it.
    progress = _result(
        cluster=["cw-rno2a"],
        run=["finished-run"],
        execution=["iris:/atqamar/done/0:attempt:0"],
        progress_time=[NOW.timestamp() - 90 * 60],
        producer_at=[NOW - timedelta(minutes=5)],
    )

    assert rl_stall_alert_rows(progress, NOW) == [
        {"cluster": "fleet", "run": "", "execution": "", "reason": "healthy", "value": 0}
    ]


def test_no_reporting_runs_returns_an_explicit_zero() -> None:
    # An empty result must be an explicit healthy row, not an absent series: `noDataState` is
    # Alerting, so returning nothing would page rather than report a quiet fleet.
    assert rl_stall_alert_rows(pa.table({}), NOW) == [
        {"cluster": "fleet", "run": "", "execution": "", "reason": "healthy", "value": 0}
    ]


@pytest.fixture
def store() -> duckdb.DuckDBPyConnection:
    database = duckdb.connect()
    database.execute(
        """
        CREATE TABLE telemetry_v1(
            cluster VARCHAR, service VARCHAR, run_id VARCHAR, execution_uid VARCHAR,
            name VARCHAR, value DOUBLE, timestamp_ms BIGINT, seq BIGINT,
            resource_attributes_json VARCHAR, attributes_json VARCHAR
        )
        """
    )
    database.execute("CREATE MACRO to_timestamp_millis(value) AS to_timestamp(value / 1000.0)::TIMESTAMP")
    return database


def _insert(database, *, service, run_id, execution_uid, progress_time, at, seq):
    database.execute(
        "INSERT INTO telemetry_v1 VALUES ('cw-rno2a', ?, ?, ?, 'progress_time_seconds', ?, ?, ?, '{}', '{}')",
        [service, run_id, execution_uid, progress_time, int(at.timestamp() * 1000), seq],
    )


def test_the_query_takes_the_latest_row_per_run_and_ignores_other_services(store) -> None:
    execution = "iris:/atqamar/e6/0:attempt:0"
    for minutes, progress, seq in ((4, 1000.0, 1), (1, 2000.0, 2)):
        _insert(
            store,
            service="marinskyrl",
            run_id="snowball-e6-muonh-0",
            execution_uid=execution,
            progress_time=progress,
            at=NOW - timedelta(minutes=minutes),
            seq=seq,
        )
    # Levanter reports the same metric name. It has its own reader and must not appear here.
    _insert(
        store,
        service="levanter",
        run_id="some-pretrain",
        execution_uid="iris:/other/0:attempt:0",
        progress_time=3000.0,
        at=NOW - timedelta(minutes=1),
        seq=3,
    )

    rows = store.execute(rl_progress_query(NOW)).fetchall()

    assert [(row[1], row[3]) for row in rows] == [("snowball-e6-muonh-0", 2000.0)]


def test_the_query_drops_rows_without_a_run_identity(store) -> None:
    # A row that never got a run id cannot be attributed to a run, and alerting on it would
    # name something nobody can look up.
    _insert(
        store,
        service="marinskyrl",
        run_id=None,
        execution_uid="iris:/atqamar/e6/0:attempt:0",
        progress_time=1000.0,
        at=NOW - timedelta(minutes=1),
        seq=1,
    )

    assert store.execute(rl_progress_query(NOW)).fetchall() == []
