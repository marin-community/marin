# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the grafana bridge: its HTTP surface over a fake finelog, and the
cache's coalescing and eviction contract."""

import threading
from datetime import UTC, datetime, timedelta

import duckdb
import pyarrow as pa
import pytest
from cache import TtlCache
from config import ClusterTarget
from conftest import FINELOG_DEPLOYMENTS_PATH, bridge_config, deployment, healthy_k8s_routes, k8s_api, make_k8s_source
from finelog.errors import QueryResultTooLargeError
from finelog_health import FinelogHealth, FinelogRole
from github_source import GithubSource
from hero_runs import HeroRun, active_hero_runs, task_state_query
from k8s_source import K8sFleet
from loom_alerts import (
    LoomAlertClient,
    LoomAlertDeliveryError,
    SlackAlertClient,
    SlackAnnouncementError,
    SlackThread,
)
from loss_spikes import loss_spike_alert_rows, loss_window_query
from server import create_app, workload_overview
from starlette.testclient import TestClient
from training_stalls import telemetry_query, training_stall_alert_rows
from wandb_source import WandbSource
from zephyr_stalls import zephyr_progress_query, zephyr_stall_alert_rows

# 2026-07-17T03:00:00Z and +1h, as Grafana sends them.
FROM_MS = 1_784_257_200_000
TO_MS = FROM_MS + 3_600_000
MARIN = ClusterTarget(
    name="marin", project="p", zone="z", instance_filter="name = finelog-marin", controller_filter="labels.x=true"
)


def finelog_result(**columns: list) -> pa.Table:
    """An Arrow table shaped like a finelog result; types are inferred from the values."""
    return pa.table(dict(columns))


_ONE_ROW = finelog_result(t=[datetime(2026, 7, 17, 3, 0, tzinfo=UTC)], value=[1.0])


class FakeSource:
    """A MetricSource that records the SQL it is handed and replays a canned table."""

    def __init__(
        self,
        table: pa.Table | None = None,
        raises: Exception | None = None,
        health: FinelogHealth | None = None,
    ) -> None:
        self._table = table if table is not None else pa.table({})
        self._raises = raises
        self._health = health or FinelogHealth(
            cluster="marin",
            server="finelog-marin",
            role=FinelogRole.HUB,
            responsive=True,
            ready=1,
            desired=1,
            latency_ms=12,
            error_class="",
            error="",
        )
        self.queries: list[str] = []

    @property
    def target(self) -> ClusterTarget:
        return MARIN

    def query(self, sql: str, *, max_rows: int) -> pa.Table:
        self.queries.append(sql)
        if self._raises is not None:
            raise self._raises
        return self._table

    def health(self) -> FinelogHealth:
        return self._health


def _client(
    source: FakeSource,
    cache_ttl: float = 20.0,
    k8s_fleet: K8sFleet | None = None,
    loom_alerts: LoomAlertClient | None = None,
    slack_alerts: SlackAlertClient | None = None,
) -> TestClient:
    github = GithubSource(auth=None, timeout=5.0)
    return TestClient(
        create_app(
            bridge_config(cache_ttl),
            {"marin": source},
            {},
            github,
            k8s_fleet or K8sFleet(()),
            WandbSource(timeout=5.0),
            loom_alerts,
            slack_alerts,
        )
    )


def _get(client: TestClient, sql: str, **params):
    return client.get("/finelog/marin/query", params={"sql": sql, "from": FROM_MS, "to": TO_MS, **params})


def test_query_returns_json_rows_with_millis_timestamps():
    resp = _get(_client(FakeSource(_ONE_ROW)), 'SELECT t, value FROM "iris.task" WHERE ts >= {{from}} AND ts < {{to}}')
    assert resp.status_code == 200
    assert resp.json() == [{"t": 1_784_257_200_000, "value": 1.0}]


def test_query_substitutes_window_macros_before_running():
    source = FakeSource(_ONE_ROW)
    _get(_client(source), "SELECT value FROM t WHERE ts >= {{from}} AND ts < {{to}}")
    assert source.queries == [
        "SELECT value FROM t WHERE ts >= TIMESTAMP '2026-07-17 03:00:00' AND ts < TIMESTAMP '2026-07-17 04:00:00'"
    ]


def test_missing_sql_is_a_400():
    resp = _client(FakeSource()).get("/finelog/marin/query", params={"from": FROM_MS, "to": TO_MS})
    assert resp.status_code == 400
    assert "sql" in resp.json()["error"]


def test_macro_without_its_bound_is_a_400():
    resp = _client(FakeSource()).get("/finelog/marin/query", params={"sql": "SELECT 1 WHERE ts >= {{from}}"})
    assert resp.status_code == 400
    assert "no matching time bound" in resp.json()["error"]


def test_unknown_cluster_is_a_400_naming_the_valid_ones():
    resp = _client(FakeSource()).get("/finelog/nope/query", params={"sql": "SELECT 1"})
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert "nope" in error and "marin" in error


def test_oversized_result_is_a_400_with_guidance():
    resp = _get(_client(FakeSource(raises=QueryResultTooLargeError("query returned 500000 rows"))), "SELECT 1")
    assert resp.status_code == 400
    assert "narrow the time range" in resp.json()["error"]


def test_repeated_identical_panels_hit_finelog_once():
    source = FakeSource(_ONE_ROW)
    client = _client(source)
    sql = "SELECT value FROM t WHERE ts >= {{from}} AND ts < {{to}}"
    assert _get(client, sql).json() == _get(client, sql).json()
    assert len(source.queries) == 1


def test_relative_window_drifting_within_the_ttl_stays_one_query():
    source = FakeSource(_ONE_ROW)
    client = _client(source, cache_ttl=60.0)
    sql = "SELECT value FROM t WHERE ts >= {{from}} AND ts < {{to}}"
    for drift in (0, 1_000, 2_500):
        _get(client, sql, **{"from": FROM_MS + drift, "to": TO_MS + drift})
    assert len(source.queries) == 1


def test_windows_further_apart_than_the_ttl_are_cached_separately():
    source = FakeSource(_ONE_ROW)
    client = _client(source, cache_ttl=20.0)
    sql = "SELECT value FROM t WHERE ts >= {{from}} AND ts < {{to}}"
    hour = 3_600_000
    _get(client, sql)
    _get(client, sql, **{"from": FROM_MS + hour, "to": TO_MS + hour})
    assert len(source.queries) == 2


def test_json_labels_flatten_into_columns():
    source = FakeSource(finelog_result(value=[3.0], labels=['{"region": "us-east5", "scope": "pool"}']))
    assert _get(_client(source), "SELECT value, labels FROM t").json() == [
        {"value": 3.0, "label_region": "us-east5", "label_scope": "pool"}
    ]


def test_native_map_labels_flatten_into_columns():
    # A native Map<Utf8,Utf8> column arrives as list[(k, v)].
    table = pa.table(
        {"value": [3.0], "labels": [[("region", "us-east5"), ("scope", "pool")]]},
        schema=pa.schema([("value", pa.float64()), ("labels", pa.map_(pa.string(), pa.string()))]),
    )
    source = FakeSource(table)
    assert _get(_client(source), "SELECT value, labels FROM t").json() == [
        {"value": 3.0, "label_region": "us-east5", "label_scope": "pool"}
    ]


def test_unparseable_labels_cell_keeps_the_row():
    # One malformed cell is schema drift; the panel still gets its row.
    source = FakeSource(finelog_result(value=[1.0], labels=["{not json"]))
    assert _get(_client(source), "SELECT value, labels FROM t").json() == [{"value": 1.0, "labels": "{not json"}]


def test_health_lists_configured_clusters():
    assert _client(FakeSource()).get("/health").json() == {"status": "ok", "clusters": ["marin"]}


def test_training_stall_alerts_distinguish_stale_missing_and_healthy_progress():
    now = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
    task_states = finelog_result(
        cluster=["cw-a", "cw-a", "cw-b", "cw-b", "cw-b"],
        job=[
            "/u/hero-stale-coord",
            "/u/hero-initializing-coord",
            "/u/hero-healthy-coord",
            "/another-user/hero-starting-coord",
            "/u/ordinary-coord",
        ],
        state_at=[now] * 5,
        running_since=[
            now - timedelta(hours=1),
            now - timedelta(hours=1),
            now - timedelta(hours=1),
            now - timedelta(minutes=30),
            now - timedelta(hours=1),
        ],
    )
    telemetry_metrics = finelog_result(
        cluster=["cw-a", "cw-a", "cw-b", "cw-b", "cw-b"],
        run_id=["hero-stale", "hero-stale", "hero-healthy", "hero-healthy", "hero-starting"],
        telemetry_job=[
            "/u/hero-stale-coord/train",
            "/u/hero-stale-coord/train",
            "/u/hero-healthy-coord/train",
            "/u/hero-healthy-coord/train",
            "/another-user/hero-starting-coord/train",
        ],
        name=["phase", "progress_time_seconds", "phase", "progress_time_seconds", "phase"],
        value=[
            1.0,
            datetime(2026, 7, 28, 11, 30, tzinfo=UTC).timestamp(),
            1.0,
            datetime(2026, 7, 28, 11, 59, tzinfo=UTC).timestamp(),
            0.0,
        ],
        ts=[now] * 5,
        execution_started_at=[now - timedelta(hours=1)] * 5,
    )

    assert training_stall_alert_rows(active_hero_runs(task_states), telemetry_metrics, now) == [
        {
            "cluster": "cw-a",
            "job": "/u/hero-stale-coord",
            "run": "hero-stale",
            "phase": "training",
            "reason": "training_stalled",
            "value": 1,
        },
        {
            "cluster": "cw-a",
            "job": "/u/hero-initializing-coord",
            "run": "hero-initializing",
            "phase": "initializing",
            "reason": "initializing_stale",
            "value": 1,
        },
        {
            "cluster": "cw-b",
            "job": "/u/hero-healthy-coord",
            "run": "hero-healthy",
            "phase": "training",
            "reason": "healthy",
            "value": 0,
        },
        {
            "cluster": "cw-b",
            "job": "/another-user/hero-starting-coord",
            "run": "hero-starting",
            "phase": "initializing",
            "reason": "initializing",
            "value": 0,
        },
    ]


def test_training_stall_alert_returns_explicit_zero_without_running_jobs():
    assert training_stall_alert_rows((), pa.table({}), datetime(2026, 7, 28, tzinfo=UTC)) == [
        {"cluster": "fleet", "job": "", "run": "", "phase": "idle", "reason": "healthy", "value": 0}
    ]


def test_training_stall_task_state_resets_running_age_after_retry():
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    database = duckdb.connect()
    database.execute(
        'CREATE TABLE "iris.task_state"(cluster VARCHAR, root_job_id VARCHAR, ts TIMESTAMP, running BIGINT)'
    )
    database.executemany(
        'INSERT INTO "iris.task_state" VALUES (?, ?, ?, ?)',
        [
            ("cw-a", "/u/hero-retry-coord", now - timedelta(minutes=50), 1),
            ("cw-a", "/u/hero-retry-coord", now - timedelta(minutes=10), 0),
            ("cw-a", "/u/hero-retry-coord", now - timedelta(minutes=5), 1),
            ("cw-a", "/u/hero-retry-coord", now - timedelta(seconds=30), 64),
        ],
    )

    (row,) = database.execute(task_state_query(now)).fetchall()
    assert row[3] == now.replace(tzinfo=None) - timedelta(minutes=5)


def test_hero_alerts_enroll_suffixed_retry_roots():
    now = datetime(2026, 8, 20, 23, 30, tzinfo=UTC)
    database = duckdb.connect()
    database.execute(
        'CREATE TABLE "iris.task_state"(cluster VARCHAR, root_job_id VARCHAR, ts TIMESTAMP, running BIGINT)'
    )
    database.executemany(
        'INSERT INTO "iris.task_state" VALUES (?, ?, ?, ?)',
        [
            ("cw-a", "/power/hero-12d8b6f0-dee637-coord-slop85", now - timedelta(seconds=30), 64),
            ("cw-a", "/power/hero-other-coord", now - timedelta(seconds=30), 64),
            ("cw-a", "/power/not-hero-coord-slop85", now - timedelta(seconds=30), 64),
        ],
    )

    task_states = database.execute(task_state_query(now)).fetch_arrow_table()
    assert [(run.root_job, run.run_id) for run in active_hero_runs(task_states)] == [
        ("/power/hero-12d8b6f0-dee637-coord-slop85", "hero-12d8b6f0-dee637"),
        ("/power/hero-other-coord", "hero-other"),
    ]


def test_training_stall_alert_selects_named_hero_run_and_resolves_on_progress():
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    database = duckdb.connect()
    database.execute(
        """
        CREATE TABLE "iris.task_state"(
            cluster VARCHAR,
            root_job_id VARCHAR,
            ts TIMESTAMP,
            running BIGINT
        )
        """
    )
    database.execute(
        """
        CREATE TABLE telemetry_v1(
            cluster VARCHAR,
            service VARCHAR,
            run_id VARCHAR,
            job_id VARCHAR,
            execution_uid VARCHAR,
            name VARCHAR,
            value DOUBLE,
            timestamp_ms BIGINT,
            seq BIGINT
        )
        """
    )
    database.execute("CREATE MACRO to_timestamp_millis(value) AS to_timestamp(value / 1000.0)")

    stalled_at = now - timedelta(minutes=20)
    phase_at = stalled_at
    progressing_at = now - timedelta(seconds=30)
    database.executemany(
        'INSERT INTO "iris.task_state" VALUES (?, ?, ?, ?)',
        [
            ("cw-a", "/rav/hero-20260819-coord", now - timedelta(hours=1), 1),
            ("cw-a", "/rav/hero-20260819-coord", now - timedelta(seconds=30), 177),
            ("cw-a", "/rav/dev-run-coord", now - timedelta(hours=1), 1),
            ("cw-a", "/rav/dev-run-coord", now - timedelta(seconds=30), 1),
        ],
    )
    database.executemany(
        "INSERT INTO telemetry_v1 VALUES (?, 'levanter', ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "cw-a",
                "hero-20260819",
                "/rav/hero-20260819-coord/grug-train-hero-20260819",
                "attempt-1",
                "phase",
                1.0,
                int(phase_at.timestamp() * 1000),
                1,
            ),
            (
                "cw-a",
                "hero-20260819",
                "/rav/hero-20260819-coord/grug-train-hero-20260819",
                "attempt-1",
                "progress_time_seconds",
                stalled_at.timestamp(),
                int(stalled_at.timestamp() * 1000),
                2,
            ),
            (
                "cw-a",
                "dev-run",
                "/rav/dev-run-coord/grug-train-dev-run",
                "attempt-1",
                "phase",
                1.0,
                int(phase_at.timestamp() * 1000),
                3,
            ),
        ],
    )

    task_states = database.execute(task_state_query(now)).fetch_arrow_table()
    runs = active_hero_runs(task_states)
    assert [(run.cluster, run.root_job, run.run_id) for run in runs] == [
        ("cw-a", "/rav/hero-20260819-coord", "hero-20260819")
    ]

    enrolled = database.execute(telemetry_query(now, runs)).fetch_arrow_table()
    firing = training_stall_alert_rows(runs, enrolled, now)
    assert firing == [
        {
            "cluster": "cw-a",
            "job": "/rav/hero-20260819-coord",
            "run": "hero-20260819",
            "phase": "training",
            "reason": "training_stalled",
            "value": 1,
        }
    ]

    database.execute(
        "INSERT INTO telemetry_v1 VALUES (?, 'levanter', ?, ?, ?, ?, ?, ?, ?)",
        (
            "cw-a",
            "hero-20260819",
            "/rav/hero-20260819-coord/grug-train-hero-20260819",
            "attempt-1",
            "progress_time_seconds",
            progressing_at.timestamp(),
            int(progressing_at.timestamp() * 1000),
            4,
        ),
    )
    recovered_metrics = database.execute(telemetry_query(now, runs)).fetch_arrow_table()
    recovered = training_stall_alert_rows(runs, recovered_metrics, now)
    assert recovered == [
        {
            "cluster": "cw-a",
            "job": "/rav/hero-20260819-coord",
            "run": "hero-20260819",
            "phase": "training",
            "reason": "healthy",
            "value": 0,
        }
    ]


def test_training_stall_alert_gives_a_new_execution_its_own_initialization_window():
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    task_states = finelog_result(
        cluster=["cw-a"],
        job=["/u/hero-retry-coord"],
        running_since=[now - timedelta(hours=1)],
    )
    telemetry_metrics = finelog_result(
        cluster=["cw-a"],
        run_id=["hero-retry"],
        telemetry_job=["/u/hero-retry-coord/train"],
        name=["phase"],
        value=[0.0],
        ts=[now - timedelta(minutes=5)],
        execution_started_at=[now - timedelta(minutes=5)],
    )

    assert training_stall_alert_rows(active_hero_runs(task_states), telemetry_metrics, now) == [
        {
            "cluster": "cw-a",
            "job": "/u/hero-retry-coord",
            "run": "hero-retry",
            "phase": "initializing",
            "reason": "initializing",
            "value": 0,
        }
    ]


def _hero_run(run_id: str) -> HeroRun:
    return HeroRun("cw-a", f"/u/{run_id}-coord", run_id, datetime(2026, 7, 28, 11, tzinfo=UTC))


def _loss_windows(now: datetime, runs: tuple[HeroRun, ...], samples: list[tuple[str, datetime, float]]) -> pa.Table:
    """Run the alert's own SQL over (run_id, observed_at, loss) rows."""
    database = duckdb.connect()
    database.execute(
        """
        CREATE TABLE telemetry_v1(
            cluster VARCHAR,
            service VARCHAR,
            run_id VARCHAR,
            name VARCHAR,
            value DOUBLE,
            timestamp_ms BIGINT
        )
        """
    )
    database.executemany(
        "INSERT INTO telemetry_v1 VALUES ('cw-a', 'levanter', ?, 'train_loss', ?, ?)",
        [(run_id, loss, int(at.timestamp() * 1000)) for run_id, at, loss in samples],
    )
    return database.execute(loss_window_query(now, runs)).fetch_arrow_table()


def _steady_baseline(run_id: str, now: datetime) -> list[tuple[str, datetime, float]]:
    """Forty baseline samples alternating one hundredth around a loss of 3.5."""
    return [(run_id, now - timedelta(minutes=50) + timedelta(seconds=30 * i), 3.5 + 0.01 * (-1) ** i) for i in range(40)]


def _recent(run_id: str, now: datetime, losses: list[float]) -> list[tuple[str, datetime, float]]:
    return [(run_id, now - timedelta(minutes=4) + timedelta(seconds=15 * i), loss) for i, loss in enumerate(losses)]


def test_loss_spike_alert_fires_on_a_sustained_rise_and_not_on_a_single_step():
    """A skipped step and a divergence look identical in the mean of the recent
    window; the floor separates them. Here the blip run's recent mean is 3.96 and
    its floor is 3.5, against a band of 3.56."""
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    runs = tuple(_hero_run(run_id) for run_id in ("hero-steady", "hero-blip", "hero-diverging"))
    samples = [
        *_steady_baseline("hero-steady", now),
        *_recent("hero-steady", now, [3.49] * 12),
        *_steady_baseline("hero-blip", now),
        *_recent("hero-blip", now, [3.5] * 6 + [9.0] + [3.5] * 5),
        *_steady_baseline("hero-diverging", now),
        *_recent("hero-diverging", now, [4.2] * 12),
    ]

    assert loss_spike_alert_rows(runs, _loss_windows(now, runs, samples)) == [
        {"cluster": "cw-a", "job": "/u/hero-steady-coord", "run": "hero-steady", "reason": "healthy", "value": 0},
        {"cluster": "cw-a", "job": "/u/hero-blip-coord", "run": "hero-blip", "reason": "healthy", "value": 0},
        {
            "cluster": "cw-a",
            "job": "/u/hero-diverging-coord",
            "run": "hero-diverging",
            "reason": "spiking",
            "value": 1,
        },
    ]


def test_loss_spike_alert_fires_on_a_loss_that_stops_being_finite():
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    runs = (_hero_run("hero-nan"),)
    samples = [
        *_steady_baseline("hero-nan", now),
        *_recent("hero-nan", now, [3.5] * 6 + [float("nan")] * 6),
    ]

    assert loss_spike_alert_rows(runs, _loss_windows(now, runs, samples)) == [
        {"cluster": "cw-a", "job": "/u/hero-nan-coord", "run": "hero-nan", "reason": "not_finite", "value": 1}
    ]


def test_loss_spike_alert_waits_for_a_baseline_before_judging_a_rise():
    # A run that just started has no history to be an outlier against, and the
    # first steps after initialization are its loudest.
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    runs = (_hero_run("hero-fresh"),)
    samples = [
        ("hero-fresh", now - timedelta(minutes=8), 11.0),
        ("hero-fresh", now - timedelta(minutes=7), 9.0),
        *_recent("hero-fresh", now, [8.0] * 12),
    ]

    assert loss_spike_alert_rows(runs, _loss_windows(now, runs, samples)) == [
        {"cluster": "cw-a", "job": "/u/hero-fresh-coord", "run": "hero-fresh", "reason": "warming_up", "value": 0}
    ]


def test_loss_spike_alert_returns_explicit_zero_without_active_hero_runs():
    assert loss_spike_alert_rows((), pa.table({})) == [
        {"cluster": "fleet", "job": "", "run": "", "reason": "healthy", "value": 0}
    ]


def test_loss_spike_query_reads_one_bounded_window_per_evaluation():
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    sql = loss_window_query(now, (_hero_run("hero-prod"),))

    assert sql.count('FROM "telemetry_v1"') == 1
    assert "name = 'train_loss'" in sql
    assert "run_id = 'hero-prod'" in sql
    assert "timestamp_ms >= CAST(EXTRACT(EPOCH FROM TIMESTAMP '2026-07-28 11:00:00') * 1000 AS BIGINT)" in sql
    assert "timestamp_ms < CAST(EXTRACT(EPOCH FROM TIMESTAMP '2026-07-28 12:00:00') * 1000 AS BIGINT)" in sql
    assert "CAST(EXTRACT(EPOCH FROM TIMESTAMP '2026-07-28 11:55:00') * 1000 AS BIGINT)" in sql


def test_zephyr_stall_alert_distinguishes_stale_healthy_and_expired_producers():
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    progress = finelog_result(
        cluster=["cw-a", "cw-a", "cw-a"],
        job=["/user/stale", "/user/healthy", "/user/expired"],
        execution=["run-stale", "run-healthy", "run-expired"],
        progress_time=[
            now.timestamp() - 46 * 60,
            now.timestamp() - 44 * 60,
            now.timestamp() - 60 * 60,
        ],
        producer_at=[
            now - timedelta(seconds=20),
            now - timedelta(seconds=20),
            now - timedelta(minutes=2),
        ],
    )

    assert zephyr_stall_alert_rows(progress, now) == [
        {
            "cluster": "cw-a",
            "job": "/user/stale",
            "execution": "run-stale",
            "reason": "shard_progress_stale",
            "value": 1,
        },
        {
            "cluster": "cw-a",
            "job": "/user/healthy",
            "execution": "run-healthy",
            "reason": "healthy",
            "value": 0,
        },
    ]


def test_zephyr_stall_alert_returns_explicit_zero_without_active_pipelines():
    assert zephyr_stall_alert_rows(pa.table({}), datetime(2026, 7, 28, tzinfo=UTC)) == [
        {"cluster": "fleet", "job": "", "execution": "", "reason": "healthy", "value": 0}
    ]


def test_alert_queries_use_int64_epoch_boundaries_and_project_timestamps():
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    run = HeroRun("cw-a", "/u/hero-prod-coord", "hero-prod", now - timedelta(hours=1))
    for sql in (telemetry_query(now, (run,)), zephyr_progress_query(now)):
        assert 'FROM "telemetry_v1"' in sql
        assert "timestamp_ms >= CAST(EXTRACT(EPOCH FROM TIMESTAMP '" in sql
        assert "timestamp_ms < CAST(EXTRACT(EPOCH FROM TIMESTAMP '" in sql
        assert "* 1000 AS BIGINT)" in sql
        assert "to_timestamp_millis(timestamp_ms)" in sql
        assert "AS origin_cluster" in sql
        assert "json_get_string" not in sql
        assert "timestamp_ms >= TIMESTAMP" not in sql


def test_zephyr_alert_query_keeps_job_identity_across_the_schema_transition():
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    database = duckdb.connect()
    database.execute("CREATE MACRO to_timestamp_millis(value) AS make_timestamp_ms(value)")
    database.execute("CREATE MACRO json_get(value, key) AS json_extract_string(value, key)")
    database.execute(
        """
        CREATE TABLE telemetry_v1(
            cluster VARCHAR,
            service VARCHAR,
            job_id VARCHAR,
            name VARCHAR,
            value DOUBLE,
            timestamp_ms BIGINT,
            seq BIGINT,
            resource_attributes_json VARCHAR,
            attributes_json VARCHAR
        )
        """
    )
    database.executemany(
        "INSERT INTO telemetry_v1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "marin",
                "zephyr",
                None,
                "progress_time_seconds",
                1_785_239_940.0,
                1_785_239_940_002,
                1,
                '{"job_id":"old-zephyr-job"}',
                '{"run":"old-execution"}',
            ),
            (
                "marin",
                "zephyr",
                "new-zephyr-job",
                "progress_time_seconds",
                1_785_239_940.0,
                1_785_239_940_003,
                2,
                "{}",
                '{"run":"new-execution"}',
            ),
        ],
    )

    zephyr_jobs = {row[0] for row in database.execute(f"SELECT job FROM ({zephyr_progress_query(now)})").fetchall()}
    assert zephyr_jobs == {"old-zephyr-job", "new-zephyr-job"}


def test_training_stall_query_bounds_each_metric_family_to_its_detection_window():
    """Wide scans read telemetry_v1 once a minute and can saturate Finelog.

    Levanter republishes `phase` every 60s. Progress needs one extra stall window
    so a metric that just became stale remains observable.
    """
    now = datetime(2026, 7, 28, 12, tzinfo=UTC)
    run = HeroRun("cw-a", "/u/hero-prod-coord", "hero-prod", now - timedelta(hours=1))
    sql = telemetry_query(now, (run,))

    assert sql.count('FROM "telemetry_v1"') == 1
    assert "name IN ('phase', 'step', 'progress_time_seconds')" in sql
    assert "run_id = 'hero-prod'" in sql
    assert "timestamp_ms >= CAST(EXTRACT(EPOCH FROM TIMESTAMP '2026-07-28 11:00:00') * 1000 AS BIGINT)" in sql
    assert (
        "name = 'phase' OR timestamp_ms >= "
        "CAST(EXTRACT(EPOCH FROM TIMESTAMP '2026-07-28 11:30:00') * 1000 AS BIGINT)" in sql
    )
    assert "WHERE name = 'phase' AND ts >= TIMESTAMP '2026-07-28 11:00:00'" in sql
    assert "root_job_id LIKE '%/hero-%-coord'" in task_state_query(now)


class FakeLoomAlerts(LoomAlertClient):
    def __init__(self, result: dict | None = None, error: LoomAlertDeliveryError | None = None) -> None:
        self.result = result
        self.error = error

    async def submit(self, payload: object) -> dict | None:
        if self.error is not None:
            raise self.error
        return self.result


def test_loom_alert_route_returns_an_accepted_run():
    resp = _client(FakeSource(), loom_alerts=FakeLoomAlerts({"id": "run-1"})).post(
        "/alerts/loom", json={"alerts": [{"status": "firing"}]}
    )
    assert resp.status_code == 202
    assert resp.json() == {"accepted": True, "run": {"id": "run-1"}}


def test_loom_alert_route_returns_retryable_failure_for_delivery_errors():
    resp = _client(
        FakeSource(),
        loom_alerts=FakeLoomAlerts(error=LoomAlertDeliveryError("loom.example returned HTTP 503")),
    ).post("/alerts/loom", json={"alerts": [{"status": "firing"}]})
    assert resp.status_code == 502
    assert resp.json() == {"error": "loom.example returned HTTP 503"}


class FakeSlackAlerts(SlackAlertClient):
    def __init__(self, thread: SlackThread | None) -> None:
        self.thread = thread
        self.payloads: list[object] = []

    async def announce(self, payload: object) -> SlackThread | None:
        self.payloads.append(payload)
        return self.thread


def test_slack_alert_route_announces_without_a_run():
    fallback = FakeSlackAlerts(SlackThread(channel="C0123ABCD", thread_ts="1700000000.000001"))
    resp = _client(FakeSource(), slack_alerts=fallback).post("/alerts/slack", json={"alerts": []})

    assert resp.status_code == 202
    assert resp.json() == {"announced": True}
    assert fallback.payloads == [{"alerts": []}]


def test_slack_alert_route_is_disabled_without_a_configured_destination():
    resp = _client(FakeSource()).post("/alerts/slack", json={"alerts": []})
    assert resp.status_code == 503


def test_slack_alert_route_reports_a_resolution_as_nothing_announced():
    resp = _client(FakeSource(), slack_alerts=FakeSlackAlerts(None)).post("/alerts/slack", json={"alerts": []})
    assert resp.status_code == 202
    assert resp.json() == {"announced": False, "reason": "no firing alerts"}


def test_slack_alert_route_asks_grafana_to_retry_a_refused_announcement():
    """This receiver has no second leg, so a dropped announcement is the whole
    notification: it must not answer 2xx."""

    class RefusingSlackAlerts(FakeSlackAlerts):
        async def announce(self, payload: object) -> SlackThread | None:
            raise SlackAnnouncementError("Slack did not accept the alert announcement")

    resp = _client(FakeSource(), slack_alerts=RefusingSlackAlerts(None)).post("/alerts/slack", json={"alerts": []})
    assert resp.status_code == 502
    assert resp.json() == {"error": "Slack did not accept the alert announcement"}


def test_finelog_fleet_health_combines_the_main_hub_and_k8s_mirrors():
    fleet = K8sFleet([make_k8s_source(k8s_api(healthy_k8s_routes()))])

    rows = _client(FakeSource(), k8s_fleet=fleet).get("/finelog/marin/fleet_health").json()

    assert [(row["cluster"], row["server"], row["role"], row["responsive"]) for row in rows] == [
        ("marin", "finelog-marin", "hub", True),
        ("cw-a", "finelog-cw-a", "mirror", True),
    ]


def test_finelog_fleet_alert_marks_slow_and_unresponsive_servers():
    slow_hub = FakeSource(
        health=FinelogHealth(
            cluster="marin",
            server="finelog-marin",
            role=FinelogRole.HUB,
            responsive=True,
            ready=1,
            desired=1,
            latency_ms=5000,
            error_class="",
            error="",
        )
    )
    routes = healthy_k8s_routes()
    routes[FINELOG_DEPLOYMENTS_PATH] = [deployment("iris", "finelog-cw-a", ready=0, containers=("finelog",))]
    fleet = K8sFleet([make_k8s_source(k8s_api(routes))])

    assert _client(slow_hub, k8s_fleet=fleet).get("/finelog/marin/alerts/fleet_health").json() == [
        {
            "cluster": "marin",
            "server": "finelog-marin",
            "role": "hub",
            "state": "slow",
            "error_class": "",
            "value": 1,
        },
        {
            "cluster": "cw-a",
            "server": "finelog-cw-a",
            "role": "mirror",
            "state": "unresponsive",
            "error_class": "readiness",
            "value": 1,
        },
    ]


def test_workload_overview_counts_issue_rows_and_keeps_explicit_zeros():
    assert workload_overview([], []) == [{"pending_pods": 0, "crashlooping_containers": 0}]
    assert workload_overview(
        [{"pod": "queued"}, {"error_class": "network"}],
        [{"container": "trainer"}, {"container": "logger"}],
    ) == [{"pending_pods": 1, "crashlooping_containers": 2}]


def test_cache_coalesces_concurrent_misses_on_one_key():
    # N callers racing a cold key compute once. Pin the order: the first caller is
    # inside compute, holding the key lock, before the rest start.
    cache: TtlCache[int] = TtlCache(ttl=60.0)
    computing = threading.Event()
    release = threading.Event()
    calls: list[int] = []
    results: list[int] = []

    def compute():
        calls.append(1)
        computing.set()
        release.wait(timeout=5)
        return 7

    def worker():
        results.append(cache.get_or_compute("k", compute))

    first = threading.Thread(target=worker)
    first.start()
    assert computing.wait(timeout=5), "first caller never entered compute"
    others = [threading.Thread(target=worker) for _ in range(3)]
    for t in others:
        t.start()
    release.set()
    for t in [first, *others]:
        t.join(timeout=10)

    assert len(calls) == 1
    assert results == [7, 7, 7, 7]


def test_cache_reuses_compute_failures_until_the_ttl_expires():
    cache: TtlCache[int] = TtlCache(ttl=60.0)
    calls: list[int] = []

    def compute() -> int:
        calls.append(1)
        raise ValueError("upstream timed out")

    for _ in range(3):
        with pytest.raises(ValueError, match="upstream timed out"):
            cache.get_or_compute("k", compute)

    assert calls == [1]


def test_cache_prunes_expired_entries_on_write():
    # Keys embed a rotating time bucket, so an insert-only cache grows without bound
    # on a long-lived process. At ttl=0 every entry is stale on arrival.
    cache: TtlCache[int] = TtlCache(ttl=0.0)
    for i in range(50):
        cache.get_or_compute(f"bucket-{i}", lambda i=i: i)
    assert len(cache) == 0
