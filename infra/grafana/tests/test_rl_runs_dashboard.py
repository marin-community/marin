# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The RL run view's own SQL, executed against a run whose producers disagree on identity.

The view claims one thing: given a run_id, it shows the trainer, the accelerators and the
inference engines together, from telemetry_v1 alone. That claim rests on identity columns
different producers do and do not stamp, so the test builds a run the way the store
actually holds one — MarinSkyRL rows carrying run_id, iris-node-agent rows carrying only
node_name, vllm rows carrying only job_id — and runs the dashboard's shipped queries over
it. A producer that stops stamping a column it is joined on turns a panel blank, which is
indistinguishable from an idle run when you are looking at a dashboard.
"""

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from dashboard_stitch import stitch_all

ROOT = Path(__file__).resolve().parent.parent
DASHBOARDS = ROOT / "dashboards"

NOW = datetime(2026, 8, 20, 12, tzinfo=UTC)
WINDOW_START = NOW - timedelta(hours=1)
CLUSTER = "cw-rno2a"
RUN_ID = "snowball-e6-muonh-0"
JOB_ID = "/atqamar/snowball-e6-muonh-0-attempt-0"
NODES = ("gb200-node-0", "gb200-node-1")

_COLUMNS = (
    "cluster",
    "service",
    "run_id",
    "job_id",
    "execution_uid",
    "node_name",
    "process_index",
    "name",
    "value",
    "timestamp_ms",
    "seq",
    "resource_attributes_json",
    "attributes_json",
)


def _millis(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def _row(
    *,
    service: str,
    name: str,
    value: float,
    moment: datetime,
    seq: int,
    run_id: str | None = None,
    job_id: str | None = None,
    node_name: str | None = None,
    role: str = "",
    attributes: dict[str, str] | None = None,
) -> tuple:
    resource = {"role": role} if role else {}
    return (
        CLUSTER,
        service,
        run_id,
        job_id,
        "iris:/atqamar/snowball-e6-muonh-0-attempt-0/0:attempt:0",
        node_name,
        "0",
        name,
        value,
        _millis(moment),
        seq,
        json.dumps(resource),
        json.dumps(attributes or {}),
    )


def _run_rows() -> list[tuple]:
    """One RL run as the three producers actually record it."""
    rows = []
    for bucket in range(6):
        moment = WINDOW_START + timedelta(minutes=5 * bucket)
        # The trainer: run_id, job_id and the node it occupies.
        for node in NODES:
            rows.append(
                _row(
                    service="marinskyrl",
                    name="policy_step",
                    value=float(bucket),
                    moment=moment,
                    seq=bucket,
                    run_id=RUN_ID,
                    job_id=JOB_ID,
                    node_name=node,
                    role="trainer",
                )
            )
        for phase, seconds in (("rollout_or_inference_wait", 44.0), ("train_step", 6.0)):
            rows.append(
                _row(
                    service="marinskyrl",
                    name="phase_duration_seconds",
                    value=seconds,
                    moment=moment,
                    seq=bucket,
                    run_id=RUN_ID,
                    job_id=JOB_ID,
                    node_name=NODES[0],
                    role="trainer",
                    attributes={"phase": phase, "clock_domain": "critical_path", "outcome": "success"},
                )
            )
        for work_kind, count in (("rollout", 64.0), ("sample", 512.0), ("generated_token", 131072.0)):
            rows.append(
                _row(
                    service="marinskyrl",
                    name="work_completed",
                    value=count,
                    moment=moment,
                    seq=bucket,
                    run_id=RUN_ID,
                    job_id=JOB_ID,
                    node_name=NODES[0],
                    role="trainer",
                    attributes={"work_kind": work_kind},
                )
            )
        for name, value in (("queue_depth", 12.0), ("capacity", 32.0)):
            rows.append(
                _row(
                    service="marinskyrl",
                    name=name,
                    value=value,
                    moment=moment,
                    seq=bucket,
                    run_id=RUN_ID,
                    job_id=JOB_ID,
                    node_name=NODES[0],
                    role="trainer",
                    attributes={"queue": "rollout_buffer"},
                )
            )
        # The Ray controller, forwarded by the same service under a different role.
        rows.append(
            _row(
                service="marinskyrl",
                name="ray_object_store_used_memory",
                value=4.0e9 + bucket * 1.0e8,
                moment=moment,
                seq=bucket,
                run_id=RUN_ID,
                job_id=JOB_ID,
                node_name=NODES[0],
                role="controller",
                attributes={"metric_source": "ray"},
            )
        )
        # The Iris node agent: node_name only. No run_id, no job_id, ever.
        for node in NODES:
            rows.append(
                _row(
                    service="iris-node-agent",
                    name="gpu_utilization_percent",
                    value=71.0 + bucket,
                    moment=moment,
                    seq=bucket,
                    node_name=node,
                    attributes={"gpu_uuid": f"GPU-{node}-0"},
                )
            )
        # The inference engines: job_id only, cumulative snapshots.
        rows.append(
            _row(
                service="vllm",
                name="generation_tokens_total",
                value=100000.0 * (bucket + 1),
                moment=moment,
                seq=bucket,
                job_id=JOB_ID,
                node_name=NODES[1],
                role="inference",
                attributes={"source_temporality": "cumulative_snapshot", "model_name": "grug-67b-a2b"},
            )
        )
    return rows


@pytest.fixture
def store() -> duckdb.DuckDBPyConnection:
    database = duckdb.connect()
    database.execute(
        """
        CREATE TABLE telemetry_v1(
            cluster VARCHAR,
            service VARCHAR,
            run_id VARCHAR,
            job_id VARCHAR,
            execution_uid VARCHAR,
            node_name VARCHAR,
            process_index VARCHAR,
            name VARCHAR,
            value DOUBLE,
            timestamp_ms BIGINT,
            seq BIGINT,
            resource_attributes_json VARCHAR,
            attributes_json VARCHAR
        )
        """
    )
    # finelog's SQL dialect, in the two spellings the dashboards use.
    database.execute("CREATE MACRO to_timestamp_millis(value) AS to_timestamp(value / 1000.0)::TIMESTAMP")
    database.execute("CREATE MACRO date_bin(width, moment) AS time_bucket(width, moment)")
    database.execute("CREATE MACRO json_get(document, key) AS json_extract_string(document, '$.' || key)")
    placeholders = ", ".join("?" for _ in _COLUMNS)
    database.executemany(f"INSERT INTO telemetry_v1 VALUES ({placeholders})", _run_rows())
    return database


def _panel_sql(title: str) -> str:
    """One panel's shipped SQL, with Grafana's macros resolved to this window."""
    dashboard = stitch_all(DASHBOARDS, DASHBOARDS / "panels")["rl_runs.json"]
    panels = {panel["title"]: panel for panel in dashboard["panels"]}
    (parameter,) = [
        param
        for param in panels[title]["targets"][0]["url_options"]["params"]
        if param["key"] == "sql"
    ]
    sql = parameter["value"]
    sql = sql.replace("{{from}}", f"TIMESTAMP '{WINDOW_START.replace(tzinfo=None)}'")
    sql = sql.replace("{{to}}", f"TIMESTAMP '{NOW.replace(tzinfo=None)}'")
    sql = sql.replace("${__interval_ms} milliseconds", "5 minutes")
    sql = sql.replace("${cluster:sqlstring}", f"'{CLUSTER}'")
    sql = sql.replace("${run:sqlstring}", f"'{RUN_ID}'")
    assert not re.search(r"\$\{|\{\{", sql), sql
    return sql


def test_the_run_variable_offers_a_run_the_trainer_reported(store) -> None:
    dashboard = stitch_all(DASHBOARDS, DASHBOARDS / "panels")["rl_runs.json"]
    (variable,) = [v for v in dashboard["templating"]["list"] if v["name"] == "run"]
    (parameter,) = [
        param
        for param in variable["query"]["infinityQuery"]["url_options"]["params"]
        if param["key"] == "sql"
    ]
    sql = parameter["value"]
    sql = sql.replace("{{from}}", f"TIMESTAMP '{WINDOW_START.replace(tzinfo=None)}'")
    sql = sql.replace("{{to}}", f"TIMESTAMP '{NOW.replace(tzinfo=None)}'")
    sql = sql.replace("${cluster:sqlstring}", f"'{CLUSTER}'")

    assert store.execute(sql).fetchall() == [(RUN_ID,)]


def test_three_producers_answer_under_one_run_id(store) -> None:
    rows = store.execute(_panel_sql("Producers reporting this run")).fetchall()

    producers = {(row[0], row[1]) for row in rows}
    assert ("marinskyrl", "trainer") in producers
    assert ("marinskyrl", "controller") in producers
    assert all(row[4] > 0 for row in rows), rows


def test_the_trainer_panels_render_for_that_run(store) -> None:
    (phases,) = store.execute(_panel_sql("Critical-path phase duration")).fetchall()[:1]
    assert phases[1] == pytest.approx(44.0)
    assert phases[2] == pytest.approx(6.0)

    work = store.execute(_panel_sql("Work completed")).fetchall()
    assert [row[1] for row in work] == [64.0] * 6

    buffer = store.execute(_panel_sql("Rollout buffer occupancy")).fetchall()
    assert [(row[1], row[2]) for row in buffer] == [(12.0, 32.0)] * 6

    ray = store.execute(_panel_sql("Ray object store and spill")).fetchall()
    assert ray[0][1] == pytest.approx(4.0e9)


def test_the_node_agent_joins_through_node_name_without_a_run_id(store) -> None:
    # The node agent stamps no run identity at all, so the run's own rows have to name
    # its nodes. This is the join that breaks first if MarinSkyRL stops stamping
    # node_name, and it breaks silently.
    rows = store.execute(_panel_sql("GPU utilisation on this run's nodes")).fetchall()

    assert rows, "no accelerator series joined to the run"
    assert {row[1] for row in rows} == {RUN_ID}
    assert rows[0][2] == pytest.approx(71.0)


def test_the_engines_join_through_the_job_the_run_names(store) -> None:
    # The serving path stamps job_id and not run_id, so the run reaches its engines only
    # because MarinSkyRL stamps job_id too.
    rows = store.execute(_panel_sql("Engine generated tokens")).fetchall()

    deltas = [row[1] for row in rows if row[1] is not None]
    assert deltas == [100000.0] * 5


def test_a_trainer_that_stops_stamping_node_name_blanks_the_accelerator_panel(store) -> None:
    # Guards the failure mode this dashboard cannot show you: an identity regression in
    # the producer reads as an idle run.
    store.execute("UPDATE telemetry_v1 SET node_name = NULL WHERE service = 'marinskyrl'")

    assert store.execute(_panel_sql("GPU utilisation on this run's nodes")).fetchall() == []


def test_a_trainer_that_stops_stamping_job_id_blanks_the_engine_panel(store) -> None:
    store.execute("UPDATE telemetry_v1 SET job_id = NULL WHERE service = 'marinskyrl'")

    assert store.execute(_panel_sql("Engine generated tokens")).fetchall() == []
