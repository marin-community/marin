# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
        # Production stamps no process_index: it is NULL on every row of a real capture, so the
        # counter windows have to separate replicas without it.
        None,
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
        for phase, seconds, outcome in (
            ("rollout_or_inference_wait", 44.0, "success"),
            ("train_step", 6.0, "success"),
            # A step that raised partway through. Its duration is a different quantity.
            ("train_step", 0.5, "failure"),
        ):
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
                    attributes={"phase": phase, "clock_domain": "critical_path", "outcome": outcome},
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
        for name, value in (("rollout_queue_depth", 12.0), ("rollout_capacity", 32.0)):
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
        # The Ray controller, forwarded under a different role. Forwarded snapshots always arrive
        # with kind "gauge"; source_temporality carries the real semantics.
        for node in NODES:
            for name, value in (
                ("ray_object_store_used_memory", 3.0e9),
                ("ray_object_store_available_memory", 1.0e9),
            ):
                rows.append(
                    _row(
                        service="marinskyrl",
                        name=name,
                        value=value,
                        moment=moment,
                        seq=bucket,
                        run_id=RUN_ID,
                        job_id=JOB_ID,
                        node_name=node,
                        role="controller",
                        attributes={"metric_source": "ray", "source_temporality": "current_snapshot"},
                    )
                )
            for state, value in (("Spilled", 2.0e9), ("Restored", 5.0e8)):
                rows.append(
                    _row(
                        service="marinskyrl",
                        name="ray_spill_manager_objects_bytes",
                        value=value,
                        moment=moment,
                        seq=bucket,
                        run_id=RUN_ID,
                        job_id=JOB_ID,
                        node_name=node,
                        role="controller",
                        attributes={
                            "metric_source": "ray",
                            "source_temporality": "current_snapshot",
                            "state": state,
                        },
                    )
                )
        # A cumulative counter from the same allowlist, which must never be averaged in.
        rows.append(
            _row(
                service="marinskyrl",
                name="ray_spill_manager_objects_bytes",
                value=9.9e12,
                moment=moment,
                seq=bucket,
                run_id=RUN_ID,
                job_id=JOB_ID,
                node_name=NODES[0],
                role="controller",
                attributes={
                    "metric_source": "ray",
                    "source_temporality": "cumulative_snapshot",
                    "state": "Spilled",
                },
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
        # Each engine actor publishes its own vLLM registry under `service='vllm'`, carrying the
        # run id it inherited from the task environment. The forwarder strips the `vllm:` prefix
        # and prometheus counters keep `_total`, so these are the names that land. The counters are
        # cumulative and the panels difference consecutive samples.
        for engine in ("0", "1"):
            for name, value in (
                ("generation_tokens_total", 1024.0 * (bucket + 1)),
                ("prompt_tokens_total", 256.0 * (bucket + 1)),
                ("num_requests_running", 48.0),
                ("num_requests_waiting", 12.0),
                ("kv_cache_usage_perc", 0.71),
                ("prefix_cache_hits_total", 42.0 * (bucket + 1)),
                ("prefix_cache_queries_total", 100.0 * (bucket + 1)),
                ("num_preemptions_total", 2.0 * (bucket + 1)),
            ):
                rows.append(
                    _row(
                        service="vllm",
                        name=name,
                        value=value,
                        moment=moment,
                        seq=bucket,
                        run_id=RUN_ID,
                        job_id=JOB_ID,
                        node_name=NODES[1],
                        role="inference",
                        attributes={"metric_source": "vllm", "engine": engine},
                    )
                )
            for stage in (
                "request_queue_time_seconds_sum",
                "request_queue_time_seconds_count",
                "request_decode_time_seconds_sum",
                "request_decode_time_seconds_count",
            ):
                rows.append(
                    _row(
                        service="vllm",
                        name=stage,
                        value=(3.5 if stage.endswith("_sum") else 1.0) * (bucket + 1),
                        moment=moment,
                        seq=bucket,
                        run_id=RUN_ID,
                        job_id=JOB_ID,
                        node_name=NODES[1],
                        role="inference",
                        attributes={"metric_source": "vllm", "engine": engine},
                    )
                )
            for reason, value in (
                ("length", 8.0 * (bucket + 1)),
                ("stop", 40.0 * (bucket + 1)),
                ("abort", 1.0 * (bucket + 1)),
            ):
                rows.append(
                    _row(
                        service="vllm",
                        name="request_success_total",
                        value=value,
                        moment=moment,
                        seq=bucket,
                        run_id=RUN_ID,
                        job_id=JOB_ID,
                        node_name=NODES[1],
                        role="inference",
                        attributes={"metric_source": "vllm", "engine": engine, "finished_reason": reason},
                    )
                )
        # Staleness as the trainer records it: one observation per admitted group, measured where
        # the consuming step is known. A group can wait in the buffer across step boundaries.
        for staleness in (0.0, 1.0, 1.0, 2.0, 4.0):
            rows.append(
                _row(
                    service="marinskyrl",
                    name="rollout_staleness_steps",
                    value=staleness,
                    moment=moment,
                    seq=bucket,
                    run_id=RUN_ID,
                    job_id=JOB_ID,
                    node_name=NODES[0],
                    role="trainer",
                    attributes={"step": str(bucket)},
                )
            )
    return rows


# `vllm` and `iris-node-agent` are declared routing rules; `marinskyrl` and `harbor` have none
# and fall through to `telemetry_v1.<service>`, so the trainer and engine panels read different
# tables.
_SEMANTIC_STREAM = {
    "vllm": "telemetry_v1.vllm",
    "iris-node-agent": "telemetry_v1.node_agent",
    "marinskyrl": "telemetry_v1.marinskyrl",
    "harbor": "telemetry_v1.harbor",
}

_SCHEMA = """(
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
)"""


@pytest.fixture
def store() -> duckdb.DuckDBPyConnection:
    database = duckdb.connect()
    # One table per semantic stream, seeded by routing each row on its service as the server does.
    for stream in sorted(set(_SEMANTIC_STREAM.values())):
        database.execute(f'CREATE TABLE "{stream}"{_SCHEMA}')
    # finelog's SQL dialect, in the two spellings the dashboards use.
    database.execute("CREATE MACRO to_timestamp_millis(value) AS to_timestamp(value / 1000.0)::TIMESTAMP")
    database.execute("CREATE MACRO date_bin(width, moment) AS time_bucket(width, moment)")
    database.execute("CREATE MACRO json_get(document, key) AS json_extract_string(document, '$.' || key)")
    # finelog runs DataFusion, which has approx_percentile_cont; duckdb spells it quantile_cont.
    database.execute("CREATE MACRO approx_percentile_cont(value, q) AS quantile_cont(value, q)")
    placeholders = ", ".join("?" for _ in _COLUMNS)
    service_index = _COLUMNS.index("service")
    routed: dict[str, list] = {}
    for row in _run_rows():
        stream = _SEMANTIC_STREAM.get(row[service_index])
        assert stream is not None, f"no semantic stream for service {row[service_index]!r}"
        routed.setdefault(stream, []).append(row)
    for stream, stream_rows in routed.items():
        database.executemany(f'INSERT INTO "{stream}" VALUES ({placeholders})', stream_rows)
    return database


def _panel_sql(title: str) -> str:
    """One panel's shipped SQL, with Grafana's macros resolved to this window."""
    dashboard = stitch_all(DASHBOARDS, DASHBOARDS / "panels")["rl_runs.json"]
    panels = {panel["title"]: panel for panel in dashboard["panels"]}
    (parameter,) = [param for param in panels[title]["targets"][0]["url_options"]["params"] if param["key"] == "sql"]
    sql = parameter["value"]
    sql = sql.replace("{{from}}", f"TIMESTAMP '{WINDOW_START.replace(tzinfo=None)}'")
    sql = sql.replace("{{to}}", f"TIMESTAMP '{NOW.replace(tzinfo=None)}'")
    sql = sql.replace("${__interval_ms} milliseconds", "5 minutes")
    # A rate panel divides by the bucket width, so the macro also appears on its own.
    sql = sql.replace("${__interval_ms}", str(5 * 60 * 1000))
    sql = sql.replace("${cluster:sqlstring}", f"'{CLUSTER}'")
    sql = sql.replace("${run:sqlstring}", f"'{RUN_ID}'")
    assert not re.search(r"\$\{|\{\{", sql), sql
    return sql


def test_the_run_variable_offers_a_run_the_trainer_reported(store) -> None:
    dashboard = stitch_all(DASHBOARDS, DASHBOARDS / "panels")["rl_runs.json"]
    (variable,) = [v for v in dashboard["templating"]["list"] if v["name"] == "run"]
    (parameter,) = [
        param for param in variable["query"]["infinityQuery"]["url_options"]["params"] if param["key"] == "sql"
    ]
    sql = parameter["value"]
    sql = sql.replace("{{from}}", f"TIMESTAMP '{WINDOW_START.replace(tzinfo=None)}'")
    sql = sql.replace("{{to}}", f"TIMESTAMP '{NOW.replace(tzinfo=None)}'")
    sql = sql.replace("${cluster:sqlstring}", f"'{CLUSTER}'")

    assert store.execute(sql).fetchall() == [(RUN_ID,)]


def test_the_trainer_panels_render_for_that_run(store) -> None:
    # A failed step ran 0.5s before raising; a successful one takes 6s. Blending them would
    # report 3.25s, which describes neither.
    phases = store.execute(_panel_sql("Rollout wait vs train step (critical path)")).fetchall()
    by_series = {row[1]: row[2] for row in phases}
    assert by_series["rollout_or_inference_wait · success"] == pytest.approx(44.0)
    assert by_series["train_step · success"] == pytest.approx(6.0)
    assert by_series["train_step · failure"] == pytest.approx(0.5)

    work = store.execute(_panel_sql("Rollouts, samples and tokens completed")).fetchall()
    assert [row[1] for row in work] == [64.0] * 6

    buffer = store.execute(_panel_sql("Rollout buffer occupancy (groups)")).fetchall()
    assert [(row[1], row[2]) for row in buffer] == [(12.0, 32.0)] * 6

    # Occupancy is a ratio, so two nodes reporting 3 GB used of 4 GB still reads 0.75 rather
    # than doubling. 6e9 used over 8e9 total.
    occupancy = store.execute(_panel_sql("Ray object store occupancy")).fetchall()
    assert [row[1] for row in occupancy] == [pytest.approx(0.75)] * 6


def test_the_node_agent_joins_through_node_name_without_a_run_id(store) -> None:
    # The node agent stamps no run identity at all, so the run's own rows have to name
    # its nodes. This is the join that breaks first if MarinSkyRL stops stamping
    # node_name, and it breaks silently.
    rows = store.execute(_panel_sql("GPU utilization on this run's nodes")).fetchall()

    assert rows, "no accelerator series joined to the run"
    assert {row[1] for row in rows} == {RUN_ID}
    assert rows[0][2] == pytest.approx(71.0)


def test_the_engine_panels_select_by_metric_name_alone(store) -> None:
    # No other MarinSkyRL producer emits these names, so the name identifies the engine path.
    throughput = store.execute(_panel_sql("Engine token throughput")).fetchall()
    # A rate over a CUMULATIVE counter: the panel takes the delta between consecutive samples,
    # so a fixture growing by 1024 per bucket per engine yields a constant rate -- and the first
    # bucket has no predecessor to difference against, so it drops out.
    rates = [round(row[1], 4) for row in throughput]
    assert rates == [round(2 * 1024.0 / 300.0, 4)] * len(rates)
    assert len(rates) == 5

    queue = store.execute(_panel_sql("Engine queue and KV-cache")).fetchall()
    assert [(row[1], row[2]) for row in queue] == [(48.0, 12.0)] * 6

    latency = store.execute(_panel_sql("Engine request latency (mean by stage)")).fetchall()
    # Prometheus histograms arrive as `_sum` and `_count`, so the panel reports a mean per stage.
    # A p90 would need bucket interpolation.
    assert {row[1] for row in latency} == {
        "request_queue_time_seconds",
        "request_decode_time_seconds",
    }


def test_a_trainer_that_stops_stamping_node_name_blanks_the_accelerator_panel(store) -> None:
    # An identity regression in the producer reads as an idle run.
    store.execute('UPDATE "telemetry_v1.marinskyrl" SET node_name = NULL')

    assert store.execute(_panel_sql("GPU utilization on this run's nodes")).fetchall() == []


def test_the_producer_census_separates_the_engine_series_from_the_trainer(store) -> None:
    # Each engine actor publishes its own registry under `service='vllm'`, so the census unions
    # the two streams.
    rows = store.execute(_panel_sql("Producers reporting this run")).fetchall()

    assert {(row[0], row[1]) for row in rows} >= {("marinskyrl", "trainer"), ("vllm", "inference")}


def _projected_columns(sql: str) -> set[str]:
    # Grafana reads the columns the outermost SELECT returns. The last SELECT in the text can sit
    # inside a subquery, and a derived table's alias is not a column, so scan at parenthesis depth
    # zero and keep only the projection list. CAST target types are spelled in caps.
    depth = 0
    start = end = None
    for match in re.finditer(r"[()]|\bSELECT\b|\bFROM\b", sql):
        token = match.group().upper()
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth == 0 and token == "SELECT":
            start, end = match.end(), None
        elif depth == 0 and token == "FROM" and start is not None and end is None:
            end = match.start()
    projection = sql[start:end]
    return {alias for alias in re.findall(r"\bAS (\w+)", projection) if not alias.isupper()}


def test_every_timeseries_panel_declares_the_columns_its_sql_returns() -> None:
    """A panel is read through its declared columns, so executing its SQL cannot see a mistake there."""
    dashboard = stitch_all(DASHBOARDS, DASHBOARDS / "panels")["rl_runs.json"]

    for panel in dashboard["panels"]:
        if panel.get("type") != "timeseries":
            continue
        for target in panel["targets"]:
            (parameter,) = [param for param in target["url_options"]["params"] if param["key"] == "sql"]
            selected = _projected_columns(parameter["value"])
            declared = {column["selector"]: column["type"] for column in target["columns"]}

            assert (
                set(declared) == selected
            ), f"{panel['title']}: declares {sorted(declared)}, SQL returns {sorted(selected)}"
            assert "number" in declared.values(), f"{panel['title']}: no numeric column to plot"


def test_every_panel_has_a_distinct_title_id_and_slot() -> None:
    # A duplicated panel renders twice and shares an id, and a test that looks panels up by
    # title cannot see it: the lookup keeps one and the dashboard keeps both.
    dashboard = stitch_all(DASHBOARDS, DASHBOARDS / "panels")["rl_runs.json"]
    panels = dashboard["panels"]

    titles = [panel["title"] for panel in panels]
    assert len(titles) == len(set(titles)), titles
    ids = [panel["id"] for panel in panels]
    assert len(ids) == len(set(ids)), ids
    slots = [(panel["gridPos"]["x"], panel["gridPos"]["y"]) for panel in panels]
    assert len(slots) == len(set(slots)), slots


def test_ray_panels_exclude_cumulative_snapshots_and_never_mix_states(store) -> None:
    # A forwarded snapshot's `kind` column is always "gauge"; source_temporality carries the real
    # semantics. The Ray allowlist includes cumulative counters, and averaging one in is silently
    # wrong. The spill states are distinct quantities and stay distinct series.
    rows = store.execute(_panel_sql("Ray spill manager bytes by state")).fetchall()

    by_state = {row[1]: row[2] for row in rows}
    assert by_state == {"Spilled": pytest.approx(2.0e9), "Restored": pytest.approx(5.0e8)}


def test_engine_rates_keep_two_actors_that_both_report_engine_zero_apart(store) -> None:
    # `engine` indexes engines within the registry that published them, so its scope depends on how
    # they were registered and two rows can carry the same index from different processes. The
    # window keys on the resource; GREATEST(delta, 0) would hide half of a cross-resource
    # subtraction as zero.
    for bucket in range(6):
        store.execute(
            f'INSERT INTO "telemetry_v1.vllm" VALUES ({", ".join("?" for _ in _COLUMNS)})',
            list(
                _row(
                    service="vllm",
                    name="generation_tokens_total",
                    value=50_000.0 + 1024.0 * (bucket + 1),
                    moment=WINDOW_START + timedelta(minutes=5 * bucket),
                    seq=bucket,
                    run_id=RUN_ID,
                    node_name=NODES[0],
                    role="inference",
                    attributes={"engine": "0"},
                )
            ),
        )

    throughput = store.execute(_panel_sql("Engine token throughput")).fetchall()
    rates = [round(row[1], 4) for row in throughput]

    assert rates == [round(3 * 1024.0 / 300.0, 4)] * len(rates)
    assert len(rates) == 5


def test_engine_panels_read_the_embedded_stream_as_well_as_the_standalone_one(store) -> None:
    # An engine embedded in the trainer publishes through the trainer's exporter, so its rows land
    # under service='marinskyrl' with metric_source='vllm'. Reading only telemetry_v1.vllm would
    # leave every engine panel blank on such a run.
    for bucket in range(6):
        store.execute(
            f'INSERT INTO "telemetry_v1.marinskyrl" VALUES ({", ".join("?" for _ in _COLUMNS)})',
            list(
                _row(
                    service="marinskyrl",
                    name="generation_tokens_total",
                    value=1024.0 * (bucket + 1),
                    moment=WINDOW_START + timedelta(minutes=5 * bucket),
                    seq=bucket,
                    run_id=RUN_ID,
                    node_name=NODES[0],
                    role="trainer",
                    attributes={"engine": "GPU-abc", "metric_source": "vllm"},
                )
            ),
        )

    rates = [round(row[1], 4) for row in store.execute(_panel_sql("Engine token throughput")).fetchall()]

    assert rates == [round(3 * 1024.0 / 300.0, 4)] * len(rates)
    assert len(rates) == 5
