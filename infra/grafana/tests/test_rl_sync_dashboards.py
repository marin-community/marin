# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The sync RL boards' panels, driven through the bridge over the rows MarinSkyRL publishes.

Driver spans carry ``clock_domain='inclusive_wall'`` and no rank. Worker spans carry a rank and a
clock domain ending in ``_wall`` with ``trainer.policy_train_spans_synchronize`` and ``_launch``
without it; both are fixtured. The two ranks are built so that a per-phase maximum across them
exceeds the parent it decomposes.
"""

import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as Record

import duckdb
import pyarrow as pa
import pytest
from config import ClusterTarget
from conftest import bridge_config, install_finelog_dialect_macros
from dashboard_stitch import stitch_all
from rl_observability import (
    RL_MAX_RESULT_ROWS,
    rl_overview_dataset,
    rl_sync_generation_dataset,
    rl_sync_train_step_dataset,
)
from server import create_app
from starlette.testclient import TestClient
from vllm_observability import VLLM_SCRAPE_INTERVAL_MS

ROOT = Path(__file__).resolve().parent.parent
DASHBOARDS = ROOT / "dashboards"

NOW = datetime(2026, 9, 2, 12, tzinfo=UTC)
WINDOW_START = NOW - timedelta(hours=1)
BUCKETS = 6
CLUSTER = "cw-rno2a"
RUN_ID = "snowball-e6-rl-7786"
NODES = ("h100-node-0", "h100-node-1")

# The E6 step, as measured over 20/20 steps of dogml/snowball_67b_a2b_rl_7786/nk0ehfrv.
STEP_SECONDS = 4209.7
DRIVER_PHASES = {
    "generate": 161.2,
    "convert_to_training_input": 8.9,
    "fwd_logprobs_values_reward": 220.2,
    "policy_train": 3805.6,
    "sync_weights": 12.6,
}
# train_critic_and_policy contains policy_train, so its own band is the Ray dispatch around it.
CONTAINER_SECONDS = 3806.0

# The generate subtree, two levels deep so that a child banded beside its parent is caught.
# generate's own exclusive time is the residual the driver publishes.
GENERATE_CHILDREN = {"rollout_collect": 156.4, "rollout_assemble": 0.1, "rollout_finalize": 4.6}
GENERATE_GRANDCHILDREN = {"rollout_tokenize": ("rollout_collect", 0.2), "rollout_retain": ("rollout_finalize", 4.5)}
GENERATE_RESIDUAL = DRIVER_PHASES["generate"] - sum(GENERATE_CHILDREN.values())

# step's own exclusive time: the driver loop outside every phase.
UNATTRIBUTED = (
    STEP_SECONDS
    - DRIVER_PHASES["generate"]
    - CONTAINER_SECONDS
    - sum(DRIVER_PHASES[phase] for phase in ("convert_to_training_input", "fwd_logprobs_values_reward", "sync_weights"))
)
DISPATCH_SECONDS = CONTAINER_SECONDS - DRIVER_PHASES["policy_train"]

# Two ranks whose barrier and compute times are anti-correlated. Rank 1 is the slowest: it arrives
# last, waits ~0 at the entry barrier, and does the full compute. A per-phase maximum over the pair
# would report 2705 s inside a 2000 s parent. Rank 1's spans overlap, so they sum past its
# policy_ppo_train and the residual it publishes is negative.
WORKER_SPANS = {
    "0": {
        "policy_entry_barrier": 700.0,
        "policy_forward": 300.0,
        "policy_backward": 700.0,
        "policy_optimizer_step": 60.0,
        "policy_entropy_allreduce": 10.0,
        "policy_metric_allreduce": 40.0,
        "policy_final_barrier": 10.0,
    },
    "1": {
        "policy_entry_barrier": 5.0,
        "policy_forward": 500.0,
        "policy_backward": 1290.0,
        "policy_optimizer_step": 90.0,
        "policy_entropy_allreduce": 15.0,
        "policy_metric_allreduce": 50.0,
        "policy_final_barrier": 60.0,
    },
}
PPO_TRAIN = {"0": 1900.0, "1": 2000.0}
CRITICAL_RANK = "1"

EXECUTION = "iris:/atqamar/snowball-e6-rl-7786-attempt-0/0:attempt:0"
# The run restarts from a checkpoint and repeats RETRIED_STEP in the same bucket, with the ranks'
# roles swapped and every worker span RETRY_SCALE times longer. A step keyed by number alone would
# give both attempts one slowest rank and one parent.
RETRY_EXECUTION = "iris:/atqamar/snowball-e6-rl-7786-attempt-0/0:attempt:1"
RETRIED_STEP = 2
RETRY_SCALE = 1.1
# One terminal event per process: the trainer, and a worker per rank. Both attempts' workers fail
# the same way.
TERMINAL_EVENTS = (
    (EXECUTION, "trainer", "failed", "ActorDiedError", (0,)),
    (EXECUTION, "worker", "failed", "ActorDiedError", (12, 5)),
    (RETRY_EXECUTION, "trainer", "completed", "normal_exit", (0,)),
    (RETRY_EXECUTION, "worker", "failed", "ActorDiedError", (30, 0)),
)
QUEUED_AT_EXIT = 3

# policy_training_step wraps these four. It ships under an inclusive clock domain, which keeps it
# out of the bands.
CONTAINED_SPANS = ("policy_forward", "policy_backward", "policy_optimizer_step", "policy_entropy_allreduce")
PUBLISHED_RESIDUAL = PPO_TRAIN[CRITICAL_RANK] - sum(WORKER_SPANS[CRITICAL_RANK].values())

# policy_span_publish ships the previous step's rows. Its parent is train_critic_and_policy, so it
# stays out of the decomposition although its clock domain is exclusive.
SPAN_PUBLISH_SECONDS = 3.0

# The driver also publishes train_step and rollout_or_inference_wait on the critical_path clock, with
# an outcome. One step fails, because a failed step renders like a fast one.
CRITICAL_PATH = {"train_step": CONTAINER_SECONDS, "rollout_or_inference_wait": DRIVER_PHASES["generate"]}
FAILED_BUCKET = 4

# The driver's rollout counters at measured magnitudes. The engine-await sum is over 512 concurrent
# coroutines and is five times the step.
TRAJECTORIES = 512.0
ENGINE_AWAIT_SUM = 23655.9
ENGINE_AWAIT_MAX = 126.6
ENV_SPLIT = {"queue": 3.3, "exec": 0.1, "resume": 88.2}
ROLLOUT_COUNTERS = {
    "rollout_trajectory_count": TRAJECTORIES,
    "rollout_engine_await_count": TRAJECTORIES,
    "rollout_engine_await_seconds_sum": ENGINE_AWAIT_SUM,
    "rollout_engine_await_seconds_max": ENGINE_AWAIT_MAX,
    "rollout_env_await_count": 3 * TRAJECTORIES,
    "rollout_env_await_seconds_max": 1.04,
    "rollout_env_await_seconds_sum": sum(ENV_SPLIT.values()),
    "rollout_env_queue_seconds_sum": ENV_SPLIT["queue"],
    "rollout_env_exec_seconds_sum": ENV_SPLIT["exec"],
    "rollout_env_resume_seconds_sum": ENV_SPLIT["resume"],
}

# vLLM, per engine and per bucket. Two engines, one publishing under each namespace.
ENGINES = ("0", "1")
GENERATED_TOKENS = 30_000.0
PROMPT_TOKENS = 12_000.0
FINISHED_REQUESTS = 100.0
TIME_PER_OUTPUT_TOKEN = 0.02
RUNNING = 48.0
QUEUE_DEPTH = 6.0
KV_CACHE_USAGE = 0.42

GPU_UTILIZATION = 97.0
SM_ACTIVE_RATIO = 0.82
TENSOR_ACTIVE_RATIO = 0.04
GPU_MEMORY_USED = 76.3 * 1024**3
NVLINK_RATE = 4.0e10
PCIE_RATE = 9.0e9
DEGRADED_GPU = ("h100-node-1", "1")
NVLINK_ERRORS_PER_BUCKET = 7.0
RESET_GPU = ("h100-node-0", "1")
RESET_BUCKET = 3
AFTER_RESET = 2.0

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
    "body_json",
    "kind",
)

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
    attributes_json VARCHAR,
    body_json VARCHAR,
    kind VARCHAR
)"""

_ARROW_SCHEMA = pa.schema(
    (column, pa.float64() if column == "value" else pa.int64() if column in ("timestamp_ms", "seq") else pa.string())
    for column in _COLUMNS
)

_SEMANTIC_STREAM = {
    "vllm": "telemetry_v1.vllm",
    "iris-node-agent": "telemetry_v1.node_agent",
    "marinskyrl": "telemetry_v1.marinskyrl",
}


def _millis(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def _row(
    *,
    name: str,
    value: float,
    moment: datetime,
    seq: int,
    service: str = "marinskyrl",
    run_id: str | None = RUN_ID,
    node_name: str | None = None,
    role: str = "",
    attributes: dict[str, str] | None = None,
    body: dict[str, object] | None = None,
    execution_uid: str = EXECUTION,
) -> tuple:
    return (
        CLUSTER,
        service,
        run_id,
        "/atqamar/snowball-e6-rl-7786-attempt-0",
        execution_uid,
        node_name,
        None,
        name,
        value,
        _millis(moment),
        seq,
        json.dumps({"role": role} if role else {}),
        json.dumps(attributes or {}),
        json.dumps(body or {}),
        # Forwarded snapshots all arrive with kind 'gauge'; source_temporality carries the semantics.
        "gauge",
    )


def _driver_rows(moment: datetime, seq: int, execution_uid: str = EXECUTION) -> list[tuple]:
    """What FinelogTimingSink publishes: one row per phase, no rank, parented to the nearest recorded ancestor."""

    def row(name: str, value: float, **attributes: str) -> tuple:
        attributes = {**attributes, "role": "trainer", "step": str(seq)}
        return _row(
            name=name,
            value=value,
            moment=moment,
            seq=seq,
            execution_uid=execution_uid,
            node_name=NODES[0],
            role="trainer",
            attributes=attributes,
        )

    tree = [
        ("step", STEP_SECONDS, ""),
        ("generate", DRIVER_PHASES["generate"], "step"),
        ("convert_to_training_input", DRIVER_PHASES["convert_to_training_input"], "step"),
        ("fwd_logprobs_values_reward", DRIVER_PHASES["fwd_logprobs_values_reward"], "step"),
        ("train_critic_and_policy", CONTAINER_SECONDS, "step"),
        ("policy_train", DRIVER_PHASES["policy_train"], "train_critic_and_policy"),
        ("sync_weights", DRIVER_PHASES["sync_weights"], "step"),
        *((phase, seconds, "generate") for phase, seconds in GENERATE_CHILDREN.items()),
        *((phase, seconds, parent) for phase, (parent, seconds) in GENERATE_GRANDCHILDREN.items()),
    ]
    outcome = "failure" if seq == FAILED_BUCKET else "success"
    return [
        *(
            row(
                "phase_duration_seconds",
                seconds,
                phase=phase,
                root="step",
                parent=parent,
                clock_domain="inclusive_wall",
            )
            for phase, seconds, parent in tree
        ),
        # A residual is what its parent's wall does not contain, so it ships exclusive.
        row(
            "phase_duration_seconds",
            GENERATE_RESIDUAL,
            phase="generate_span_residual",
            root="step",
            parent="generate",
            clock_domain="exclusive_wall",
        ),
        *(
            row("phase_duration_seconds", seconds, phase=phase, clock_domain="critical_path", outcome=outcome)
            for phase, seconds in CRITICAL_PATH.items()
        ),
        *(
            row("rollout_count" if counter.endswith("_count") else "rollout_wait_seconds", value, counter=counter)
            for counter, value in ROLLOUT_COUNTERS.items()
        ),
    ]


def _worker_rows(moment: datetime, seq: int, clock: str, execution_uid: str = EXECUTION) -> list[tuple]:
    """What WorkerTimingSink publishes: exclusive spans per rank and the inclusive parents.

    Ranks sit on different nodes, so the DCGM join credits both nodes to the run.
    """

    def row(worker_rank: str, name: str, value: float, **attributes: str) -> tuple:
        return _row(
            name=name,
            value=value,
            moment=moment,
            seq=seq,
            execution_uid=execution_uid,
            node_name=NODES[int(worker_rank) % len(NODES)],
            role="worker",
            attributes={**attributes, "role": "worker", "rank": worker_rank, "step": str(seq)},
        )

    def span(worker_rank: str, phase: str, seconds: float, parent: str, domain: str) -> tuple:
        return row(
            worker_rank,
            "phase_duration_seconds",
            seconds,
            phase=phase,
            root="step",
            parent=parent,
            clock_domain=f"{domain}_{clock}",
        )

    rows = []
    retried = execution_uid == RETRY_EXECUTION
    scale = RETRY_SCALE if retried else 1.0
    for rank, spans in WORKER_SPANS.items():
        worker_rank = str(len(WORKER_SPANS) - 1 - int(rank)) if retried else rank
        spans = {phase: seconds * scale for phase, seconds in spans.items()}
        contained = sum(spans[phase] for phase in CONTAINED_SPANS)
        residual = PPO_TRAIN[rank] * scale - sum(spans.values())
        rows += [span(worker_rank, phase, seconds, "policy_ppo_train", "exclusive") for phase, seconds in spans.items()]
        rows += [
            span(worker_rank, "policy_span_residual", residual, "policy_ppo_train", "exclusive"),
            span(worker_rank, "policy_span_publish", SPAN_PUBLISH_SECONDS, "train_critic_and_policy", "exclusive"),
            span(worker_rank, "policy_training_step", contained, "policy_ppo_train", "inclusive"),
            span(worker_rank, "policy_ppo_train", PPO_TRAIN[rank] * scale, "train_critic_and_policy", "inclusive"),
        ]
    return rows


def _node_agent_rows(moment: datetime, seq: int) -> list[tuple]:
    """DCGM through the Iris node agent: node_name only, no run identity, ever."""
    rows = []
    for node in NODES:
        for gpu in ("0", "1"):
            gauges = {
                "gpu_utilization_percent": GPU_UTILIZATION,
                "gpu_sm_active_ratio": SM_ACTIVE_RATIO,
                "gpu_tensor_active_ratio": TENSOR_ACTIVE_RATIO,
                "gpu_memory_used_bytes": GPU_MEMORY_USED,
                "gpu_nvlink_receive_bytes_per_second": NVLINK_RATE,
                "gpu_nvlink_transmit_bytes_per_second": NVLINK_RATE,
                "gpu_pcie_receive_bytes_per_second": PCIE_RATE,
                "gpu_pcie_transmit_bytes_per_second": PCIE_RATE,
                "gpu_power_watts": 620.0,
            }
            gauges["gpu_pcie_replay_errors"] = 3.0
            identity = {"gpu_uuid": f"GPU-{node}-{gpu}", "gpu_index": gpu}
            series = [(name, value, identity) for name, value in gauges.items()]
            # One cumulative NVLink series per error kind. A flat series, zero or not, is no new fault.
            # One GPU keeps counting, and another's counter resets.
            nvlink = {"crc_flit": 100.0, "crc_data": 0.0, "replay": 0.0, "recovery": 0.0}
            if (node, gpu) == DEGRADED_GPU:
                nvlink["replay"] = NVLINK_ERRORS_PER_BUCKET * seq
            if (node, gpu) == RESET_GPU:
                nvlink["recovery"] = 40.0 if seq < RESET_BUCKET else AFTER_RESET
            series += [("gpu_nvlink_errors", value, {**identity, "error_kind": kind}) for kind, value in nvlink.items()]
            for name, value, attributes in series:
                rows.append(
                    _row(
                        service="iris-node-agent",
                        run_id=None,
                        name=name,
                        value=value,
                        moment=moment,
                        seq=seq,
                        node_name=node,
                        attributes=attributes,
                    )
                )
    return rows


def _vllm_rows(moment: datetime, seq: int) -> list[tuple]:
    """The engine registry as inference_observability publishes it: cumulative counters and
    histograms, and gauges as current snapshots."""
    rows = []
    for engine in ENGINES:
        # Engine 0 is forwarded by the MarinSkyRL process under its own service name; engine 1
        # publishes its own registry as service='vllm'.
        identity = {"metric_source": "vllm", "engine": engine, "engine_index": engine}
        cumulative = {**identity, "source_temporality": "cumulative_snapshot"}
        current = {**identity, "source_temporality": "current_snapshot"}
        finished = FINISHED_REQUESTS * seq
        samples = [
            ("generation_tokens_total", GENERATED_TOKENS * seq, cumulative),
            ("prompt_tokens_total", PROMPT_TOKENS * seq, cumulative),
            ("request_time_per_output_token_seconds_count", finished, cumulative),
            ("request_time_per_output_token_seconds_sum", finished * TIME_PER_OUTPUT_TOKEN, cumulative),
            ("request_time_per_output_token_seconds_bucket", finished, {**cumulative, "le": "+Inf"}),
            ("num_requests_running", RUNNING, current),
            ("num_requests_waiting", QUEUE_DEPTH, current),
            ("kv_cache_usage_perc", KV_CACHE_USAGE, current),
        ]
        rows += [
            _row(
                service="marinskyrl" if engine == "0" else "vllm",
                name=name,
                value=value,
                moment=moment,
                seq=seq,
                node_name=NODES[1],
                role="inference",
                attributes=attributes,
            )
            for name, value, attributes in samples
        ]
    return rows


def _run_rows(clock: str) -> list[tuple]:
    rows = []
    for bucket in range(BUCKETS):
        moment = WINDOW_START + timedelta(minutes=5 * bucket)
        rows += _driver_rows(moment, bucket)
        rows += _worker_rows(moment, bucket, clock)
        rows += _node_agent_rows(moment, bucket)
        rows += _vllm_rows(moment, bucket)
        # The run variable reads policy_step, so the run has to report one.
        rows.append(
            _row(
                name="policy_step",
                value=float(bucket),
                moment=moment,
                seq=bucket,
                node_name=NODES[0],
                role="trainer",
            )
        )
    retry = WINDOW_START + timedelta(minutes=5 * RETRIED_STEP + 2)
    rows += _driver_rows(retry, RETRIED_STEP, RETRY_EXECUTION)
    rows += _worker_rows(retry, RETRIED_STEP, clock, RETRY_EXECUTION)
    for execution_uid, role, status, reason, lost in TERMINAL_EVENTS:
        for process, records in enumerate(lost):
            rows.append(
                _row(
                    name="terminal",
                    value=0.0,
                    moment=NOW - timedelta(seconds=1),
                    seq=BUCKETS,
                    execution_uid=execution_uid,
                    node_name=NODES[process],
                    role=role,
                    attributes={"role": role},
                    body={
                        "status": status,
                        "reason": reason,
                        "export_lost_records": records,
                        "export_queued_records": QUEUED_AT_EXIT,
                    },
                )
            )
    return rows


def _empty_store() -> duckdb.DuckDBPyConnection:
    database = duckdb.connect()
    for stream in sorted(set(_SEMANTIC_STREAM.values())):
        database.execute(f'CREATE TABLE "{stream}"{_SCHEMA}')
    install_finelog_dialect_macros(database)
    # Finelog and DuckDB name the struct constructor the vLLM sample query uses differently.
    database.execute(
        """CREATE MACRO named_struct(k1, v1, k2, v2, k3, v3)
                   AS struct_pack(timestamp_ms := v1, seq := v2, value := v3)"""
    )
    return database


def _store(clock: str) -> duckdb.DuckDBPyConnection:
    database = _empty_store()
    service_index = _COLUMNS.index("service")
    routed: dict[str, list] = {}
    for row in _run_rows(clock):
        stream = _SEMANTIC_STREAM[row[service_index]]
        routed.setdefault(stream, []).append(row)
    for stream, stream_rows in routed.items():
        # DuckDB's executemany costs milliseconds a row; one Arrow batch costs microseconds.
        columns = [list(column) for column in zip(*stream_rows, strict=True)]
        database.register("seeded_rows", pa.table(columns, schema=_ARROW_SCHEMA))
        database.execute(f'INSERT INTO "{stream}" SELECT * FROM seeded_rows')
        database.unregister("seeded_rows")
    return database


@pytest.fixture
def store(request) -> duckdb.DuckDBPyConnection:
    """A synchronised run, whose worker spans ship as ``*_wall``. A test parametrized with "launch"
    gets an unsynchronised run, whose worker spans ship as ``*_launch``."""
    return _store(getattr(request, "param", "wall"))


def _stitched() -> dict:
    return stitch_all(DASHBOARDS, DASHBOARDS / "panels")


def _dashboard(name: str = "rl_sync_train_step.json") -> dict:
    return _stitched()[name]


def _rl_dashboards() -> dict:
    return {name: _stitched()[name] for name in ("rl_sync_train_step.json", "rl_sync_generation.json", "rl_runs.json")}


def _all_panels(title: str) -> list[dict]:
    return [panel for board in _rl_dashboards().values() for panel in board["panels"] if panel["title"] == title]


def _resolve(sql: str) -> str:
    sql = sql.replace("{{from}}", f"TIMESTAMP '{WINDOW_START.replace(tzinfo=None)}'")
    sql = sql.replace("{{to}}", f"TIMESTAMP '{NOW.replace(tzinfo=None)}'")
    sql = sql.replace("${__interval_ms} milliseconds", "5 minutes")
    sql = sql.replace("${__interval_ms}", str(5 * 60 * 1000))
    sql = sql.replace("${cluster:sqlstring}", f"'{CLUSTER}'")
    sql = sql.replace("${run:sqlstring}", f"'{RUN_ID}'")
    assert not re.search(r"\$\{|\{\{", sql), sql
    return sql


BUCKET_MS = 5 * 60 * 1000
BUCKET_TIMES = [_millis(WINDOW_START) + bucket * BUCKET_MS for bucket in range(BUCKETS)]
# What a worker panel's mean over one bucket's steps reads, relative to a single attempt's step.
BUCKET_SCALE = {t: (1.0 + RETRY_SCALE) / 2 if bucket == RETRIED_STEP else 1.0 for bucket, t in enumerate(BUCKET_TIMES)}
_DATASETS = {
    "/v1/rl/overview": rl_overview_dataset,
    "/v1/rl/generation": rl_sync_generation_dataset,
    "/v1/rl/train-step": rl_sync_train_step_dataset,
}
_TEMPLATE = {
    "${cluster:csv}": CLUSTER,
    "${run}": RUN_ID,
    "${__from}": str(_millis(WINDOW_START)),
    "${__to}": str(_millis(NOW)),
    "${__interval_ms}": str(BUCKET_MS),
}


def _params(target: dict) -> dict[str, str]:
    """The target's query parameters, with Grafana's macros resolved to this window."""
    params = {}
    for param in target["url_options"]["params"]:
        value = param["value"]
        for macro, resolved in _TEMPLATE.items():
            value = value.replace(macro, resolved)
        assert "${" not in value, value
        params[param["key"]] = value
    return params


def _bridge(database: duckdb.DuckDBPyConnection):
    """The bridge app over this store."""

    def query(sql: str, *, max_rows: int):
        table = database.execute(sql).fetch_arrow_table()
        assert table.num_rows <= max_rows, (table.num_rows, max_rows)
        return table

    source = Record(target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"), query=query)
    app = create_app(replace(bridge_config(), max_rows=RL_MAX_RESULT_ROWS), {"marin": source}, {}, None, None, None)
    return app


def _matches(expression: str | None, row: dict) -> bool:
    """Infinity's filterExpression, which Grafana applies to the rows the bridge returns."""
    if not expression:
        return True
    return eval(expression.replace("&&", " and ").replace("||", " or "), {"__builtins__": {}}, row)


def _responses(database: duckdb.DuckDBPyConnection, targets: list[dict]) -> list[list[dict]]:
    """Each target's rows through one bridge app, after the target's own filter."""
    app = _bridge(database)
    with TestClient(app) as client:
        responses = [client.get(f"/finelog/marin{target['url']}", params=_params(target)) for target in targets]
    assert [response.status_code for response in responses] == [200] * len(targets), [r.text for r in responses]
    return [
        [row for row in response.json() if _matches(target.get("filterExpression"), row)]
        for target, response in zip(targets, responses, strict=True)
    ]


def _target_rows(database: duckdb.DuckDBPyConnection, target: dict) -> list[tuple]:
    """What one target renders: its rows, in the columns it declares."""
    (rows,) = _responses(database, [target])
    return [tuple(row[column["selector"]] for column in target["columns"]) for row in rows]


def _panel_rows(database: duckdb.DuckDBPyConnection, title: str) -> list[tuple]:
    """The rows of the one panel with this title on the three sync RL boards."""
    ((target,),) = [panel["targets"] for panel in _all_panels(title)]
    return _target_rows(database, target)


def test_the_run_variable_offers_the_run_the_trainer_reported(store) -> None:
    (variable,) = [v for v in _dashboard()["templating"]["list"] if v["name"] == "run"]
    (parameter,) = [
        param for param in variable["query"]["infinityQuery"]["url_options"]["params"] if param["key"] == "sql"
    ]

    assert store.execute(_resolve(parameter["value"])).fetchall() == [(RUN_ID,)]


def test_the_step_bands_are_exclusive_and_they_close_on_the_step(store) -> None:
    rows = _panel_rows(store, "Step composition: exclusive seconds per span")

    bands = {series: seconds for _, series, seconds in rows}
    # A phase's band is its wall minus its children's. Banded at its own wall, train_critic_and_policy
    # would put 3806 s beside the 3805.6 s of policy_train it contains.
    assert bands["train_critic_and_policy"] == pytest.approx(DISPATCH_SECONDS)
    assert bands["policy_train"] == pytest.approx(DRIVER_PHASES["policy_train"])
    assert bands["step"] == pytest.approx(UNATTRIBUTED)
    assert sum(bands.values()) == pytest.approx(STEP_SECONDS)


def test_the_generate_subtree_is_subtracted_from_generate(store) -> None:
    bands = {
        series: seconds for _, series, seconds in _panel_rows(store, "Step composition: exclusive seconds per span")
    }

    # generate's own band equals the residual the driver publishes.
    assert bands["generate"] == pytest.approx(GENERATE_RESIDUAL)
    assert bands["rollout_collect"] == pytest.approx(
        GENERATE_CHILDREN["rollout_collect"] - GENERATE_GRANDCHILDREN["rollout_tokenize"][1]
    )
    assert bands["rollout_tokenize"] == pytest.approx(GENERATE_GRANDCHILDREN["rollout_tokenize"][1])
    # The whole subtree still sums to generate, two levels deep.
    subtree = ["generate", *GENERATE_CHILDREN, *GENERATE_GRANDCHILDREN]
    assert sum(bands[phase] for phase in subtree) == pytest.approx(DRIVER_PHASES["generate"])


def test_the_train_step_panel_is_the_mean_step_split_by_outcome(store) -> None:
    rows = _panel_rows(store, "train_step duration")

    # The retried step's bucket holds both attempts' train_step, and a sum would read twice the step.
    assert rows == [
        (t, "train_step · failure" if bucket == FAILED_BUCKET else "train_step · success", CRITICAL_PATH["train_step"])
        for bucket, t in enumerate(BUCKET_TIMES)
    ]


def test_the_accelerator_panels_join_dcgm_to_the_run_through_its_nodes(store) -> None:
    sm = _panel_rows(store, "SM and tensor-pipe activity on this run's nodes")
    by_series = {series: value for _, series, value in sm}
    assert by_series["SM active"] == pytest.approx(SM_ACTIVE_RATIO * 100.0)
    assert by_series["tensor pipe active"] == pytest.approx(TENSOR_ACTIVE_RATIO * 100.0)

    memory = _panel_rows(store, "GPU memory in use on this run's nodes")
    assert memory == [(t, pytest.approx(GPU_MEMORY_USED), pytest.approx(GPU_MEMORY_USED)) for t in BUCKET_TIMES]

    fabric = _panel_rows(store, "NVLink against PCIe receive traffic")
    # Four GPUs across the run's two nodes, summed per direction.
    assert {series for _, series, _ in fabric} == {"NVLink receive", "PCIe receive"}
    assert {round(value) for _, series, value in fabric if series == "NVLink receive"} == {round(4 * NVLINK_RATE)}


def test_a_trainer_that_stops_stamping_node_name_blanks_the_accelerator_panels(store) -> None:
    # DCGM rows carry no run identity, so without node_name on the run's rows the join finds nothing.
    store.execute('UPDATE "telemetry_v1.marinskyrl" SET node_name = NULL')

    assert _panel_rows(store, "SM and tensor-pipe activity on this run's nodes") == []


def test_the_fault_table_differences_the_counters_and_hides_healthy_gpus(store) -> None:
    rows = _panel_rows(store, "Link faults and power on this run's GPUs")

    # The two healthy GPUs hold only flat counters, one of them nonzero, and do not appear. A reset
    # counts what the counter holds after it.
    assert rows == [
        (DEGRADED_GPU[0], "GPU-{}-{}".format(*DEGRADED_GPU), 620.0, NVLINK_ERRORS_PER_BUCKET * (BUCKETS - 1), 0.0),
        (RESET_GPU[0], "GPU-{}-{}".format(*RESET_GPU), 620.0, AFTER_RESET, 0.0),
    ]


def test_the_generation_board_shows_the_inference_panels_for_the_run(store) -> None:
    """The vLLM row mounts the Inference diagnostics fragments with their identity bound to the run."""
    panels = [panel for panel in _dashboard("rl_sync_generation.json")["panels"] if panel["id"] in range(231, 236)]
    status, requests, tokens, tpot, kv = (_target_rows(store, target) for panel in panels for target in panel["targets"])
    engines = len(ENGINES)
    # Each engine's counter increase is divided by its bin: the dashboard's 15 s for the engine the
    # MarinSkyRL process forwards, the scrape interval for the standalone one.
    per_second = 1000 / 15_000 + 1000 / VLLM_SCRAPE_INTERVAL_MS

    assert [(metric, state) for metric, state, _ in status] == [
        ("query", "detail"),
        ("run attention", "no_conclusion"),
    ]
    assert requests == [
        (t, series, value, series)
        for t in BUCKET_TIMES
        for series, value in (
            ("num_requests_in_flight", engines * (RUNNING + QUEUE_DEPTH)),
            ("num_requests_running", engines * RUNNING),
            ("num_requests_waiting", engines * QUEUE_DEPTH),
        )
    ]
    assert tokens == [
        (t, series, pytest.approx(count * per_second))
        for t in BUCKET_TIMES[1:]
        for series, count in (("generated tokens/s", GENERATED_TOKENS), ("prompt tokens/s", PROMPT_TOKENS))
    ]
    assert tpot == [
        (t, "time per output token", pytest.approx(TIME_PER_OUTPUT_TOKEN), "tpot", "mean_over_time")
        for t in BUCKET_TIMES[1:]
    ]
    assert kv == [
        (t, series, pytest.approx(KV_CACHE_USAGE), series)
        for t in BUCKET_TIMES
        for series in ("kv_cache_usage", "kv_cache_usage_peak")
    ]


def test_every_panel_says_on_its_face_why_it_would_be_blank() -> None:
    """An empty panel and a broken producer look the same, so each panel says what empty means."""
    boards = _rl_dashboards()
    span_panels_on_rl_runs = [panel for panel in boards["rl_runs.json"]["panels"] if panel["id"] in (12, 20, 21, 23)]
    for panel in [
        *(panel for name in ("rl_sync_train_step.json", "rl_sync_generation.json") for panel in boards[name]["panels"]),
        *span_panels_on_rl_runs,
    ]:
        if panel["type"] != "row":
            assert panel["fieldConfig"]["defaults"].get("noValue"), panel["title"]


def test_the_tail_is_reported_against_the_per_trajectory_mean(store) -> None:
    rows = _panel_rows(store, "rollout_engine_await: slowest trajectory ÷ mean")

    expected = ENGINE_AWAIT_MAX / (ENGINE_AWAIT_SUM / TRAJECTORIES)
    assert expected > 1.0, "the fixture no longer has a tail"
    assert {round(value, 6) for _, value in rows} == {round(expected, 6)}


def test_the_vitals_table_names_the_clock_domain_the_ranks_and_the_failed_steps(store) -> None:
    """Which clock each sink stamped, whether any worker reported, and whether a step failed are all
    invisible in a duration. Each clock domain is its own row, so a run that mixed two shows both."""
    steps = BUCKETS + 1  # the retried attempt's repeat of a step is a step of its own
    assert _panel_rows(store, "Span coverage: clock domain, ranks, failed steps") == [
        ("trainer", "critical_path", None, steps, 1),
        ("trainer", "exclusive_wall", None, steps, None),
        ("trainer", "inclusive_wall", None, steps, None),
        ("worker", "exclusive_wall", len(WORKER_SPANS), steps, None),
        ("worker", "inclusive_wall", len(WORKER_SPANS), steps, None),
    ]


def test_the_outcome_table_reports_each_attempts_terminal_events_by_role(store) -> None:
    title = "Terminal event and lost records"

    # One row per attempt and role, summed over the role's processes.
    assert _panel_rows(store, title) == [
        (EXECUTION, "trainer", "failed", "ActorDiedError", 0, 3),
        (EXECUTION, "worker", "failed", "ActorDiedError", 17, 6),
        (RETRY_EXECUTION, "trainer", "completed", "normal_exit", 0, 3),
        (RETRY_EXECUTION, "worker", "failed", "ActorDiedError", 30, 6),
    ]
    # A count that is not an integer is unknown, and so is a sum that includes it. A CAST would raise
    # and fail the overview's whole span source, taking every panel on it down with this one.
    store.execute(
        """UPDATE "telemetry_v1.marinskyrl" SET body_json = replace(body_json, ': 12', ': "unknown"')
           WHERE name = 'terminal'"""
    )
    assert _panel_rows(store, title) == [
        (EXECUTION, "trainer", "failed", "ActorDiedError", 0, 3),
        (EXECUTION, "worker", "failed", "ActorDiedError", None, 6),
        (RETRY_EXECUTION, "trainer", "completed", "normal_exit", 0, 3),
        (RETRY_EXECUTION, "worker", "failed", "ActorDiedError", 30, 6),
    ]
    store.execute("""DELETE FROM "telemetry_v1.marinskyrl" WHERE name = 'terminal'""")
    assert _panel_rows(store, title) == []


def test_the_residual_panel_reports_both_trees_signed(store) -> None:
    (panel,) = _all_panels("Signed span residuals: generate and policy_ppo_train")
    driver, worker = (_target_rows(store, target) for target in panel["targets"])

    assert {round(value, 6) for _, value in driver} == {round(GENERATE_RESIDUAL, 6)}

    assert PUBLISHED_RESIDUAL < 0, "the fixture no longer has overlapping spans"
    # Signed, from the slowest rank. A negative residual is the only sign of overlapping spans.
    assert worker == [(t, pytest.approx(PUBLISHED_RESIDUAL * BUCKET_SCALE[t])) for t in BUCKET_TIMES]


def test_the_generate_shares_partition_the_phase(store) -> None:
    rows = _panel_rows(store, "generate: share of each child span")

    # Grandchildren count against their own parents. The shares sum to one in the retried step's
    # bucket too.
    expected = {child: seconds / DRIVER_PHASES["generate"] for child, seconds in GENERATE_CHILDREN.items()}
    expected["generate_span_residual"] = GENERATE_RESIDUAL / DRIVER_PHASES["generate"]
    assert sum(expected.values()) == pytest.approx(1.0)
    shares = {(t, series): value for t, series, value in rows}
    assert shares == pytest.approx({(t, band): share for t in BUCKET_TIMES for band, share in expected.items()})


def test_the_generate_shares_are_empty_without_the_subtree(store) -> None:
    """Most runs measure generate as one span. A 100% generate_span_residual band would look like a
    defect in generate."""
    store.execute(
        """DELETE FROM "telemetry_v1.marinskyrl"
           WHERE json_extract_string(attributes_json, '$.parent') = 'generate'"""
    )

    assert _panel_rows(store, "generate: share of each child span") == []


def test_the_rollout_waits_are_divided_by_the_trajectory_count(store) -> None:
    rows = _panel_rows(store, "Per-trajectory wait: rollout_engine_await and rollout_env_await")

    waits = (ENGINE_AWAIT_SUM / TRAJECTORIES, sum(ENV_SPLIT.values()) / TRAJECTORIES, ENGINE_AWAIT_MAX)
    assert rows == [(t, *map(pytest.approx, waits)) for t in BUCKET_TIMES]


def test_the_environment_split_is_a_partition_with_an_audit_band(store) -> None:
    rows = _panel_rows(store, "Environment wait: queue, exec and resume")

    shares = {series: value for _, series, value in rows}
    awaited = sum(ENV_SPLIT.values())
    assert shares["rollout_env_resume"] == pytest.approx(ENV_SPLIT["resume"] / awaited)
    assert shares["rollout_env_exec"] == pytest.approx(ENV_SPLIT["exec"] / awaited)
    # The producer states the three terms partition the wait exactly, so the audit band is zero
    # until they stop doing so.
    assert shares["remainder"] == pytest.approx(0.0)
    assert sum(shares.values()) == pytest.approx(1.0)


LONG_RUN_STEPS = 500
LONG_RUN_RANKS = 64
RANKS_PER_NODE = 8
GPUS_PER_NODE = 8
LONG_RUN_STEP_MS = 60_000


def _long_run_store(ranks: int, steps: int = LONG_RUN_STEPS) -> duckdb.DuckDBPyConnection:
    """`steps` steps spread over the window of LONG_RUN_STEPS one-minute steps, across `ranks` worker
    ranks eight to a node.

    The first step of this suite's run is the template: its worker rows are copied onto every rank
    of the same parity, its node agent rows onto every GPU, and everything onto every step. Copied
    in SQL, because a million rows built in Python would dominate the suite.
    """
    database = _empty_store()
    template = [row for row in _run_rows("wall") if row[_COLUMNS.index("seq")] == 0]
    database.register(
        "template_rows", pa.table([list(column) for column in zip(*template, strict=True)], schema=_ARROW_SCHEMA)
    )
    columns = ", ".join(_COLUMNS)
    for stream, predicate, copies, node, attributes in (
        # The driver, the engines and the run's other rows: one copy per step.
        (
            "telemetry_v1.marinskyrl",
            "service = 'marinskyrl' AND json_get(attributes_json, 'rank') IS NULL",
            "range(0)",
            "node_name",
            """CASE WHEN json_get(attributes_json, 'step') IS NULL THEN attributes_json
                    ELSE json_merge_patch(attributes_json, json_object('step', CAST(step AS VARCHAR))) END""",
        ),
        (
            "telemetry_v1.vllm",
            "service = 'vllm'",
            "range(0)",
            "node_name",
            "attributes_json",
        ),
        # A worker's spans: one copy per rank of the template rank's parity.
        (
            "telemetry_v1.marinskyrl",
            f"json_get(attributes_json, 'rank') = CAST(copy % 2 AS VARCHAR) AND copy < {ranks}",
            f"range({ranks})",
            f"'long-node-' || CAST(copy // {RANKS_PER_NODE} AS VARCHAR)",
            """json_merge_patch(attributes_json, json_object(
                   'step', CAST(step AS VARCHAR), 'rank', CAST(copy AS VARCHAR)))""",
        ),
        # DCGM on every GPU of every node the ranks occupy.
        (
            "telemetry_v1.node_agent",
            f"node_name = '{NODES[0]}' AND json_get(attributes_json, 'gpu_index') = '0'",
            f"range({ranks // RANKS_PER_NODE * GPUS_PER_NODE})",
            f"'long-node-' || CAST(copy // {GPUS_PER_NODE} AS VARCHAR)",
            f"""json_merge_patch(attributes_json, json_object(
                   'gpu_uuid', 'GPU-' || CAST(copy AS VARCHAR),
                   'gpu_index', CAST(copy % {GPUS_PER_NODE} AS VARCHAR)))""",
        ),
    ):
        single = copies == "range(0)"
        database.execute(
            f"""INSERT INTO "{stream}" ({columns})
                SELECT cluster, service, run_id, job_id, execution_uid, {node}, process_index, name, value,
                       timestamp_ms + step * {LONG_RUN_STEP_MS * LONG_RUN_STEPS // steps}, step,
                       resource_attributes_json, CAST({attributes} AS VARCHAR), body_json, kind
                FROM template_rows,
                     (SELECT range AS step FROM range({steps})) AS steps,
                     (SELECT range AS copy FROM {"range(1)" if single else copies}) AS copies
                WHERE {predicate}"""
        )
    return database


def test_no_source_grows_with_the_rank_count_on_a_long_run() -> None:
    """Span data is steps x ranks x phases x clock domains, so every source reduces the ranks in
    Finelog. At 500 steps across 64 ranks, each source holds under its cap, and a span or counter
    source returns exactly as many rows as it does across eight.

    The overview also reduces the steps: ten times as many steps in the same window return the same
    rows. A per-step overview source would pass its cap on a long window of a fast run, and the cap
    fails the whole overview, including panels that do not read spans."""
    wide, narrow = _long_run_store(LONG_RUN_RANKS), _long_run_store(RANKS_PER_NODE)
    dense = _long_run_store(RANKS_PER_NODE, steps=10 * LONG_RUN_STEPS)
    start_ms = _millis(WINDOW_START)
    end_ms = start_ms + (LONG_RUN_STEPS + 1) * LONG_RUN_STEP_MS

    for build in _DATASETS.values():
        dataset = build((CLUSTER,), RUN_ID, start_ms, end_ms, BUCKET_MS)
        for source in dataset.sources:
            rows = wide.execute(source.sql).fetch_arrow_table().num_rows
            assert 0 < rows <= source.max_rows, (dataset.name, source.name, rows)
            if source.name != "gpu":
                assert rows == narrow.execute(source.sql).fetch_arrow_table().num_rows, (dataset.name, source.name)
            if build is rl_overview_dataset:
                assert dense.execute(source.sql).fetch_arrow_table().num_rows == (
                    narrow.execute(source.sql).fetch_arrow_table().num_rows
                ), source.name

    targets = [
        target
        for board in _rl_dashboards().values()
        for panel in board["panels"]
        for target in panel.get("targets", [])
        if target["url"] in _DATASETS
    ]
    window = {"from": str(start_ms), "to": str(end_ms)}
    for database, served in ((wide, targets), (dense, [t for t in targets if t["url"] == "/v1/rl/overview"])):
        app = _bridge(database)
        with TestClient(app) as client:
            statuses = {
                client.get(f"/finelog/marin{target['url']}", params={**_params(target), **window}).status_code
                for target in served
            }
        assert statuses == {200}
