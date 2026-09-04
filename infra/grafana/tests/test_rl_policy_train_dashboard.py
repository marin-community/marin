# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The policy_train dashboard's SQL, run against the rows MarinSkyRL actually publishes.

The fixture is built from the emitting code rather than from the panels: driver spans carry
``clock_domain='inclusive_wall'`` and no rank, worker spans carry a rank and one of two clock
domains, and the two ranks are constructed so that a per-phase maximum across them would exceed
the parent it is supposed to decompose. That is the mistake these panels exist to avoid, so it is
the one the fixture makes available.

``WorkerTimingSink._clock_domain`` composes the worker domain from the containment and the
synchronise mode: ``exclusive_wall`` with ``trainer.policy_train_spans_synchronize``, and
``exclusive_launch`` without it. Both are fixtured, because a panel that names only one renders
empty on every run made the other way and reads exactly like a producer that stopped publishing.
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
# train_critic_and_policy contains policy_train, so its own band is the Ray dispatch around it and
# never the 3806 s it wraps.
CONTAINER_SECONDS = 3806.0

# The generate subtree, in the proportions the pr488 run measured: the fan-out is essentially the
# whole phase and generate's own exclusive time is the published residual. Two levels deep, because
# one level would not catch a query that bands a child beside the parent that contains it.
GENERATE_CHILDREN = {"rollout_collect": 156.4, "rollout_assemble": 0.1, "rollout_finalize": 4.6}
GENERATE_GRANDCHILDREN = {"rollout_tokenize": ("rollout_collect", 0.2), "rollout_retain": ("rollout_finalize", 4.5)}
GENERATE_RESIDUAL = DRIVER_PHASES["generate"] - sum(GENERATE_CHILDREN.values())

# step's own exclusive time. The old panel lumped this together with train_critic_and_policy's, and
# the two are different costs: one is the driver's step loop, the other is the Ray round trip.
UNATTRIBUTED = (
    STEP_SECONDS
    - DRIVER_PHASES["generate"]
    - CONTAINER_SECONDS
    - sum(DRIVER_PHASES[phase] for phase in ("convert_to_training_input", "fwd_logprobs_values_reward", "sync_weights"))
)
DISPATCH_SECONDS = CONTAINER_SECONDS - DRIVER_PHASES["policy_train"]

# Two ranks whose barrier and compute time are anti-correlated. Rank 1 is r*: it arrives last, so
# it waits ~0 at the entry barrier and then does the full compute. Taking a per-phase maximum over
# the pair would report 2645 s inside a 2000 s parent.
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
        "policy_backward": 1200.0,
        "policy_optimizer_step": 90.0,
        "policy_entropy_allreduce": 15.0,
        "policy_metric_allreduce": 50.0,
        "policy_final_barrier": 60.0,
    },
}
PPO_TRAIN = {"0": 1900.0, "1": 2000.0}
CRITICAL_RANK = "1"

# policy_training_step wraps these four, and the fixture carries both of the ways it has arrived --
# which no single run does, so one store exercises both exclusions at once. The current spelling
# ships under an inclusive clock domain and is excluded by that. The first instrumented run
# published it as policy_training_step_other, which is absent from TIMING_PARENTS, so the sink
# stamped an empty parent on it and it arrives looking exactly like a leaf.
CONTAINER_SPAN = "policy_training_step_other"
CONTAINED_SPANS = (
    "policy_forward",
    "policy_backward",
    "policy_optimizer_step",
    "policy_entropy_allreduce",
)

# policy_span_publish is the cost of shipping the PREVIOUS step's rows. It is measured after
# policy_ppo_train's wall is taken and declares a parent outside it, so it is a worker span that
# does not belong in this decomposition however exclusive its clock domain looks.
SPAN_PUBLISH_SECONDS = 3.0

# The token counters were renamed on 2026-09-03 to say that they are one rank's shard rather than
# the run total. The fixture publishes the current spelling; the back catalogue carries the old one.
WORKER_COUNTERS = {
    "0": {
        "micro_step_count": 64.0,
        "rank_tokens_real": 6000.0,
        "rank_tokens_padded": 8000.0,
        "attention_work_ratio": 1.9,
    },
    "1": {
        "micro_step_count": 64.0,
        "rank_tokens_real": 6400.0,
        "rank_tokens_padded": 8000.0,
        "attention_work_ratio": 1.7,
    },
}

# The critical-path twins the driver publishes beside the tree, carrying an outcome and no place in
# it: train_step == train_critic_and_policy and rollout_or_inference_wait == generate. One step ends
# in a failure, because a truncated step renders exactly like a fast one.
CRITICAL_PATH = {"train_step": CONTAINER_SECONDS, "rollout_or_inference_wait": DRIVER_PHASES["generate"]}
FAILED_BUCKET = 4

# The driver's rollout counters, at the magnitudes the pr488 run measured. The engine-await total is
# a CONCURRENT SUM over 512 coroutines and is five times the whole step; a panel that plots it raw is
# the failure the per-trajectory division exists to prevent.
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

# Torch's own memory, per rank. Rank 1 holds most and is the one that binds the micro-batch, and it
# is the only rank whose allocator had to retry.
WORKER_MEMORY = {
    "0": {"peak_allocated_bytes": 61.0 * 1024**3, "peak_reserved_bytes": 71.0 * 1024**3},
    "1": {"peak_allocated_bytes": 63.0 * 1024**3, "peak_reserved_bytes": 74.0 * 1024**3},
}
WORKER_ALLOCATOR = {"0": {"alloc_retries": 0.0, "alloc_ooms": 0.0}, "1": {"alloc_retries": 5.0, "alloc_ooms": 0.0}}

# A cumulative Prometheus histogram: counts are cumulative in `le`, so +Inf carries the total.
GENERATION_TOKEN_BUCKETS = {"64": 10.0, "256": 50.0, "1024": 90.0, "4096": 99.0, "+Inf": 100.0}
LATENCY_BUCKETS = {"0.5": 20.0, "2": 60.0, "8": 95.0, "32": 99.0, "+Inf": 100.0}

SM_ACTIVE_RATIO = 0.82
TENSOR_ACTIVE_RATIO = 0.04
GPU_MEMORY_USED = 76.3 * 1024**3
NVLINK_RATE = 4.0e10
PCIE_RATE = 9.0e9

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

_SEMANTIC_STREAM = {
    "vllm": "telemetry_v1.vllm",
    "iris-node-agent": "telemetry_v1.node_agent",
    "marinskyrl": "telemetry_v1.marinskyrl",
}


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
    node_name: str | None = None,
    role: str = "",
    attributes: dict[str, str] | None = None,
) -> tuple:
    return (
        CLUSTER,
        service,
        run_id,
        "/atqamar/snowball-e6-rl-7786-attempt-0",
        "iris:/atqamar/snowball-e6-rl-7786-attempt-0/0:attempt:0",
        node_name,
        None,
        name,
        value,
        _millis(moment),
        seq,
        json.dumps({"role": role} if role else {}),
        json.dumps(attributes or {}),
    )


def _driver_rows(moment: datetime, seq: int) -> list[tuple]:
    """What FinelogTimingSink publishes: one inclusive-wall row per recorded phase, no rank.

    The parent is the nearest *recorded* ancestor, so the synchronous trainer — which never opens
    a run_training timer — attaches fwd_logprobs_values_reward and train_critic_and_policy straight
    to step, exactly as timing_observability.nearest_recorded_parent resolves them.
    """
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
    # EXCLUSIVE_DRIVER_SPANS: a residual is what its parent's wall does not contain, so it ships
    # exclusive while every other driver span ships inclusive.
    return (
        [
            _row(
                service="marinskyrl",
                name="phase_duration_seconds",
                value=seconds,
                moment=moment,
                seq=seq,
                run_id=RUN_ID,
                node_name=NODES[0],
                role="trainer",
                attributes={
                    "phase": phase,
                    "root": "step",
                    "parent": parent,
                    "clock_domain": "inclusive_wall",
                    "role": "trainer",
                    "step": str(seq),
                },
            )
            for phase, seconds, parent in tree
        ]
        + [
            _row(
                service="marinskyrl",
                name="phase_duration_seconds",
                value=GENERATE_RESIDUAL,
                moment=moment,
                seq=seq,
                run_id=RUN_ID,
                node_name=NODES[0],
                role="trainer",
                attributes={
                    "phase": "generate_span_residual",
                    "root": "step",
                    "parent": "generate",
                    "clock_domain": "exclusive_wall",
                    "role": "trainer",
                    "step": str(seq),
                },
            )
        ]
        + [
            _row(
                service="marinskyrl",
                name="phase_duration_seconds",
                value=seconds,
                moment=moment,
                seq=seq,
                run_id=RUN_ID,
                node_name=NODES[0],
                role="trainer",
                attributes={
                    "phase": phase,
                    "clock_domain": "critical_path",
                    "role": "trainer",
                    "outcome": "failure" if seq == FAILED_BUCKET else "success",
                    "step": str(seq),
                },
            )
            for phase, seconds in CRITICAL_PATH.items()
        ]
        + [
            _row(
                service="marinskyrl",
                # The counts and the seconds go to different instruments, as publish_rollout_counters
                # sends them.
                name="rollout_count" if counter.endswith("_count") else "rollout_wait_seconds",
                value=value,
                moment=moment,
                seq=seq,
                run_id=RUN_ID,
                node_name=NODES[0],
                role="trainer",
                attributes={"counter": counter, "role": "trainer", "step": str(seq)},
            )
            for counter, value in ROLLOUT_COUNTERS.items()
        ]
    )


def _worker_rows(moment: datetime, seq: int, clock: str) -> list[tuple]:
    """What WorkerTimingSink publishes: exclusive spans per rank, plus one inclusive parent.

    The ranks sit on different nodes, which is what lets the DCGM join credit both of the run's
    nodes to it: those rows carry a node and no run, and the run's own rows carry the reverse.
    """
    rows = []
    for rank, spans in WORKER_SPANS.items():
        rank_node = NODES[int(rank) % len(NODES)]
        emitted = dict(spans)
        # The container span, as the first instrumented run actually published it: an exclusive
        # clock domain, an empty parent, and four of the spans beside it inside its own wall.
        # Banding it counts that time twice, and the producer's own residual goes sharply negative.
        contained_seconds = sum(spans[phase] for phase in CONTAINED_SPANS)
        emitted[CONTAINER_SPAN] = contained_seconds
        emitted["policy_span_residual"] = PPO_TRAIN[rank] - sum(emitted.values())
        parents = dict.fromkeys(emitted, "policy_ppo_train")
        parents[CONTAINER_SPAN] = ""
        parents["policy_span_publish"] = "policy_train"
        emitted["policy_span_publish"] = SPAN_PUBLISH_SECONDS
        for phase, seconds in emitted.items():
            rows.append(
                _row(
                    service="marinskyrl",
                    name="phase_duration_seconds",
                    value=seconds,
                    moment=moment,
                    seq=seq,
                    run_id=RUN_ID,
                    node_name=rank_node,
                    role="worker",
                    attributes={
                        "phase": phase,
                        "root": "step",
                        "parent": parents[phase],
                        "clock_domain": f"exclusive_{clock}",
                        "role": "worker",
                        "rank": rank,
                        "step": str(seq),
                    },
                )
            )
        # The current spelling of the container, under the inclusive domain the sink gives it.
        rows.append(
            _row(
                service="marinskyrl",
                name="phase_duration_seconds",
                value=contained_seconds,
                moment=moment,
                seq=seq,
                run_id=RUN_ID,
                node_name=rank_node,
                role="worker",
                attributes={
                    "phase": "policy_training_step",
                    "root": "step",
                    "parent": "policy_ppo_train",
                    "clock_domain": f"inclusive_{clock}",
                    "role": "worker",
                    "rank": rank,
                    "step": str(seq),
                },
            )
        )
        rows.append(
            _row(
                service="marinskyrl",
                name="phase_duration_seconds",
                value=PPO_TRAIN[rank],
                moment=moment,
                seq=seq,
                run_id=RUN_ID,
                node_name=rank_node,
                role="worker",
                attributes={
                    "phase": "policy_ppo_train",
                    "root": "step",
                    "parent": "policy_train",
                    "clock_domain": f"inclusive_{clock}",
                    "role": "worker",
                    "rank": rank,
                    "step": str(seq),
                },
            )
        )
        # policy_train_bytes was split off the unit-1 counter instrument on 2026-09-03; the byte
        # gauges moved to it and the allocator deltas stayed behind.
        published = [
            ("policy_train_count", WORKER_COUNTERS[rank]),
            ("policy_train_count", WORKER_ALLOCATOR[rank]),
            ("policy_train_bytes", WORKER_MEMORY[rank]),
        ]
        for instrument, counters in published:
            for counter, value in counters.items():
                rows.append(
                    _row(
                        service="marinskyrl",
                        name=instrument,
                        value=value,
                        moment=moment,
                        seq=seq,
                        run_id=RUN_ID,
                        node_name=rank_node,
                        role="worker",
                        attributes={"counter": counter, "role": "worker", "rank": rank, "step": str(seq)},
                    )
                )
    return rows


def _node_agent_rows(moment: datetime, seq: int) -> list[tuple]:
    """DCGM through the Iris node agent: node_name only, no run identity, ever."""
    rows = []
    for node in NODES:
        for gpu in ("0", "1"):
            gauges = {
                "gpu_sm_active_ratio": SM_ACTIVE_RATIO,
                "gpu_tensor_active_ratio": TENSOR_ACTIVE_RATIO,
                "gpu_memory_used_bytes": GPU_MEMORY_USED,
                "gpu_nvlink_receive_bytes_per_second": NVLINK_RATE,
                "gpu_nvlink_transmit_bytes_per_second": NVLINK_RATE,
                "gpu_pcie_receive_bytes_per_second": PCIE_RATE,
                "gpu_pcie_transmit_bytes_per_second": PCIE_RATE,
                "gpu_power_watts": 620.0,
            }
            # Cumulative fault counters. Only one GPU is actually degraded.
            degraded = node == NODES[1] and gpu == "1"
            gauges["gpu_nvlink_errors"] = 100.0 + (7.0 * seq if degraded else 0.0)
            gauges["gpu_pcie_replay_errors"] = 3.0
            for name, value in gauges.items():
                rows.append(
                    _row(
                        service="iris-node-agent",
                        name=name,
                        value=value,
                        moment=moment,
                        seq=seq,
                        node_name=node,
                        attributes={"gpu_uuid": f"GPU-{node}-{gpu}", "gpu_index": gpu},
                    )
                )
    return rows


def _vllm_rows(moment: datetime, seq: int) -> list[tuple]:
    """The engine registry, split across the two namespaces a run's metrics can land in."""
    rows = []
    histograms = {
        "request_generation_tokens_bucket": GENERATION_TOKEN_BUCKETS,
        "iteration_tokens_total_bucket": GENERATION_TOKEN_BUCKETS,
        "time_to_first_token_seconds_bucket": LATENCY_BUCKETS,
        "inter_token_latency_seconds_bucket": LATENCY_BUCKETS,
        "request_queue_time_seconds_bucket": LATENCY_BUCKETS,
        "request_prefill_time_seconds_bucket": LATENCY_BUCKETS,
        "request_decode_time_seconds_bucket": LATENCY_BUCKETS,
        "e2e_request_latency_seconds_bucket": LATENCY_BUCKETS,
    }
    for engine in ("0", "1"):
        # Engine 0 is forwarded by the MarinSkyRL process under its own service name, as the
        # first instrumented run's engine rows actually were; engine 1 publishes its own
        # registry as service='vllm'. The panels have to read both.
        engine_service = "marinskyrl" if engine == "0" else "vllm"
        for name, buckets in histograms.items():
            for upper_bound, count in buckets.items():
                rows.append(
                    _row(
                        service=engine_service,
                        name=name,
                        value=count * (seq + 1),
                        moment=moment,
                        seq=seq,
                        run_id=RUN_ID,
                        node_name=NODES[1],
                        role="inference",
                        attributes={
                            "metric_source": "vllm",
                            "engine": engine,
                            "le": upper_bound,
                            "source_temporality": "cumulative_snapshot",
                        },
                    )
                )
        for name, value in (("queue_depth", 6.0), ("kv_cache_usage_perc", 0.42)):
            rows.append(
                _row(
                    service=engine_service,
                    name=name,
                    value=value,
                    moment=moment,
                    seq=seq,
                    run_id=RUN_ID,
                    node_name=NODES[1],
                    role="inference",
                    attributes={"metric_source": "vllm", "engine": engine},
                )
            )
        for reason, value in (("kv_cache", 3.0), ("scheduler", 1.0)):
            rows.append(
                _row(
                    service=engine_service,
                    name="num_requests_waiting_by_reason",
                    value=value,
                    moment=moment,
                    seq=seq,
                    run_id=RUN_ID,
                    node_name=NODES[1],
                    role="inference",
                    attributes={"metric_source": "vllm", "engine": engine, "reason": reason},
                )
            )
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
                service="marinskyrl",
                name="policy_step",
                value=float(bucket),
                moment=moment,
                seq=bucket,
                run_id=RUN_ID,
                node_name=NODES[0],
                role="trainer",
            )
        )
    return rows


def _store(clock: str) -> duckdb.DuckDBPyConnection:
    database = duckdb.connect()
    for stream in sorted(set(_SEMANTIC_STREAM.values())):
        database.execute(f'CREATE TABLE "{stream}"{_SCHEMA}')
    # finelog's SQL dialect, in the spellings the dashboards use.
    database.execute("CREATE MACRO to_timestamp_millis(value) AS to_timestamp(value / 1000.0)::TIMESTAMP")
    database.execute("CREATE MACRO date_bin(width, moment) AS time_bucket(width, moment)")
    database.execute("CREATE MACRO json_get(document, key) AS json_extract_string(document, '$.' || key)")
    database.execute("CREATE MACRO approx_percentile_cont(value, q) AS quantile_cont(value, q)")
    placeholders = ", ".join("?" for _ in _COLUMNS)
    service_index = _COLUMNS.index("service")
    routed: dict[str, list] = {}
    for row in _run_rows(clock):
        stream = _SEMANTIC_STREAM[row[service_index]]
        routed.setdefault(stream, []).append(row)
    for stream, stream_rows in routed.items():
        database.executemany(f'INSERT INTO "{stream}" VALUES ({placeholders})', stream_rows)
    return database


@pytest.fixture
def store() -> duckdb.DuckDBPyConnection:
    """A synchronised run: worker spans measure execution and ship as ``*_wall``."""
    return _store("wall")


@pytest.fixture
def launch_store() -> duckdb.DuckDBPyConnection:
    """An unsynchronised run: the same spans measure launch and ship as ``*_launch``."""
    return _store("launch")


# The panels this branch adds live in dashboards/panels/rl_*.json and are mounted by panelRef, so
# they can move between dashboards without their body moving. These tests follow the panel, not the
# dashboard: a fragment is the single source of truth for everything except id and gridPos.
def _stitched() -> dict:
    return stitch_all(DASHBOARDS, DASHBOARDS / "panels")


def _dashboard(name: str = "rl_policy_train.json") -> dict:
    return _stitched()[name]


def _rl_dashboards() -> dict:
    return {name: _stitched()[name] for name in ("rl_policy_train.json", "rl_runs.json")}


def _our_panels() -> list[dict]:
    """Every panel this branch owns: the shared fragments, plus rl_policy_train's own panels.

    rl_runs.json's other panels came from marin#8562 and are deliberately left alone -- holding
    them to a rule written after they shipped would only make that PR harder to rebase onto.
    """
    ours = {path.stem for path in (DASHBOARDS / "panels").glob("rl_*.json")}
    mounted = [
        panel
        for name, board in _rl_dashboards().items()
        for panel, source in zip(board["panels"], json.loads((DASHBOARDS / name).read_text())["panels"], strict=True)
        if source.get("panelRef") in ours or (name == "rl_policy_train.json" and panel["type"] != "row")
    ]
    assert len(mounted) >= len(ours), (len(mounted), len(ours))
    return mounted


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


def _panel_sql(title: str) -> str:
    """One panel's shipped SQL, with Grafana's macros resolved to this window.

    Searched across both RL dashboards, because a shared fragment has one body wherever it is
    mounted -- and asserted unique, so a copy-pasted second body cannot pass as the same panel.
    """
    matches = _all_panels(title)
    assert matches, f"no panel titled {title!r} on either RL dashboard"
    sql = {next(p["value"] for p in m["targets"][0]["url_options"]["params"] if p["key"] == "sql") for m in matches}
    assert len(sql) == 1, f"{title} is mounted twice with different SQL"
    return _resolve(sql.pop())


def test_the_run_variable_offers_the_run_the_trainer_reported(store) -> None:
    (variable,) = [v for v in _dashboard()["templating"]["list"] if v["name"] == "run"]
    (parameter,) = [
        param for param in variable["query"]["infinityQuery"]["url_options"]["params"] if param["key"] == "sql"
    ]

    assert store.execute(_resolve(parameter["value"])).fetchall() == [(RUN_ID,)]


def test_the_step_bands_are_exclusive_and_they_close_on_the_step(store) -> None:
    rows = store.execute(_panel_sql("Step composition — exclusive seconds per phase")).fetchall()

    bands = {series: seconds for _, series, seconds in rows}
    # Every phase gets a band, and it is the wall it did not spend inside a child. A parent banded
    # at its own wall would double-count: train_critic_and_policy would put 3806 s beside the
    # 3805.6 s of policy_train it contains, in a 4210 s step.
    assert bands["train_critic_and_policy"] == pytest.approx(DISPATCH_SECONDS)
    assert bands["policy_train"] == pytest.approx(DRIVER_PHASES["policy_train"])
    assert bands["unattributed"] == pytest.approx(UNATTRIBUTED)
    assert sum(bands.values()) == pytest.approx(STEP_SECONDS)


def test_the_generate_subtree_is_subtracted_from_generate_and_not_stacked_beside_it(store) -> None:
    """The tree grew a level under generate after this panel shipped, and a hardcoded exclusion
    list could not see it: rollout_collect alone is 97% of generate, so banding both put 162% of
    the phase on the stack with nothing to say so."""
    bands = {
        series: seconds
        for _, series, seconds in store.execute(_panel_sql("Step composition — exclusive seconds per phase")).fetchall()
    }

    # generate's own band is the orchestration it does outside its children, which is what the
    # producer publishes as generate_span_residual.
    assert bands["generate"] == pytest.approx(GENERATE_RESIDUAL)
    assert bands["rollout_collect"] == pytest.approx(
        GENERATE_CHILDREN["rollout_collect"] - GENERATE_GRANDCHILDREN["rollout_tokenize"][1]
    )
    assert bands["rollout_tokenize"] == pytest.approx(GENERATE_GRANDCHILDREN["rollout_tokenize"][1])
    # The whole subtree still sums to generate, two levels deep.
    subtree = ["generate", *GENERATE_CHILDREN, *GENERATE_GRANDCHILDREN]
    assert sum(bands[phase] for phase in subtree) == pytest.approx(DRIVER_PHASES["generate"])


def test_policy_train_share_reproduces_the_measured_ninety_percent(store) -> None:
    rows = store.execute(_panel_sql("policy_train share of the step")).fetchall()

    assert {round(share, 4) for _, share in rows} == {round(DRIVER_PHASES["policy_train"] / STEP_SECONDS, 4)}


def test_the_share_still_has_a_denominator_when_the_run_publishes_no_step_span(store) -> None:
    """No run has published a `step` span since the generate tree landed, so a panel that needs one
    is permanently blank. The step's direct children partition it, so their sum stands in."""
    store.execute(
        """DELETE FROM "telemetry_v1.marinskyrl"
           WHERE json_extract_string(attributes_json, '$.phase') = 'step'"""
    )
    rows = store.execute(_panel_sql("policy_train share of the step")).fetchall()

    direct_children = (
        DRIVER_PHASES["generate"]
        + CONTAINER_SECONDS
        + sum(
            DRIVER_PHASES[phase] for phase in ("convert_to_training_input", "fwd_logprobs_values_reward", "sync_weights")
        )
    )
    assert direct_children != pytest.approx(STEP_SECONDS), "the fixture no longer distinguishes the two denominators"
    assert {round(share, 6) for _, share in rows} == {round(DRIVER_PHASES["policy_train"] / direct_children, 6)}


def test_the_decomposition_reads_the_critical_rank_and_never_a_per_phase_maximum(store) -> None:
    rows = store.execute(_panel_sql("policy_ppo_train decomposition at the critical rank")).fetchall()

    bands = {series: seconds for _, series, seconds in rows}
    expected = dict(WORKER_SPANS[CRITICAL_RANK])
    expected["unattributed"] = PPO_TRAIN[CRITICAL_RANK] - sum(WORKER_SPANS[CRITICAL_RANK].values())
    assert bands == pytest.approx(expected)

    # Only spans that name policy_ppo_train as their parent are banded, so the two container
    # spellings and the publish cost drop out by construction rather than by a list of names.
    # policy_span_publish is exclusive, it is a worker row, and it belongs to another parent; a
    # rule that read the clock domain alone would band it and quietly shrink the residual by its
    # three seconds.
    assert CONTAINER_SPAN not in bands
    assert "policy_training_step" not in bands
    assert "policy_span_publish" not in bands
    assert bands["unattributed"] == pytest.approx(PPO_TRAIN[CRITICAL_RANK] - sum(WORKER_SPANS[CRITICAL_RANK].values()))
    # The producer's own residual is excluded and recomputed. Reading the published one would put
    # a -1949 s band in a 2000 s stack.
    assert "policy_span_residual" not in bands
    published = (
        PPO_TRAIN[CRITICAL_RANK]
        - sum(WORKER_SPANS[CRITICAL_RANK].values())
        - sum(WORKER_SPANS[CRITICAL_RANK][phase] for phase in CONTAINED_SPANS)
    )
    assert published < 0, "the fixture no longer reproduces the double-count"

    # The bands close on the parent they decompose. A per-phase maximum over the two ranks would
    # sum to 2645 s inside a 2000 s span, because the barrier and the compute come from different
    # ranks -- which is the failure this panel is built to avoid.
    assert sum(bands.values()) == pytest.approx(PPO_TRAIN[CRITICAL_RANK])
    per_phase_max = sum(max(WORKER_SPANS["0"][phase], WORKER_SPANS["1"][phase]) for phase in WORKER_SPANS["0"])
    assert per_phase_max > PPO_TRAIN[CRITICAL_RANK]

    # And it is the slow rank's row set, not the fast one's.
    assert bands["policy_entry_barrier"] == pytest.approx(WORKER_SPANS[CRITICAL_RANK]["policy_entry_barrier"])
    assert bands["policy_entry_barrier"] != pytest.approx(WORKER_SPANS["0"]["policy_entry_barrier"])


def test_the_skew_panel_reports_the_spread_and_names_the_same_slowest_rank(store) -> None:
    rows = store.execute(_panel_sql("Rank skew: policy_ppo_train across ranks")).fetchall()

    for _, slowest, _p95, _p50, fastest in rows:
        assert slowest == pytest.approx(PPO_TRAIN[CRITICAL_RANK])
        assert fastest == pytest.approx(min(PPO_TRAIN.values()))


def test_the_derived_ratios_divide_the_quantities_they_name(store) -> None:
    micro = store.execute(_panel_sql("policy_train ÷ micro-step count")).fetchall()
    for _, seconds_per_micro_step, micro_steps in micro:
        assert micro_steps == pytest.approx(64.0)
        assert seconds_per_micro_step == pytest.approx(DRIVER_PHASES["policy_train"] / 64.0)

    ratio = store.execute(_panel_sql("backward ÷ forward at the critical rank")).fetchall()
    expected = WORKER_SPANS[CRITICAL_RANK]["policy_backward"] / WORKER_SPANS[CRITICAL_RANK]["policy_forward"]
    assert [round(value, 6) for _, value in ratio] == [round(expected, 6)] * len(ratio)

    waiting = store.execute(_panel_sql("Waiting and collective share at the critical rank")).fetchall()
    barriers = sum(
        WORKER_SPANS[CRITICAL_RANK][phase]
        for phase in (
            "policy_entry_barrier",
            "policy_final_barrier",
            "policy_metric_allreduce",
            "policy_entropy_allreduce",
        )
    )
    assert [round(value, 6) for _, value in waiting] == [round(barriers / PPO_TRAIN[CRITICAL_RANK], 6)] * len(waiting)


def test_the_worker_panels_read_whichever_clock_the_sink_stamped(launch_store) -> None:
    """A run made without ``policy_train_spans_synchronize`` ships ``*_launch`` and nothing else.

    Naming ``exclusive_wall`` alone renders all four worker panels empty on every such run, which
    is indistinguishable from a producer that stopped publishing — and every arm of this campaign
    ran unsynchronised.
    """
    bands = {
        series: seconds
        for _, series, seconds in launch_store.execute(
            _panel_sql("policy_ppo_train decomposition at the critical rank")
        ).fetchall()
    }
    assert bands["policy_backward"] == pytest.approx(WORKER_SPANS[CRITICAL_RANK]["policy_backward"])
    assert sum(bands.values()) == pytest.approx(PPO_TRAIN[CRITICAL_RANK])

    skew = launch_store.execute(_panel_sql("Rank skew: policy_ppo_train across ranks")).fetchall()
    assert {round(slowest, 6) for _, slowest, _, _, _ in skew} == {round(PPO_TRAIN[CRITICAL_RANK], 6)}

    ratio = launch_store.execute(_panel_sql("backward ÷ forward at the critical rank")).fetchall()
    expected = WORKER_SPANS[CRITICAL_RANK]["policy_backward"] / WORKER_SPANS[CRITICAL_RANK]["policy_forward"]
    assert {round(value, 6) for _, value in ratio} == {round(expected, 6)}

    waiting = launch_store.execute(_panel_sql("Waiting and collective share at the critical rank")).fetchall()
    assert {value for _, value in waiting} != {None}


def test_padding_is_a_per_rank_ratio_rather_than_a_ratio_of_summed_tokens(store) -> None:
    rows = store.execute(_panel_sql("Padding waste and attention work")).fetchall()

    # Averaging the per-rank fractions (0.25 and 0.20) is unaffected by how the batch is sharded;
    # a ratio of summed tokens would not be.
    expected_padding = sum(
        1.0 - counters["rank_tokens_real"] / counters["rank_tokens_padded"] for counters in WORKER_COUNTERS.values()
    ) / len(WORKER_COUNTERS)
    expected_work = sum(counters["attention_work_ratio"] for counters in WORKER_COUNTERS.values()) / len(WORKER_COUNTERS)
    for _, padded_fraction, attention_work_ratio in rows:
        assert padded_fraction == pytest.approx(expected_padding)
        assert attention_work_ratio == pytest.approx(expected_work)


def test_the_padding_panel_reads_the_old_spelling_of_the_token_counters(store) -> None:
    """Runs from before the 2026-09-03 rename publish tokens_real and tokens_padded. Reading only
    the current spelling empties this panel across the whole back catalogue, and an empty padding
    panel reads as an unpadded batch."""
    fresh = store.execute(_panel_sql("Padding waste and attention work")).fetchall()
    store.execute(
        """UPDATE "telemetry_v1.marinskyrl"
           SET attributes_json = replace(attributes_json, 'rank_tokens_', 'tokens_')"""
    )
    renamed = store.execute(_panel_sql("Padding waste and attention work")).fetchall()

    assert [row[1] for row in renamed] == [pytest.approx(row[1]) for row in fresh]
    assert all(row[1] is not None for row in renamed)


def test_the_accelerator_panels_join_dcgm_to_the_run_through_its_nodes(store) -> None:
    sm = store.execute(_panel_sql("SM and tensor-pipe activity on this run's nodes")).fetchall()
    by_series = {series: value for _, series, value in sm}
    assert by_series["SM active"] == pytest.approx(SM_ACTIVE_RATIO * 100.0)
    assert by_series["tensor pipe active"] == pytest.approx(TENSOR_ACTIVE_RATIO * 100.0)

    memory = store.execute(_panel_sql("GPU memory in use on this run's nodes")).fetchall()
    assert [row[2] for row in memory] == [pytest.approx(GPU_MEMORY_USED)] * len(memory)

    fabric = store.execute(_panel_sql("NVLink against PCIe receive traffic")).fetchall()
    # Four GPUs across the run's two nodes, summed per direction.
    assert {series for _, series, _ in fabric} == {"NVLink receive", "PCIe receive"}
    assert {round(value) for _, series, value in fabric if series == "NVLink receive"} == {round(4 * NVLINK_RATE)}


def test_a_trainer_that_stops_stamping_node_name_blanks_the_accelerator_panels(store) -> None:
    # The DCGM rows carry no run identity, so an identity regression in the producer reads as an
    # idle fleet rather than as a broken join.
    store.execute('UPDATE "telemetry_v1.marinskyrl" SET node_name = NULL')

    assert store.execute(_panel_sql("SM and tensor-pipe activity on this run's nodes")).fetchall() == []


def test_the_fault_table_differences_the_counters_and_hides_healthy_gpus(store) -> None:
    rows = store.execute(_panel_sql("Link faults and power on this run's GPUs")).fetchall()

    # One GPU is degraded; the other three have flat counters and must not appear.
    assert len(rows) == 1
    node, gpu, peak_power, nvlink_increase, pcie_increase = rows[0]
    assert (node, gpu) == (NODES[1], f"GPU-{NODES[1]}-1")
    assert peak_power == pytest.approx(620.0)
    assert nvlink_increase == pytest.approx(7.0 * (BUCKETS - 1))
    assert pcie_increase == pytest.approx(0.0)


def test_the_engine_histograms_interpolate_quantiles_from_cumulative_buckets(store) -> None:
    rows = store.execute(_panel_sql("Generated tokens per request")).fetchall()

    by_series = {series: value for _, series, value in rows}
    # Counts are cumulative in `le`: 50 of 100 requests are at or below 256 tokens, 90 at or
    # below 1024, 99 at or below 4096.
    assert by_series["generated tokens · p50"] == pytest.approx(256.0)
    assert by_series["generated tokens · p90"] == pytest.approx(1024.0)
    assert by_series["generated tokens · p99"] == pytest.approx(4096.0)
    # The first sample of a cumulative series has nothing to difference against and drops out.
    assert len({t for t, _, _ in rows}) == BUCKETS - 1

    stages = store.execute(_panel_sql("Request latency by stage")).fetchall()
    assert {series for _, series, _ in stages} == {
        f"{stage} · {quantile}" for stage in ("queue", "decode", "end to end") for quantile in ("p50", "p99")
    }

    tokens = store.execute(_panel_sql("Time to first token and inter-token latency")).fetchall()
    assert {series.split(" · ")[0] for _, series, _ in tokens} == {
        "time to first token",
        "inter-token latency",
    }

    iteration = store.execute(_panel_sql("Tokens per engine iteration")).fetchall()
    assert {series for _, series, _ in iteration} == {
        "iteration tokens · p50",
        "iteration tokens · p90",
        "iteration tokens · p99",
    }


def test_a_counter_reset_drops_the_sample_rather_than_reading_as_a_giant_delta(store) -> None:
    # An engine that restarts republishes its histogram from zero. Clamping the negative step to
    # zero would keep the sample and understate the bucket; the panel drops it.
    for stream in ("telemetry_v1.vllm", "telemetry_v1.marinskyrl"):
        store.execute(
            f"""UPDATE "{stream}" SET value = 1.0
                WHERE name = 'request_generation_tokens_bucket' AND seq >= 3"""
        )
    rows = store.execute(_panel_sql("Generated tokens per request")).fetchall()

    # Buckets 1 and 2 still difference cleanly; 3 is the reset and 4-5 are flat at 1.0, so no
    # quantile survives there.
    assert len({t for t, _, value in rows if value is not None}) == 2


def test_engine_rows_are_read_from_whichever_namespace_the_run_wrote_them_to(store) -> None:
    # An RL run's engine metrics are forwarded by the MarinSkyRL process under its own service
    # name, so they land in telemetry_v1.marinskyrl rather than telemetry_v1.vllm. Reading only
    # the latter renders every engine panel blank for exactly the runs this dashboard is for.
    both = store.execute(_panel_sql("Generated tokens per request")).fetchall()
    assert both

    store.execute('DELETE FROM "telemetry_v1.vllm"')
    marinskyrl_only = store.execute(_panel_sql("Generated tokens per request")).fetchall()

    assert {series for _, series, _ in marinskyrl_only} == {series for _, series, _ in both}
    assert {round(v, 6) for _, _, v in marinskyrl_only} == {round(v, 6) for _, _, v in both}


def test_the_engine_gauges_are_averaged_and_never_differenced(store) -> None:
    queue = store.execute(_panel_sql("Engine queue depth and why requests are waiting")).fetchall()
    by_series = {series: value for _, series, value in queue}
    assert by_series["queue depth"] == pytest.approx(6.0)
    assert by_series["waiting · kv_cache"] == pytest.approx(3.0)
    assert by_series["waiting · scheduler"] == pytest.approx(1.0)

    cache = store.execute(_panel_sql("KV-cache utilisation")).fetchall()
    assert [(row[1], row[2]) for row in cache] == [(pytest.approx(0.42), pytest.approx(0.42))] * len(cache)


def test_every_timeseries_panel_declares_the_columns_its_sql_returns() -> None:
    """A panel is read through its declared columns, so executing its SQL cannot see a mistake there."""
    for panel in (p for board in _rl_dashboards().values() for p in board["panels"]):
        if panel.get("type") != "timeseries":
            continue
        for target in panel["targets"]:
            (parameter,) = [param for param in target["url_options"]["params"] if param["key"] == "sql"]
            final_select = re.split(r"\bSELECT\b", parameter["value"])[-1]
            selected = {alias for alias in re.findall(r"\bAS (\w+)", final_select) if not alias.isupper()}
            declared = {column["selector"]: column["type"] for column in target["columns"]}

            assert (
                set(declared) == selected
            ), f"{panel['title']}: declares {sorted(declared)}, SQL returns {sorted(selected)}"
            assert "number" in declared.values(), f"{panel['title']}: no numeric column to plot"


def test_the_hover_text_stays_short_enough_to_be_read() -> None:
    """A description nobody reads is a description that is not there. Fifty words is about four
    lines in the tooltip; past that the trap at the end is the part that gets skipped."""
    for panel in _our_panels():
        for field, cap in (("description", 50), ("noValue", 40)):
            text = panel.get(field) or panel["fieldConfig"]["defaults"].get(field, "")
            assert len(text.split()) <= cap, f"{panel['title']}: {field} is {len(text.split())} words"


def test_every_panel_says_on_its_face_why_it_would_be_blank() -> None:
    """An empty panel and a broken producer render identically, and half of these panels are empty
    on a run made by a build that predates their series. That distinction belongs on the panel face,
    not behind a description hover."""
    for panel in _our_panels():
        assert panel["fieldConfig"]["defaults"].get("noValue"), panel["title"]


def test_every_rank_fills_the_grid_on_both_rl_dashboards() -> None:
    """A rank with a gap floats the next panel up into a section that does not ask its question."""
    for name, board in _rl_dashboards().items():
        widths: dict[int, int] = {}
        for panel in board["panels"]:
            if panel["type"] == "row":
                continue
            widths[panel["gridPos"]["y"]] = widths.get(panel["gridPos"]["y"], 0) + panel["gridPos"]["w"]
        assert set(widths.values()) == {24}, (name, widths)
        rows = [panel["gridPos"]["y"] for panel in board["panels"] if panel["type"] == "row"]
        assert rows == sorted(rows), name


def test_the_training_rows_are_questions() -> None:
    """Each row states the question it answers, in the order an engineer needs them."""
    rows = [panel["title"] for panel in _dashboard()["panels"] if panel["type"] == "row"]
    assert rows == [
        "Why is the training step slow?",
        "Were the accelerators doing arithmetic, or waiting?",
        "Is the engine slow, or is nothing reaching it?",
    ]


def test_the_new_panels_answer_the_question_of_the_row_they_sit_in() -> None:
    """Panels answering "can I believe this" sit above panels answering "what exactly is wrong".

    The fragments mount into marin#8562's five questions rather than into a taxonomy of their own,
    which is both the right reading order and the smallest possible diff to that dashboard.
    """
    board = _dashboard("rl_runs.json")
    rows = [panel for panel in board["panels"] if panel["type"] == "row"]
    section = {}
    for panel in board["panels"]:
        if panel["type"] == "row":
            continue
        owner = max(
            (row for row in rows if row["gridPos"]["y"] < panel["gridPos"]["y"]), key=lambda row: row["gridPos"]["y"]
        )
        section[panel["title"]] = owner["title"]

    assert section["Span coverage: clock, ranks, truncated steps"] == "Alive and moving?"
    assert section["How the run ended, and whether telemetry kept up"] == "Alive and moving?"
    assert section["Step composition — exclusive seconds per phase"] == "Where did the wall clock go?"
    assert section["policy_train share of the step"] == "Where did the wall clock go?"
    assert section["Signed span residuals — both trees"] == "Where did the wall clock go?"
    assert section["Inside generate — where the fan-out goes"] == "Why is generation slow?"
    assert section["A trajectory's wait: the engine against the environment"] == "Why is generation slow?"
    assert section["Is the environment slow, or the loop around it?"] == "Why is generation slow?"

    # The run picker only offers runs that reported a policy_step, so a lead panel going blank is
    # always a true alarm rather than a filter that did not match.
    (variable,) = [v for v in board["templating"]["list"] if v["name"] == "run"]
    (parameter,) = [
        param for param in variable["query"]["infinityQuery"]["url_options"]["params"] if param["key"] == "sql"
    ]
    assert "'policy_step'" in parameter["value"]


def test_every_panel_has_a_distinct_title_id_and_slot() -> None:
    """id and gridPos are dashboard-local -- the two things a panelRef legitimately varies."""
    for name, board in _rl_dashboards().items():
        panels = board["panels"]
        titles = [panel["title"] for panel in panels]
        assert len(titles) == len(set(titles)), (name, titles)
        ids = [panel["id"] for panel in panels]
        assert len(ids) == len(set(ids)), (name, ids)
        slots = [(panel["gridPos"]["x"], panel["gridPos"]["y"]) for panel in panels]
        assert len(slots) == len(set(slots)), (name, slots)


def test_a_span_query_names_one_sink_or_keeps_the_two_apart() -> None:
    """The two sinks publish different clock domains, and summing across them double-counts.

    A worker span is exclusive of its siblings and a driver span is inclusive of everything it
    contains, so a query pins one sink. The audit table is the one query that has to read both, and
    it may only do so by returning the role as a column, so the two never merge into one number.
    """
    for panel in (p for board in _rl_dashboards().values() for p in board["panels"]):
        for target in panel.get("targets", []):
            (parameter,) = [param for param in target["url_options"]["params"] if param["key"] == "sql"]
            sql = parameter["value"]
            if "phase_duration_seconds" not in sql:
                continue
            if "'trainer'" in sql or "'worker'" in sql:
                assert ("'trainer'" in sql) != ("'worker'" in sql), panel["title"]
            elif "'critical_path'" in sql:
                # WorkerTimingSink._clock_domain can only produce {inclusive,exclusive}_{wall,launch},
                # so a query pinned to the critical-path domain is driver-only by construction.
                continue
            else:
                assert "AS role" in sql, panel["title"]


def test_the_vitals_table_names_the_clock_domain_the_ranks_and_the_truncated_steps(store) -> None:
    """Three things decide whether anything below can be read, and all three are invisible in a
    duration: which clock the worker sink stamped, whether any worker reported at all, and whether a
    step ended in a failure -- a truncated step renders exactly like a fast one."""
    rows = store.execute(_panel_sql("Span coverage: clock, ranks, truncated steps")).fetchall()

    by_sink = {(role, clock): (ranks, steps, failed) for role, clock, ranks, steps, failed in rows}
    assert by_sink[("worker", "exclusive_wall")][0] == len(WORKER_SPANS)
    assert by_sink[("trainer", "inclusive_wall")][0] == 0, "driver rows carry no rank"
    assert by_sink[("trainer", "critical_path")] == (0, BUCKETS, 1)
    # Nothing but the critical-path rows carries an outcome, so nothing else may report a failure.
    assert {failed for (_, clock), (_, _, failed) in by_sink.items() if clock != "critical_path"} == {0}


def test_the_vitals_table_shows_a_run_that_stamped_two_clock_domains_as_two_rows(store) -> None:
    """No run publishes both today, because the synchronise flag is fixed for its lifetime. If one
    ever does, the panels below would average execution time against launch time into one series,
    and this table is where that becomes visible rather than a number that quietly moved."""
    store.execute(
        """UPDATE "telemetry_v1.marinskyrl"
           SET attributes_json = replace(attributes_json, 'exclusive_wall', 'exclusive_launch')
           WHERE seq >= 3 AND json_extract_string(attributes_json, '$.role') = 'worker'"""
    )
    rows = store.execute(_panel_sql("Span coverage: clock, ranks, truncated steps")).fetchall()

    worker_clocks = {clock for role, clock, *_ in rows if role == "worker"}
    assert worker_clocks == {"exclusive_wall", "exclusive_launch", "inclusive_wall"}


def test_the_residual_panel_reports_both_trees_signed(store) -> None:
    (panel,) = _all_panels("Signed span residuals — both trees")
    targets = panel["targets"]
    queries = [
        _resolve(next(p["value"] for p in target["url_options"]["params"] if p["key"] == "sql")) for target in targets
    ]

    driver = store.execute(queries[0]).fetchall()
    assert {round(value, 6) for _, value in driver} == {round(GENERATE_RESIDUAL, 6)}

    worker = store.execute(queries[1]).fetchall()
    published = (
        PPO_TRAIN[CRITICAL_RANK]
        - sum(WORKER_SPANS[CRITICAL_RANK].values())
        - sum(WORKER_SPANS[CRITICAL_RANK][phase] for phase in CONTAINED_SPANS)
    )
    assert published < 0, "the fixture no longer reproduces the double-count"
    # Signed, and read from r*. Clamping it at zero would retire the one series that can report a
    # child being counted inside its parent.
    assert {round(value, 6) for _, value in worker} == {round(published, 6)}


def test_the_generate_shares_partition_the_phase(store) -> None:
    rows = store.execute(_panel_sql("Inside generate — where the fan-out goes")).fetchall()

    shares = {series: value for _, series, value in rows}
    assert shares["rollout_collect"] == pytest.approx(GENERATE_CHILDREN["rollout_collect"] / DRIVER_PHASES["generate"])
    # The grandchildren belong to their own parents' walls, not to generate's.
    assert set(shares) == {*GENERATE_CHILDREN, "unaccounted"}
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["unaccounted"] == pytest.approx(GENERATE_RESIDUAL / DRIVER_PHASES["generate"])


def test_the_generate_shares_are_blank_rather_than_a_single_full_band_without_the_subtree(store) -> None:
    """156 of the 167 runs in finelog measure generate as one wall. Reporting 100% unaccounted for
    those would read as a defect in generate rather than as an absent instrument."""
    store.execute(
        """DELETE FROM "telemetry_v1.marinskyrl"
           WHERE json_extract_string(attributes_json, '$.parent') = 'generate'"""
    )

    assert store.execute(_panel_sql("Inside generate — where the fan-out goes")).fetchall() == []


def test_the_rollout_waits_are_divided_by_the_trajectory_count(store) -> None:
    rows = store.execute(_panel_sql("A trajectory's wait: the engine against the environment")).fetchall()

    for _, engine, environment, slowest in rows:
        assert engine == pytest.approx(ENGINE_AWAIT_SUM / TRAJECTORIES)
        assert environment == pytest.approx(sum(ENV_SPLIT.values()) / TRAJECTORIES)
        assert slowest == pytest.approx(ENGINE_AWAIT_MAX)


def test_no_panel_plots_a_concurrent_await_sum_undivided(store) -> None:
    """rollout_*_seconds_sum is a sum over up to 4,096 coroutines. It exceeds its own parent by
    design -- 23,656 s against a 4,210 s step in this fixture -- so any panel reading these counters
    has to divide before plotting. Banding one is the single easiest way for this dashboard to
    publish a number nobody should believe."""
    readers = [
        panel
        for board in _rl_dashboards().values()
        for panel in board["panels"]
        for target in panel.get("targets", [])
        for param in target["url_options"]["params"]
        if param["key"] == "sql" and "rollout_wait_seconds" in param["value"]
    ]
    assert len(readers) == 2, [panel["title"] for panel in readers]
    assert ENGINE_AWAIT_SUM > STEP_SECONDS, "the fixture no longer makes the raw sum implausible"

    for panel in readers:
        for row in store.execute(_panel_sql(panel["title"])).fetchall():
            plotted = [cell for cell in row[1:] if isinstance(cell, float)]
            assert plotted, panel["title"]
            assert max(plotted) < STEP_SECONDS, f"{panel['title']} plots {max(plotted)}"


def test_the_environment_split_is_a_partition_with_an_audit_band(store) -> None:
    rows = store.execute(_panel_sql("Is the environment slow, or the loop around it?")).fetchall()

    shares = {series: value for _, series, value in rows}
    awaited = sum(ENV_SPLIT.values())
    assert shares["resuming on the event loop"] == pytest.approx(ENV_SPLIT["resume"] / awaited)
    assert shares["running the environment"] == pytest.approx(ENV_SPLIT["exec"] / awaited)
    # The producer states the three terms partition the wait exactly, so the audit band is zero
    # until they stop doing so.
    assert shares["unaccounted"] == pytest.approx(0.0)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_memory_is_the_worst_rank_and_allocator_events_are_the_run_total(store) -> None:
    rows = store.execute(_panel_sql("Allocator pressure and peak memory on the worst rank")).fetchall()

    for _, reserved, allocated, retries, ooms in rows:
        # The binding constraint on the micro-batch is the rank that used most, never the mean.
        assert reserved == pytest.approx(max(m["peak_reserved_bytes"] for m in WORKER_MEMORY.values()))
        assert allocated == pytest.approx(max(m["peak_allocated_bytes"] for m in WORKER_MEMORY.values()))
        assert retries == pytest.approx(sum(a["alloc_retries"] for a in WORKER_ALLOCATOR.values()))
        assert ooms == pytest.approx(0.0)


def test_the_memory_panel_reads_the_instrument_the_byte_gauges_moved_to(store) -> None:
    """peak_allocated_bytes and peak_reserved_bytes were split off policy_train_count onto
    policy_train_bytes on 2026-09-03. Naming either instrument alone empties the series on half the
    runs, and an empty memory series reads as headroom."""
    store.execute(
        """UPDATE "telemetry_v1.marinskyrl" SET name = 'policy_train_count'
           WHERE name = 'policy_train_bytes'"""
    )
    rows = store.execute(_panel_sql("Allocator pressure and peak memory on the worst rank")).fetchall()

    assert all(row[1] is not None for row in rows)
    assert rows[0][1] == pytest.approx(max(m["peak_reserved_bytes"] for m in WORKER_MEMORY.values()))
