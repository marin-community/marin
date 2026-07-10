# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pluggable StageRunner strategies (zephyr.runners)."""

import os
import uuid
from contextlib import suppress

import pytest
from finelog.client import LogClient
from finelog.embedded import EmbeddedServer
from fray.types import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext, ZephyrWorkerError
from zephyr.runners import InlineRunner, SubprocessRunner
from zephyr.stats import (
    ZEPHYR_STAGE_STATS_NAMESPACE,
    ZEPHYR_WORKER_STATS_NAMESPACE,
    StatsWriter,
)


def _ctx(local_client, tmp_path, *, stage_runner_factory) -> ZephyrContext:
    return ZephyrContext(
        client=local_client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"test-runner-{uuid.uuid4().hex[:8]}",
        stage_runner_factory=stage_runner_factory,
    )


@pytest.fixture(
    params=[
        pytest.param(lambda: InlineRunner(), id="inline"),
        pytest.param(lambda: SubprocessRunner(), id="subprocess"),
    ]
)
def runner_factory(request):
    """Run each test against both shipped runners."""
    return request.param


def test_simple_map(local_client, tmp_path, runner_factory):
    """Both runners produce correct results for a basic map pipeline."""
    ctx = _ctx(local_client, tmp_path, stage_runner_factory=runner_factory)
    try:
        ds = Dataset.from_list([1, 2, 3, 4, 5]).map(lambda x: x * 3)
        results = ctx.execute(ds).results
    finally:
        ctx.shutdown()
    assert sorted(results) == [3, 6, 9, 12, 15]


def test_user_counters_propagate(local_client, tmp_path, runner_factory):
    """User counters flow back from the worker (or its subprocess child) to the coordinator."""

    def increment(x: int) -> int:
        counters.pipeline.update_counter("docs", 1)
        counters.pipeline.update_counter("doubled_sum", x * 2)
        return x

    ctx = _ctx(local_client, tmp_path, stage_runner_factory=runner_factory)
    try:
        ds = Dataset.from_list([1, 2, 3, 4, 5]).map(increment)
        outcome = ctx.execute(ds)
    finally:
        ctx.shutdown()
    assert sorted(outcome.results) == [1, 2, 3, 4, 5]
    assert outcome.counters.get("docs") == 5
    assert outcome.counters.get("doubled_sum") == 30


def test_exception_preserves_user_frame(local_client, tmp_path, runner_factory):
    """A Python exception in user code surfaces with the user's frame visible.

    Inline path raises directly; subprocess path attaches a ``__notes__``
    breadcrumb so the user frame survives cloudpickling.
    """

    def buggy(_: int) -> int:
        empty: tuple = ()
        return empty[0]

    ctx = _ctx(local_client, tmp_path, stage_runner_factory=runner_factory)
    try:
        ds = Dataset.from_list([0]).map(buggy)
        with pytest.raises(ZephyrWorkerError) as exc_info:
            ctx.execute(ds)
    finally:
        ctx.shutdown()

    chained = ""
    cur: BaseException | None = exc_info.value
    while cur is not None:
        chained += str(cur) + "".join(getattr(cur, "__notes__", []))
        cur = cur.__cause__ or cur.__context__
    assert "buggy" in chained or "tuple index out of range" in chained, chained


@pytest.mark.parametrize(
    "runner_factory_fn",
    [
        pytest.param(lambda: InlineRunner(), id="inline"),
        pytest.param(
            lambda: SubprocessRunner(),
            id="subprocess",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="subprocess gives each shard a unique PID; strict=True catches silent fallback to inline.",
            ),
        ),
    ],
)
def test_runner_parametrization_isolates_processes(local_client, tmp_path, runner_factory_fn):
    """Regression guard that subprocess parametrization actually spawns subprocesses.

    Inline reuses the worker actor (≤ max_workers PIDs); subprocess gets one PID per shard.
    """

    def record_pid(x: int) -> int:
        counters.pipeline.update_counter(f"shard_pid_{os.getpid()}", 1)
        return x

    ctx = _ctx(local_client, tmp_path, stage_runner_factory=runner_factory_fn)
    try:
        ds = Dataset.from_list(list(range(5))).map(record_pid)
        outcome = ctx.execute(ds)
    finally:
        ctx.shutdown()

    pid_counters = {k: v for k, v in outcome.counters.items() if k.startswith("shard_pid_")}
    assert sum(pid_counters.values()) == 5, pid_counters
    assert len(pid_counters) <= 2, pid_counters


def test_subprocess_runner_isolates_native_crash(local_client, tmp_path):
    """A native abort in one shard surfaces as a deterministic TASK error.

    Forcibly terminate the child via ``os._exit(139)`` (the exit code SIGSEGV
    would produce); the inline runner has no way to recover from that, but
    the subprocess runner sees ``returncode != 0`` and routes to
    ``report_error`` so the pipeline aborts cleanly after MAX_SHARD_FAILURES.
    """

    def crash(_: int) -> int:
        os._exit(139)

    ctx = _ctx(local_client, tmp_path, stage_runner_factory=lambda: SubprocessRunner())
    try:
        ds = Dataset.from_list([0]).map(crash)
        with pytest.raises(ZephyrWorkerError) as exc_info:
            ctx.execute(ds)
    finally:
        ctx.shutdown()

    rendered = str(exc_info.value)
    assert "Shard 0" in rendered
    assert "exited with code 139" in rendered or "failed" in rendered


@pytest.fixture()
def finelog_server(tmp_path):
    """Start an embedded finelog server and yield its URL."""
    server = EmbeddedServer(log_dir=str(tmp_path / "finelog"))
    yield f"http://127.0.0.1:{server.port}"
    server.stop()


def test_finelog_stats_emitted(local_client, tmp_path, finelog_server, monkeypatch):
    """Pipeline emits rows to both zephyr.stage and zephyr.worker finelog tables."""
    writers: list[StatsWriter] = []

    def make_writer(url: str | None = None) -> StatsWriter:
        w = StatsWriter(LogClient.connect(finelog_server))
        writers.append(w)
        return w

    monkeypatch.setattr(StatsWriter, "connect", staticmethod(make_writer))

    ctx = _ctx(local_client, tmp_path, stage_runner_factory=lambda: InlineRunner())
    try:
        ds = Dataset.from_list(list(range(10))).map(lambda x: x)
        ctx.execute(ds)
    finally:
        ctx.shutdown()

    # ctx.shutdown() closes the coordinator's writer; close any runner writers too.
    for w in writers:
        with suppress(Exception):
            w.close()

    query_client = LogClient.connect(finelog_server)
    try:
        stage_rows = query_client.query(f'SELECT * FROM "{ZEPHYR_STAGE_STATS_NAMESPACE}"')
        worker_rows = query_client.query(f'SELECT * FROM "{ZEPHYR_WORKER_STATS_NAMESPACE}"')
    finally:
        query_client.close()

    assert stage_rows.num_rows >= 1, "Expected stage stat rows, got none"
    assert worker_rows.num_rows >= 1, "Expected worker stat rows, got none"

    stage_names = stage_rows.column("stage_name").to_pylist()
    assert any("map" in s.lower() for s in stage_names), f"No map stage in {stage_names}"

    # ---- stage stat correctness ----
    total_items = sum(stage_rows.column("items").to_pylist())
    assert total_items >= 10, f"Expected >= 10 items across stage rows, got {total_items}"

    total_bytes = sum(stage_rows.column("bytes_processed").to_pylist())
    assert total_bytes > 0, f"Expected non-zero bytes_processed in stage rows, got {total_bytes}"

    statuses = stage_rows.column("status").to_pylist()
    assert all(s == "END" for s in statuses), f"Unexpected stage statuses: {statuses}"

    elapsed_values = stage_rows.column("elapsed").to_pylist()
    assert all(e >= 0 for e in elapsed_values), f"Negative elapsed in stage rows: {elapsed_values}"

    item_rates = stage_rows.column("item_rate").to_pylist()
    assert all(r >= 0 for r in item_rates), f"Negative item_rate in stage rows: {item_rates}"

    byte_rates = stage_rows.column("byte_rate").to_pylist()
    assert all(r >= 0 for r in byte_rates), f"Negative byte_rate in stage rows: {byte_rates}"

    total_shards_values = stage_rows.column("total_shards").to_pylist()
    assert all(n > 0 for n in total_shards_values), f"total_shards <= 0: {total_shards_values}"

    # Resource counters are >= 0; they may be 0 if the sampler never fired.
    assert all(v >= 0 for v in stage_rows.column("cpu_pct_avg").to_pylist())
    assert all(v >= 0 for v in stage_rows.column("cpu_time_total").to_pylist())
    assert all(v >= 0 for v in stage_rows.column("mem_bytes_avg").to_pylist())
    assert all(v >= 0 for v in stage_rows.column("mem_peak_bytes_max").to_pylist())

    # ---- worker stat correctness ----
    worker_statuses = worker_rows.column("status").to_pylist()
    assert "START" in worker_statuses, f"No START worker rows: {worker_statuses}"
    assert "END" in worker_statuses, f"No END worker rows: {worker_statuses}"

    shard_indices = worker_rows.column("shard_idx").to_pylist()
    assert all(i >= 0 for i in shard_indices), f"Negative shard_idx: {shard_indices}"

    # END rows must have processed all items (sum across shards = 10) and
    # non-zero byte counts.
    end_mask = [s == "END" for s in worker_statuses]
    end_items = [v for v, m in zip(worker_rows.column("items").to_pylist(), end_mask, strict=True) if m]
    assert sum(end_items) >= 10, f"END rows account for fewer than 10 items: {end_items}"
    end_bytes = [v for v, m in zip(worker_rows.column("bytes_processed").to_pylist(), end_mask, strict=True) if m]
    assert sum(end_bytes) > 0, f"END rows have zero bytes_processed: {end_bytes}"

    # Resource counters: final sample fires in the execute() finally block, so
    # END rows must have positive memory readings.
    end_mem_current = [
        v for v, m in zip(worker_rows.column("mem_current_bytes").to_pylist(), end_mask, strict=True) if m
    ]
    assert all(v > 0 for v in end_mem_current), f"END rows have non-positive mem_current_bytes: {end_mem_current}"
    end_mem_peak = [v for v, m in zip(worker_rows.column("mem_peak_bytes").to_pylist(), end_mask, strict=True) if m]
    assert all(v > 0 for v in end_mem_peak), f"END rows have non-positive mem_peak_bytes: {end_mem_peak}"
    end_cpu_time = [v for v, m in zip(worker_rows.column("cpu_time_total").to_pylist(), end_mask, strict=True) if m]
    assert all(v >= 0 for v in end_cpu_time), f"Negative cpu_time_total in END rows: {end_cpu_time}"
    end_cpu_avg_pct = [v for v, m in zip(worker_rows.column("cpu_avg_pct").to_pylist(), end_mask, strict=True) if m]
    assert all(v >= 0 for v in end_cpu_avg_pct)
    end_mem_avg_bytes = [v for v, m in zip(worker_rows.column("mem_avg_bytes").to_pylist(), end_mask, strict=True) if m]
    assert all(v >= 0 for v in end_mem_avg_bytes)

    # ---- cross-validation: stage aggregates must match worker END rows ----
    # Single-stage pipeline: exactly one stage row expected.
    assert stage_rows.num_rows == 1, f"Expected 1 stage row, got {stage_rows.num_rows}"
    stage = {col: stage_rows.column(col).to_pylist()[0] for col in stage_rows.column_names}

    # Items and bytes are SUM across shards.
    assert sum(end_items) == stage["items"], f"sum(worker END items)={sum(end_items)} != stage items={stage['items']}"
    assert (
        sum(end_bytes) == stage["bytes_processed"]
    ), f"sum(worker END bytes)={sum(end_bytes)} != stage bytes_processed={stage['bytes_processed']}"

    # Memory peak is MAX across shards.
    assert (
        max(end_mem_peak) == stage["mem_peak_bytes_max"]
    ), f"max(worker END mem_peak_bytes)={max(end_mem_peak)} != stage mem_peak_bytes_max={stage['mem_peak_bytes_max']}"

    # CPU time is SUM across shards (float arithmetic, allow tiny rounding).
    assert sum(end_cpu_time) == pytest.approx(
        stage["cpu_time_total"], rel=1e-4
    ), f"sum(worker END cpu_time_total)={sum(end_cpu_time)} != stage cpu_time_total={stage['cpu_time_total']}"

    # Average CPU% and average RSS are AVERAGE (mean of per-shard END values).
    n_end = len(end_cpu_avg_pct)
    assert stage["cpu_pct_avg"] == pytest.approx(
        sum(end_cpu_avg_pct) / n_end, rel=1e-4
    ), f"stage cpu_pct_avg={stage['cpu_pct_avg']} != mean(worker END cpu_avg_pct)={sum(end_cpu_avg_pct) / n_end}"
    assert stage["mem_bytes_avg"] == pytest.approx(
        sum(end_mem_avg_bytes) / n_end, rel=1e-4
    ), f"stage mem_bytes_avg={stage['mem_bytes_avg']} != mean(worker END mem_avg_bytes)={sum(end_mem_avg_bytes) / n_end}"
