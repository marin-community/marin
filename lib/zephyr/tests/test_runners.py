# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pluggable StageRunner strategies (zephyr.runners)."""

from __future__ import annotations

import os
import uuid

import pytest
from fray import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext, ZephyrWorkerError
from zephyr.runners import InlineRunner, SubprocessRunner


def _ctx(local_client, tmp_path, *, stage_runner_factory) -> ZephyrContext:
    return ZephyrContext(
        client=local_client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"test-runner-{uuid.uuid4().hex[:8]}",
        stage_runner_factory=stage_runner_factory,
    )


@pytest.fixture(params=[InlineRunner, SubprocessRunner], ids=["inline", "subprocess"])
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
        counters.increment("docs", 1)
        counters.increment("doubled_sum", x * 2)
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
    "runner_cls",
    [
        pytest.param(InlineRunner, id="inline"),
        pytest.param(
            SubprocessRunner,
            id="subprocess",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="subprocess gives each shard a unique PID; strict=True catches silent fallback to inline.",
            ),
        ),
    ],
)
def test_runner_parametrization_isolates_processes(local_client, tmp_path, runner_cls):
    """Regression guard that subprocess parametrization actually spawns subprocesses.

    Inline reuses the worker actor (≤ max_workers PIDs); subprocess gets one PID per shard.
    """

    def record_pid(x: int) -> int:
        counters.increment(f"shard_pid_{os.getpid()}", 1)
        return x

    ctx = _ctx(local_client, tmp_path, stage_runner_factory=runner_cls)
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
        import os

        os._exit(139)

    ctx = _ctx(local_client, tmp_path, stage_runner_factory=SubprocessRunner)
    try:
        ds = Dataset.from_list([0]).map(crash)
        with pytest.raises(ZephyrWorkerError) as exc_info:
            ctx.execute(ds)
    finally:
        ctx.shutdown()

    rendered = str(exc_info.value)
    assert "Shard 0" in rendered
    assert "exited with code 139" in rendered or "failed" in rendered
