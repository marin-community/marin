# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared (long-lived) ZephyrContext.

A shared context starts one coordinator + worker pool via ``start()`` and
serves multiple pipelines concurrently; connecting contexts submit pipelines
through ``coordinator_endpoint``. These tests use ``LocalClient`` end-to-end so
they exercise the real ``start()`` → ``execute()`` → ``shutdown()`` path (the
coordinator runs in a background job thread, workers are in-process actors).
"""

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import _TEST_TASK_COST, _make_test_coordinator
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import (
    ZEPHYR_COORDINATOR_ENDPOINT_ENV,
    ZephyrContext,
    ZephyrWorkerError,
    _PipelineExecution,
)
from zephyr.shuffle import ListShard
from zephyr.stage_io import ShardTask


def _count_items(x):
    counters.pipeline.update_counter("items", 1)
    return x


@pytest.fixture
def shared_ctx(tmp_path):
    """A started shared context (2 workers) and its own LocalClient.

    Function-scoped and torn down after each test so the serve job's thread
    is released and no coordinator lingers between tests.
    """
    client = LocalClient(max_threads=8)
    ctx = ZephyrContext(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"shared-{uuid.uuid4().hex[:8]}",
    )
    ctx.start()
    yield ctx
    ctx.shutdown()
    client.shutdown(wait=True)


def _connect(shared_ctx: ZephyrContext) -> ZephyrContext:
    """A separate driver connected to the shared coordinator (models a step)."""
    return ZephyrContext(
        client=shared_ctx.client,
        resources=ResourceConfig(cpu=1, ram="512m"),
        coordinator_endpoint=shared_ctx.coordinator_endpoint,
        name=f"driver-{uuid.uuid4().hex[:8]}",
    )


def test_shared_context_runs_concurrent_pipelines_with_isolated_results(shared_ctx):
    """Three pipelines submitted concurrently to one shared coordinator each
    get their own correct results and per-pipeline counters — no cross-talk."""
    sizes = [4, 7, 10]

    def run_one(n: int):
        driver = _connect(shared_ctx)
        ds = Dataset.from_list(list(range(n))).map(_count_items).map(lambda x: x * 2)
        return driver.execute(ds)

    with ThreadPoolExecutor(max_workers=len(sizes)) as pool:
        outcomes = list(pool.map(run_one, sizes))

    for n, outcome in zip(sizes, outcomes, strict=True):
        assert sorted(outcome.results) == [x * 2 for x in range(n)]
        # Each pipeline's counter reflects only its own item count.
        assert outcome.counters.get("items") == n


def test_shared_context_pipeline_failure_does_not_break_the_pool(shared_ctx):
    """One pipeline failing (a shard raises) fails only its own execute();
    a concurrent healthy pipeline and a later pipeline both still succeed."""

    def explode(x):
        raise ValueError(f"bad value: {x}")

    def run_failing():
        driver = _connect(shared_ctx)
        return driver.execute(Dataset.from_list([1, 2, 3]).map(explode))

    def run_healthy():
        driver = _connect(shared_ctx)
        return driver.execute(Dataset.from_list([1, 2, 3, 4]).map(lambda x: x + 100))

    with ThreadPoolExecutor(max_workers=2) as pool:
        failing = pool.submit(run_failing)
        healthy = pool.submit(run_healthy)

        with pytest.raises(ZephyrWorkerError, match="ValueError"):
            failing.result()

        # The concurrent healthy pipeline is unaffected.
        assert sorted(healthy.result().results) == [101, 102, 103, 104]

    # The coordinator survived the failure: a fresh pipeline still runs.
    later = _connect(shared_ctx)
    assert sorted(later.execute(Dataset.from_list([5, 6]).map(lambda x: x * 10)).results) == [50, 60]


def test_shared_context_rejects_pipeline_that_cannot_fit_a_worker(shared_ctx):
    """A pipeline whose per-task cost exceeds the shared worker's resources is
    rejected up front rather than deadlocking forever unscheduled."""
    # Workers have 512m; demand 4g per task so it can never be scheduled.
    driver = ZephyrContext(
        client=shared_ctx.client,
        resources=ResourceConfig(cpu=1, ram="4g"),
        coordinator_endpoint=shared_ctx.coordinator_endpoint,
        name=f"toobig-{uuid.uuid4().hex[:8]}",
    )
    with pytest.raises(ValueError, match="exceeds per-worker resources"):
        driver.execute(Dataset.from_list([1, 2, 3]).map(lambda x: x))


def test_coordinator_shutdown_fails_in_flight_run_promptly(tmp_path, actor_context):
    """shutdown() fails an in-flight execution so its run_pipeline caller returns
    a clean error instead of blocking until no_workers_timeout."""
    coordinator = _make_test_coordinator(tmp_path)
    try:
        # Register an execution with a queued task but no workers, so its
        # _wait_for_stage would otherwise block indefinitely.
        run = _PipelineExecution(execution_id="stuck", map_cost=_TEST_TASK_COST, reduce_cost=_TEST_TASK_COST)
        coordinator._executions["stuck"] = run
        task = ShardTask(shard_idx=0, total_shards=1, shard=ListShard(refs=[]), operations=[], cost=_TEST_TASK_COST)
        coordinator._start_stage(run, "test", 0, [task])

        waiter = ThreadPoolExecutor(max_workers=1)
        fut = waiter.submit(coordinator._wait_for_stage, run)

        coordinator.shutdown()

        # The wait returns (raises) promptly rather than hanging on no_workers_timeout.
        with pytest.raises(ZephyrWorkerError, match="shutting down"):
            fut.result(timeout=10.0)
        waiter.shutdown(wait=True)
    finally:
        coordinator.shutdown()


def test_shared_context_shutdown_disconnects(tmp_path):
    """After shutdown() the context is disconnected and the serve job stops."""
    client = LocalClient(max_threads=8)
    ctx = ZephyrContext(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"shared-{uuid.uuid4().hex[:8]}",
    )
    endpoint = ctx.start()
    assert endpoint and ctx.coordinator_endpoint == endpoint

    # Works while up.
    assert sorted(ctx.execute(Dataset.from_list([1, 2]).map(lambda x: x + 1)).results) == [2, 3]

    ctx.shutdown()
    assert ctx.coordinator_endpoint is None
    assert ctx._serve_job is None

    client.shutdown(wait=True)


def test_with_block_starts_pool_yields_endpoint_and_tears_down(tmp_path):
    """`with pool as endpoint` starts the pool, yields its endpoint, tears down on exit.

    Inside the block, a plain ZephyrContext with no endpoint is still a
    dedicated context (unchanged), while one given the endpoint connects.
    """
    client = LocalClient(max_threads=8)
    pool = ZephyrContext(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"shared-{uuid.uuid4().hex[:8]}",
    )
    with pool as endpoint:
        assert endpoint == pool.coordinator_endpoint

        # Connecting driver → runs on the shared pool.
        shared_driver = ZephyrContext(
            client=client, resources=ResourceConfig(cpu=1, ram="512m"), coordinator_endpoint=endpoint
        )
        assert sorted(shared_driver.execute(Dataset.from_list([1, 2, 3]).map(lambda x: x * 3)).results) == [3, 6, 9]

        # Plain usage inside the block (no endpoint, env unset) stays dedicated.
        dedicated = ZephyrContext(
            client=client,
            max_workers=2,
            resources=ResourceConfig(cpu=1, ram="512m"),
            chunk_storage_prefix=str(tmp_path / "dedicated"),
            name=f"dedicated-{uuid.uuid4().hex[:8]}",
        )
        assert dedicated.coordinator_endpoint is None
        assert sorted(dedicated.execute(Dataset.from_list([4, 5]).map(lambda x: x + 1)).results) == [5, 6]

    # On block exit the pool is torn down.
    assert pool.coordinator_endpoint is None
    assert pool._serve_job is None
    client.shutdown(wait=True)


def test_env_endpoint_is_picked_up_by_a_plain_context(tmp_path, monkeypatch):
    """A context with no explicit endpoint connects to the pool named in the env var."""
    client = LocalClient(max_threads=8)
    pool = ZephyrContext(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"shared-{uuid.uuid4().hex[:8]}",
    )
    with pool as endpoint:
        monkeypatch.setenv(ZEPHYR_COORDINATOR_ENDPOINT_ENV, endpoint)
        # No coordinator_endpoint passed — it must come from the env.
        driver = ZephyrContext(client=client, resources=ResourceConfig(cpu=1, ram="512m"))
        assert driver.coordinator_endpoint == endpoint
        assert sorted(driver.execute(Dataset.from_list([2, 5]).map(lambda x: x + 1)).results) == [3, 6]
    client.shutdown(wait=True)
