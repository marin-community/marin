# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for a standing pool serving several pipelines.

End-to-end on ``LocalClient``, so the coordinator runs in a background job
thread and workers are in-process actors.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import _TEST_TASK_COST, _make_test_coordinator
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from zephyr import counters
from zephyr.coordinator import _PipelineExecution
from zephyr.dataset import Dataset
from zephyr.execution import (
    ZEPHYR_COORDINATOR_ENDPOINT_ENV,
    ZEPHYR_POOL_ENV,
    PoolMode,
    ZephyrContext,
)
from zephyr.plan import compute_plan
from zephyr.pool import _job_environment, _ZephyrPool
from zephyr.shuffle import ListShard
from zephyr.stage_io import ShardTask, ZephyrWorkerError


def _count_items(x):
    counters.pipeline.update_counter("items", 1)
    return x


@pytest.fixture
def shared_pool(tmp_path):
    """A started _ZephyrPool (2 workers) and its own LocalClient.

    Function-scoped and torn down after each test so the serve job's thread
    is released and no coordinator lingers between tests.
    """
    client = LocalClient(max_threads=8)
    pool = _ZephyrPool(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"shared-{uuid.uuid4().hex[:8]}",
    )
    pool.start()
    yield pool
    pool.shutdown()
    client.shutdown(wait=True)


def _connect(pool: _ZephyrPool) -> ZephyrContext:
    """A separate driver connected to the shared pool (models a step)."""
    return ZephyrContext(
        client=pool.client,
        resources=ResourceConfig(cpu=1, ram="512m"),
        coordinator_endpoint=pool.endpoint,
        name=f"driver-{uuid.uuid4().hex[:8]}",
    )


def test_shared_pool_runs_concurrent_pipelines_with_isolated_results(shared_pool):
    """Three pipelines submitted concurrently to one pool each get their own
    correct results and per-pipeline counters — no cross-talk."""
    sizes = [4, 7, 10]

    def run_one(n: int):
        driver = _connect(shared_pool)
        ds = Dataset.from_list(list(range(n))).map(_count_items).map(lambda x: x * 2)
        return driver.execute(ds)

    with ThreadPoolExecutor(max_workers=len(sizes)) as pool:
        outcomes = list(pool.map(run_one, sizes))

    for n, outcome in zip(sizes, outcomes, strict=True):
        assert sorted(outcome.results) == [x * 2 for x in range(n)]
        # Each pipeline's counter reflects only its own item count.
        assert outcome.counters.get("items") == n


def test_shared_pool_pipeline_failure_does_not_break_the_pool(shared_pool):
    """One pipeline failing (a shard raises) fails only its own execute();
    a concurrent healthy pipeline and a later pipeline both still succeed."""

    def explode(x):
        raise ValueError(f"bad value: {x}")

    def run_failing():
        driver = _connect(shared_pool)
        return driver.execute(Dataset.from_list([1, 2, 3]).map(explode))

    def run_healthy():
        driver = _connect(shared_pool)
        return driver.execute(Dataset.from_list([1, 2, 3, 4]).map(lambda x: x + 100))

    with ThreadPoolExecutor(max_workers=2) as pool:
        failing = pool.submit(run_failing)
        healthy = pool.submit(run_healthy)

        with pytest.raises(ZephyrWorkerError, match="ValueError"):
            failing.result()

        assert sorted(healthy.result().results) == [101, 102, 103, 104]

    later = _connect(shared_pool)
    assert sorted(later.execute(Dataset.from_list([5, 6]).map(lambda x: x * 10)).results) == [50, 60]


def test_shared_pool_rejects_pipeline_that_cannot_fit_a_worker(shared_pool):
    """A pipeline whose per-task cost exceeds the pool worker's resources is
    rejected up front rather than deadlocking forever unscheduled."""
    # Workers have 512m; demand 4g per task so it can never be scheduled.
    driver = ZephyrContext(
        client=shared_pool.client,
        resources=ResourceConfig(cpu=1, ram="4g"),
        coordinator_endpoint=shared_pool.endpoint,
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


def test_pool_shutdown_disconnects(tmp_path):
    """After shutdown() a driver holding the old endpoint cannot reach the coordinator."""
    client = LocalClient(max_threads=8)
    pool = _ZephyrPool(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"shared-{uuid.uuid4().hex[:8]}",
    )
    endpoint = pool.start()
    assert endpoint and pool.endpoint == endpoint

    driver = ZephyrContext(client=client, resources=ResourceConfig(cpu=1, ram="512m"), coordinator_endpoint=endpoint)
    assert sorted(driver.execute(Dataset.from_list([1, 2]).map(lambda x: x + 1)).results) == [2, 3]

    pool.shutdown()
    assert pool.endpoint is None
    assert pool._serve_job is None

    # The stale endpoint must stop resolving; otherwise a late driver would
    # register a pipeline on a dead pool and block until no_workers_timeout.
    stale = ZephyrContext(client=client, resources=ResourceConfig(cpu=1, ram="512m"), coordinator_endpoint=endpoint)
    with pytest.raises(RuntimeError, match="not found in registry"):
        stale.execute(Dataset.from_list([1, 2]).map(lambda x: x + 1))

    client.shutdown(wait=True)


def test_coordinator_rejects_pipelines_after_shutdown(tmp_path, actor_context):
    """A pipeline submitted to an already-shut-down coordinator fails fast.

    Guards the direct handle path: without the check the call registers an
    execution and then waits out no_workers_timeout for workers that already
    exited.
    """
    coordinator = _make_test_coordinator(tmp_path)
    coordinator.shutdown()

    with pytest.raises(ZephyrWorkerError, match="shut down"):
        coordinator.run_pipeline(
            compute_plan(Dataset.from_list([1, 2]).map(lambda x: x + 1)),
            "exec-after-shutdown",
            _TEST_TASK_COST,
            _TEST_TASK_COST,
        )


def test_pool_restart_waits_for_the_new_coordinator(tmp_path):
    """start() after shutdown() waits for the new coordinator, not a stale address.

    The endpoint is derived from the pool's name, so a restart returns the same
    string by construction. What must not regress is the wait: start() has to
    block until the *new* coordinator reports ready, rather than returning as
    soon as an address can be computed.
    """
    client = LocalClient(max_threads=8)
    prefix = tmp_path / "chunks"
    pool = _ZephyrPool(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(prefix),
        name=f"shared-{uuid.uuid4().hex[:8]}",
    )
    pool.start()
    pool.shutdown()

    endpoint = pool.start()
    try:
        assert endpoint == pool.endpoint
        driver = ZephyrContext(client=client, resources=ResourceConfig(cpu=1, ram="512m"), coordinator_endpoint=endpoint)
        assert sorted(driver.execute(Dataset.from_list([1, 2]).map(lambda x: x + 1)).results) == [2, 3]
    finally:
        pool.shutdown()
        client.shutdown(wait=True)


def test_pool_context_manager_yields_endpoint_and_tears_down(tmp_path):
    """`with _ZephyrPool(...) as endpoint` starts the pool, yields its endpoint,
    and tears it down on exit.

    A plain ZephyrContext with no endpoint stays a dedicated context (unchanged),
    while one given the endpoint connects to the pool.
    """
    client = LocalClient(max_threads=8)
    pool = _ZephyrPool(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"shared-{uuid.uuid4().hex[:8]}",
    )
    with pool as endpoint:
        assert endpoint == pool.endpoint

        shared_driver = ZephyrContext(
            client=client, resources=ResourceConfig(cpu=1, ram="512m"), coordinator_endpoint=endpoint
        )
        assert sorted(shared_driver.execute(Dataset.from_list([1, 2, 3]).map(lambda x: x * 3)).results) == [3, 6, 9]

        # Plain usage (no endpoint, env unset) stays dedicated.
        dedicated = ZephyrContext(
            client=client,
            max_workers=2,
            resources=ResourceConfig(cpu=1, ram="512m"),
            chunk_storage_prefix=str(tmp_path / "dedicated"),
            name=f"dedicated-{uuid.uuid4().hex[:8]}",
        )
        assert dedicated.coordinator_endpoint is None
        assert sorted(dedicated.execute(Dataset.from_list([4, 5]).map(lambda x: x + 1)).results) == [5, 6]

    assert pool.endpoint is None
    assert pool._serve_job is None
    client.shutdown(wait=True)


def test_job_environment_none_when_unset_else_carries_extras_and_env_vars():
    """_job_environment inherits the parent env (None) unless extras/env vars are set."""
    # Backward-compatible: no extras and no env vars → inherit the parent job's env.
    assert _job_environment(None, None) is None
    assert _job_environment([], {}) is None

    env = _job_environment(["datakit"], {"JAX_PLATFORMS": "cpu"})
    assert env is not None
    assert env.extras == ["datakit"]
    assert env.env_vars["JAX_PLATFORMS"] == "cpu"


def test_pool_launches_job_with_requested_extras_and_env_vars(tmp_path):
    """A pool started with pip_dependency_groups / job_env_vars launches its
    coordinator + worker job carrying those extras and env vars (so the generic
    pool workers can import each stage's deps)."""
    client = LocalClient(max_threads=8)
    submitted = []
    original_submit = client.submit

    def recording_submit(request, adopt_existing=True):
        submitted.append(request)
        return original_submit(request, adopt_existing)

    client.submit = recording_submit
    pool = _ZephyrPool(
        client=client,
        max_workers=1,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"env-{uuid.uuid4().hex[:8]}",
        pip_dependency_groups=["datakit"],
        job_env_vars={"JAX_PLATFORMS": "cpu"},
    )
    with pool:
        serve_request = next(r for r in submitted if r.name.endswith("-pool"))
        assert serve_request.environment is not None
        assert serve_request.environment.extras == ["datakit"]
        assert serve_request.environment.env_vars["JAX_PLATFORMS"] == "cpu"
    client.shutdown(wait=True)


def test_isolated_mode_declines_a_pool_offered_by_the_environment(tmp_path, monkeypatch):
    """mode=ISOLATED runs on its own pool even when the env names a shared one.

    A step whose stages need something the shared workers lack has to be able
    to opt out, and it must not have to know whether a parent set the env.
    """
    client = LocalClient(max_threads=8)
    monkeypatch.setenv(ZEPHYR_COORDINATOR_ENDPOINT_ENV, "local/some-other-pool-coord-0")
    monkeypatch.setenv(ZEPHYR_POOL_ENV, "some-other-pool")

    ctx = ZephyrContext(
        client=client,
        mode=PoolMode.ISOLATED,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"isolated-{uuid.uuid4().hex[:8]}",
    )
    try:
        assert ctx.coordinator_endpoint is None
        assert ctx.pool_name is None
        assert sorted(ctx.execute(Dataset.from_list([1, 2]).map(lambda x: x * 5)).results) == [5, 10]
    finally:
        ctx.shutdown()
        client.shutdown(wait=True)


def test_isolated_mode_rejects_an_explicitly_configured_pool(local_client):
    """Asking for ISOLATED and naming a pool is a contradiction, not a precedence puzzle."""
    with pytest.raises(ValueError, match="contradicts"):
        ZephyrContext(client=local_client, mode=PoolMode.ISOLATED, pool_name="ingest")


def test_shared_mode_fails_fast_without_a_pool(local_client, monkeypatch):
    """mode=INHERIT means 'I expect a pool'; starting a private one would hide a misconfiguration."""
    monkeypatch.delenv(ZEPHYR_COORDINATOR_ENDPOINT_ENV, raising=False)
    monkeypatch.delenv(ZEPHYR_POOL_ENV, raising=False)
    with pytest.raises(ValueError, match="requires a pool"):
        ZephyrContext(client=local_client, mode=PoolMode.INHERIT, resources=ResourceConfig(cpu=1, ram="512m"))


def test_context_start_owns_a_pool_that_other_contexts_join_by_name(tmp_path, monkeypatch):
    """Entering a mode=HOST context starts its pool; others join it by name."""
    client = LocalClient(max_threads=8)
    name = f"ingest-{uuid.uuid4().hex[:8]}"
    owner = ZephyrContext(
        client=client,
        mode=PoolMode.HOST,
        pool_name=name,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"owner-{uuid.uuid4().hex[:8]}",
    )
    with owner:
        assert owner.coordinator_endpoint is not None

        monkeypatch.setenv(ZEPHYR_POOL_ENV, name)
        joiner = ZephyrContext(client=client, resources=ResourceConfig(cpu=1, ram="512m"))
        assert joiner.coordinator_endpoint == owner.coordinator_endpoint
        assert sorted(joiner.execute(Dataset.from_list([3, 4]).map(lambda x: x + 1)).results) == [4, 5]

    assert owner.coordinator_endpoint is None
    client.shutdown(wait=True)


def test_env_endpoint_is_picked_up_by_a_plain_context(tmp_path, monkeypatch):
    """A context with no explicit endpoint connects to the pool named in the env var."""
    client = LocalClient(max_threads=8)
    pool = _ZephyrPool(
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
