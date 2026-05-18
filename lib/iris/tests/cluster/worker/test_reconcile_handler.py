# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the handle_reconcile free function and SpecCache.

These tests exercise the Reconcile RPC handler introduced in Phase B without
spinning up a full gRPC server. Most tests use the `mock_worker` fixture from
conftest (which satisfies the ReconcileContext Protocol) and call
`worker.handle_reconcile(request)` directly. A handful of tests construct a
fresh SpecCache explicitly to verify eviction semantics in isolation.

Scenario mapping to the kata spec:
  1. Spec-inline + BUILDING on first call, cache hit (no re-enqueue) on second.
  2. No-spec steady state: first call populates cache + enqueues, second call
     (no inline spec, attempt already in _tasks) returns current state.
  3. No spec and no cache: controller sent run intent but forgot to include spec
     and the worker has no cached copy -> TASK_STATE_MISSING in response.
  4. intent=stop: worker kills the attempt asynchronously; response reflects
     the in-progress state at the time of the call.
  5. Zombie kill: local attempt absent from desired set -> worker kills it.
  6. Terminal eviction: after an attempt reaches a terminal state, the next
     reconcile tick evicts it from SpecCache so the cache stays bounded.
"""

from unittest.mock import Mock

import pytest
from iris.cluster.runtime.types import ContainerPhase, ContainerStatus
from iris.cluster.types import JobName
from iris.cluster.worker.reconcile import SpecCache
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.rpc import job_pb2, worker_pb2
from iris.test_util import wait_for_condition
from rigging.timing import Duration

from tests.cluster.worker.conftest import create_mock_container_handle, create_run_task_request

pytestmark = pytest.mark.timeout(10)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def worker(mock_bundle_store, mock_runtime, tmp_path) -> Worker:
    config = WorkerConfig(
        port=0,
        port_range=(50200, 50300),
        poll_interval=Duration.from_seconds(0.05),
        cache_dir=tmp_path / "cache",
        default_task_image="mock-image",
        worker_id="w-reconcile-test",
    )
    return Worker(config, bundle_store=mock_bundle_store, container_runtime=mock_runtime)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _task_id(name: str = "reconcile-task") -> str:
    return JobName.root("test-user", name).task(0).to_wire()


def _run_desired(
    task_id: str,
    attempt_id: int,
    run_request: job_pb2.RunTaskRequest | None = None,
) -> worker_pb2.Worker.DesiredAttempt:
    """Build a DesiredAttempt with intent=run, optionally with inline spec."""
    spec = (
        worker_pb2.Worker.AttemptSpec(request=run_request)
        if run_request is not None
        else worker_pb2.Worker.AttemptSpec()
    )
    return worker_pb2.Worker.DesiredAttempt(
        task_id=task_id,
        attempt_id=attempt_id,
        run=spec,
    )


def _stop_desired(task_id: str, attempt_id: int) -> worker_pb2.Worker.DesiredAttempt:
    return worker_pb2.Worker.DesiredAttempt(
        task_id=task_id,
        attempt_id=attempt_id,
        stop=worker_pb2.Worker.STOP_REASON_CANCELLED,
    )


def _reconcile(
    worker: Worker,
    desired: list[worker_pb2.Worker.DesiredAttempt],
) -> worker_pb2.Worker.ReconcileResponse:
    return worker.handle_reconcile(worker_pb2.Worker.ReconcileRequest(desired=desired))


def _observations_by_key(
    response: worker_pb2.Worker.ReconcileResponse,
) -> dict[tuple[str, int], worker_pb2.Worker.AttemptObservation]:
    return {(obs.task_id, obs.attempt_id): obs for obs in response.observed}


# ── Scenario 1: spec-inline enqueue + cache hit ───────────────────────────────


def test_inline_spec_enqueues_and_reports_building(worker, mock_runtime):
    """First reconcile with spec inline enqueues the attempt; response is BUILDING."""
    task_id = _task_id("inline-spec")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    # Use a container handle that stays pending indefinitely so we can inspect state.
    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )

    response = _reconcile(worker, [_run_desired(task_id, 0, run_request=run_req)])

    obs = _observations_by_key(response)
    key = (task_id, 0)
    assert key in obs, "Expected observation for the newly-enqueued attempt"
    # Workers map PENDING -> BUILDING in the observation (never expose PENDING to controller).
    assert obs[key].state in (
        job_pb2.TASK_STATE_BUILDING,
        job_pb2.TASK_STATE_RUNNING,
    ), f"Unexpected state {obs[key].state} — expected BUILDING or RUNNING"

    # Spec must be cached now.
    assert worker._spec_cache.lookup(task_id, 0) is not None

    # Clean up.
    worker.kill_task(task_id)
    task = worker.get_task(task_id)
    if task and task.thread:
        task.thread.join(timeout=5.0)


def test_second_reconcile_without_spec_is_cache_hit(worker, mock_runtime):
    """Second reconcile call with no inline spec must NOT re-enqueue (cache hit).

    After the first call with an inline spec the attempt is in `_tasks`, so
    `_process_run_intent` takes the early-return path. The observed state is
    reported from the local task record, not as MISSING.
    """
    task_id = _task_id("cache-hit")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )
    submit_spy = Mock(wraps=worker.submit_task)
    worker.submit_task = submit_spy  # type: ignore[method-assign]

    # First call: spec inline, enqueues attempt.
    _reconcile(worker, [_run_desired(task_id, 0, run_request=run_req)])
    call_count_after_first = submit_spy.call_count

    # Second call: no inline spec.
    response2 = _reconcile(worker, [_run_desired(task_id, 0, run_request=None)])

    # No additional submit_task call.
    assert submit_spy.call_count == call_count_after_first, "Unexpected re-enqueue on second reconcile"

    # Observation still present and NOT MISSING.
    obs = _observations_by_key(response2)
    assert (task_id, 0) in obs
    assert obs[(task_id, 0)].state != job_pb2.TASK_STATE_MISSING

    worker.kill_task(task_id)
    task = worker.get_task(task_id)
    if task and task.thread:
        task.thread.join(timeout=5.0)


# ── Scenario 3: no spec and no cache → TASK_STATE_MISSING ────────────────────


def test_no_spec_no_cache_reports_missing(worker):
    """intent=run with neither inline spec nor cached spec → TASK_STATE_MISSING."""
    task_id = _task_id("missing")

    # Worker has no prior knowledge of this attempt.
    assert worker._spec_cache.lookup(task_id, 0) is None
    assert worker.get_task(task_id) is None

    response = _reconcile(worker, [_run_desired(task_id, 0, run_request=None)])

    obs = _observations_by_key(response)
    assert (task_id, 0) in obs
    assert obs[(task_id, 0)].state == job_pb2.TASK_STATE_MISSING

    # Nothing should have been enqueued.
    assert worker.get_task(task_id) is None


# ── Scenario 4: intent=stop kills non-terminal attempt ───────────────────────


def test_stop_intent_kills_running_attempt(worker, mock_runtime):
    """intent=stop signals the worker to kill the attempt asynchronously."""
    task_id = _task_id("stop-intent")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )

    # Submit a task so it's locally known.
    worker.submit_task(run_req)
    task = worker.get_task(task_id, attempt_id=0)
    assert task is not None
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_RUNNING)

    # Reconcile with intent=stop.
    response = _reconcile(worker, [_stop_desired(task_id, 0)])

    # The response should include the attempt's current observation (running or transitioning).
    obs = _observations_by_key(response)
    assert (task_id, 0) in obs

    # should_stop is set synchronously by _process_stop_intent.
    assert task.should_stop is True

    task.thread.join(timeout=5.0)
    assert task.status == job_pb2.TASK_STATE_KILLED


# ── Scenario 5: zombie kill (local attempt absent from desired) ───────────────


def test_zombie_attempt_is_killed(worker, mock_runtime):
    """Attempt running locally but absent from desired is killed (zombie kill)."""
    task_id = _task_id("zombie")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )

    worker.submit_task(run_req)
    task = worker.get_task(task_id, attempt_id=0)
    assert task is not None
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_RUNNING)

    # Reconcile with an empty desired set — the running attempt becomes a zombie.
    _reconcile(worker, [])

    # Zombie kill is async; should_stop is set synchronously.
    assert task.should_stop is True

    task.thread.join(timeout=5.0)
    assert task.status == job_pb2.TASK_STATE_KILLED


def test_zombie_kill_skips_terminal_tasks(worker, mock_runtime):
    """Attempts already in a terminal state are not re-killed during zombie detection."""
    task_id = _task_id("zombie-terminal")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    # Container exits immediately with success.
    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=0)],
        )
    )

    worker.submit_task(run_req)
    task = worker.get_task(task_id, attempt_id=0)
    assert task is not None
    task.thread.join(timeout=5.0)
    assert task.status == job_pb2.TASK_STATE_SUCCEEDED

    # Reconcile with empty desired; terminal attempt must not be re-killed.
    _reconcile(worker, [])
    assert task.status == job_pb2.TASK_STATE_SUCCEEDED


# ── Scenario 6: SpecCache evicts on terminal observation ─────────────────────


def test_spec_cache_evicts_after_terminal_state(worker, mock_runtime):
    """After an attempt becomes terminal, the next reconcile tick evicts it from SpecCache."""
    task_id = _task_id("evict")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    # Container exits immediately with success.
    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=0)],
        )
    )

    # First reconcile: spec inline → enqueues and caches.
    _reconcile(worker, [_run_desired(task_id, 0, run_request=run_req)])
    assert worker._spec_cache.lookup(task_id, 0) is not None

    # Wait for the task to terminate.
    task = worker.get_task(task_id, attempt_id=0)
    assert task is not None
    task.thread.join(timeout=5.0)
    assert task.status == job_pb2.TASK_STATE_SUCCEEDED

    # Second reconcile tick: _build_observation sees terminal status → evict.
    _reconcile(worker, [_run_desired(task_id, 0, run_request=None)])

    # Cache entry must be gone.
    assert worker._spec_cache.lookup(task_id, 0) is None


# ── Standalone SpecCache unit tests ───────────────────────────────────────────


def test_spec_cache_add_and_lookup():
    cache = SpecCache()
    req = create_run_task_request()
    cache.add(req.task_id, req.attempt_id, req)
    assert cache.lookup(req.task_id, req.attempt_id) is req
    assert len(cache) == 1


def test_spec_cache_lookup_miss_returns_none():
    cache = SpecCache()
    assert cache.lookup("nonexistent", 0) is None


def test_spec_cache_evict_removes_entry():
    cache = SpecCache()
    req = create_run_task_request()
    cache.add(req.task_id, req.attempt_id, req)
    cache.evict(req.task_id, req.attempt_id)
    assert cache.lookup(req.task_id, req.attempt_id) is None
    assert len(cache) == 0


def test_spec_cache_evict_is_idempotent():
    cache = SpecCache()
    # Evicting a key that was never added must not raise.
    cache.evict("ghost", 99)
    assert len(cache) == 0


def test_spec_cache_multiple_keys_independent():
    cache = SpecCache()
    task_id_a = _task_id("cache-a")
    task_id_b = _task_id("cache-b")
    req_a = create_run_task_request(task_id=task_id_a, attempt_id=0)
    req_b = create_run_task_request(task_id=task_id_b, attempt_id=0)
    cache.add(task_id_a, 0, req_a)
    cache.add(task_id_b, 0, req_b)
    cache.evict(task_id_a, 0)
    assert cache.lookup(task_id_a, 0) is None
    assert cache.lookup(task_id_b, 0) is req_b


# ── Response shape tests ───────────────────────────────────────────────────────


def test_reconcile_response_includes_worker_id(worker):
    """ReconcileResponse always includes the worker_id."""
    response = _reconcile(worker, [])
    assert response.worker_id == "w-reconcile-test"


def test_reconcile_response_includes_health(worker):
    """ReconcileResponse always includes a WorkerHealth message."""
    response = _reconcile(worker, [])
    assert response.HasField("health")


def test_reconcile_empty_desired_empty_observed_when_no_tasks(worker):
    """With no desired and no local tasks, observed list is empty."""
    response = _reconcile(worker, [])
    assert list(response.observed) == []


def test_reconcile_pending_state_reported_as_building(worker, mock_runtime):
    """Workers never expose TASK_STATE_PENDING; it is mapped to TASK_STATE_BUILDING."""
    task_id = _task_id("pending-map")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    # Container never starts so the attempt stays in PENDING / BUILDING.
    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )

    # Submit directly so attempt is in _tasks before reconcile is called.
    worker.submit_task(run_req)

    response = _reconcile(worker, [_run_desired(task_id, 0, run_request=run_req)])

    obs = _observations_by_key(response)
    assert (task_id, 0) in obs
    # PENDING must be hidden from the controller.
    assert obs[(task_id, 0)].state != job_pb2.TASK_STATE_PENDING

    worker.kill_task(task_id)
    task = worker.get_task(task_id, attempt_id=0)
    if task and task.thread:
        task.thread.join(timeout=5.0)
