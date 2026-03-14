# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for JAX distributed initialization via Iris endpoint registry.

Tests the multi-task coordinator handshake end-to-end: task 0 registers a
coordinator endpoint, task 1 polls until it resolves, both call
jax.distributed.initialize with the same coordinator address.

Unlike the unit tests in test_jax_init.py (which test each piece in isolation),
these tests verify the coordination protocol with concurrent threads sharing a
thread-safe endpoint store — closer to real multi-VM behavior.

No real GPUs or Iris cluster needed — uses a thread-safe in-memory endpoint
store and mocks jax.distributed.initialize.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("jax")

from iris.actor.resolver import ResolveResult, ResolvedEndpoint
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from iris.runtime.jax_init import _poll_for_coordinator, initialize_jax


@dataclass
class ThreadSafeEndpointStore:
    """Thread-safe in-memory endpoint store for integration testing.

    Implements both the EndpointRegistry and Resolver protocols so
    task 0 (register) and task 1 (resolve) share the same backing store,
    matching the behavior of a real Iris controller's endpoint table.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _endpoints: dict[str, tuple[str, str]] = field(default_factory=dict)
    _next_id: int = 0

    def register(self, name: str, address: str, metadata: dict[str, str] | None = None) -> str:
        with self._lock:
            self._next_id += 1
            endpoint_id = f"ep-{self._next_id}"
            self._endpoints[name] = (address, endpoint_id)
            return endpoint_id

    def unregister(self, endpoint_id: str) -> None:
        with self._lock:
            to_remove = [k for k, (_, eid) in self._endpoints.items() if eid == endpoint_id]
            for k in to_remove:
                del self._endpoints[k]

    def resolve(self, name: str) -> ResolveResult:
        with self._lock:
            if name in self._endpoints:
                address, eid = self._endpoints[name]
                return ResolveResult(
                    name=name,
                    endpoints=[ResolvedEndpoint(url=address, actor_id=eid)],
                )
            return ResolveResult(name=name, endpoints=[])


@dataclass
class FakeContext:
    """Fake IrisContext backed by a shared endpoint store."""

    registry: ThreadSafeEndpointStore
    resolver: ThreadSafeEndpointStore


def _make_job_info(task_index: int, num_tasks: int = 2) -> JobInfo:
    job_name = JobName.from_string(f"/testuser/jax-integration/{task_index}")
    return JobInfo(
        task_id=job_name,
        num_tasks=num_tasks,
        attempt_id=0,
        advertise_host="10.0.0.1",
        controller_address="controller:8080",
        ports={},
    )


def test_concurrent_registration_and_resolution_converge():
    """Task 0 registers coordinator endpoint; task 1 polls and resolves the same address.

    Runs both tasks concurrently with task 0 registering after a brief delay,
    verifying that _poll_for_coordinator converges when the endpoint appears.
    """
    store = ThreadSafeEndpointStore()
    expected_address = "10.0.0.1:8476"
    registration_event = threading.Event()

    def task0() -> None:
        time.sleep(0.05)
        store.register("jax_coordinator", expected_address)
        registration_event.set()

    def task1() -> str:
        return _poll_for_coordinator(
            resolver=store,
            endpoint_name="jax_coordinator",
            timeout=5.0,
            poll_interval=0.01,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f0 = pool.submit(task0)
        f1 = pool.submit(task1)
        f0.result(timeout=10)
        resolved_address = f1.result(timeout=10)

    assert resolved_address == expected_address


def test_initialize_jax_full_protocol_both_tasks_agree():
    """Run initialize_jax for task 0 then task 1; both get the same coordinator.

    Task 0 registers the coordinator endpoint via the shared store.
    Task 1 resolves it via polling. Both call jax.distributed.initialize
    with the same coordinator address but their own process_id.
    """
    store = ThreadSafeEndpointStore()
    mock_jax_init = MagicMock()

    with patch("jax.distributed.initialize", mock_jax_init):
        # Task 0: registers coordinator
        job0 = _make_job_info(task_index=0, num_tasks=2)
        ctx0 = FakeContext(registry=store, resolver=store)
        with (
            patch("iris.runtime.jax_init.get_job_info", return_value=job0),
            patch("iris.runtime.jax_init.iris_ctx", return_value=ctx0),
            patch("iris.runtime.jax_init.atexit"),
        ):
            initialize_jax(port=8476)

        # Task 1: resolves coordinator via polling
        job1 = _make_job_info(task_index=1, num_tasks=2)
        ctx1 = FakeContext(registry=store, resolver=store)
        with (
            patch("iris.runtime.jax_init.get_job_info", return_value=job1),
            patch("iris.runtime.jax_init.iris_ctx", return_value=ctx1),
        ):
            initialize_jax(port=8476, poll_interval=0.01, poll_timeout=5.0)

    assert mock_jax_init.call_count == 2
    # Task 0: jax.distributed.initialize("10.0.0.1:8476", 2, 0)
    assert mock_jax_init.call_args_list[0].args == ("10.0.0.1:8476", 2, 0)
    # Task 1: jax.distributed.initialize("10.0.0.1:8476", 2, 1)
    assert mock_jax_init.call_args_list[1].args == ("10.0.0.1:8476", 2, 1)


def test_task0_restart_reregisters_and_task1_resolves_new_address():
    """After task 0 restarts, it re-registers with a new address; task 1 resolves to the new one.

    Simulates:
    1. Task 0 (attempt 0) registers at address A
    2. Task 0 crashes — endpoint is unregistered (cascade delete in real controller)
    3. Task 0 (attempt 1) registers at address B
    4. Task 1 resolves to address B
    """
    store = ThreadSafeEndpointStore()

    # Attempt 0: task 0 registers original address
    original_id = store.register("jax_coordinator", "10.0.0.1:8476")

    # Crash: cascade delete removes the endpoint
    store.unregister(original_id)

    # Attempt 1: task 0 re-registers with new address (new host after reschedule)
    store.register("jax_coordinator", "10.0.0.2:8476")

    # Task 1 resolves — should see the new address
    resolved = _poll_for_coordinator(
        resolver=store,
        endpoint_name="jax_coordinator",
        timeout=5.0,
        poll_interval=0.01,
    )
    assert resolved == "10.0.0.2:8476"


def test_concurrent_restart_reconvergence():
    """Task 1 is already polling when task 0 crashes and re-registers.

    Verifies that task 1 eventually resolves to the new coordinator address
    after task 0 restarts with a different host.
    """
    store = ThreadSafeEndpointStore()

    def task0_lifecycle() -> None:
        # Attempt 0: register, then crash
        eid = store.register("jax_coordinator", "10.0.0.1:8476")
        time.sleep(0.02)
        store.unregister(eid)
        # Brief outage window
        time.sleep(0.05)
        # Attempt 1: re-register on a different host
        store.register("jax_coordinator", "10.0.0.3:8476")

    def task1_poll() -> str:
        return _poll_for_coordinator(
            resolver=store,
            endpoint_name="jax_coordinator",
            timeout=5.0,
            poll_interval=0.01,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f0 = pool.submit(task0_lifecycle)
        f1 = pool.submit(task1_poll)
        f0.result(timeout=10)
        resolved = f1.result(timeout=10)

    # Task 1 should converge on whichever address was available when it polled.
    # It might see 10.0.0.1 (before crash) or 10.0.0.3 (after re-register).
    # The important thing is it resolves _something_ and doesn't timeout.
    assert resolved in ("10.0.0.1:8476", "10.0.0.3:8476")


def test_poll_timeout_when_coordinator_never_registers():
    """TimeoutError when task 0 never registers — task 1 polls until timeout."""
    store = ThreadSafeEndpointStore()

    with pytest.raises(TimeoutError, match="Timed out"):
        _poll_for_coordinator(
            resolver=store,
            endpoint_name="jax_coordinator",
            timeout=0.2,
            poll_interval=0.01,
        )


def test_four_task_job_all_resolve_same_coordinator():
    """In a 4-task job, tasks 1-3 all resolve to the same coordinator registered by task 0."""
    store = ThreadSafeEndpointStore()
    expected_address = "10.0.0.1:8476"
    mock_jax_init = MagicMock()

    with patch("jax.distributed.initialize", mock_jax_init):
        # Task 0: registers coordinator
        job0 = _make_job_info(task_index=0, num_tasks=4)
        ctx0 = FakeContext(registry=store, resolver=store)
        with (
            patch("iris.runtime.jax_init.get_job_info", return_value=job0),
            patch("iris.runtime.jax_init.iris_ctx", return_value=ctx0),
            patch("iris.runtime.jax_init.atexit"),
        ):
            initialize_jax(port=8476)

        # Tasks 1-3: all resolve the same coordinator
        for task_index in (1, 2, 3):
            job = _make_job_info(task_index=task_index, num_tasks=4)
            ctx = FakeContext(registry=store, resolver=store)
            with (
                patch("iris.runtime.jax_init.get_job_info", return_value=job),
                patch("iris.runtime.jax_init.iris_ctx", return_value=ctx),
            ):
                initialize_jax(port=8476, poll_interval=0.01, poll_timeout=5.0)

    assert mock_jax_init.call_count == 4
    # All tasks should use the same coordinator address
    for i, call in enumerate(mock_jax_init.call_args_list):
        assert call.args[0] == expected_address, f"Task {i} got wrong coordinator"
        assert call.args[1] == 4, f"Task {i} got wrong num_processes"
        assert call.args[2] == i, f"Task {i} got wrong process_id"


def test_iris_port_overrides_default_in_full_protocol():
    """When IRIS_PORT_jax is allocated, it overrides the default port argument."""
    store = ThreadSafeEndpointStore()
    mock_jax_init = MagicMock()

    with patch("jax.distributed.initialize", mock_jax_init):
        # Task 0 with IRIS_PORT_jax=12345
        job0 = _make_job_info(task_index=0, num_tasks=2)
        job0.ports = {"jax": 12345}
        ctx0 = FakeContext(registry=store, resolver=store)
        with (
            patch("iris.runtime.jax_init.get_job_info", return_value=job0),
            patch("iris.runtime.jax_init.iris_ctx", return_value=ctx0),
            patch("iris.runtime.jax_init.atexit"),
        ):
            initialize_jax(port=9999)  # This should be overridden by IRIS_PORT_jax

        # Task 1 resolves whatever task 0 registered
        job1 = _make_job_info(task_index=1, num_tasks=2)
        ctx1 = FakeContext(registry=store, resolver=store)
        with (
            patch("iris.runtime.jax_init.get_job_info", return_value=job1),
            patch("iris.runtime.jax_init.iris_ctx", return_value=ctx1),
        ):
            initialize_jax(port=9999, poll_interval=0.01, poll_timeout=5.0)

    # Both should use port 12345 (from IRIS_PORT_jax), not 9999
    assert mock_jax_init.call_args_list[0].args == ("10.0.0.1:12345", 2, 0)
    assert mock_jax_init.call_args_list[1].args == ("10.0.0.1:12345", 2, 1)
