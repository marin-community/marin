# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("jax")

from iris.actor.resolver import ResolvedEndpoint, ResolveResult
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from iris.runtime.jax_init import _poll_for_coordinator, initialize_jax


@dataclass
class FakeRegistry:
    registered: list[tuple[str, str]] = field(default_factory=list)
    unregistered: list[str] = field(default_factory=list)
    next_id: str = "endpoint-1"

    def register(self, name: str, address: str, metadata: dict[str, str] | None = None) -> str:
        self.registered.append((name, address))
        return self.next_id

    def unregister(self, endpoint_id: str) -> None:
        self.unregistered.append(endpoint_id)


@dataclass
class FakeResolver:
    results: list[ResolveResult] = field(default_factory=list)
    results_by_name: dict[str, list[ResolveResult]] = field(default_factory=dict)
    call_count: int = 0
    call_count_by_name: dict[str, int] = field(default_factory=dict)

    def resolve(self, name: str) -> ResolveResult:
        sequence = self.results_by_name.get(name, self.results)
        idx = min(self.call_count_by_name.get(name, 0), len(sequence) - 1)
        result = sequence[idx]
        self.call_count += 1
        self.call_count_by_name[name] = self.call_count_by_name.get(name, 0) + 1
        return result


@dataclass
class FakeContext:
    registry: FakeRegistry = field(default_factory=FakeRegistry)
    resolver: FakeResolver = field(default_factory=FakeResolver)


def _make_job_info(task_index: int = 0, num_tasks: int = 1) -> JobInfo:
    """Create a JobInfo with the given task_index and num_tasks."""
    job_name = JobName.from_string(f"/testuser/testjob/{task_index}")
    return JobInfo(
        task_id=job_name,
        num_tasks=num_tasks,
        attempt_id=0,
        advertise_host="10.0.0.1",
        controller_address="controller:8080",
        ports={},
    )


def _found_endpoint(name: str, url: str = "10.0.0.1:8476") -> ResolveResult:
    return ResolveResult(
        name=name,
        endpoints=[ResolvedEndpoint(url=url, actor_id=f"{name}-ep")],
    )


def _ready_results(endpoint_name: str, num_tasks: int) -> dict[str, list[ResolveResult]]:
    return {f"{endpoint_name}_ready_{i}": [_found_endpoint(f"{endpoint_name}_ready_{i}")] for i in range(num_tasks)}


@patch("iris.runtime.jax_init.atexit")
@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_single_task(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    mock_atexit: MagicMock,
) -> None:
    """Single-task jobs call jax.distributed.initialize with explicit args."""
    mock_get_job_info.return_value = _make_job_info(task_index=0, num_tasks=1)

    initialize_jax()

    mock_jax_init.assert_called_once_with(
        "10.0.0.1:8476",
        1,
        0,
        initialization_timeout=1800,
    )
    mock_iris_ctx.assert_not_called()


@pytest.mark.parametrize("env_key,env_val", [("PJRT_DEVICE", "TPU"), ("JAX_PLATFORMS", "tpu")])
@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_tpu_is_noop(
    mock_get_job_info: MagicMock,
    mock_jax_init: MagicMock,
    env_key: str,
    env_val: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On TPU, initialize_jax is a no-op — TPU runtime handles distributed init."""
    monkeypatch.setenv(env_key, env_val)
    initialize_jax()
    mock_jax_init.assert_not_called()
    mock_get_job_info.assert_not_called()


@patch("iris.runtime.jax_init.atexit")
@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_no_job_info(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    mock_atexit: MagicMock,
) -> None:
    """No job info means we're not in an Iris job — skip distributed init."""
    mock_get_job_info.return_value = None

    initialize_jax()

    mock_jax_init.assert_not_called()
    mock_iris_ctx.assert_not_called()


@patch("iris.runtime.jax_init.atexit")
@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_task0_registers(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    mock_atexit: MagicMock,
) -> None:
    """Task 0 registers the coordinator endpoint and calls jax.distributed.initialize."""
    mock_get_job_info.return_value = _make_job_info(task_index=0, num_tasks=4)
    fake_ctx = FakeContext(resolver=FakeResolver(results_by_name=_ready_results("jax_coordinator", 4)))
    mock_iris_ctx.return_value = fake_ctx

    initialize_jax(port=9999)

    assert fake_ctx.registry.registered == [
        ("jax_coordinator_ready_0", "10.0.0.1:0"),
        ("jax_coordinator", "10.0.0.1:9999"),
    ]
    mock_jax_init.assert_called_once_with("10.0.0.1:9999", 4, 0, initialization_timeout=1800)
    assert mock_atexit.register.call_count == 2
    mock_atexit.register.assert_any_call(fake_ctx.registry.unregister, "endpoint-1")


@patch("iris.runtime.jax_init.atexit")
@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_task0_uses_iris_port(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    mock_atexit: MagicMock,
) -> None:
    """Task 0 uses IRIS_PORT_jax when available, ignoring the port argument."""
    info = _make_job_info(task_index=0, num_tasks=2)
    info.ports = {"jax": 12345}
    mock_get_job_info.return_value = info
    fake_ctx = FakeContext(resolver=FakeResolver(results_by_name=_ready_results("jax_coordinator", 2)))
    mock_iris_ctx.return_value = fake_ctx

    initialize_jax(port=9999)

    assert fake_ctx.registry.registered == [
        ("jax_coordinator_ready_0", "10.0.0.1:0"),
        ("jax_coordinator", "10.0.0.1:12345"),
    ]
    mock_jax_init.assert_called_once_with("10.0.0.1:12345", 2, 0, initialization_timeout=1800)


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_taskN_polls(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
) -> None:
    """Task N polls for the coordinator endpoint and calls jax.distributed.initialize."""
    mock_get_job_info.return_value = _make_job_info(task_index=2, num_tasks=4)

    empty = ResolveResult(name="jax_coordinator", endpoints=[])
    found = _found_endpoint("jax_coordinator")
    results_by_name = {
        **_ready_results("jax_coordinator", 4),
        "jax_coordinator": [empty, empty, found],
    }
    fake_ctx = FakeContext(resolver=FakeResolver(results_by_name=results_by_name))
    mock_iris_ctx.return_value = fake_ctx

    initialize_jax(poll_timeout=10.0, poll_interval=0.01)

    assert fake_ctx.resolver.call_count >= 3
    assert fake_ctx.registry.registered == [("jax_coordinator_ready_2", "10.0.0.1:0")]
    mock_jax_init.assert_called_once_with("10.0.0.1:8476", 4, 2, initialization_timeout=10)


@patch("iris.runtime.jax_init.atexit")
@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_task0_waits_for_peer_ready_before_coordinator(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    mock_atexit: MagicMock,
) -> None:
    """Task 0 waits for every peer before registering the JAX coordinator."""
    mock_get_job_info.return_value = _make_job_info(task_index=0, num_tasks=2)
    missing = ResolveResult(name="jax_coordinator_ready_1", endpoints=[])
    ready = _found_endpoint("jax_coordinator_ready_1")
    results_by_name = {
        "jax_coordinator_ready_0": [_found_endpoint("jax_coordinator_ready_0")],
        "jax_coordinator_ready_1": [missing, ready],
    }
    fake_ctx = FakeContext(resolver=FakeResolver(results_by_name=results_by_name))
    mock_iris_ctx.return_value = fake_ctx

    initialize_jax(poll_timeout=10.0, poll_interval=0.01)

    assert fake_ctx.resolver.call_count_by_name["jax_coordinator_ready_1"] >= 2
    assert fake_ctx.registry.registered[-1] == ("jax_coordinator", "10.0.0.1:8476")
    mock_jax_init.assert_called_once_with("10.0.0.1:8476", 2, 0, initialization_timeout=10)


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_poll_timeout(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
) -> None:
    """TimeoutError is raised when coordinator endpoint is not found within timeout."""
    mock_get_job_info.return_value = _make_job_info(task_index=1, num_tasks=2)

    empty_ready = ResolveResult(name="jax_coordinator_ready_0", endpoints=[])
    results_by_name = {
        "jax_coordinator_ready_0": [empty_ready],
        "jax_coordinator_ready_1": [_found_endpoint("jax_coordinator_ready_1")],
    }
    fake_ctx = FakeContext(resolver=FakeResolver(results_by_name=results_by_name))
    mock_iris_ctx.return_value = fake_ctx

    with pytest.raises(TimeoutError, match="pre-initialize barrier"):
        initialize_jax(poll_timeout=0.1, poll_interval=0.01)

    mock_jax_init.assert_not_called()


def test_poll_for_coordinator_default_interval() -> None:
    """_poll_for_coordinator works with the default poll_interval=2.0 (must not crash on ExponentialBackoff)."""
    found = ResolveResult(
        name="coord",
        endpoints=[ResolvedEndpoint(url="1.2.3.4:8476", actor_id="ep-1")],
    )
    resolver = FakeResolver(results=[found])
    address = _poll_for_coordinator(resolver, "coord", timeout=10.0, poll_interval=2.0)
    assert address == "1.2.3.4:8476"


def test_poll_for_coordinator_returns_url() -> None:
    """_poll_for_coordinator returns the url from the first resolved endpoint."""
    found = ResolveResult(
        name="coord",
        endpoints=[ResolvedEndpoint(url="1.2.3.4:8476", actor_id="ep-1")],
    )
    resolver = FakeResolver(results=[found])
    address = _poll_for_coordinator(resolver, "coord", timeout=5.0, poll_interval=0.01)
    assert address == "1.2.3.4:8476"


def test_poll_for_coordinator_timeout() -> None:
    """_poll_for_coordinator raises TimeoutError when endpoint never appears."""
    empty = ResolveResult(name="coord", endpoints=[])
    resolver = FakeResolver(results=[empty])

    with pytest.raises(TimeoutError, match="Timed out"):
        _poll_for_coordinator(resolver, "coord", timeout=0.1, poll_interval=0.01)
