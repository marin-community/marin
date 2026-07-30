# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from unittest.mock import MagicMock, call, patch

import pytest

pytest.importorskip("jax")

import jax
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.actor.resolver import ResolvedEndpoint, ResolveResult
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from iris.runtime.jax_init import (
    _JAX_DIST_INIT_TIMEOUT,
    _poll_for_coordinator,
    configure_jax_compilation_cache,
    initialize_jax,
)


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
    call_count: int = 0

    def resolve(self, name: str) -> ResolveResult:
        idx = min(self.call_count, len(self.results) - 1)
        result = self.results[idx]
        self.call_count += 1
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


@pytest.fixture(autouse=True)
def _mock_compilation_cache_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """initialize_jax() always calls configure_jax_compilation_cache() first.

    Unmocked, that call resolves the real marin_prefix() on whatever host runs
    the test; on a host where it resolves a remote gs:// path, it permanently
    flips jax_persistent_cache_enable_xla_caches to "none" for the rest of the
    process (no test here restores it), breaking later compilation-cache
    tests. The tests below that exercise configure_jax_compilation_cache()
    directly call it through their own imported reference, which this
    module-attribute patch does not touch.
    """
    monkeypatch.setattr("iris.runtime.jax_init.configure_jax_compilation_cache", MagicMock())


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

    mock_jax_init.assert_called_once_with("10.0.0.1:8476", num_processes=1, process_id=0)
    mock_iris_ctx.assert_not_called()


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_tpu_multitask_uses_iris_registry(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TPU Iris jobs use the same explicit coordinator and rank wiring as other multi-task jobs."""
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    monkeypatch.setenv("JAX_PLATFORMS", "tpu,cpu")
    mock_get_job_info.side_effect = [
        _make_job_info(task_index=0, num_tasks=2),
        _make_job_info(task_index=1, num_tasks=2),
    ]
    found = ResolveResult(
        name="jax_coordinator",
        endpoints=[ResolvedEndpoint(url="10.0.0.1:8476", actor_id="ep-1")],
    )
    fake_ctx = FakeContext(resolver=FakeResolver(results=[found]))
    mock_iris_ctx.return_value = fake_ctx

    initialize_jax()
    initialize_jax(poll_timeout=10.0, poll_interval=0.01)

    assert fake_ctx.registry.registered == [("jax_coordinator", "10.0.0.1:8476")]
    assert mock_jax_init.call_args_list == [
        call("10.0.0.1:8476", 2, 0, initialization_timeout=_JAX_DIST_INIT_TIMEOUT),
        call("10.0.0.1:8476", 2, 1, initialization_timeout=_JAX_DIST_INIT_TIMEOUT),
    ]


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
    fake_ctx = FakeContext()
    mock_iris_ctx.return_value = fake_ctx

    initialize_jax(port=9999)

    assert fake_ctx.registry.registered == [("jax_coordinator", "10.0.0.1:9999")]
    mock_jax_init.assert_called_once_with("10.0.0.1:9999", 4, 0, initialization_timeout=_JAX_DIST_INIT_TIMEOUT)
    mock_atexit.register.assert_called_once_with(fake_ctx.registry.unregister, "endpoint-1")


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
    fake_ctx = FakeContext()
    mock_iris_ctx.return_value = fake_ctx

    initialize_jax(port=9999)

    assert fake_ctx.registry.registered == [("jax_coordinator", "10.0.0.1:12345")]
    mock_jax_init.assert_called_once_with("10.0.0.1:12345", 2, 0, initialization_timeout=_JAX_DIST_INIT_TIMEOUT)


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
    found = ResolveResult(
        name="jax_coordinator",
        endpoints=[ResolvedEndpoint(url="10.0.0.1:8476", actor_id="ep-1")],
    )
    fake_ctx = FakeContext(resolver=FakeResolver(results=[empty, empty, found]))
    mock_iris_ctx.return_value = fake_ctx

    initialize_jax(poll_timeout=10.0, poll_interval=0.01)

    assert fake_ctx.resolver.call_count >= 3
    mock_jax_init.assert_called_once_with("10.0.0.1:8476", 4, 2, initialization_timeout=_JAX_DIST_INIT_TIMEOUT)


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

    empty = ResolveResult(name="jax_coordinator", endpoints=[])
    fake_ctx = FakeContext(resolver=FakeResolver(results=[empty]))
    mock_iris_ctx.return_value = fake_ctx

    with pytest.raises(TimeoutError, match="Timed out"):
        initialize_jax(poll_timeout=0.1, poll_interval=0.01)

    mock_jax_init.assert_not_called()


@patch("iris.runtime.jax_init.atexit")
@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_supervised_single_host(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    mock_atexit: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supervised non-zero rank on a single host joins via advertise_host, no registry."""
    mock_get_job_info.return_value = _make_job_info(task_index=0, num_tasks=1)
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_COUNT", "8")
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_INDEX", "3")
    monkeypatch.setenv("IRIS_MULTIGPU_LOCAL_DEVICE_IDS", "3")

    initialize_jax()

    mock_jax_init.assert_called_once_with(
        "10.0.0.1:8476", 8, 3, local_device_ids=[3], initialization_timeout=_JAX_DIST_INIT_TIMEOUT
    )
    mock_iris_ctx.assert_not_called()


@patch("iris.runtime.jax_init.atexit")
@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_supervised_global_rank0_registers(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    mock_atexit: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global rank 0 on a multi-host supervised job registers the coordinator."""
    mock_get_job_info.return_value = _make_job_info(task_index=0, num_tasks=2)
    fake_ctx = FakeContext()
    mock_iris_ctx.return_value = fake_ctx
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_COUNT", "16")
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_INDEX", "0")
    monkeypatch.setenv("IRIS_MULTIGPU_LOCAL_DEVICE_IDS", "0")

    initialize_jax()

    assert fake_ctx.registry.registered == [("jax_coordinator", "10.0.0.1:8476")]
    mock_jax_init.assert_called_once_with(
        "10.0.0.1:8476", 16, 0, local_device_ids=[0], initialization_timeout=_JAX_DIST_INIT_TIMEOUT
    )


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_supervised_other_host_polls(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supervised rank on host != 0 polls the registry for rank 0's address."""
    mock_get_job_info.return_value = _make_job_info(task_index=1, num_tasks=2)
    found = ResolveResult(
        name="jax_coordinator",
        endpoints=[ResolvedEndpoint(url="10.0.0.9:8476", actor_id="ep-1")],
    )
    fake_ctx = FakeContext(resolver=FakeResolver(results=[found]))
    mock_iris_ctx.return_value = fake_ctx
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_COUNT", "16")
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_INDEX", "8")
    monkeypatch.setenv("IRIS_MULTIGPU_LOCAL_DEVICE_IDS", "0")

    initialize_jax()

    mock_jax_init.assert_called_once_with(
        "10.0.0.9:8476", 16, 8, local_device_ids=[0], initialization_timeout=_JAX_DIST_INIT_TIMEOUT
    )
    assert fake_ctx.registry.registered == []


# NOT_FOUND is the proper Connect "name absent"; UNIMPLEMENTED is what an older
# controller's bare HTTP 404 decodes to for the same condition. Both must retry.
@pytest.mark.parametrize("pending_code", [Code.NOT_FOUND, Code.UNIMPLEMENTED])
def test_poll_for_coordinator_retries_until_registered(pending_code: Code) -> None:
    """The lookup reports the name absent until rank 0 registers; the poller retries, not crashes."""
    found = ResolveResult(
        name="jax_coordinator",
        endpoints=[ResolvedEndpoint(url="10.0.0.9:8476", actor_id="ep-1")],
    )

    class NotYetRegisteredResolver:
        def __init__(self) -> None:
            self.calls = 0

        def resolve(self, name: str) -> ResolveResult:
            self.calls += 1
            if self.calls < 3:
                raise ConnectError(pending_code, f"{name} not registered")
            return found

    resolver = NotYetRegisteredResolver()
    url = _poll_for_coordinator(resolver, "jax_coordinator", timeout=5.0, poll_interval=0.001)
    assert url == "10.0.0.9:8476"
    assert resolver.calls == 3


def test_poll_for_coordinator_propagates_real_connect_errors() -> None:
    """A Connect error outside the pending set (e.g. PERMISSION_DENIED) is real and propagates."""

    class DeniedResolver:
        def resolve(self, name: str) -> ResolveResult:
            raise ConnectError(Code.PERMISSION_DENIED, "not allowed")

    with pytest.raises(ConnectError):
        _poll_for_coordinator(DeniedResolver(), "jax_coordinator", timeout=5.0, poll_interval=0.001)


@contextmanager
def _isolated_jax_cache_config():
    """Restore ``jax.config`` and cache-related env vars around a test."""
    original_cache_dir = jax.config.jax_compilation_cache_dir
    original_enable_xla_caches = jax.config.jax_persistent_cache_enable_xla_caches
    jax.config.update("jax_compilation_cache_dir", None)
    try:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
            os.environ.pop("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", None)
            yield
    finally:
        jax.config.update("jax_compilation_cache_dir", original_cache_dir)
        jax.config.update("jax_persistent_cache_enable_xla_caches", original_enable_xla_caches)


def test_configure_compilation_cache_derives_from_marin_prefix() -> None:
    """With nothing set, the cache dir is ``marin_prefix()`` + subdir, written to env and jax.config."""
    with _isolated_jax_cache_config():
        with patch("iris.runtime.jax_init.marin_prefix", return_value="gs://marin-eu/marin/"):
            configure_jax_compilation_cache()

        assert os.environ["JAX_COMPILATION_CACHE_DIR"] == "gs://marin-eu/marin/compilation-cache"
        assert jax.config.jax_compilation_cache_dir == "gs://marin-eu/marin/compilation-cache"


@pytest.mark.parametrize("source", ["env", "jax_config"])
def test_configure_compilation_cache_keeps_explicit_dir(source: str) -> None:
    """An explicit cache dir (env var or jax.config) is preserved; the prefix default is not derived."""
    with _isolated_jax_cache_config():
        if source == "env":
            os.environ["JAX_COMPILATION_CACHE_DIR"] = "gs://explicit/cache"
        else:
            jax.config.update("jax_compilation_cache_dir", "gs://explicit/cache")

        with patch("iris.runtime.jax_init.marin_prefix") as mock_prefix:
            configure_jax_compilation_cache()

        mock_prefix.assert_not_called()
        if source == "env":
            assert os.environ["JAX_COMPILATION_CACHE_DIR"] == "gs://explicit/cache"
        else:
            assert jax.config.jax_compilation_cache_dir == "gs://explicit/cache"
            assert "JAX_COMPILATION_CACHE_DIR" not in os.environ


@pytest.mark.parametrize("scheme", ["gs://", "s3://"])
def test_configure_compilation_cache_clears_xla_autotune_compile_option(scheme: str) -> None:
    """A remote cache dir clears the XLA compile option that would otherwise crash.

    Without the guard, JAX hands XLA's C++ debug_options a raw
    ``xla_gpu_per_fusion_autotune_cache_dir`` built from the remote URL, which
    XLA's `tsl::Env` cannot open (UNIMPLEMENTED: file system scheme not
    implemented). This drives JAX's own compile-options builder to confirm the
    field is empty after ``configure_jax_compilation_cache`` runs.
    """
    from jax._src import compiler as jax_compiler  # noqa: PLC0415

    with _isolated_jax_cache_config():
        with patch("iris.runtime.jax_init.marin_prefix", return_value=f"{scheme}marin-eu/marin/"):
            configure_jax_compilation_cache()

        options = jax_compiler.get_compile_options(num_replicas=1, num_partitions=1)
        assert options.executable_build_options.debug_options.xla_gpu_per_fusion_autotune_cache_dir == ""


def test_configure_compilation_cache_keeps_xla_autotune_for_local_dir() -> None:
    """A local cache dir leaves XLA's autotune sub-cache derivation untouched."""
    from jax._src import compiler as jax_compiler  # noqa: PLC0415

    with _isolated_jax_cache_config():
        with patch("iris.runtime.jax_init.marin_prefix", return_value="/mnt/local/marin/"):
            configure_jax_compilation_cache()

        options = jax_compiler.get_compile_options(num_replicas=1, num_partitions=1)
        assert options.executable_build_options.debug_options.xla_gpu_per_fusion_autotune_cache_dir != ""


def test_configure_compilation_cache_keeps_explicit_xla_autotune_setting() -> None:
    """An explicit JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES env var is not overridden."""
    with _isolated_jax_cache_config():
        original = jax.config.jax_persistent_cache_enable_xla_caches
        os.environ["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] = "all"
        with patch("iris.runtime.jax_init.marin_prefix", return_value="s3://marin-eu/marin/"):
            configure_jax_compilation_cache()

        # We never call jax.config.update for this setting when the env var is
        # already present, so jax.config keeps whatever it already had; the env
        # var itself (which JAX reads directly) is left as the caller set it.
        assert jax.config.jax_persistent_cache_enable_xla_caches == original
        assert os.environ["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] == "all"


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
