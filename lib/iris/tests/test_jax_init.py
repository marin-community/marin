# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import rigging.timing as timing

pytest.importorskip("jax")

import iris.runtime.jax_init as jax_init_module
import jax
from iris.actor.resolver import ResolvedEndpoint, ResolveResult
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from iris.env_resources import _read_iris_resource_proto
from iris.runtime.jax_init import configure_jax_compilation_cache, initialize_jax, resolve_coordinator_port

EXPECTED_JAX_INITIALIZATION_TIMEOUT = 1800
EXPECTED_JAX_HEARTBEAT_TIMEOUT = 100


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


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0

    def monotonic(self) -> float:
        return self.current

    def sleep(self, interval: float) -> None:
        self.current += interval


@dataclass
class FakeExitHooks:
    callbacks: list[tuple[Callable[..., Any], tuple[Any, ...]]] = field(default_factory=list)

    def register(self, callback: Callable[..., Any], *args: Any) -> None:
        self.callbacks.append((callback, args))

    def run(self) -> None:
        for callback, args in self.callbacks:
            callback(*args)


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(jax_init_module, "time", clock)
    monkeypatch.setattr(timing, "time", clock)
    return clock


@pytest.fixture(autouse=True)
def exit_hooks(monkeypatch: pytest.MonkeyPatch) -> FakeExitHooks:
    hooks = FakeExitHooks()
    monkeypatch.setattr(jax_init_module, "atexit", hooks)
    return hooks


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


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_single_task(
    mock_get_job_info: MagicMock,
    _mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
) -> None:
    """Single-task jobs call jax.distributed.initialize with explicit args."""
    mock_get_job_info.return_value = _make_job_info(task_index=0, num_tasks=1)

    with patch("iris.runtime.jax_init.find_free_port", return_value=45678):
        initialize_jax()

    mock_jax_init.assert_called_once_with(
        "10.0.0.1:45678",
        num_processes=1,
        process_id=0,
    )


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
    coordinator_info = _make_job_info(task_index=0, num_tasks=2)
    coordinator_info.ports = {"jax": 12345}
    mock_get_job_info.side_effect = [coordinator_info, _make_job_info(task_index=1, num_tasks=2)]
    found = ResolveResult(
        name="jax_coordinator",
        endpoints=[ResolvedEndpoint(url="10.0.0.1:12345", actor_id="ep-1")],
    )
    fake_ctx = FakeContext(resolver=FakeResolver(results=[found]))
    mock_iris_ctx.return_value = fake_ctx

    initialize_jax()
    initialize_jax(poll_timeout=10.0, poll_interval=0.01)

    assert fake_ctx.registry.registered == [("jax_coordinator", "10.0.0.1:12345")]
    assert mock_jax_init.call_args_list == [
        call(
            "10.0.0.1:12345",
            2,
            0,
            initialization_timeout=EXPECTED_JAX_INITIALIZATION_TIMEOUT,
            heartbeat_timeout_seconds=EXPECTED_JAX_HEARTBEAT_TIMEOUT,
        ),
        call(
            "10.0.0.1:12345",
            2,
            1,
            initialization_timeout=EXPECTED_JAX_INITIALIZATION_TIMEOUT,
            heartbeat_timeout_seconds=EXPECTED_JAX_HEARTBEAT_TIMEOUT,
        ),
    ]


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_no_job_info(
    mock_get_job_info: MagicMock,
    _mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
) -> None:
    """No job info means we're not in an Iris job — skip distributed init."""
    mock_get_job_info.return_value = None

    initialize_jax()

    mock_jax_init.assert_not_called()


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_task0_registers(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    exit_hooks: FakeExitHooks,
) -> None:
    """Task 0 registers the coordinator endpoint and calls jax.distributed.initialize."""
    mock_get_job_info.return_value = _make_job_info(task_index=0, num_tasks=4)
    fake_ctx = FakeContext()
    mock_iris_ctx.return_value = fake_ctx

    initialize_jax(port=9999, heartbeat_timeout=37)

    assert fake_ctx.registry.registered == [("jax_coordinator", "10.0.0.1:9999")]
    mock_jax_init.assert_called_once_with(
        "10.0.0.1:9999",
        4,
        0,
        initialization_timeout=EXPECTED_JAX_INITIALIZATION_TIMEOUT,
        heartbeat_timeout_seconds=37,
    )
    exit_hooks.run()
    assert fake_ctx.registry.unregistered == ["endpoint-1"]


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_task0_uses_iris_port(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
) -> None:
    """Task 0 uses IRIS_PORT_JAX when available, ignoring the port argument."""
    info = _make_job_info(task_index=0, num_tasks=2)
    info.ports = {"jax": 12345}
    mock_get_job_info.return_value = info
    fake_ctx = FakeContext()
    mock_iris_ctx.return_value = fake_ctx

    initialize_jax(port=9999)

    assert fake_ctx.registry.registered == [("jax_coordinator", "10.0.0.1:12345")]
    mock_jax_init.assert_called_once_with(
        "10.0.0.1:12345",
        2,
        0,
        initialization_timeout=EXPECTED_JAX_INITIALIZATION_TIMEOUT,
        heartbeat_timeout_seconds=EXPECTED_JAX_HEARTBEAT_TIMEOUT,
    )


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_taskN_polls(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    fake_clock: FakeClock,
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

    mock_jax_init.assert_called_once_with(
        "10.0.0.1:8476",
        4,
        2,
        initialization_timeout=EXPECTED_JAX_INITIALIZATION_TIMEOUT,
        heartbeat_timeout_seconds=EXPECTED_JAX_HEARTBEAT_TIMEOUT,
    )


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_poll_timeout(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    fake_clock: FakeClock,
) -> None:
    """TimeoutError is raised when coordinator endpoint is not found within timeout."""
    mock_get_job_info.return_value = _make_job_info(task_index=1, num_tasks=2)

    empty = ResolveResult(name="jax_coordinator", endpoints=[])
    fake_ctx = FakeContext(resolver=FakeResolver(results=[empty]))
    mock_iris_ctx.return_value = fake_ctx

    with pytest.raises(TimeoutError, match="Timed out"):
        initialize_jax(poll_timeout=0.1, poll_interval=0.01)

    mock_jax_init.assert_not_called()
    assert fake_clock.current == 0.1


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_supervised_single_host(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local peer resolves the address published by global rank 0."""
    mock_get_job_info.return_value = _make_job_info(task_index=0, num_tasks=1)
    found = ResolveResult(
        name="jax_coordinator",
        endpoints=[ResolvedEndpoint(url="10.0.0.1:12345", actor_id="ep-1")],
    )
    fake_ctx = FakeContext(resolver=FakeResolver(results=[found]))
    mock_iris_ctx.return_value = fake_ctx
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_COUNT", "8")
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_INDEX", "3")
    monkeypatch.setenv("IRIS_MULTIGPU_LOCAL_DEVICE_IDS", "3")

    initialize_jax()

    mock_jax_init.assert_called_once_with(
        "10.0.0.1:12345",
        8,
        3,
        local_device_ids=[3],
        initialization_timeout=EXPECTED_JAX_INITIALIZATION_TIMEOUT,
        heartbeat_timeout_seconds=EXPECTED_JAX_HEARTBEAT_TIMEOUT,
    )
    assert fake_ctx.registry.registered == []


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_supervised_global_rank0_registers(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global rank 0 on a multi-host supervised job registers the coordinator."""
    info = _make_job_info(task_index=0, num_tasks=2)
    info.ports = {"jax": 12345}
    mock_get_job_info.return_value = info
    fake_ctx = FakeContext()
    mock_iris_ctx.return_value = fake_ctx
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_COUNT", "16")
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_INDEX", "0")
    monkeypatch.setenv("IRIS_MULTIGPU_LOCAL_DEVICE_IDS", "0")

    initialize_jax()

    assert fake_ctx.registry.registered == [("jax_coordinator", "10.0.0.1:12345")]
    mock_jax_init.assert_called_once_with(
        "10.0.0.1:12345",
        16,
        0,
        local_device_ids=[0],
        initialization_timeout=EXPECTED_JAX_INITIALIZATION_TIMEOUT,
        heartbeat_timeout_seconds=EXPECTED_JAX_HEARTBEAT_TIMEOUT,
    )


@patch("jax.distributed.initialize")
@patch("iris.runtime.jax_init.iris_ctx")
@patch("iris.runtime.jax_init.get_job_info")
def test_initialize_jax_supervised_global_rank0_picks_port(
    mock_get_job_info: MagicMock,
    mock_iris_ctx: MagicMock,
    mock_jax_init: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global rank 0 chooses and publishes the port without supervisor help."""
    mock_get_job_info.return_value = _make_job_info(task_index=0, num_tasks=2)
    fake_ctx = FakeContext()
    mock_iris_ctx.return_value = fake_ctx
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_COUNT", "16")
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_INDEX", "0")
    monkeypatch.setenv("IRIS_MULTIGPU_LOCAL_DEVICE_IDS", "0")

    with patch("iris.runtime.jax_init.find_free_port", return_value=45678):
        initialize_jax()

    coordinator = "10.0.0.1:45678"
    assert fake_ctx.registry.registered == [("jax_coordinator", coordinator)]
    mock_jax_init.assert_called_once_with(
        coordinator,
        16,
        0,
        local_device_ids=[0],
        initialization_timeout=EXPECTED_JAX_INITIALIZATION_TIMEOUT,
        heartbeat_timeout_seconds=EXPECTED_JAX_HEARTBEAT_TIMEOUT,
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
        "10.0.0.9:8476",
        16,
        8,
        local_device_ids=[0],
        initialization_timeout=EXPECTED_JAX_INITIALIZATION_TIMEOUT,
        heartbeat_timeout_seconds=EXPECTED_JAX_HEARTBEAT_TIMEOUT,
    )
    assert fake_ctx.registry.registered == []


@pytest.mark.parametrize("assigned", [{}, {"jax": 0}], ids=["unassigned", "k8s-placeholder"])
def test_resolve_coordinator_port_uses_kernel_fallback(assigned: dict[str, int]) -> None:
    info = _make_job_info()
    info.ports = assigned

    with patch("iris.runtime.jax_init.find_free_port", return_value=45678) as find_port:
        assert resolve_coordinator_port(info) == 45678
    find_port.assert_called_once_with()


def test_resolve_coordinator_port_prefers_assigned_port() -> None:
    info = _make_job_info()
    info.ports = {"jax": 12345}

    assert resolve_coordinator_port(info, 9999) == 12345
    assert resolve_coordinator_port(_make_job_info(), 9999) == 9999


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
            os.environ.pop("XLA_FLAGS", None)
            os.environ.pop("IRIS_TASK_RESOURCES", None)
            # Off a real launch by default, so the autotune cache stays node-local
            # and never reaches for object storage.
            os.environ.pop("MARIN_PROVENANCE", None)
            _read_iris_resource_proto.cache_clear()
            yield
    finally:
        _read_iris_resource_proto.cache_clear()
        jax.config.update("jax_compilation_cache_dir", original_cache_dir)
        jax.config.update("jax_persistent_cache_enable_xla_caches", original_enable_xla_caches)


@contextmanager
def _gpu_task(tmp_path):
    """Present this process as an Iris GPU task with the scratch cache mount in place."""
    scratch_cache_dir = tmp_path / "scratch-cache"
    scratch_cache_dir.mkdir()
    os.environ["IRIS_TASK_RESOURCES"] = '{"device": {"gpu": {"count": 4, "variant": "GB200"}}}'
    _read_iris_resource_proto.cache_clear()
    with patch.object(jax_init_module, "SCRATCH_CACHE_PATH", str(scratch_cache_dir)):
        yield scratch_cache_dir


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

        configure_jax_compilation_cache()
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


def test_remote_cache_points_xla_autotune_at_the_node_mount(tmp_path) -> None:
    """A GPU task with a remote JAX cache autotunes into the node-local mount."""
    with _isolated_jax_cache_config(), _gpu_task(tmp_path) as scratch_cache_dir:
        os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="
        with patch("iris.runtime.jax_init.marin_prefix", return_value="s3://marin-eu/marin/"):
            configure_jax_compilation_cache()

        autotune_dir = f"{scratch_cache_dir}/xla/per-fusion-autotune"
        assert f"--xla_gpu_per_fusion_autotune_cache_dir={autotune_dir}" in os.environ["XLA_FLAGS"].split()
        # The pre-existing flag survives; XLA_FLAGS is additive, not replaced.
        assert "--xla_gpu_enable_command_buffer=" in os.environ["XLA_FLAGS"].split()
        assert os.path.isdir(autotune_dir)


def test_remote_cache_leaves_xla_flags_alone_without_gpus(tmp_path) -> None:
    """A TPU or CPU jaxlib aborts on an unknown --xla_gpu flag, so never set one there."""
    with _isolated_jax_cache_config(), _gpu_task(tmp_path):
        os.environ["IRIS_TASK_RESOURCES"] = '{"device": {"tpu": {"count": 4, "variant": "v5p-8"}}}'
        _read_iris_resource_proto.cache_clear()
        with patch("iris.runtime.jax_init.marin_prefix", return_value="s3://marin-eu/marin/"):
            configure_jax_compilation_cache()

        assert "XLA_FLAGS" not in os.environ


@pytest.mark.parametrize("mount", ["missing", "file-shadowed"])
def test_remote_cache_skips_the_autotune_flag_when_the_mount_is_unusable(tmp_path, mount) -> None:
    """No mount, or a file where the autotune dir belongs: skip the flag, don't abort JAX init."""
    with _isolated_jax_cache_config(), _gpu_task(tmp_path) as scratch_cache_dir:
        if mount == "missing":
            scratch_cache_path = str(tmp_path / "never-mounted")
        else:
            (scratch_cache_dir / "xla").write_bytes(b"")
            scratch_cache_path = str(scratch_cache_dir)
        with patch.object(jax_init_module, "SCRATCH_CACHE_PATH", scratch_cache_path):
            with patch("iris.runtime.jax_init.marin_prefix", return_value="s3://marin-eu/marin/"):
                configure_jax_compilation_cache()

        assert "XLA_FLAGS" not in os.environ


def test_remote_cache_keeps_an_explicit_xla_autotune_dir(tmp_path) -> None:
    """An operator-chosen autotune directory wins over the node mount."""
    with _isolated_jax_cache_config(), _gpu_task(tmp_path):
        os.environ["XLA_FLAGS"] = "--xla_gpu_per_fusion_autotune_cache_dir=/mnt/elsewhere"
        with patch("iris.runtime.jax_init.marin_prefix", return_value="s3://marin-eu/marin/"):
            configure_jax_compilation_cache()

        assert os.environ["XLA_FLAGS"] == "--xla_gpu_per_fusion_autotune_cache_dir=/mnt/elsewhere"


def test_launch_provenance_mirrors_the_autotune_cache_to_object_storage(tmp_path) -> None:
    """A launched GPU task hands the node-local autotune dir to sync_kv_cache for mirroring."""
    calls: list[tuple[str, str]] = []

    with _isolated_jax_cache_config(), _gpu_task(tmp_path) as scratch_cache_dir:
        os.environ["MARIN_PROVENANCE"] = "{}"
        with (
            patch.object(jax_init_module, "sync_kv_cache", lambda prefix, local: calls.append((prefix, local))),
            patch("iris.runtime.jax_init.marin_prefix", return_value="s3://marin-eu/marin/"),
        ):
            configure_jax_compilation_cache()

    autotune_dir = f"{scratch_cache_dir}/xla/per-fusion-autotune"
    assert calls == [(jax_init_module._XLA_AUTOTUNE_REMOTE_PREFIX, autotune_dir)]


def test_autotune_cache_stays_node_local_without_a_launch_provenance(tmp_path) -> None:
    """Off a real launch (no MARIN_PROVENANCE), the autotune cache never reaches object storage."""
    calls: list = []

    with _isolated_jax_cache_config(), _gpu_task(tmp_path):
        with (
            patch.object(jax_init_module, "sync_kv_cache", lambda *args: calls.append(args)),
            patch("iris.runtime.jax_init.marin_prefix", return_value="s3://marin-eu/marin/"),
        ):
            configure_jax_compilation_cache()

    assert calls == []


def test_explicit_remote_cache_dir_still_gets_the_xla_guard(tmp_path) -> None:
    """The remote-URL guard follows the cache dir, not who set it.

    An explicitly configured ``gs://``/``s3://`` cache dir used to skip the
    guard entirely, so JAX handed XLA a URL and the first compile died with
    "UNIMPLEMENTED: File system scheme not implemented".
    """
    from jax._src import compiler as jax_compiler  # noqa: PLC0415

    with _isolated_jax_cache_config(), _gpu_task(tmp_path) as scratch_cache_dir:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = "s3://explicit/cache"
        configure_jax_compilation_cache()

        options = jax_compiler.get_compile_options(num_replicas=1, num_partitions=1)
        # Not `== ""`: jaxlib folds XLA_FLAGS into the same field, and whether it
        # has already parsed them depends on what ran earlier in the process. The
        # invariant is that XLA is never handed a URL.
        assert "://" not in options.executable_build_options.debug_options.xla_gpu_per_fusion_autotune_cache_dir
        autotune_dir = f"{scratch_cache_dir}/xla/per-fusion-autotune"
        assert f"--xla_gpu_per_fusion_autotune_cache_dir={autotune_dir}" in os.environ["XLA_FLAGS"]


def test_xla_autotune_flag_is_excluded_from_the_compilation_cache_key() -> None:
    """The split depends on JAX ignoring this flag when it hashes XLA_FLAGS.

    If a JAX upgrade drops the exclusion, every node's autotune directory name
    enters the compilation cache key and the shared cache stops hitting.
    """
    from jax._src import cache_key  # noqa: PLC0415

    assert jax_init_module._XLA_AUTOTUNE_CACHE_DIR_FLAG in cache_key.xla_flags_to_exclude_from_cache_key
