"""Unit tests for DaytonaEnvironment strategy selection and DinD compose logic."""

import asyncio
import io
import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from daytona.common.errors import (
    DaytonaConflictError,
    DaytonaError,
    DaytonaValidationError,
)
from tenacity import stop_after_attempt, wait_none
from upath import UPath

from harbor.environments.daytona import (
    DaytonaClientManager,
    DaytonaDockerSetupError,
    DaytonaEnvironment,
    DaytonaProcessError,
    DaytonaSandboxStopError,
    _DaytonaDinD,
    _DaytonaDirect,
)
from harbor.environments.daytona import environment as daytona_environment
from harbor.environments.base import (
    ContainerRegistryThrottledError,
    ExecResult,
    SandboxBuildFailedError,
    SandboxDiskFullError,
)
from harbor_config import ErrorCategory, error_category
from harbor_config.errors import is_resume_retryable
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import ResourceMode, ServiceVolumeConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.models.trial.result import ExceptionInfo


class _ChunkBoundedRemoteFile:
    """UPath-like boundary that rejects whole-file convenience methods."""

    def __init__(self, max_chunk_bytes: int):
        self.max_chunk_bytes = max_chunk_bytes
        self.data = b""
        self.parent = self

    def mkdir(self, *, parents: bool, exist_ok: bool) -> None:
        pass

    def write_bytes(self, data: bytes) -> None:
        if len(data) > self.max_chunk_bytes:
            raise OverflowError("remote write received an unbounded payload")
        self.data = data

    def read_bytes(self) -> bytes:
        if len(self.data) > self.max_chunk_bytes:
            raise OverflowError("remote read returned an unbounded payload")
        return self.data

    def open(self, mode: str):
        return _ChunkBoundedBuffer(self, mode)


class _ChunkBoundedBuffer(io.BytesIO):
    def __init__(self, remote_file: _ChunkBoundedRemoteFile, mode: str):
        if mode not in ("rb", "wb"):
            raise ValueError(f"unsupported mode: {mode}")
        super().__init__(remote_file.data if mode == "rb" else b"")
        self._remote_file = remote_file
        self._mode = mode

    def write(self, data: bytes) -> int:
        if len(data) > self._remote_file.max_chunk_bytes:
            raise OverflowError("remote write received an unbounded payload")
        return super().write(data)

    def read(self, size: int = -1) -> bytes:
        if size < 0 or size > self._remote_file.max_chunk_bytes:
            raise OverflowError("remote read requested an unbounded payload")
        return super().read(size)

    def __exit__(self, *args) -> None:
        if self._mode == "wb":
            self._remote_file.data = self.getvalue()
        super().__exit__(*args)


def test_remote_file_staging_writes_in_bounded_chunks(temp_dir: Path) -> None:
    chunk_bytes = daytona_environment._REMOTE_COPY_CHUNK_BYTES
    payload = bytes(range(256)) * (chunk_bytes * 3 // 256)
    local_file = temp_dir / "payload.bin"
    local_file.write_bytes(payload)
    remote_file = _ChunkBoundedRemoteFile(max_chunk_bytes=chunk_bytes)

    daytona_environment._copy_local_file_to_remote(local_file, remote_file)  # type: ignore[arg-type]

    assert remote_file.data == payload


def test_remote_file_staging_reads_in_bounded_chunks(temp_dir: Path) -> None:
    chunk_bytes = daytona_environment._REMOTE_COPY_CHUNK_BYTES
    payload = bytes(range(255, -1, -1)) * (chunk_bytes * 3 // 256)
    remote_file = _ChunkBoundedRemoteFile(max_chunk_bytes=chunk_bytes)
    remote_file.data = payload
    local_file = temp_dir / "payload.bin"

    daytona_environment._copy_remote_file_to_local(remote_file, local_file)  # type: ignore[arg-type]

    assert local_file.read_bytes() == payload


def _make_env(
    temp_dir: Path,
    *,
    compose: bool = False,
    allow_internet: bool = True,
    mounts: list[ServiceVolumeConfig] | None = None,
    extra_docker_compose: list[Path] | None = None,
    cpu_mode: ResourceMode = ResourceMode.AUTO,
    memory_mode: ResourceMode = ResourceMode.AUTO,
    ttl_minutes: int | None = None,
    network_policy: dict[str, str] | None = None,
):
    """Create a DaytonaEnvironment with a minimal valid setup."""
    env_dir = temp_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    if compose:
        (env_dir / "docker-compose.yaml").write_text(
            "services:\n  main:\n    build: .\n"
        )
    else:
        (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")

    trial_dir = temp_dir / "trial"
    trial_dir.mkdir(exist_ok=True)
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    if mounts is None:
        mounts = [
            {
                "type": "bind",
                "source": trial_paths.verifier_dir.resolve().absolute().as_posix(),
                "target": str(EnvironmentPaths.verifier_dir),
            },
            {
                "type": "bind",
                "source": trial_paths.agent_dir.resolve().absolute().as_posix(),
                "target": str(EnvironmentPaths.agent_dir),
            },
            {
                "type": "bind",
                "source": trial_paths.artifacts_dir.resolve().absolute().as_posix(),
                "target": str(EnvironmentPaths.artifacts_dir),
            },
        ]
    kwargs: dict = {}
    kwargs["mounts"] = mounts

    return DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="test-task",
        session_id="Test.Session.123",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            allow_internet=allow_internet,
            cpus=2,
            memory_mb=4096,
        ),
        extra_docker_compose=extra_docker_compose,
        cpu_enforcement_policy=cpu_mode,
        memory_enforcement_policy=memory_mode,
        ttl_minutes=ttl_minutes,
        network_policy=network_policy,
        **kwargs,
    )


# ── Strategy selection ────────────────────────────────────────────────


class TestStrategySelection:
    def test_dockerfile_selects_direct(self, temp_dir):
        env = _make_env(temp_dir, compose=False)
        assert isinstance(env._strategy, _DaytonaDirect)
        assert not env._compose_mode

    def test_compose_selects_dind(self, temp_dir):
        env = _make_env(temp_dir, compose=True)
        assert isinstance(env._strategy, _DaytonaDinD)
        assert env._compose_mode

    def test_extra_compose_selects_dind(self, temp_dir):
        extra = temp_dir / "extra.yaml"
        extra.write_text("services:\n  sidecar:\n    image: redis:7\n")
        env = _make_env(temp_dir, compose=False, extra_docker_compose=[extra])
        assert isinstance(env._strategy, _DaytonaDinD)
        assert env._compose_mode

    def test_validate_raises_when_no_definition(self, temp_dir):
        env_dir = temp_dir / "empty_env"
        env_dir.mkdir()
        trial_dir = temp_dir / "trial"
        trial_dir.mkdir(exist_ok=True)
        trial_paths = TrialPaths(trial_dir=trial_dir)
        trial_paths.mkdir()

        with pytest.raises(FileNotFoundError, match="no environment definition"):
            DaytonaEnvironment(
                environment_dir=env_dir,
                environment_name="bad",
                session_id="s.1",
                trial_paths=trial_paths,
                task_env_config=EnvironmentConfig(),
            )


class TestResourceCapabilities:
    def test_daytona_supports_requests_not_limits(self, temp_dir):
        caps = type(_make_env(temp_dir)).resource_capabilities()
        assert caps is not None
        assert caps.cpu_request is True
        assert caps.memory_request is True
        assert caps.cpu_limit is False
        assert caps.memory_limit is False

    def test_cpu_request_policy_succeeds(self, temp_dir):
        env = _make_env(temp_dir, cpu_mode=ResourceMode.REQUEST)
        assert env._cpu_resource_mode == ResourceMode.REQUEST

    def test_memory_guarantee_policy_rejected(self, temp_dir):
        with pytest.raises(ValueError, match="memory resource limits"):
            _make_env(temp_dir, memory_mode=ResourceMode.GUARANTEE)


class TestNetworkPolicy:
    @pytest.mark.parametrize(
        ("environment_kwargs", "expected"),
        [
            (
                {
                    "network_policy": {
                        "mode": "domain_allow_list",
                        "value": "iris.oa.dev",
                    }
                },
                (None, "iris.oa.dev", None),
            ),
            (
                {
                    "network_policy": {
                        "mode": "network_allow_list",
                        "value": "10.0.0.0/8,192.168.1.0/24",
                    }
                },
                ("10.0.0.0/8,192.168.1.0/24", None, None),
            ),
            ({}, (None, None, True)),
        ],
    )
    def test_network_policy_reaches_snapshot_params(
        self, temp_dir, environment_kwargs, expected
    ):
        env = _make_env(temp_dir, allow_internet=False, **environment_kwargs)

        params = env._snapshot_sandbox_params("test-snapshot")

        assert (
            params.network_allow_list,
            params.domain_allow_list,
            params.network_block_all,
        ) == expected

    def test_domain_allowlist_reaches_image_params(self, temp_dir):
        from daytona import Image

        env = _make_env(
            temp_dir,
            allow_internet=False,
            network_policy={"mode": "domain_allow_list", "value": "iris.oa.dev"},
        )

        params = env._image_sandbox_params(
            image=Image.base("ubuntu:22.04"), resources=None
        )

        assert params.domain_allow_list == "iris.oa.dev"
        assert params.network_allow_list is None
        assert params.network_block_all is None

    @pytest.mark.parametrize(
        ("environment_kwargs", "message"),
        [
            (
                {"network_policy": {"mode": "block_all", "value": "invalid"}},
                "does not accept a value",
            ),
            (
                {"network_policy": {"mode": "domain_allow_list"}},
                "requires a value",
            ),
            (
                {
                    "compose": True,
                    "network_policy": {
                        "mode": "domain_allow_list",
                        "value": "iris.oa.dev",
                    },
                },
                "direct-sandbox",
            ),
        ],
    )
    def test_invalid_network_policy_is_rejected(
        self, temp_dir, environment_kwargs, message
    ):
        with pytest.raises(ValueError, match=message):
            _make_env(temp_dir, **environment_kwargs)


# ── DinD compose command building ─────────────────────────────────────


class TestDinDComposeCmd:
    @pytest.fixture
    def dind(self, temp_dir):
        env = _make_env(temp_dir, compose=True)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        return strategy

    def test_project_name_lowercased_and_dashes(self, dind):
        assert dind._project_name == "test-session-123"

    def test_compose_cmd_is_shlex_safe(self, dind):
        cmd = dind._compose_cmd(["up", "-d"])
        # Should round-trip through shlex.split
        parts = shlex.split(cmd)
        assert parts[0] == "docker"
        assert parts[1] == "compose"
        assert "up" in parts
        assert "-d" in parts

    def test_compose_cmd_includes_project_directory(self, dind):
        cmd = dind._compose_cmd(["build"])
        parts = shlex.split(cmd)
        idx = parts.index("--project-directory")
        assert parts[idx + 1] == "/harbor/environment"

    def test_compose_cmd_includes_compose_files(self, dind):
        cmd = dind._compose_cmd(["build"])
        parts = shlex.split(cmd)
        f_indices = [i for i, p in enumerate(parts) if p == "-f"]
        file_paths = [parts[i + 1] for i in f_indices]
        assert any("docker-compose-resources.json" in p for p in file_paths)
        assert any("docker-compose-build.yaml" in p for p in file_paths)
        assert any("docker-compose-mounts.json" in p for p in file_paths)
        assert any(
            p.endswith("/harbor/environment/docker-compose.yaml") for p in file_paths
        )

    def test_compose_cmd_uses_prebuilt_when_set(self, dind):
        dind._use_prebuilt = True
        cmd = dind._compose_cmd(["build"])
        parts = shlex.split(cmd)
        f_indices = [i for i, p in enumerate(parts) if p == "-f"]
        file_paths = [parts[i + 1] for i in f_indices]
        assert any("docker-compose-prebuilt.yaml" in p for p in file_paths)
        assert not any("docker-compose-build.yaml" in p for p in file_paths)


class TestDinDComposeFileFlags:
    @pytest.fixture
    def dind(self, temp_dir):
        env = _make_env(temp_dir, compose=True)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        return strategy

    def test_flags_are_flat_list_of_pairs(self, dind):
        flags = dind._compose_file_flags()
        # Every odd index should be "-f"
        for i in range(0, len(flags), 2):
            assert flags[i] == "-f"
        # Even indices are paths
        assert len(flags) % 2 == 0

    def test_no_network_appended_when_internet_disabled(self, temp_dir):
        env = _make_env(temp_dir, compose=True, allow_internet=False)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        flags = strategy._compose_file_flags()
        file_paths = [flags[i + 1] for i in range(0, len(flags), 2)]
        assert any("docker-compose-no-network.yaml" in p for p in file_paths)

    def test_no_network_absent_when_internet_allowed(self, dind):
        flags = dind._compose_file_flags()
        file_paths = [flags[i + 1] for i in range(0, len(flags), 2)]
        assert not any("docker-compose-no-network.yaml" in p for p in file_paths)

    def test_mounts_compose_positioned_between_build_and_task_compose(self, dind):
        flags = dind._compose_file_flags()
        file_paths = [flags[i + 1] for i in range(0, len(flags), 2)]
        resources_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-resources.json")
        )
        build_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-build.yaml")
        )
        mounts_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-mounts.json")
        )
        env_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("/harbor/environment/docker-compose.yaml")
        )
        assert resources_idx < build_idx < mounts_idx < env_idx

    def test_extra_compose_positioned_after_task_compose(self, temp_dir):
        extra = temp_dir / "extra.yaml"
        extra.write_text("services:\n  sidecar:\n    image: redis:7\n")
        env = _make_env(
            temp_dir,
            compose=True,
            extra_docker_compose=[extra],
        )
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        flags = strategy._compose_file_flags()
        file_paths = [flags[i + 1] for i in range(0, len(flags), 2)]
        env_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("/harbor/environment/docker-compose.yaml")
        )
        extra_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-extra-0.yaml")
        )
        mounts_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-mounts.json")
        )
        assert mounts_idx < env_idx < extra_idx

    def test_extra_compose_positioned_after_mounts_without_task_compose(self, temp_dir):
        extra = temp_dir / "extra.yaml"
        extra.write_text("services:\n  sidecar:\n    image: redis:7\n")
        env = _make_env(
            temp_dir,
            compose=False,
            extra_docker_compose=[extra],
        )
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        flags = strategy._compose_file_flags()
        file_paths = [flags[i + 1] for i in range(0, len(flags), 2)]
        extra_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-extra-0.yaml")
        )
        mounts_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-mounts.json")
        )
        assert mounts_idx < extra_idx


@pytest.mark.parametrize(
    ("compose_results", "expected_error"),
    [
        (
            [
                ExecResult(
                    return_code=1,
                    stderr='failed to fetch anonymous token: Get "https://auth.docker.io/token": dial tcp [2606:4700:4403::ac40:904e]:443: connect: network is unreachable',
                )
            ],
            DaytonaProcessError,
        ),
        (
            [
                ExecResult(
                    return_code=1,
                    stderr="failed to fetch oauth token: https://auth.docker.io/token: 504 Gateway Timeout",
                )
            ],
            DaytonaProcessError,
        ),
        (
            [
                ExecResult(
                    return_code=1,
                    stderr="failed to resolve source metadata for docker.io/library/python:3.12-slim: unexpected status from HEAD request to https://registry-1.docker.io/v2/library/python/manifests/3.12-slim: 502 Bad Gateway",
                )
            ],
            DaytonaProcessError,
        ),
        (
            [
                ExecResult(
                    return_code=1,
                    stderr="loaded metadata from registry-1.docker.io\nRUN curl https://example.com: network is unreachable",
                )
            ],
            SandboxBuildFailedError,
        ),
        (
            [
                ExecResult(
                    return_code=1,
                    stderr="failed to resolve source metadata: 429 Too Many Requests",
                )
            ],
            ContainerRegistryThrottledError,
        ),
        (
            [ExecResult(return_code=1, stderr="Dockerfile parse error on line 3")],
            SandboxBuildFailedError,
        ),
        (
            [
                ExecResult(
                    return_code=1,
                    stderr=(
                        "pip._vendor.urllib3.exceptions.ReadTimeoutError: "
                        "HTTPSConnectionPool(host='files.pythonhosted.org', port=443): "
                        "Read timed out."
                    ),
                )
            ],
            SandboxBuildFailedError,
        ),
        (
            [
                ExecResult(return_code=0),
                ExecResult(return_code=1, stderr="429 Too Many Requests"),
            ],
            ContainerRegistryThrottledError,
        ),
    ],
)
async def test_dind_start_classifies_compose_failures(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    compose_results: list[ExecResult],
    expected_error: type[Exception],
) -> None:
    env = _make_env(temp_dir, compose=True)
    strategy = env._strategy
    assert isinstance(strategy, _DaytonaDinD)

    monkeypatch.setattr(
        DaytonaClientManager, "get_instance", AsyncMock(return_value=MagicMock())
    )
    monkeypatch.setattr(env, "_configure_daytona_client", AsyncMock())
    monkeypatch.setattr(env, "_create_sandbox", AsyncMock())
    monkeypatch.setattr(env, "_sdk_upload_file", AsyncMock())
    monkeypatch.setattr(env, "_sdk_upload_dir", AsyncMock())
    monkeypatch.setattr(strategy, "_wait_for_docker_daemon", AsyncMock())
    monkeypatch.setattr(strategy, "_stage_resources_compose_file", AsyncMock())
    monkeypatch.setattr(strategy, "_stage_extra_compose_files", AsyncMock())
    monkeypatch.setattr(strategy, "_stage_mounts_compose_file", AsyncMock())
    monkeypatch.setattr(
        strategy, "_vm_exec", AsyncMock(return_value=ExecResult(return_code=0))
    )
    monkeypatch.setattr(
        strategy,
        "_compose_exec",
        AsyncMock(side_effect=compose_results),
    )

    with pytest.raises(expected_error):
        await strategy.start(force_build=False)


async def test_dind_docker_daemon_start_failure_is_retryable_infrastructure(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_env(temp_dir, compose=True)
    strategy = env._strategy
    assert isinstance(strategy, _DaytonaDinD)
    monkeypatch.setattr(strategy, "_DOCKER_DAEMON_TIMEOUT_SEC", 2)
    monkeypatch.setattr(
        strategy,
        "_vm_exec",
        AsyncMock(
            return_value=ExecResult(
                return_code=1,
                stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
            )
        ),
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    with pytest.raises(DaytonaDockerSetupError):
        await strategy._wait_for_docker_daemon()


@pytest.fixture
def dind_startup(temp_dir, monkeypatch):
    env = _make_env(temp_dir, compose=True)
    env._auto_stop_interval = 0
    sandboxes = []
    deleted = []

    async def create(**kwargs):
        # A replacement must never leak the previous failed sandbox.
        assert len(deleted) == len(sandboxes)
        sandbox_id = f"dind-{len(sandboxes)}"

        async def delete():
            deleted.append(sandbox_id)

        sandboxes.append(SimpleNamespace(id=sandbox_id, delete=delete, fs=AsyncMock()))
        return sandbox_id

    client = SimpleNamespace(get=AsyncMock(side_effect=lambda sid: sandboxes[-1]))
    manager = SimpleNamespace(get_client=AsyncMock(return_value=client))
    monkeypatch.setattr(
        DaytonaClientManager, "get_instance", AsyncMock(return_value=manager)
    )
    monkeypatch.setattr(daytona_environment, "create_in_subprocess", create)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(_DaytonaDinD.start.retry, "sleep", _no_sleep)
    return env, sandboxes, deleted


@pytest.mark.unit
@pytest.mark.parametrize("phase", ["launcher", "daemon", "build", "up", "main"])
@pytest.mark.parametrize("failures", [1, 3])
async def test_dind_start_retries_setup_and_surfaces_diagnostics(
    dind_startup, monkeypatch, phase, failures
):
    env, sandboxes, deleted = dind_startup

    async def execute(command, **kwargs):
        argv = shlex.split(command)
        if argv[0] == "tail":
            return ExecResult(return_code=0, stdout="x" * 20000 + "daemon-log-end")
        failing = len(sandboxes) <= failures and (
            (phase == "launcher" and argv[0] == "dockerd-entrypoint.sh")
            or (phase == "daemon" and command == "docker info")
            or (phase in {"build", "up"} and phase in argv)
            or (phase == "main" and argv[-1] == "true")
        )
        return ExecResult(
            return_code=1 if failing else 0,
            stderr="Cannot connect to the Docker daemon" if failing else "",
            stdout="docker-info-output" if command == "docker info" else "",
        )

    monkeypatch.setattr(env, "_sandbox_exec", execute)
    if failures == 1:
        await env.start(force_build=False)
        assert len(sandboxes) == 2
        assert deleted == ["dind-0"]
        await env.stop(delete=True)
        assert deleted == ["dind-0", "dind-1"]
        return

    with pytest.raises(DaytonaDockerSetupError) as error:
        try:
            await env.start(force_build=False)
        except DaytonaDockerSetupError as exc:
            persisted = ExceptionInfo.from_exception(exc)
            raise
    assert len(sandboxes) == 3
    assert deleted == ["dind-0", "dind-1"]
    assert "Cannot connect to the Docker daemon" in str(error.value)
    assert "docker-info-output" in persisted.exception_traceback
    assert "daemon-log-end" in persisted.exception_traceback
    assert "x" * 8193 not in persisted.exception_traceback
    assert error_category(persisted.exception_type) is ErrorCategory.INFRASTRUCTURE
    assert is_resume_retryable(persisted.exception_type)
    await env.stop(delete=True)
    assert deleted == ["dind-0", "dind-1", "dind-2"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "error_type", [ValueError, DaytonaDockerSetupError, asyncio.CancelledError]
)
async def test_dind_start_preserves_original_failure_when_diagnostics_fail(
    dind_startup, monkeypatch, error_type
):
    env, sandboxes, deleted = dind_startup
    original = error_type("startup failed")

    async def execute(command, **kwargs):
        if command == "docker info" or command.startswith("tail "):
            raise OSError("diagnostic transport unavailable")
        raise original

    monkeypatch.setattr(env, "_sandbox_exec", execute)
    with pytest.raises(type(original)) as error:
        await env.start(force_build=False)
    assert error.value is original
    attempts = 3 if error_type is DaytonaDockerSetupError else 1
    assert len(sandboxes) == attempts
    assert deleted == [f"dind-{i}" for i in range(attempts - 1)]
    if error_type is not asyncio.CancelledError:
        assert any(
            "diagnostic transport unavailable" in note for note in original.__notes__
        )


# ── DinD compose env vars ─────────────────────────────────────────────


class TestDinDComposeEnvVars:
    @pytest.fixture
    def dind(self, temp_dir):
        env = _make_env(temp_dir, compose=True)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        return strategy

    def test_contains_required_keys(self, dind):
        env_vars = dind._compose_env_vars()
        required = {
            "CONTEXT_DIR",
            "MAIN_IMAGE_NAME",
            "CPUS",
            "MEMORY",
        }
        assert required <= set(env_vars.keys())

    def test_legacy_path_keys_are_self_bound(self, dind):
        env_vars = dind._compose_env_vars()
        assert env_vars["HOST_VERIFIER_LOGS_PATH"] == str(EnvironmentPaths.verifier_dir)
        assert env_vars["ENV_VERIFIER_LOGS_PATH"] == str(EnvironmentPaths.verifier_dir)
        assert env_vars["HOST_AGENT_LOGS_PATH"] == str(EnvironmentPaths.agent_dir)
        assert env_vars["ENV_AGENT_LOGS_PATH"] == str(EnvironmentPaths.agent_dir)
        assert env_vars["HOST_ARTIFACTS_PATH"] == str(EnvironmentPaths.artifacts_dir)
        assert env_vars["ENV_ARTIFACTS_PATH"] == str(EnvironmentPaths.artifacts_dir)

    def test_context_dir_points_to_environment(self, dind):
        assert dind._compose_env_vars()["CONTEXT_DIR"] == "/harbor/environment"

    def test_image_name_includes_env_name(self, dind):
        assert dind._compose_env_vars()["MAIN_IMAGE_NAME"] == "hb__test-task"

    def test_resources_from_config(self, dind):
        env_vars = dind._compose_env_vars()
        assert env_vars["CPUS"] == "2"
        assert env_vars["MEMORY"] == "4096M"

    def test_prebuilt_image_included_when_set(self, dind):
        dind._use_prebuilt = True
        dind._env.task_env_config = EnvironmentConfig(docker_image="myimage:latest")
        env_vars = dind._compose_env_vars()
        assert env_vars["PREBUILT_IMAGE_NAME"] == "myimage:latest"

    def test_prebuilt_image_absent_when_not_set(self, dind):
        env_vars = dind._compose_env_vars()
        assert "PREBUILT_IMAGE_NAME" not in env_vars

    def test_infra_vars_win_over_task_and_persistent_env(self, dind, caplog):
        dind._resolved_task_env = {"CPUS": "999", "CONTEXT_DIR": "/wrong"}
        dind._env._persistent_env = {"MEMORY": "1G", "MAIN_IMAGE_NAME": "wrong-image"}

        with caplog.at_level(logging.WARNING):
            env_vars = dind._compose_env_vars()

        assert env_vars["CPUS"] == "2"
        assert env_vars["MEMORY"] == "4096M"
        assert env_vars["CONTEXT_DIR"] == "/harbor/environment"
        assert env_vars["MAIN_IMAGE_NAME"] == "hb__test-task"
        assert any("CPUS" in rec.message for rec in caplog.records)


# ── DinD log path mapping ─────────────────────────────────────────────


class TestSandboxLogPath:
    @pytest.fixture
    def dind(self, temp_dir):
        env = _make_env(temp_dir, compose=True)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        return strategy

    def test_verifier_dir_returns_self(self, dind):
        path = str(EnvironmentPaths.verifier_dir)
        assert dind._sandbox_log_path(path) == path

    def test_agent_dir_returns_self(self, dind):
        path = str(EnvironmentPaths.agent_dir)
        assert dind._sandbox_log_path(path) == path

    def test_artifacts_dir_returns_self(self, dind):
        path = str(EnvironmentPaths.artifacts_dir)
        assert dind._sandbox_log_path(path) == path

    def test_subpath_returns_self(self, dind):
        path = str(EnvironmentPaths.verifier_dir) + "/reward.txt"
        assert dind._sandbox_log_path(path) == path

    def test_non_log_path_returns_none(self, dind):
        assert dind._sandbox_log_path("/home/user/code") is None

    def test_partial_prefix_no_match(self, dind):
        # e.g. /logs/verifier_extra should NOT match /logs/verifier
        path = str(EnvironmentPaths.verifier_dir) + "_extra"
        assert dind._sandbox_log_path(path) is None


# ── Self-bind volume resolution ───────────────────────────────────────


class TestResolveVolumes:
    def test_self_binds_trial_bind_mounts(self, temp_dir):
        mounts: list[ServiceVolumeConfig] = [
            {
                "type": "bind",
                "source": "/host/never/applies/agent",
                "target": str(EnvironmentPaths.agent_dir),
            },
            {
                "type": "bind",
                "source": "/host/never/applies/verifier",
                "target": str(EnvironmentPaths.verifier_dir),
            },
        ]
        env = _make_env(temp_dir, compose=True, mounts=mounts)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        volumes = strategy._resolve_volumes()
        assert [v["source"] for v in volumes] == [v["target"] for v in volumes]
        assert {v["target"] for v in volumes} == {
            str(EnvironmentPaths.agent_dir),
            str(EnvironmentPaths.verifier_dir),
        }

    def test_self_binds_every_mount(self, temp_dir):
        """Every bind mount in `mounts` (base or user-additive) gets
        self-bound — the trial now passes the combined list."""
        combined: list[ServiceVolumeConfig] = [
            {
                "type": "bind",
                "source": "/discarded",
                "target": str(EnvironmentPaths.verifier_dir),
            },
            {
                "type": "bind",
                "source": "/discarded",
                "target": "/in/container/extra",
            },
        ]
        env = _make_env(temp_dir, compose=True, mounts=combined)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        volumes = strategy._resolve_volumes()
        assert [v["source"] for v in volumes] == [v["target"] for v in volumes]


class TestStageMountsComposeFile:
    async def test_writes_json_locally_and_uploads_to_vm(self, temp_dir):
        mounts: list[ServiceVolumeConfig] = [
            {
                "type": "bind",
                "source": "/discarded",
                "target": str(EnvironmentPaths.verifier_dir),
            }
        ]
        env = _make_env(temp_dir, compose=True, mounts=mounts)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)

        uploaded: list[tuple[str, str, dict]] = []

        async def _fake_upload(source, target):
            source = Path(source)
            assert source.name == "docker-compose-mounts.json"
            assert source.parent != env.trial_paths.trial_dir
            uploaded.append((str(source), target, json.loads(source.read_text())))

        env._sdk_upload_file = _fake_upload  # type: ignore[method-assign]

        volumes = strategy._resolve_volumes()
        await strategy._stage_mounts_compose_file(volumes)

        source, target, body = uploaded[0]
        assert not Path(source).exists()
        assert not list(env.trial_paths.trial_dir.glob("*docker-compose-mounts.json"))
        assert body["services"]["main"]["volumes"] == cast(list, volumes)

        # Uploaded under the shared compose dir on the VM with the canonical name.
        assert target == "/harbor/compose/docker-compose-mounts.json"


# ── _sandbox_exec shell parameter ─────────────────────────────────────


class TestSandboxExecShellParam:
    def test_direct_strategy_properties(self, temp_dir):
        """Direct strategy should use default shell (bash -lc)."""
        env = _make_env(temp_dir, compose=False)
        assert isinstance(env._strategy, _DaytonaDirect)

    def test_dind_strategy_properties(self, temp_dir):
        """DinD strategy should exist and have compose mode."""
        env = _make_env(temp_dir, compose=True)
        assert isinstance(env._strategy, _DaytonaDinD)
        assert env._compose_mode


# ── Client configuration kwarg plumbing ───────────────────────────────


class _StubClientManager:
    """Records calls to ``configure`` without spinning up a real client."""

    def __init__(self):
        self.configure_calls: list[dict] = []

    async def configure(self, **kwargs):
        self.configure_calls.append(kwargs)


class TestConfigureDaytonaClient:
    async def test_absent_kwarg_does_not_call_configure(self, temp_dir):
        env = _make_env(temp_dir)
        stub = _StubClientManager()
        env._client_manager = stub
        await env._configure_daytona_client()
        assert stub.configure_calls == []

    async def test_int_kwarg_forwards_to_configure(self, temp_dir):
        env = _make_env(temp_dir)
        env._kwargs["connection_pool_maxsize"] = 500
        stub = _StubClientManager()
        env._client_manager = stub
        await env._configure_daytona_client()
        assert stub.configure_calls == [{"connection_pool_maxsize": 500}]

    async def test_none_kwarg_forwards_explicit_none(self, temp_dir):
        env = _make_env(temp_dir)
        env._kwargs["connection_pool_maxsize"] = None
        stub = _StubClientManager()
        env._client_manager = stub
        await env._configure_daytona_client()
        assert stub.configure_calls == [{"connection_pool_maxsize": None}]


# ── DaytonaClientManager first-wins semantics ─────────────────────────


async def test_concurrent_daytona_accounts_keep_sdk_requests_and_cleanup_separate(
    monkeypatch,
):
    class AccountClient:
        def __init__(self, config):
            self.key = config.api_key
            self.closed = False

        async def create(self):
            assert not self.closed
            return {"account": self.key}

        async def close(self):
            self.closed = True

    monkeypatch.setattr(daytona_environment, "AsyncDaytona", AccountClient)
    monkeypatch.setenv("DAYTONA_API_KEY", "ambient-account")
    train = await DaytonaClientManager.get_instance(api_key="test-training-account")
    evaluation = await DaytonaClientManager.get_instance(
        api_key="test-evaluation-account"
    )
    async with asyncio.TaskGroup() as group:
        train_future = group.create_task(train.get_client())
        eval_future = group.create_task(evaluation.get_client())
    train_client, eval_client = train_future.result(), eval_future.result()
    try:
        assert await train_client.create() == {"account": "test-training-account"}
        assert await eval_client.create() == {"account": "test-evaluation-account"}
        await train._cleanup()
        assert train_client.closed
        assert await eval_client.create() == {"account": "test-evaluation-account"}
    finally:
        await train._cleanup()
        await evaluation._cleanup()


class TestDaytonaClientManagerConfigure:
    async def test_first_call_stores_value(self):
        mgr = DaytonaClientManager()
        await mgr.configure(connection_pool_maxsize=500)
        assert mgr._client_config_set is True
        assert mgr._connection_pool_maxsize == 500

    async def test_repeated_same_value_is_silent(self, caplog):
        mgr = DaytonaClientManager()
        await mgr.configure(connection_pool_maxsize=500)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            await mgr.configure(connection_pool_maxsize=500)
        assert caplog.records == []
        assert mgr._connection_pool_maxsize == 500

    async def test_conflicting_value_warns_and_keeps_first(self, caplog):
        mgr = DaytonaClientManager()
        await mgr.configure(connection_pool_maxsize=500)
        with caplog.at_level(logging.WARNING):
            await mgr.configure(connection_pool_maxsize=999)
        assert "already configured" in caplog.text
        assert mgr._connection_pool_maxsize == 500

    async def test_configure_after_client_built_warns(self, caplog):
        mgr = DaytonaClientManager()
        # Simulate a client that was built before any configure() call.
        # configure() only checks ``is not None``; it never dereferences.
        mgr._client = object()  # type: ignore[assignment]
        with caplog.at_level(logging.WARNING):
            await mgr.configure(connection_pool_maxsize=500)
        assert "before any explicit configuration" in caplog.text
        assert mgr._client_config_set is False
        assert mgr._connection_pool_maxsize is None

    async def test_explicit_none_is_preserved(self):
        mgr = DaytonaClientManager()
        await mgr.configure(connection_pool_maxsize=None)
        assert mgr._client_config_set is True
        assert mgr._connection_pool_maxsize is None

    async def test_cleanup_resets_config_so_reconfigure_takes_effect(self):
        """Cleanup must clear recorded config; otherwise a process that closes
        and reopens the client (notebooks, test suites, library embedding)
        would keep using the first-ever value even after reconfiguration."""
        mgr = DaytonaClientManager()
        await mgr.configure(connection_pool_maxsize=5)
        await mgr._cleanup()
        assert mgr._client_config_set is False
        assert mgr._connection_pool_maxsize is None
        await mgr.configure(connection_pool_maxsize=9)
        assert mgr._connection_pool_maxsize == 9

    def test_atexit_cleanup_handles_cancelled_sdk_close(self):
        class CancellingClient:
            def __init__(self):
                self.close_started = False

            async def close(self):
                self.close_started = True
                raise asyncio.CancelledError

        mgr = DaytonaClientManager()
        client = CancellingClient()
        mgr._client = client  # type: ignore[assignment]

        # An exception escaping an atexit callback is printed as an ignored
        # traceback, so returning normally is part of the callback contract.
        mgr._cleanup_sync()

        assert client.close_started


# ── SDK download to local vs. remote (GCS-style) targets ──────────────


@dataclass
class _FakeFileInfo:
    is_dir: bool


class _FakeDaytonaFs:
    """Minimal stand-in for ``AsyncSandbox.fs``.

    Holds an in-sandbox tree of ``{path: bytes}`` and mimics the real Daytona
    SDK contract: the download primitives write to the *local* filesystem and
    the upload primitives read from it.
    """

    def __init__(self, files: dict[str, bytes]):
        self._files = files

    async def search_files(self, source_dir: str, pattern: str):
        prefix = source_dir.rstrip("/") + "/"
        matched = [p for p in self._files if p.startswith(prefix)]
        return type("SearchResult", (), {"files": matched})()

    async def get_file_info(self, path: str):
        return _FakeFileInfo(is_dir=path not in self._files)

    async def download_file(self, source_path: str, destination: str):
        Path(destination).write_bytes(self._files[source_path])

    async def download_files(self, files):
        for req in files:
            Path(req.destination).write_bytes(self._files[req.source])

    async def upload_file(self, source_path: str, target_path: str):
        self._files[target_path] = Path(source_path).read_bytes()

    async def upload_files(self, files):
        for req in files:
            self._files[req.destination] = Path(req.source).read_bytes()


class _FailedSearchDaytonaFs(_FakeDaytonaFs):
    async def search_files(self, source_dir: str, pattern: str):
        raise DaytonaError("Failed to search files: 502 Bad Gateway")


class _FakeSandbox:
    def __init__(self, files: dict[str, bytes]):
        self.fs = _FakeDaytonaFs(files)


class TestSdkDownloadTargets:
    """``_sdk_download_*`` must land files at both local and remote targets.

    Regression for the UPath/GCS trial-dir migration: a ``gs://`` trial dir
    makes ``TrialPaths.verifier_dir`` a remote ``UPath`` so the verifier's
    ``download_dir`` target was no longer a local path. The Daytona SDK only
    writes locally, so remote targets must be staged through a temp dir and
    copied into object storage.
    """

    @pytest.fixture
    def env(self, temp_dir: Path):
        e = _make_env(temp_dir, compose=False)
        e._sandbox = cast(
            object,
            _FakeSandbox(
                {
                    "/logs/verifier/reward.json": b'{"reward": 1.0}',
                    "/logs/verifier/test-stdout.txt": b"ok\n",
                }
            ),
        )
        return e

    async def test_download_dir_to_local_target(self, env, temp_dir: Path):
        target = temp_dir / "out_verifier"
        await env._sdk_download_dir("/logs/verifier", target)
        assert json.loads((target / "reward.json").read_text()) == {"reward": 1.0}
        assert (target / "test-stdout.txt").read_text() == "ok\n"

    async def test_download_dir_to_remote_target(self, env):
        # ``str(GCSPath)`` is what the verifier passes; emulate with memory://,
        # a non-local fsspec protocol that needs no network or credentials.
        target = "memory://trial-remote/verifier"
        await env._sdk_download_dir(target_dir=target, source_dir="/logs/verifier")
        reward = UPath("memory://trial-remote/verifier/reward.json")
        assert json.loads(reward.read_text()) == {"reward": 1.0}
        assert UPath("memory://trial-remote/verifier/test-stdout.txt").read_text() == (
            "ok\n"
        )

    async def test_download_file_to_remote_target(self, env):
        target = "memory://trial-remote-file/reward.json"
        await env._sdk_download_file("/logs/verifier/reward.json", target)
        assert json.loads(UPath(target).read_text()) == {"reward": 1.0}

    async def test_download_dir_normalizes_exhausted_daytona_error(
        self, env, temp_dir: Path, monkeypatch
    ):
        monkeypatch.setattr(asyncio, "sleep", _no_sleep)
        sandbox = _FakeSandbox({})
        sandbox.fs = _FailedSearchDaytonaFs({})
        env._sandbox = cast(object, sandbox)

        with pytest.raises(DaytonaProcessError) as error:
            await env.download_dir("/logs/verifier", temp_dir / "verifier")

        assert isinstance(error.value.__cause__, DaytonaError)


class TestSdkUploadSources:
    """``_sdk_upload_*`` must read from both local and remote (CloudPath) sources.

    Regression for harbor#49: with an ``s3://`` trials dir the upload source
    arrives as a cloud ``CloudPath``/``UPath``, and ``Path(source)`` raised
    ``TypeError`` before the upload started (244 occurrences in 40 min on one
    RL job). The Daytona SDK only reads from the local filesystem, so a remote
    source must be staged into a local temp dir and that dir uploaded. This is
    the upload-side analogue of the ``gs://`` download fix in 06cf57cb.
    """

    @pytest.fixture
    def env(self, temp_dir: Path):
        e = _make_env(temp_dir, compose=False)
        e._sandbox = cast(object, _FakeSandbox({}))
        return e

    async def test_upload_dir_from_local_source(self, env, temp_dir: Path):
        src = temp_dir / "tests"
        (src / "sub").mkdir(parents=True)
        (src / "sub" / "a.txt").write_bytes(b"alpha")
        (src / "b.txt").write_bytes(b"beta")

        await env._sdk_upload_dir(source_dir=src, target_dir="/srv/tests")

        uploaded = cast(_FakeSandbox, env._sandbox).fs._files
        assert uploaded["/srv/tests/b.txt"] == b"beta"
        assert uploaded["/srv/tests/sub/a.txt"] == b"alpha"

    async def test_upload_dir_from_remote_source(self, env):
        # Issue #49 saw a cloudpathlib S3Path; emulate with memory://, a
        # non-local fsspec protocol that needs no network or credentials.
        # Path(UPath(...)) raises the same TypeError as Path(S3Path(...)).
        root = UPath("memory://upload-src/tests")
        (root / "sub").mkdir(parents=True, exist_ok=True)
        (root / "sub" / "a.txt").write_bytes(b"alpha")
        (root / "b.txt").write_bytes(b"beta")

        await env._sdk_upload_dir(source_dir=root, target_dir="/srv/tests")

        uploaded = cast(_FakeSandbox, env._sandbox).fs._files
        assert uploaded["/srv/tests/b.txt"] == b"beta"
        assert uploaded["/srv/tests/sub/a.txt"] == b"alpha"

    async def test_upload_file_from_remote_source(self, env):
        src = UPath("memory://upload-file/reward.json")
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b'{"reward": 1.0}')

        await env._sdk_upload_file(source_path=src, target_path="/srv/reward.json")

        uploaded = cast(_FakeSandbox, env._sandbox).fs._files
        assert uploaded["/srv/reward.json"] == b'{"reward": 1.0}'


# ── Orphaned-sandbox reaping on build failure ─────────────────────────────


class TestOrphanReaping:
    """The SDK can lose a server-side sandbox handle during create failure.

    Each environment stamps a unique label on its sandboxes and reaps the
    orphan by that label when creation fails or is cancelled.
    """

    def test_instance_label_is_unique_and_stamped(self, temp_dir):
        env1 = _make_env(temp_dir)
        env2 = _make_env(temp_dir)
        assert env1._instance_label != env2._instance_label
        assert env1._instance_label.startswith("harbor-")

        from daytona import Image

        labels = env1._image_sandbox_params(
            image=Image.base("ubuntu:22.04"),
            resources=None,
        ).labels
        assert labels["harbor.instance"] == env1._instance_label

        snap_kwargs = env1._sandbox_common_kwargs()
        assert snap_kwargs["labels"]["harbor.instance"] == env1._instance_label

    async def test_reap_deletes_orphans_by_label(self, temp_dir):
        from unittest.mock import AsyncMock, MagicMock

        env = _make_env(temp_dir)

        orphan = MagicMock()
        orphan.id = "orphan-1"
        orphan.state = "build_failed"
        orphan.delete = AsyncMock()

        async def _fake_list(query):
            # Confirm we filter by this instance's unique label.
            assert query.labels == {"harbor.instance": env._instance_label}
            yield orphan

        daytona = MagicMock()
        daytona.list = _fake_list

        await env._reap_orphaned_sandbox(daytona)

        orphan.delete.assert_awaited_once()

    async def test_reap_swallows_list_errors(self, temp_dir):
        from unittest.mock import MagicMock

        env = _make_env(temp_dir)

        def _raising_list(query):
            raise RuntimeError("transient list failure")

        daytona = MagicMock()
        daytona.list = _raising_list

        # Must not raise — reaping is best-effort and must not mask the
        # original build error.
        await env._reap_orphaned_sandbox(daytona)

    async def test_cancelled_create_deletes_server_side_orphan(
        self, temp_dir, monkeypatch
    ):
        env = _make_env(temp_dir)

        orphan = MagicMock()
        orphan.id = "orphan-1"
        orphan.state = "started"
        orphan.delete = AsyncMock()

        async def _cancelled_create(**kwargs):
            raise asyncio.CancelledError()

        async def _fake_list(query):
            assert query.labels == {"harbor.instance": env._instance_label}
            yield orphan

        daytona = MagicMock()
        monkeypatch.setattr(
            daytona_environment, "create_in_subprocess", _cancelled_create
        )
        daytona.list = _fake_list

        with pytest.raises(asyncio.CancelledError):
            await env._create_sandbox(cast("object", SimpleNamespace()), daytona)

        orphan.delete.assert_awaited_once()


@pytest.mark.parametrize(
    "reason",
    [
        "429 Too Many Requests",
        "504 Gateway Timeout",
        "Image processing cancelled",
        "bad Dockerfile",
        "None",
    ],
)
@pytest.mark.parametrize("failures", [3, 4])
async def test_sandbox_build_retries_cleanup_and_back_off(
    temp_dir, reason, failures, monkeypatch
):
    env = _make_env(temp_dir)
    created = []
    deleted = []
    waits = []
    transient = reason in {
        "429 Too Many Requests",
        "504 Gateway Timeout",
        "Image processing cancelled",
    }

    async def create(**kwargs):
        # Every failed remote sandbox must be deleted before another is created.
        assert len(deleted) == len(created)
        sandbox = SimpleNamespace(id=f"sandbox-{len(created)}", state="build_failed")

        async def delete():
            deleted.append(sandbox.id)

        sandbox.delete = delete
        created.append(sandbox)
        if len(created) <= failures:
            raise DaytonaError(
                f"Sandbox entered error state: BUILD_FAILED, error reason: {reason}"
            )
        return sandbox.id

    async def list_sandboxes(query):
        yield created[-1]

    async def sleep(delay):
        waits.append(float(delay))

    async def get(sandbox_id):
        assert sandbox_id == created[-1].id
        return created[-1]

    monkeypatch.setattr(daytona_environment, "create_in_subprocess", create)
    daytona = SimpleNamespace(get=get, list=list_sandboxes)
    create_sandbox = env._create_sandbox.retry_with(sleep=sleep)
    try:
        if transient and failures == 3:
            await create_sandbox(env, cast("object", SimpleNamespace()), daytona)
        else:
            with pytest.raises(SandboxBuildFailedError):
                await create_sandbox(env, cast("object", SimpleNamespace()), daytona)
    finally:
        await env._stop_keepalive()
    assert len(created) == (4 if transient else 1)
    assert len(deleted) == (failures if transient else 1)
    assert len(waits) == (3 if transient else 0)
    if transient:
        assert all(
            lower <= delay <= lower + 30 for lower, delay in zip((30, 60, 120), waits)
        )


async def test_created_sandbox_is_deleted_when_parent_reconnect_fails(
    temp_dir, monkeypatch
):
    from daytona import CreateSandboxFromImageParams, Image

    env = _make_env(temp_dir)
    live_sandboxes = {"created-sandbox"}
    get_attempts = 0

    async def create_in_child(params, *, timeout, api_key):
        return "created-sandbox"

    class Sandbox:
        id = "created-sandbox"

        async def delete(self):
            live_sandboxes.remove(self.id)

    class Daytona:
        async def get(self, sandbox_id):
            nonlocal get_attempts
            get_attempts += 1
            if get_attempts == 1:
                raise DaytonaError("reconnect failed", status_code=503)
            assert sandbox_id == Sandbox.id
            return Sandbox()

        async def list(self, query):
            for _ in live_sandboxes:
                yield Sandbox()

    monkeypatch.setattr(daytona_environment, "create_in_subprocess", create_in_child)
    params = CreateSandboxFromImageParams(image=Image.base("ubuntu:22.04"))
    create = env._create_sandbox.retry_with(stop=stop_after_attempt(1))
    with pytest.raises(DaytonaError):
        await create(env, params, Daytona())
    assert not live_sandboxes


def test_wall_clock_ttl_reaches_all_daytona_create_requests(temp_dir):
    from daytona import Image, Resources

    env = _make_env(temp_dir, ttl_minutes=90)

    image_params = [
        env._image_sandbox_params(
            image=Image.base("ubuntu:22.04"),
            resources=resources,
        )
        for resources in (None, Resources(cpu=2, memory=4))
    ]
    snapshot_params = env._snapshot_sandbox_params("calendar-snapshot")

    assert [params.ttl_minutes for params in image_params] == [90, 90]
    assert snapshot_params.ttl_minutes == 90


# ── auto_snapshot declarative-build fall-through (org snapshot cap full) ──


class TestAutoSnapshotQuotaFallThrough:
    """When the org snapshot cap is full and the env's shared snapshot does not
    exist, the auto_snapshot path must fall through to a declarative Dockerfile
    build (so the trial still runs + gets scored), not fail unscored."""

    @pytest.mark.asyncio
    async def test_quota_exceeded_falls_through_to_dockerfile_build(
        self, temp_dir, monkeypatch, caplog
    ):
        from daytona import (
            CreateSandboxFromImageParams,
            CreateSandboxFromSnapshotParams,
            Resources,
        )

        from harbor.environments.daytona.snapshots import (
            DaytonaSnapshotService,
            SnapshotQuotaExceeded,
        )

        env = _make_env(temp_dir)
        env._auto_snapshot = True

        async def _boom(self, daytona, resources):  # noqa: ANN001
            raise SnapshotQuotaExceeded("Snapshot limit exceeded for org")

        monkeypatch.setattr(DaytonaSnapshotService, "ensure_auto", _boom)

        with caplog.at_level(logging.WARNING):
            params = await env._resolve_start_sandbox_params(
                daytona=None, resources=Resources(), force_build=True
            )

        # Fell through to a declarative image build, NOT a snapshot reuse.
        assert isinstance(params, CreateSandboxFromImageParams)
        assert not isinstance(params, CreateSandboxFromSnapshotParams)
        # Loud about the degraded path.
        assert any(
            "DECLARATIVE" in r.message and "cap is FULL" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_happy_path_reuses_snapshot(self, temp_dir, monkeypatch):
        from daytona import CreateSandboxFromSnapshotParams, Resources

        from harbor.environments.daytona.snapshots import DaytonaSnapshotService

        env = _make_env(temp_dir)
        env._auto_snapshot = True

        async def _ok(self, daytona, resources):  # noqa: ANN001
            return "harbor__deadbeef__snapshot"

        monkeypatch.setattr(DaytonaSnapshotService, "ensure_auto", _ok)

        params = await env._resolve_start_sandbox_params(
            daytona=None, resources=Resources(), force_build=False
        )
        assert isinstance(params, CreateSandboxFromSnapshotParams)
        assert params.snapshot == "harbor__deadbeef__snapshot"

    @pytest.mark.asyncio
    async def test_raw_quota_error_is_caught_and_falls_through(
        self, temp_dir, monkeypatch, caplog
    ):
        """A quota error that surfaces RAW (not wrapped into SnapshotQuotaExceeded)
        must ALSO be caught (via is_snapshot_quota_error) and fall through to a
        declarative build — covers any path/phrasing the source wrapping misses."""
        from daytona import (
            CreateSandboxFromImageParams,
            CreateSandboxFromSnapshotParams,
            Resources,
        )
        from daytona.common.errors import DaytonaValidationError

        from harbor.environments.daytona.snapshots import DaytonaSnapshotService

        env = _make_env(temp_dir)
        env._auto_snapshot = True

        async def _raw_quota(self, daytona, resources):  # noqa: ANN001
            # The exact prod string, raised RAW (NOT wrapped).
            raise DaytonaValidationError(
                "Snapshot quota exceeded. Maximum allowed: 100"
            )

        monkeypatch.setattr(DaytonaSnapshotService, "ensure_auto", _raw_quota)

        with caplog.at_level(logging.WARNING):
            params = await env._resolve_start_sandbox_params(
                daytona=None, resources=Resources(), force_build=True
            )

        assert isinstance(params, CreateSandboxFromImageParams)
        assert not isinstance(params, CreateSandboxFromSnapshotParams)
        assert any(
            "DECLARATIVE" in r.message and "cap is FULL" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_non_quota_error_reraises(self, temp_dir, monkeypatch):
        """The broadened catch must NOT swallow non-quota errors — an unrelated
        failure out of ensure_auto re-raises unchanged (no spurious fall-through)."""
        from daytona import Resources

        from harbor.environments.daytona.snapshots import DaytonaSnapshotService

        env = _make_env(temp_dir)
        env._auto_snapshot = True

        async def _other(self, daytona, resources):  # noqa: ANN001
            raise RuntimeError("totally unrelated boom")

        monkeypatch.setattr(DaytonaSnapshotService, "ensure_auto", _other)

        with pytest.raises(RuntimeError, match="totally unrelated boom"):
            await env._resolve_start_sandbox_params(
                daytona=None, resources=Resources(), force_build=True
            )

    @pytest.mark.asyncio
    async def test_multiple_trials_of_capped_env_all_fall_through(
        self, temp_dir, monkeypatch, caplog
    ):
        """MULTI-TRIAL-PER-ENV: ≥2 trials of the SAME cap-full env BOTH fall
        through to a declarative build, not just the first (lock-winner).

        Drives the REAL ensure_auto path (GET 404 -> locked create -> quota) so
        the first trial discovers the cap + memoizes, and a sibling trial then
        short-circuits straight to the declarative build WITHOUT re-hitting the
        doomed snapshot.create()."""
        from daytona import (
            CreateSandboxFromImageParams,
            CreateSandboxFromSnapshotParams,
            Resources,
        )
        from daytona.common.errors import (
            DaytonaNotFoundError,
            DaytonaValidationError,
        )

        from harbor.environments.daytona import snapshots as snapshots_mod
        from harbor.environments.daytona.snapshots import _CapFullRegistry

        # Isolate the process-wide cap-full memo for this test.
        monkeypatch.setattr(_CapFullRegistry, "_names", set())

        get_calls = {"n": 0}
        create_calls = {"n": 0}

        class _FakeSnap:
            async def get(self, name):  # noqa: ANN001
                get_calls["n"] += 1
                raise DaytonaNotFoundError("snapshot not found")

            async def create(self, params):  # noqa: ANN001
                create_calls["n"] += 1
                raise DaytonaValidationError(
                    "Snapshot quota exceeded. Maximum allowed: 100"
                )

            async def delete(self, snapshot):  # noqa: ANN001
                pass

        class _FakeDaytona:
            snapshot = _FakeSnap()

        # Two SEPARATE env instances = two trials of the same task/env (distinct
        # DaytonaEnvironment objects, identical environment_dir -> identical
        # auto_snapshot name -> shared cap-full memo).
        env_a = _make_env(temp_dir)
        env_a._auto_snapshot = True
        env_b = _make_env(temp_dir)
        env_b._auto_snapshot = True

        async def create_snapshot(params, *, timeout, target, api_key):
            return await _FakeSnap().create(params)

        # Replace the process boundary rather than creating a remote snapshot.
        monkeypatch.setattr(
            snapshots_mod,
            "create_in_subprocess",
            create_snapshot,
        )

        daytona = _FakeDaytona()
        results = []
        with caplog.at_level(logging.WARNING):
            for env in (env_a, env_b):
                params = await env._resolve_start_sandbox_params(
                    daytona=daytona, resources=Resources(), force_build=True
                )
                results.append(params)

        # BOTH trials fell through to a declarative image build.
        for params in results:
            assert isinstance(params, CreateSandboxFromImageParams)
            assert not isinstance(params, CreateSandboxFromSnapshotParams)

        # The doomed snapshot.create() was attempted EXACTLY ONCE (first trial);
        # the sibling short-circuited via the cap-full memo.
        assert create_calls["n"] == 1, (
            f"sibling re-hit the doomed create (create_calls={create_calls['n']}); "
            "the cap-full memo did not short-circuit it"
        )
        # Loud fall-through fired for BOTH trials.
        fallthroughs = [
            r
            for r in caplog.records
            if "DECLARATIVE" in r.message and "cap is FULL" in r.message
        ]
        assert len(fallthroughs) >= 2


# ── Persistent per-sandbox exec session ───────────────────────────────


class _FakeCmd:
    def __init__(self, cmd_id: str = "cmd-1", exit_code: int = 0):
        self.cmd_id = cmd_id
        self.id = cmd_id
        self.exit_code = exit_code


class _FakeLogs:
    stdout = "out"
    stderr = ""


async def _no_sleep(*args, **kwargs):
    """Instant sleep so SANDBOX_RETRY backoff doesn't slow the unit tests."""
    return None


class _Auth401(Exception):
    """Mimics Daytona's intermittent toolbox 401 flake ("Bearer token is invalid").

    ``is_daytona_auth_flake`` detects it by HTTP ``status_code``, not message
    substrings, so a bare exception carrying ``status_code == 401`` is enough.
    """

    status_code = 401


class _FakeProcess:
    """Records create_session / execute_session_command calls for a fake sandbox.

    ``fail_execute_times`` makes the first N execute_session_command calls raise
    ``exc_factory()`` (default: a generic RuntimeError) on the SAME live session —
    modelling a transient blip/auth-flake that the retry tier rides out in place.

    ``dead_first_session`` marks the FIRST-created session id as permanently dead
    (execute always raises for it); only a *recreated* session revives it, which
    drives ``_sandbox_exec``'s recreate-on-death path.
    """

    def __init__(
        self,
        fail_execute_times: int = 0,
        exc_factory=None,
        dead_first_session: bool = False,
    ):
        self.created_sessions: list[str] = []
        self.executed_sessions: list[str] = []
        self._fail_execute_times = fail_execute_times
        self._exc_factory = exc_factory or (lambda: RuntimeError("session gone"))
        self._dead_first_session = dead_first_session
        self._dead_sessions: set[str] = set()

    async def create_session(self, session_id: str):
        if self._dead_first_session and not self.created_sessions:
            self._dead_sessions.add(session_id)
        self.created_sessions.append(session_id)

    async def execute_session_command(self, session_id, request, timeout=None):
        if session_id in self._dead_sessions:
            raise RuntimeError("session gone")
        if self._fail_execute_times > 0:
            self._fail_execute_times -= 1
            raise self._exc_factory()
        self.executed_sessions.append(session_id)
        return _FakeCmd()

    async def get_session_command(self, session_id, command_id):
        return _FakeCmd(cmd_id=command_id, exit_code=0)

    async def get_session_command_logs(self, session_id, command_id):
        return _FakeLogs()


class _FakeSandboxProc:
    def __init__(self, process: _FakeProcess):
        self.process = process
        self.id = "sandbox-1"


class _FailedPollProcess(_FakeProcess):
    async def get_session_command(self, session_id, command_id):
        raise DaytonaError("Failed to get session command")


class _CreatedButConflictProcess(_FakeProcess):
    async def create_session(self, session_id: str):
        if session_id not in self.created_sessions:
            self.created_sessions.append(session_id)
        raise DaytonaConflictError(
            "Failed to create session: conflict: session already exists",
            status_code=409,
        )


class _ContendedDeleteSandbox(_FakeSandboxProc):
    def __init__(self):
        super().__init__(_FakeProcess())
        self.id = "sandbox-delete"
        self.delete_blocked = True
        self.deleted = False

    async def delete(self) -> None:
        if self.delete_blocked:
            raise RuntimeError("Sandbox state change in progress")
        self.deleted = True


@pytest.mark.parametrize("compose", [False, True], ids=["direct", "docker-in-daytona"])
async def test_stop_retains_sandbox_for_retry_after_delete_failure(
    temp_dir: Path, monkeypatch, compose: bool
) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    env = _make_env(temp_dir, compose=compose)
    sandbox = _ContendedDeleteSandbox()
    env._sandbox = cast(object, sandbox)

    with pytest.raises(
        DaytonaSandboxStopError,
        match="sandbox-delete",
    ) as error:
        await env.stop(delete=True)

    assert isinstance(error.value.__cause__, RuntimeError)
    assert not sandbox.deleted

    sandbox.delete_blocked = False
    await env.stop(delete=True)
    assert sandbox.deleted


@pytest.mark.parametrize("recreate", [False, True], ids=["initial", "replacement"])
async def test_session_creation_failure_is_retryable_infrastructure(
    temp_dir: Path, monkeypatch, caplog, recreate: bool
) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    process = _FakeProcess()
    env = _make_env(temp_dir, compose=False)
    env._sandbox = cast(object, _FakeSandboxProc(process))
    execution_error = DaytonaError("Failed to execute command", status_code=502)
    if recreate:
        await env.exec("echo ready")
        monkeypatch.setattr(
            process,
            "execute_session_command",
            AsyncMock(side_effect=execution_error),
        )
    creation_error = DaytonaError("Failed to create session: ", status_code=503)
    monkeypatch.setattr(
        process, "create_session", AsyncMock(side_effect=creation_error)
    )

    with pytest.raises(DaytonaProcessError) as error:
        await env.exec("echo hi")

    assert error_category(type(error.value).__name__) == ErrorCategory.INFRASTRUCTURE
    assert is_resume_retryable(type(error.value).__name__)
    assert error.value.__cause__ is creation_error
    assert "503" in str(error.value)
    if recreate:
        assert any(
            record.exc_info and record.exc_info[1] is execution_error
            for record in caplog.records
        )


class TestPersistentExecSession:
    """``_sandbox_exec`` reuses ONE Daytona session per sandbox (``create_session``
    paid once per sandbox, not once per exec), recreating it only if it dies."""

    def _env(self, temp_dir: Path, process: _FakeProcess):
        e = _make_env(temp_dir, compose=False)
        e._sandbox = cast(object, _FakeSandboxProc(process))
        return e

    async def test_create_session_called_once_across_execs(self, temp_dir):
        proc = _FakeProcess()
        env = self._env(temp_dir, proc)
        for _ in range(3):
            await env._sandbox_exec("echo hi")
        # The whole point: create_session ONCE, not once per exec.
        assert len(proc.created_sessions) == 1
        assert len(proc.executed_sessions) == 3
        # Every exec reused the one session id.
        assert set(proc.executed_sessions) == set(proc.created_sessions)
        assert env._exec_session_id == proc.created_sessions[0]

    async def test_ensure_exec_session_reuses_cached_id(self, temp_dir):
        proc = _FakeProcess()
        env = self._env(temp_dir, proc)
        first = await env._ensure_exec_session()
        second = await env._ensure_exec_session()
        assert first == second
        assert len(proc.created_sessions) == 1

    async def test_create_session_conflict_uses_existing_session(self, temp_dir):
        proc = _CreatedButConflictProcess()
        env = self._env(temp_dir, proc)

        result = await env._sandbox_exec("echo hi")

        assert result.return_code == 0
        assert proc.executed_sessions == proc.created_sessions

    async def test_recreate_on_session_death_and_retry(self, temp_dir, monkeypatch):
        # The cached session genuinely died: execute keeps failing on it. The
        # SANDBOX_RETRY tier exhausts against the dead session, then _sandbox_exec
        # drops the dead id, recreates once, and succeeds on the fresh session.
        monkeypatch.setattr(asyncio, "sleep", _no_sleep)
        proc = _FakeProcess(dead_first_session=True)
        env = self._env(temp_dir, proc)
        result = await env._sandbox_exec("echo hi")
        assert result.return_code == 0
        # create_session called twice: initial + recreate-after-death.
        assert len(proc.created_sessions) == 2
        # The cached id points at the fresh session, never a dead one.
        assert env._exec_session_id == proc.created_sessions[-1]
        assert proc.executed_sessions == [proc.created_sessions[-1]]

    async def test_transient_auth_flake_on_exec_is_retried_in_place(
        self, temp_dir, monkeypatch
    ):
        # The verifier grading exec (verify() -> exec() -> _sandbox_exec) hits a
        # transient toolbox 401 on a still-alive session. The SANDBOX_RETRY
        # auth-flake tier must ride it out IN PLACE — no session recreate — so the
        # exec SUCCEEDS instead of escaping as DaytonaAuthenticationError (the
        # root-cause fix for the flawed_summ tb2 gate-fails).
        monkeypatch.setattr(asyncio, "sleep", _no_sleep)
        proc = _FakeProcess(fail_execute_times=2, exc_factory=_Auth401)
        env = self._env(temp_dir, proc)
        result = await env._sandbox_exec("echo hi")
        assert result.return_code == 0
        # Rode out the 401 on the ORIGINAL session: created once, never recreated.
        assert len(proc.created_sessions) == 1
        assert proc.executed_sessions == [proc.created_sessions[0]]
        assert env._exec_session_id == proc.created_sessions[0]

    async def test_teardown_resets_cached_session_id(self, temp_dir):
        proc = _FakeProcess()
        env = self._env(temp_dir, proc)
        await env._sandbox_exec("echo hi")
        assert env._exec_session_id is not None
        # Direct keep-alive stop detaches the sandbox; the cached id must clear
        # so a reattached env re-establishes its own session.
        await env._strategy.stop(delete=False)
        assert env._exec_session_id is None

    async def test_poll_failure_uses_stable_infrastructure_error(
        self, temp_dir, monkeypatch
    ):
        monkeypatch.setattr(asyncio, "sleep", _no_sleep)
        env = self._env(temp_dir, _FailedPollProcess())

        with pytest.raises(DaytonaProcessError) as error:
            await env._poll_response("session-1", "command-1")

        assert isinstance(error.value.__cause__, DaytonaError)


class _BoundedSleep:
    """Patched ``asyncio.sleep`` for the keepalive loop: record each requested
    interval and return immediately for the first ``max_ticks - 1`` calls so the
    loop pings a bounded number of times, then set ``reached`` and park forever
    (so the loop can't spin unboundedly and the test can await a known state)."""

    def __init__(self, max_ticks: int):
        self.max_ticks = max_ticks
        self.intervals: list[float] = []
        self.reached = asyncio.Event()

    async def __call__(self, interval, *args, **kwargs):
        self.intervals.append(interval)
        if len(self.intervals) >= self.max_ticks:
            self.reached.set()
            await asyncio.Event().wait()  # park; the task is cancelled on teardown
        return None


class TestSandboxKeepalive:
    """The background keepalive bumps the live sandbox's last-activity timestamp
    so an HTTP-only agent (opencode) can't idle-reap it mid-trial, and the task is
    cancelled cleanly on teardown."""

    def _env_with_sandbox(self, temp_dir: Path, auto_stop_mins: int):
        from unittest.mock import AsyncMock, MagicMock

        env = _make_env(temp_dir, compose=False)
        env._auto_stop_interval = auto_stop_mins
        sandbox = MagicMock()
        sandbox.id = "sandbox-keepalive"
        sandbox.refresh_activity = AsyncMock()
        env._sandbox = cast(object, sandbox)
        return env, sandbox

    async def test_pings_refresh_activity_on_interval(self, temp_dir, monkeypatch):
        # auto_stop=4min -> interval = max(30, (4*60)//2 = 120) = 120s.
        env, sandbox = self._env_with_sandbox(temp_dir, auto_stop_mins=4)
        sleeper = _BoundedSleep(max_ticks=4)
        monkeypatch.setattr(asyncio, "sleep", sleeper)

        env._start_keepalive()
        assert env._keepalive_task is not None
        await asyncio.wait_for(sleeper.reached.wait(), timeout=5)

        # 3 immediate ticks fired a refresh each before the loop parked.
        assert sandbox.refresh_activity.await_count == 3
        assert sleeper.intervals[:3] == [120, 120, 120]

        await env._stop_keepalive()
        assert env._keepalive_task is None

    async def test_interval_clamped_to_minimum(self, temp_dir, monkeypatch):
        # auto_stop=1min -> (1*60)//2 = 30, clamped floor is also 30 -> 30s.
        env, _ = self._env_with_sandbox(temp_dir, auto_stop_mins=1)
        sleeper = _BoundedSleep(max_ticks=2)
        monkeypatch.setattr(asyncio, "sleep", sleeper)

        env._start_keepalive()
        await asyncio.wait_for(sleeper.reached.wait(), timeout=5)
        assert sleeper.intervals[0] == 30

        await env._stop_keepalive()

    async def test_no_keepalive_when_auto_stop_disabled(self, temp_dir):
        # auto_stop=0 means Daytona never idle-reaps, so no keepalive is needed.
        env, _ = self._env_with_sandbox(temp_dir, auto_stop_mins=0)
        env._start_keepalive()
        assert env._keepalive_task is None

    async def test_transient_refresh_failure_is_retried(
        self, temp_dir, monkeypatch, caplog
    ):
        # A transient refresh_activity error must be logged and retried on the next
        # tick, never crash the loop.
        env, sandbox = self._env_with_sandbox(temp_dir, auto_stop_mins=4)
        sandbox.refresh_activity.side_effect = [RuntimeError("flake"), None, None]
        sleeper = _BoundedSleep(max_ticks=4)
        monkeypatch.setattr(asyncio, "sleep", sleeper)

        with caplog.at_level(logging.WARNING):
            env._start_keepalive()
            await asyncio.wait_for(sleeper.reached.wait(), timeout=5)

        # It kept pinging after the failure (3 attempts across 3 ticks).
        assert sandbox.refresh_activity.await_count == 3
        assert not env._keepalive_task.done()
        assert any("refresh_activity failed" in r.message for r in caplog.records)

        await env._stop_keepalive()

    async def test_stop_cancels_keepalive(self, temp_dir, monkeypatch):
        env, _ = self._env_with_sandbox(temp_dir, auto_stop_mins=4)
        sleeper = _BoundedSleep(max_ticks=2)
        monkeypatch.setattr(asyncio, "sleep", sleeper)

        env._start_keepalive()
        task = env._keepalive_task
        await asyncio.wait_for(sleeper.reached.wait(), timeout=5)

        # The top-level teardown (delete=False detaches without a real delete call)
        # must cancel the keepalive task.
        await env.stop(delete=False)
        assert env._keepalive_task is None
        assert task.cancelled()

    async def test_start_is_idempotent(self, temp_dir, monkeypatch):
        env, _ = self._env_with_sandbox(temp_dir, auto_stop_mins=4)
        sleeper = _BoundedSleep(max_ticks=10)
        monkeypatch.setattr(asyncio, "sleep", sleeper)

        env._start_keepalive()
        first = env._keepalive_task
        env._start_keepalive()  # a second call must not spawn a duplicate task
        assert env._keepalive_task is first

        await env._stop_keepalive()


# ── F2: reap a runaway detached command on a cancelled poll ────────────


class _ReapFakeProcess:
    """Records session-command calls for the reap-on-cancel tests."""

    def __init__(self):
        self.created: list[str] = []
        self.executed: list[tuple[str, str, object]] = []
        self.deleted: list[str] = []

    async def create_session(self, session_id):
        self.created.append(session_id)

    async def execute_session_command(self, session_id, request, timeout=None):
        self.executed.append((session_id, request.command, timeout))
        return SimpleNamespace(cmd_id="cmd-1")

    async def delete_session(self, session_id):
        self.deleted.append(session_id)


class _ReapFakeSandbox:
    def __init__(self):
        self.id = "sbx-1"
        self.process = _ReapFakeProcess()


@pytest.mark.parametrize(
    "message, expected_error",
    [
        (
            "Failed to execute session command: bad request: failed to create log directory: "
            "mkdir /root/.daytona/sessions/session-id: no space left on device",
            SandboxDiskFullError,
        ),
        (
            "Failed to execute session command: bad request: invalid session",
            DaytonaValidationError,
        ),
    ],
)
async def test_session_command_classifies_only_disk_exhaustion(
    temp_dir, message, expected_error
):
    env = _make_env(temp_dir)
    original = DaytonaValidationError(message)
    env._sandbox = SimpleNamespace(
        process=SimpleNamespace(execute_session_command=AsyncMock(side_effect=original))
    )
    execute = env._execute_session_command_with_retry.retry_with(
        stop=stop_after_attempt(1), wait=wait_none()
    )

    with pytest.raises(expected_error) as raised:
        await execute(env, "session-id", SimpleNamespace(command="true"), 60)

    if expected_error is SandboxDiskFullError:
        assert raised.value.__cause__ is original
        assert (
            error_category(type(raised.value).__name__) is ErrorCategory.INFRASTRUCTURE
        )
    else:
        assert raised.value is original


async def test_disk_exhaustion_is_not_masked_by_session_recreation(
    temp_dir, monkeypatch
):
    env = _make_env(temp_dir)
    original = DaytonaValidationError(
        "failed to create log directory: no space left on device"
    )
    env._sandbox = SimpleNamespace(
        process=SimpleNamespace(
            execute_session_command=AsyncMock(side_effect=original),
            create_session=AsyncMock(
                side_effect=DaytonaValidationError("cannot create session")
            ),
        )
    )
    env._exec_session_id = "session-id"
    retrying = env._execute_session_command_with_retry.retry
    monkeypatch.setattr(retrying, "stop", stop_after_attempt(1))
    monkeypatch.setattr(retrying, "wait", wait_none())

    with pytest.raises(SandboxDiskFullError) as raised:
        await env._sandbox_exec("true")

    assert raised.value.__cause__ is original


class TestReapInflightAgentCommand:
    """``_sandbox_exec`` launches the command ``run_async`` (detached
    server-side); a cancelled poll must reap it, else a runaway opencode keeps
    running and orphans the sandbox (auto_stop=0)."""

    async def test_reap_creates_session_pkills_and_deletes(self, temp_dir):
        env = _make_env(temp_dir)
        env._sandbox = cast("object", _ReapFakeSandbox())
        await env._reap_inflight_agent_command()
        proc = env._sandbox.process
        # One FRESH short-lived session, a scoped pkill, then torn down.
        assert len(proc.created) == 1
        assert proc.executed and "pkill -9 -f opencode" in proc.executed[0][1]
        assert proc.deleted == proc.created

    async def test_reap_is_noop_without_sandbox(self, temp_dir):
        env = _make_env(temp_dir)
        env._sandbox = None
        # Must not raise (best-effort on a teardown path).
        await env._reap_inflight_agent_command()

    async def test_sandbox_exec_reaps_on_cancel_and_reraises(
        self, temp_dir, monkeypatch
    ):
        env = _make_env(temp_dir)
        env._sandbox = cast("object", _ReapFakeSandbox())
        reaped = {"called": False}

        async def _fake_reap():
            reaped["called"] = True

        async def _fake_ensure_session():
            return "sess-1"

        async def _fake_execute(session_id, request, timeout_sec):
            return SimpleNamespace(cmd_id="cmd-1")

        async def _fake_poll(session_id, cmd_id):
            raise asyncio.CancelledError()

        monkeypatch.setattr(env, "_reap_inflight_agent_command", _fake_reap)
        monkeypatch.setattr(env, "_ensure_exec_session", _fake_ensure_session)
        monkeypatch.setattr(env, "_execute_session_command_with_retry", _fake_execute)
        monkeypatch.setattr(env, "_poll_response", _fake_poll)

        with pytest.raises(asyncio.CancelledError):
            await env._sandbox_exec("sleep 999")

        assert reaped["called"] is True
