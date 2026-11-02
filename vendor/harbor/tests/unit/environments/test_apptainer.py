"""Unit tests for the bridge-backed ApptainerEnvironment.

These tests cover construction-time invariants and the HTTP request/response
plumbing against a stubbed bridge. They do not require a real bridge server or
any HPC node — the bridge HTTP helpers are patched. End-to-end validation
against a live bridge + apptainer workers is described in the package docs and
must run on the cluster.
"""

from __future__ import annotations

import json
import subprocess
from collections import deque
from pathlib import Path
from unittest.mock import patch

import pytest

from harbor.environments.apptainer import ApptainerEnvironment, BridgeOutageError
from harbor.environments.apptainer import server as server_module
from harbor.environments.apptainer import worker as worker_module
from harbor.environments.base import ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

BRIDGE_URL = "http://bridge.test:9910"


def _make_env(temp_dir, **kwargs):
    env_dir = temp_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")

    trial_dir = temp_dir / "trial"
    trial_dir.mkdir(exist_ok=True)
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    defaults = dict(
        environment_dir=env_dir,
        environment_name="test-task",
        session_id="test-task__abc123",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(docker_image="ubuntu:22.04"),
        bridge_url=BRIDGE_URL,
    )
    defaults.update(kwargs)
    return ApptainerEnvironment(**defaults)


@pytest.fixture
def apptainer_env(temp_dir):
    return _make_env(temp_dir)


class TestProperties:
    def test_type(self, apptainer_env):
        assert apptainer_env.type() == EnvironmentType.APPTAINER

    def test_capabilities(self, apptainer_env):
        caps = apptainer_env.capabilities
        # CPU-only HPC nodes that can run isolated containers; results live on
        # the remote cluster (not a host bind mount), so `mounted` is False.
        assert caps.gpus is False
        assert caps.disable_internet is True
        assert caps.mounted is False


class TestConstruction:
    def test_missing_bridge_url_raises(self, temp_dir):
        with pytest.raises(ValueError, match="bridge_url"):
            _make_env(temp_dir, bridge_url="")

    def test_bridge_url_from_env_var(self, temp_dir, monkeypatch):
        monkeypatch.setenv("APPTAINER_BRIDGE_URL", BRIDGE_URL)
        env = _make_env(temp_dir, bridge_url="")
        assert env._bridge_url == BRIDGE_URL

    def test_explicit_bridge_url_wins_over_env(self, temp_dir, monkeypatch):
        monkeypatch.setenv("APPTAINER_BRIDGE_URL", "http://env-var:1234")
        env = _make_env(temp_dir, bridge_url=BRIDGE_URL)
        assert env._bridge_url == BRIDGE_URL


class TestTaskFiles:
    def test_collect_task_files_base64(self, apptainer_env):
        files = apptainer_env._collect_task_files()
        assert "Dockerfile" in files
        import base64

        assert base64.b64decode(files["Dockerfile"]).decode() == "FROM ubuntu:22.04\n"

    def test_base_image_parsed_from_dockerfile(self, apptainer_env):
        assert apptainer_env._base_image_from_dockerfile() == "ubuntu:22.04"


class TestBridgeHealth:
    def test_unreachable_bridge_raises_outage(self, apptainer_env):
        def boom(url, timeout=10):
            raise OSError("connection refused")

        with patch(
            "harbor.environments.apptainer.apptainer._http_get", side_effect=boom
        ):
            with pytest.raises(BridgeOutageError, match="unreachable"):
                apptainer_env._check_bridge_health()

    def test_workers_down_under_timeout_does_not_raise(self, apptainer_env):
        apptainer_env._workers_dead_timeout = 1000.0
        with patch(
            "harbor.environments.apptainer.apptainer._http_get",
            return_value={"workers_alive": False},
        ):
            # First observation of down workers starts the clock but tolerates.
            apptainer_env._check_bridge_health()
        assert apptainer_env._workers_dead_since is not None

    def test_workers_recover_clears_dead_clock(self, apptainer_env):
        apptainer_env._workers_dead_since = 1.0
        with patch(
            "harbor.environments.apptainer.apptainer._http_get",
            return_value={"workers_alive": True},
        ):
            apptainer_env._check_bridge_health()
        assert apptainer_env._workers_dead_since is None


class TestExec:
    async def test_exec_not_started_raises(self, apptainer_env):
        with pytest.raises(RuntimeError, match="not started"):
            await apptainer_env.exec("echo hi")

    async def test_exec_submits_and_polls(self, apptainer_env):
        apptainer_env._env_id = "env-1"

        async def fake_post(url, data, timeout=60):
            assert url.endswith("/env/exec")
            assert data["env_id"] == "env-1"
            assert data["command"] == "echo hi"
            return {"job_id": "job-1"}

        async def fake_poll(bridge_url, job_id, timeout_sec=600):
            assert job_id == "job-1"
            return {"stdout": "hi\n", "stderr": "", "return_code": 0}

        with (
            patch(
                "harbor.environments.apptainer.apptainer._async_http_post",
                side_effect=fake_post,
            ),
            patch(
                "harbor.environments.apptainer.apptainer._poll_job",
                side_effect=fake_poll,
            ),
        ):
            result = await apptainer_env.exec("echo hi")

        assert isinstance(result, ExecResult)
        assert result.return_code == 0
        assert result.stdout == "hi\n"

    async def test_exec_surfaces_worker_truncation(self, apptainer_env):
        apptainer_env._env_id = "env-1"

        async def fake_post(url, data, timeout=60):
            return {"job_id": "job-1"}

        async def fake_poll(bridge_url, job_id, timeout_sec=600):
            return {
                "stdout": "recent output",
                "stderr": "",
                "return_code": 0,
                "stdout_truncated": True,
                "stderr_truncated": False,
            }

        with (
            patch(
                "harbor.environments.apptainer.apptainer._async_http_post",
                side_effect=fake_post,
            ),
            patch(
                "harbor.environments.apptainer.apptainer._poll_job",
                side_effect=fake_poll,
            ),
        ):
            result = await apptainer_env.exec("long command")

        assert result.stdout == "recent output"
        assert result.stdout_truncated is True
        assert result.stderr_truncated is False

    async def test_exec_merges_persistent_env(self, temp_dir):
        env = _make_env(temp_dir, persistent_env={"FOO": "bar"})
        env._env_id = "env-1"
        captured = {}

        async def fake_post(url, data, timeout=60):
            captured.update(data)
            return {"job_id": "job-1"}

        async def fake_poll(bridge_url, job_id, timeout_sec=600):
            return {"stdout": "", "stderr": "", "return_code": 0}

        with (
            patch(
                "harbor.environments.apptainer.apptainer._async_http_post",
                side_effect=fake_post,
            ),
            patch(
                "harbor.environments.apptainer.apptainer._poll_job",
                side_effect=fake_poll,
            ),
        ):
            await env.exec("echo $FOO", env={"BAZ": "qux"})

        assert captured["env_vars"] == {"FOO": "bar", "BAZ": "qux"}


class TestStop:
    async def test_stop_without_env_id_is_noop(self, apptainer_env):
        # Should not attempt any HTTP call when nothing was started.
        with patch("harbor.environments.apptainer.apptainer._async_http_post") as post:
            await apptainer_env.stop(delete=True)
        post.assert_not_called()


@pytest.mark.parametrize(
    ("worker_id", "node_id"),
    [
        ("jwc07n056-3", "jwc07n056"),
        ("julia.hpc.tu-dresden.de", "julia.hpc.tu-dresden.de"),
        ("julia.hpc.tu-dresden.de-12", "julia.hpc.tu-dresden.de"),
    ],
)
def test_worker_routing_preserves_dashes_in_hostnames(worker_id, node_id):
    assert server_module._node_id_from_worker_id(worker_id) == node_id


def test_followup_job_routes_to_node_recorded_at_start():
    worker_queues = {}
    with patch.multiple(
        server_module,
        _envs={
            "env-1": {
                "worker_id": "julia.hpc.tu-dresden.de-3",
                "node_id": "julia.hpc.tu-dresden.de",
            }
        },
        _jobs={},
        _job_queue=deque(),
        _worker_queues=worker_queues,
        _stats={"jobs_submitted": 0},
    ):
        job_id = server_module._submit_job("env-1", server_module.JOB_EXEC)

    assert list(worker_queues) == ["julia.hpc.tu-dresden.de"]
    assert list(worker_queues["julia.hpc.tu-dresden.de"]) == [job_id]


def test_cached_base_sif_keeps_multiline_run_as_one_command(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM ubuntu:22.04\n"
        "RUN printf one && \\\n"
        "    printf two && \\\n"
        "    printf three\n"
    )
    instance = worker_module.ApptainerInstance("env-1", str(tmp_path), str(tmp_path))

    assert instance._extract_run_commands(str(tmp_path)) == [
        "printf one && printf two && printf three"
    ]


def test_sif_definition_includes_copy_and_complete_run(tmp_path, monkeypatch):
    source = tmp_path / "task_file.txt"
    source.write_text("payload")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM ubuntu:22.04\n"
        "COPY task_file.txt /app/input.txt\n"
        'ENV APP_MODE="test value"\n'
        "RUN printf one && \\\n"
        "    printf two\n"
        "WORKDIR /app\n"
    )
    output_sif = tmp_path / "built.sif"
    definitions = []

    def run(command, **kwargs):
        definition = Path(command[-1])
        definitions.append(definition.read_text())
        Path(command[-2]).write_bytes(b"sif")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(worker_module, "APPTAINER", "apptainer")
    monkeypatch.setattr(worker_module.subprocess, "run", run)
    instance = worker_module.ApptainerInstance("env-1", str(tmp_path), str(tmp_path))

    assert instance._build_sif(str(dockerfile), str(output_sif)) == str(output_sif)
    assert output_sif.read_bytes() == b"sif"
    assert f"    {source} /app/input.txt" in definitions[0]
    assert "    export APP_MODE='test value'" in definitions[0]
    assert "    printf one && printf two" in definitions[0]


def test_sif_build_rejects_missing_copy_source(tmp_path, monkeypatch):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM ubuntu:22.04\nCOPY missing.txt /app/input.txt\n")
    monkeypatch.setattr(worker_module, "APPTAINER", "apptainer")
    instance = worker_module.ApptainerInstance("env-1", str(tmp_path), str(tmp_path))

    with pytest.raises(FileNotFoundError, match="missing.txt"):
        instance._build_sif(str(dockerfile), str(tmp_path / "built.sif"))


def test_fakeroot_mode_rejects_unknown_values():
    with pytest.raises(ValueError, match="BRIDGE_USE_FAKEROOT"):
        worker_module._fakeroot_attempts("sometimes")


def test_worker_exec_retains_tail_and_reports_truncated_streams(tmp_path, monkeypatch):
    def run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="stdout-012345",
            stderr="stderr-abcdef",
        )

    monkeypatch.setattr(worker_module, "APPTAINER", "apptainer")
    monkeypatch.setattr(worker_module.subprocess, "run", run)
    instance = worker_module.ApptainerInstance(
        "env-1", str(tmp_path), str(tmp_path), max_exec_output_chars=6
    )

    result = instance.exec({"command": "produce output"})

    assert result == {
        "stdout": "012345",
        "stderr": "abcdef",
        "return_code": 0,
        "stdout_truncated": True,
        "stderr_truncated": True,
    }


def test_worker_rejects_nonpositive_exec_output_limit(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        worker_module.ApptainerInstance(
            "env-1", str(tmp_path), str(tmp_path), max_exec_output_chars=0
        )


def test_deferred_build_env_reaches_fallback_instance(tmp_path, monkeypatch):
    sif = tmp_path / "image.sif"
    sif.write_bytes(b"sif")
    Path(worker_module._deferred_sidecar(str(sif))).write_text(
        json.dumps(
            {
                "run": [],
                "env": {"APP_MODE": "test value"},
                "workdir": "/",
                "copy": [],
            }
        )
    )
    overlay = tmp_path / "image.overlay.img"
    overlay.write_bytes(b"overlay")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[:3] == ["apptainer", "overlay", "create"]:
            Path(command[-1]).write_bytes(b"overlay")
        if command[:3] == ["apptainer", "instance", "start"]:
            return_code = 1 if "--fakeroot" in command else 0
            return subprocess.CompletedProcess(
                command,
                return_code,
                stdout="",
                stderr="fakeroot unavailable" if return_code else "",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(worker_module, "APPTAINER", "apptainer")
    monkeypatch.setenv("BRIDGE_USE_FAKEROOT", "auto")
    monkeypatch.setattr(worker_module.subprocess, "run", run)
    monkeypatch.setattr(
        worker_module, "_bake_deferred_overlay", lambda _sif: str(overlay)
    )
    instance = worker_module.ApptainerInstance("env-1", str(tmp_path), str(tmp_path))

    result = instance.start(
        {
            "sif_path": str(sif),
            "task_name": "test-task",
            "files_b64": {},
        }
    )

    starts = [
        command
        for command in commands
        if command[:3] == ["apptainer", "instance", "start"]
    ]
    assert len(starts) == 2
    assert "--fakeroot" in starts[0]
    assert "--fakeroot" not in starts[1]
    for command in starts:
        env_index = command.index("APP_MODE=test value")
        assert command[env_index - 1] == "--env"
    assert result["fakeroot"] is False
    assert result["deferred_overlay"] is True
