# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Harbor adapter through a local Iris exec transport."""

import asyncio
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from iris.cluster.types import JobName
from iris.resources.state import TaskState
from iris.rpc import job_pb2
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.iris import machine as iris_backend
from harbor.environments.iris import IrisEnvironment

IMAGE = "example.test/task@sha256:" + "a" * 64


def environment(tmp_path: Path, **kwargs) -> IrisEnvironment:
    config = kwargs.pop(
        "task_env_config", EnvironmentConfig(docker_image=IMAGE, allow_internet=False, workdir="/workspace")
    )
    return IrisEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="task",
        session_id="trial",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=config,
        cluster="test",
        **kwargs,
    )


@pytest.mark.parametrize(
    "filename", ["Dockerfile", "docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"]
)
def test_rejects_runtime_builds_and_services(tmp_path: Path, filename: str) -> None:
    task_environment = tmp_path / "environment"
    task_environment.mkdir()
    (task_environment / filename).write_text("services: {}")
    with pytest.raises(ValueError):
        environment(tmp_path)


def test_task_cannot_import_driver_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "driver-private")
    config = EnvironmentConfig(docker_image=IMAGE, allow_internet=False, env={"STOLEN": "${AWS_SECRET_ACCESS_KEY}"})
    with pytest.raises(ValueError, match="literal"):
        environment(tmp_path, task_env_config=config)
    with pytest.raises(ValueError, match="driver environment"):
        environment(tmp_path, persistent_env={"STOLEN": "driver-private"})


@pytest.mark.parametrize(
    "override", [{"network": "allow"}, {"container_profile": "privileged"}, {"secret_env": {"KEY": "secret"}}]
)
def test_cannot_override_isolation(tmp_path: Path, override: dict) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        environment(tmp_path, **override)


def test_cancelled_start_reclaims_job_that_finishes_submitting(tmp_path: Path, monkeypatch) -> None:
    adapter = environment(tmp_path)

    async def run() -> None:
        submitting = asyncio.Event()
        finish_submission = asyncio.Event()
        reclaimed = asyncio.Event()

        async def close() -> None:
            reclaimed.set()

        async def create(spec):
            submitting.set()
            await finish_submission.wait()
            return SimpleNamespace(close=close)

        monkeypatch.setattr(adapter.factory, "create", create)
        start = asyncio.create_task(adapter.start(force_build=False))
        await submitting.wait()
        start.cancel()
        finish_submission.set()
        with pytest.raises(asyncio.CancelledError):
            await start
        assert reclaimed.is_set()

    asyncio.run(run())


class ExecTransport:
    """Run trusted fixture commands locally in place of the Iris container."""

    def __init__(self, guest_root: Path | None = None):
        self.guest_root = guest_root

    def exec_in_container(self, request, timeout_ms):
        del timeout_ms
        command = list(request.command)
        if self.guest_root is not None:
            for prefix in ("/logs", "/tests", "/solution", "/workspace", "/setup_files"):
                command = [argument.replace(prefix, str(self.guest_root / prefix.lstrip("/"))) for argument in command]
            command = [argument.replace("id -u && id -un", "printf '0\\nroot\\n'") for argument in command]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return SimpleNamespace(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr, error="")


class ControllerJob:
    def __init__(self):
        self.cancelled = False
        self.shutdown = False
        self.endpoint_closed = False
        self.submission: dict = {}

    def submit(self, **kwargs):
        self.submission = kwargs
        return self

    def tasks(self):
        return [
            SimpleNamespace(
                task_id=JobName.from_wire("/test/job/0"), status=lambda: SimpleNamespace(state=TaskState.RUNNING)
            )
        ]

    def cancel(self):
        self.cancelled = True


@pytest.mark.parametrize(
    ("resource_mode", "expected_cpu", "expected_memory_mb"),
    [("auto", 3, 3072), ("request", 3, 3072), ("ignore", 1, 2048)],
)
def test_offline_lifecycle_commands_and_host_artifacts(
    tmp_path: Path, monkeypatch, resource_mode: str, expected_cpu: int, expected_memory_mb: int
) -> None:
    job = ControllerJob()
    client = SimpleNamespace(submit=job.submit, shutdown=lambda: setattr(job, "shutdown", True))
    endpoint = SimpleNamespace(url="http://test", credentials=None, close=lambda: setattr(job, "endpoint_closed", True))
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: endpoint)
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: client)
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: ExecTransport())
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "driver-private")
    workspace = tmp_path / "workspace"
    config = EnvironmentConfig(
        docker_image=IMAGE,
        allow_internet=False,
        workdir=str(workspace),
        env={"PUBLIC": "task-visible"},
        cpus=3,
        memory_mb=3072,
    )
    adapter = environment(
        tmp_path, task_env_config=config, cpu_enforcement_policy=resource_mode, memory_enforcement_policy=resource_mode
    )

    async def run() -> None:
        await adapter.start(force_build=False)
        # The local fixture transport inherits the test process; the submitted Iris
        # environment below is the credential boundary exercised by this regression.
        result = await adapter.exec('printf "%s" "$PUBLIC" > answer.txt; cat answer.txt')
        assert result.return_code == 0
        assert result.stdout == "task-visible"
        assert (await adapter.exec("pwd")).stdout.strip() == str(workspace)
        downloaded = tmp_path / "answer.txt"
        await adapter.download_file(str(workspace / "answer.txt"), downloaded)
        assert downloaded.read_text() == "task-visible"
        uploads = tmp_path / "inputs"
        uploads.mkdir()
        (uploads / "input.txt").write_text("fresh input")
        await adapter.upload_dir(uploads, str(workspace / "inputs"))
        assert (workspace / "inputs" / "input.txt").read_text() == "fresh input"
        await adapter.download_dir(str(workspace / "inputs"), f"memory://iris-{tmp_path.name}/inputs")
        assert StoragePath(f"memory://iris-{tmp_path.name}/inputs/input.txt").read_text() == "fresh input"
        private = tmp_path / "driver-private.txt"
        private.write_text("private")
        (workspace / "inputs" / "escape").symlink_to(private)
        with pytest.raises(tarfile.FilterError):
            await adapter.download_dir(str(workspace / "inputs"), tmp_path / "unsafe-download")
        assert private.read_text() == "private"
        await adapter.stop(delete=True)

    asyncio.run(run())
    assert job.submission["container_profile"] == job_pb2.CONTAINER_PROFILE_SANDBOX
    assert job.submission["egress_policy"] == job_pb2.EGRESS_POLICY_NONE
    assert job.submission["environment"].env_vars == {}
    assert job.submission["environment"].setup_scripts == []
    assert job.submission["task_image"] == IMAGE
    assert job.submission["resources"].cpu == expected_cpu
    assert job.submission["resources"].memory == expected_memory_mb * 1024 * 1024
    assert job.cancelled and job.shutdown and job.endpoint_closed


def test_oracle_harbor_trial_uses_real_mounts_and_verifier_lifecycle(tmp_path: Path, monkeypatch) -> None:
    controller_job = ControllerJob()
    client = SimpleNamespace(submit=controller_job.submit, shutdown=lambda: None)
    endpoint = SimpleNamespace(url="http://test", credentials=None, close=lambda: None)
    guest = tmp_path / "guest"
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: endpoint)
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: client)
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: ExecTransport(guest))
    task = tmp_path / "task"
    for directory in ("environment", "solution", "tests"):
        (task / directory).mkdir(parents=True)
    (task / "instruction.md").write_text("Write the requested answer to answer.txt.")
    (task / "task.toml").write_text(
        f' version = "1.0"\n[environment]\ndocker_image = "{IMAGE}"\nallow_internet = false\nworkdir = "/workspace"\n'
    )
    (task / "solution" / "solve.sh").write_text("#!/bin/sh\nprintf correct > answer.txt\n")
    (task / "tests" / "test.sh").write_text(
        f"#!/bin/sh\n[ \"$(cat '{guest}/workspace/answer.txt')\" = correct ] && "
        f"printf 1 > '{guest}/logs/verifier/reward.txt'\n"
    )
    config = JobConfig.model_validate(
        {
            "job_name": "offline-iris-oracle",
            "jobs_dir": str(tmp_path / "jobs"),
            "n_concurrent_trials": 1,
            "tasks": [{"path": str(task)}],
            "agents": [{"name": "oracle"}],
            "environment": {
                "import_path": "harbor.environments.iris:IrisEnvironment",
                "kwargs": {"cluster": "test"},
            },
        }
    )

    async def run() -> None:
        result = await (await Job.create(config)).run()
        assert result.stats.n_errored_trials == 0
        evaluation = next(iter(result.stats.evals.values()))
        assert evaluation.metrics[0]["mean"] == 1.0

    asyncio.run(run())
    assert controller_job.cancelled
