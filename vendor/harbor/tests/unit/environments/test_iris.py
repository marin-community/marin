# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Harbor adapter through a local direct Iris process broker."""

import asyncio
import os
import signal
import socket
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.environments.iris import IrisEnvironment
from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName, NetworkAttachment
from iris.resources.state import TaskState
from iris.rpc import controller_pb2, job_pb2
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.iris import machine as iris_backend
from shellbox.backends.iris import process_guest

IMAGE = "example.test/task@sha256:" + "a" * 64


def environment(tmp_path: Path, **kwargs) -> IrisEnvironment:
    config = kwargs.pop(
        "task_env_config",
        EnvironmentConfig(
            docker_image=IMAGE, allow_internet=False, workdir="/workspace"
        ),
    )
    return IrisEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="task",
        session_id="trial",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=config,
        **kwargs,
    )


def test_rejects_runtime_builds_without_prepared_images(tmp_path: Path) -> None:
    task_environment = tmp_path / "environment"
    task_environment.mkdir()
    (task_environment / "Dockerfile").write_text("FROM python:3.12-slim")
    with pytest.raises(ValueError):
        environment(
            tmp_path,
            task_env_config=EnvironmentConfig(
                allow_internet=False, workdir="/workspace"
            ),
        )


def test_task_cannot_import_driver_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "driver-private")
    config = EnvironmentConfig(
        docker_image=IMAGE,
        allow_internet=False,
        env={"STOLEN": "${AWS_SECRET_ACCESS_KEY}"},
    )
    with pytest.raises(ValueError):
        environment(tmp_path, task_env_config=config)
    with pytest.raises(ValueError, match="driver environment"):
        environment(tmp_path, persistent_env={"STOLEN": "driver-private"})


@pytest.mark.parametrize(
    "override",
    [
        {"network": "allow"},
        {"container_profile": "privileged"},
        {"secret_env": {"KEY": "secret"}},
    ],
)
def test_cannot_override_isolation(tmp_path: Path, override: dict) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        environment(tmp_path, **override)


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_cancelled_start_reclaims_job_that_finishes_submitting(
    tmp_path: Path, monkeypatch, cleanup_failure: bool
) -> None:
    adapter = environment(tmp_path)

    async def run() -> None:
        submitting = asyncio.Event()
        finish_submission = asyncio.Event()
        closing = asyncio.Event()
        finish_cleanup = asyncio.Event()
        reclaimed = asyncio.Event()

        async def close() -> None:
            closing.set()
            await finish_cleanup.wait()
            reclaimed.set()
            if cleanup_failure:
                raise RuntimeError("Backend cleanup failed")

        async def create(spec):
            submitting.set()
            await finish_submission.wait()
            return SimpleNamespace(close=close)

        monkeypatch.setattr(adapter.factory, "create_task", create)
        start = asyncio.create_task(adapter.start(force_build=False))
        await submitting.wait()
        start.cancel("initial cancellation")
        finish_submission.set()
        await asyncio.wait_for(closing.wait(), 5)
        start.cancel("repeated cancellation")
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError) as error:
            await asyncio.wait_for(start, 5)
        assert error.value.args == ("initial cancellation",)
        assert reclaimed.is_set()

    asyncio.run(run())


@pytest.fixture(autouse=True)
def in_cluster_driver(monkeypatch):
    monkeypatch.setattr(
        iris_backend,
        "get_job_info",
        lambda: JobInfo(
            task_id=JobName.from_wire("/test/driver/0"),
            attempt_uid="driver-attempt",
            controller_address="http://test",
            task_token="fixture-task-token",
        ),
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    monkeypatch.setattr(process_guest, "PORT", port)


class ConnectionTransport:
    def get_task_connection(self, request, timeout_ms):
        return controller_pb2.Controller.GetTaskConnectionResponse(
            pod_ip="127.0.0.1",
            port=process_guest.PORT,
            pod_uid="fixture-pod",
            attempt_uid="fixture-attempt",
        )

    def exec_in_container(self, request, timeout_ms):
        raise AssertionError("Primary commands must use the direct process broker")


def mapped_broker_source(source: str, guest_root: Path) -> str:
    # The local broker has no container filesystem: rewrite only its process
    # spawn boundary so Harbor's fixed guest paths stay under the test directory.
    shim = f"""
_original_spawn = asyncio.create_subprocess_exec

def _guest_path(value):
    for prefix in ("/logs", "/tests", "/solution", "/workspace", "/setup_files"):
        value = value.replace(prefix, {str(guest_root)!r} + prefix)
    return value.replace("id -u && id -un", "printf '0\\nroot\\n'")

async def _guest_spawn(*argv, **kwargs):
    kwargs["cwd"] = _guest_path(kwargs["cwd"])
    return await _original_spawn(*(_guest_path(arg) for arg in argv), **kwargs)

asyncio.create_subprocess_exec = _guest_spawn
"""
    return source.replace(
        'if __name__ == "__main__":', shim + '\nif __name__ == "__main__":'
    )


class ControllerJob:
    def __init__(self, guest_root: Path | None = None):
        self.guest_root = guest_root
        self.cancelled = False
        self.shutdown = False
        self.submission: dict = {}
        self.process: subprocess.Popen | None = None

    def submit_task(self, **kwargs):
        self.submission = kwargs
        command = list(kwargs["task"].primary_service.command.command)
        if self.guest_root is not None:
            command[3] = mapped_broker_source(command[3], self.guest_root)
            command[-1] = str(self.guest_root / command[-1].lstrip("/"))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env={"PATH": os.environ["PATH"]},
        )
        return self

    def task(self, task_index: int):
        return SimpleNamespace(
            task_id=JobName.from_wire("/test/driver/child").task(task_index),
            status=lambda: SimpleNamespace(
                state=TaskState.RUNNING,
                error_message="",
                current_attempt_number=0,
                attempts=(
                    SimpleNamespace(attempt_number=0, attempt_uid="fixture-attempt"),
                ),
            ),
        )

    def cancel(self):
        self.cancelled = True
        if self.process is not None:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=10)
            self.process.stderr.close()


@pytest.fixture
def local_controller(monkeypatch):
    jobs = []

    def configure(guest_root: Path | None = None):
        job = ControllerJob(guest_root)
        jobs.append(job)
        client = SimpleNamespace(
            submit_task=job.submit_task, shutdown=lambda: setattr(job, "shutdown", True)
        )
        monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: client)
        monkeypatch.setattr(
            iris_backend,
            "ControllerServiceClientSync",
            lambda **_: ConnectionTransport(),
        )
        return job

    yield configure
    for job in jobs:
        job.cancel()


@pytest.mark.parametrize(
    ("resource_mode", "expected_cpu", "expected_memory_mb"),
    [("auto", 3, 3072), ("request", 3, 3072), ("ignore", 1, 2048)],
)
def test_offline_lifecycle_commands_and_host_artifacts(
    tmp_path: Path,
    monkeypatch,
    local_controller,
    resource_mode: str,
    expected_cpu: int,
    expected_memory_mb: int,
) -> None:
    job = local_controller()
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
        tmp_path,
        task_env_config=config,
        cpu_enforcement_policy=resource_mode,
        memory_enforcement_policy=resource_mode,
    )

    async def run() -> None:
        await adapter.start(force_build=False)
        # The submitted environment is the credential boundary; the local broker
        # also starts with a clean environment to model the isolated container.
        result = await adapter.exec(
            'printf "%s" "$PUBLIC" > answer.txt; cat answer.txt'
        )
        assert result.return_code == 0
        assert result.stdout == "task-visible"
        assert (await adapter.exec("pwd")).stdout.strip() == str(workspace)
        with pytest.raises(ValueError):
            await adapter.exec(
                "touch credential-leak.txt",
                env={"AWS_SECRET_ACCESS_KEY": "driver-private"},
            )
        with pytest.raises(ValueError):
            await adapter.exec(
                "touch credential-leak.txt", env={"INNOCENT_ALIAS": "driver-private"}
            )
        assert not (workspace / "credential-leak.txt").exists()
        downloaded = tmp_path / "answer.txt"
        await adapter.download_file(str(workspace / "answer.txt"), downloaded)
        assert downloaded.read_text() == "task-visible"
        uploads = tmp_path / "inputs"
        uploads.mkdir()
        (uploads / "input.txt").write_text("fresh input")
        await adapter.upload_dir(uploads, str(workspace / "inputs"))
        assert (workspace / "inputs" / "input.txt").read_text() == "fresh input"
        remote = StoragePath(f"memory://iris-{tmp_path.name}/inputs")
        (remote / "existing.txt").write_text("previous trial artifact")
        await adapter.download_dir(str(workspace / "inputs"), str(remote))
        assert (remote / "input.txt").read_text() == "fresh input"
        assert (remote / "existing.txt").read_text() == "previous trial artifact"
        assert not (remote / "download").exists()
        private = tmp_path / "driver-private.txt"
        private.write_text("private")
        (workspace / "inputs" / "escape").symlink_to(private)
        with pytest.raises(tarfile.FilterError):
            await adapter.download_dir(
                str(workspace / "inputs"), tmp_path / "unsafe-download"
            )
        assert private.read_text() == "private"
        await adapter.stop(delete=True)

    asyncio.run(run())
    assert job.submission["container_profile"] == job_pb2.CONTAINER_PROFILE_SANDBOX
    assert job.submission["egress_policy"] == job_pb2.EGRESS_POLICY_NONE
    primary = job.submission["task"].primary_service.to_proto(
        job_pb2.EnvironmentConfig()
    )
    assert dict(primary.env_vars) == {"PUBLIC": "task-visible"}
    assert not primary.setup_scripts and not primary.secrets
    assert primary.image == IMAGE
    assert primary.resources.cpu_millicores == expected_cpu * 1000
    assert primary.resources.memory_bytes == expected_memory_mb * 1024 * 1024
    assert job.submission["task"].network_attachment is NetworkAttachment.POD
    assert (
        job.submission["task"].driver_connection.driver_attempt_uid == "driver-attempt"
    )
    assert job.cancelled and job.shutdown


def test_oracle_harbor_trial_uses_real_mounts_and_verifier_lifecycle(
    tmp_path: Path, local_controller
) -> None:
    guest = tmp_path / "guest"
    controller_job = local_controller(guest)
    task = tmp_path / "task"
    for directory in ("environment", "solution", "tests"):
        (task / directory).mkdir(parents=True)
    (task / "instruction.md").write_text("Write the requested answer to answer.txt.")
    (task / "task.toml").write_text(
        f' version = "1.0"\n[environment]\ndocker_image = "{IMAGE}"\nallow_internet = false\nworkdir = "/workspace"\n'
    )
    (task / "solution" / "solve.sh").write_text(
        "#!/bin/sh\nprintf correct > answer.txt\n"
    )
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
                "type": "iris",
                "kwargs": {},
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
