# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Iris backend with a local subprocess exec provider and a scripted controller."""

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from iris.cluster.types import JobName
from iris.resources.state import TaskState
from shellbox.backends.iris import machine as iris_backend
from shellbox.backends.iris.machine import IrisMachine, IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, MachineTerminated, NetworkPolicy, UnsupportedMachineSpec

# Linux MAX_ARG_STRLEN: the worker passes the exec command as argv to `docker exec` or `kubectl exec`.
LINUX_ARGUMENT_LIMIT_BYTES = 128 * 1024


class LocalRpc:
    def exec_in_container(self, request, timeout_ms):
        del timeout_ms
        if max(len(argument.encode()) + 1 for argument in request.command) > LINUX_ARGUMENT_LIMIT_BYTES:
            return SimpleNamespace(exit_code=0, stdout="", stderr="", error="[Errno 7] Argument list too long: 'docker'")
        result = subprocess.run(request.command, capture_output=True, text=True, timeout=30)
        return SimpleNamespace(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr, error="")


class FailingRpc:
    def exec_in_container(self, request, timeout_ms):
        del request, timeout_ms
        return SimpleNamespace(exit_code=0, stdout="", stderr="", error="Task /user/shellbox/0 is not running")


class LocalTask:
    """A sandbox task in one fixed state."""

    def __init__(self, state: TaskState):
        self.state = state
        self.task_id = JobName.from_wire("/user/shellbox/0")

    def status(self):
        return SimpleNamespace(
            state=self.state, error_message="" if self.state is TaskState.RUNNING else "container exited"
        )


class LocalJob:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


class LocalClient:
    def shutdown(self):
        pass


class SubmissionRecorded(Exception):
    """Raised by ``RecordingClient.submit`` so ``create`` stops before polling the task."""


class RecordingClient(LocalClient):
    def __init__(self):
        self.submitted: dict = {}

    def submit(self, **kwargs):
        self.submitted = kwargs
        raise SubmissionRecorded


class LocalEndpoint:
    url = "http://controller"
    credentials = None

    def close(self):
        pass


def local_machine(tmp_path: Path, rpc=None, task: LocalTask | None = None) -> tuple[IrisMachine, LocalJob]:
    job = LocalJob()
    spec = MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
    machine = IrisMachine(
        LocalEndpoint(),  # type: ignore[arg-type]
        LocalClient(),  # type: ignore[arg-type]
        rpc or LocalRpc(),  # type: ignore[arg-type]
        job,  # type: ignore[arg-type]
        task or LocalTask(TaskState.RUNNING),  # type: ignore[arg-type]
        spec,
    )
    return machine, job


def submitted_job(monkeypatch, factory: IrisMachineFactory) -> dict:
    """Return the keyword arguments ``create`` passes to ``IrisClient.submit``."""
    client = RecordingClient()
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: LocalEndpoint())
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: client)
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: LocalRpc())
    with pytest.raises(SubmissionRecorded):
        asyncio.run(factory.create(MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir="/tmp")))
    return client.submitted


@pytest.mark.parametrize("resource", ["cpus", "storage_mb"])
def test_zero_resource_requests_cannot_silently_select_iris_defaults(resource):
    with pytest.raises(ValueError, match=resource):
        MachineSpec(source=RegistryImage("ubuntu:24.04"), **{resource: 0})


def test_iris_binary_command_and_file_round_trip(tmp_path: Path) -> None:
    async def scenario() -> None:
        machine, job = local_machine(tmp_path)
        try:
            result = await machine.run(
                Command(("/bin/sh", "-c", "cat; printf '\\000\\377' >&2"), stdin=b"abc\x00\xff", output_limit_bytes=4)
            )
            assert result.exit_code == 0
            assert result.stdout == b"abc\x00"
            assert result.stdout_truncated
            assert result.stderr == b"\x00\xff"

            source = tmp_path / "input.bin"
            source.write_bytes(b"\x00\xffpayload")
            await machine.upload(source, str(tmp_path / "remote.bin"))
            target = tmp_path / "download.bin"
            await machine.download(str(tmp_path / "remote.bin"), target)
            assert target.read_bytes() == source.read_bytes()
        finally:
            await machine.close()
        assert job.terminated

    asyncio.run(scenario())


def test_file_larger_than_one_exec_argument_round_trips(tmp_path: Path) -> None:
    async def scenario() -> None:
        machine, _ = local_machine(tmp_path)
        payload = bytes(range(256)) * 2048
        source = tmp_path / "large.bin"
        source.write_bytes(payload)
        try:
            await machine.upload(source, str(tmp_path / "remote.bin"))
            await machine.download(str(tmp_path / "remote.bin"), tmp_path / "back.bin")
        finally:
            await machine.close()
        assert (tmp_path / "back.bin").read_bytes() == payload

    asyncio.run(scenario())


def test_create_keeps_the_submitters_tokens_out_of_the_sandbox(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "submitter-hf-token")
    monkeypatch.setenv("WANDB_API_KEY", "submitter-wandb-key")
    factory = IrisMachineFactory(controller_url="http://controller", cluster_network=NetworkPolicy.DENY)

    env_vars = submitted_job(monkeypatch, factory)["environment"].to_proto().env_vars

    assert env_vars["HF_TOKEN"] == ""
    assert env_vars["WANDB_API_KEY"] == ""


def test_create_refuses_a_network_policy_the_cluster_does_not_provide() -> None:
    factory = IrisMachineFactory(controller_url="http://controller", cluster_network=NetworkPolicy.DENY)

    with pytest.raises(UnsupportedMachineSpec, match="provides deny, not allow"):
        asyncio.run(factory.create(MachineSpec(source=RegistryImage("ubuntu:24.04"), network=NetworkPolicy.ALLOW)))


@pytest.mark.parametrize("state", [TaskState.KILLED, TaskState.FAILED, TaskState.PREEMPTED, TaskState.WORKER_FAILED])
def test_command_on_an_ended_sandbox_raises_machine_terminated(tmp_path: Path, state: TaskState) -> None:
    machine, _ = local_machine(tmp_path, FailingRpc(), LocalTask(state))

    with pytest.raises(MachineTerminated, match=f"is {state}"):
        asyncio.run(machine.run(Command(("true",))))


def test_exec_error_on_a_running_sandbox_is_not_machine_terminated(tmp_path: Path) -> None:
    machine, _ = local_machine(tmp_path, FailingRpc(), LocalTask(TaskState.RUNNING))

    with pytest.raises(RuntimeError, match="Iris exec failed") as raised:
        asyncio.run(machine.run(Command(("true",))))
    assert not isinstance(raised.value, MachineTerminated)
