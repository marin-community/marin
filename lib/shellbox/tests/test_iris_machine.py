# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Iris backend with a local subprocess exec provider and a scripted controller."""

import asyncio
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("iris")

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.client.workload_codec import task_status_from_proto
from iris.cluster.types import JobName
from iris.resources.state import TaskState
from iris.rpc import job_pb2
from rigging.timing import ExponentialBackoff
from shellbox.backends.iris import machine as iris_backend
from shellbox.backends.iris.machine import IrisMachine, IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, MachineTerminated, NetworkPolicy

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


class RefusingRpc(LocalRpc):
    """Fails the first ``refusals`` execs with ``code`` before running anything, then runs them locally."""

    def __init__(self, code: Code, refusals: int):
        self.code = code
        self.refusals = refusals
        self.scripts: list[str] = []

    def exec_in_container(self, request, timeout_ms):
        self.scripts.append(request.command[-1])
        if len(self.scripts) <= self.refusals:
            raise ConnectError(self.code, "refused")
        return super().exec_in_container(request, timeout_ms)

    def sent(self, marker: str) -> int:
        return sum(marker in script for script in self.scripts)


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

    def cancel(self):
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


@pytest.mark.parametrize("private_credentials", [False, True])
def test_factory_uses_typed_iris_states_and_cancels_the_job(tmp_path, monkeypatch, caplog, private_credentials):
    task_id = JobName.from_wire("/fixture/sandbox/0")
    status = task_status_from_proto(job_pb2.TaskStatus(task_id=task_id.to_wire(), state=job_pb2.TASK_STATE_RUNNING))
    job = LocalJob()
    job.tasks = lambda: [SimpleNamespace(task_id=task_id, status=lambda: status)]
    client = LocalClient()
    submitted_environments = []

    def submit(**kwargs):
        submitted_environments.append(kwargs["environment"].env_vars)
        return job

    client.submit = submit
    endpoint = LocalEndpoint()
    endpoint.url = "http://fixture"
    endpoint.credentials = None
    monkeypatch.setattr("shellbox.backends.iris.machine.connect_controller", lambda **kwargs: endpoint)
    monkeypatch.setattr("shellbox.backends.iris.machine.IrisClient.remote", lambda *args, **kwargs: client)
    monkeypatch.setattr("shellbox.backends.iris.machine.ControllerServiceClientSync", lambda **kwargs: LocalRpc())
    secret = "private-fixture-credential"
    monkeypatch.setenv("SHELLBOX_TEST_PRIVATE_KEY", secret)
    factory = IrisMachineFactory(
        controller_url="http://fixture",
        secret_env={"JUDGE_KEY": ("env:SHELLBOX_TEST_PRIVATE_KEY",)} if private_credentials else None,
    )
    spec = MachineSpec(RegistryImage("fixture"), workdir=str(tmp_path), network=NetworkPolicy.ALLOW)

    async def scenario():
        machine = await factory.create(spec)
        try:
            assert secret not in json.dumps(asdict(machine.spec), default=str)
            result = await machine.run(Command(("sh", "-c", "printf ready")))
            assert result.stdout == b"ready"
        finally:
            await machine.close()

    asyncio.run(scenario())
    assert job.terminated
    assert submitted_environments == [{"JUDGE_KEY": secret} if private_credentials else {}]
    assert secret not in json.dumps(asdict(spec), default=str)
    assert secret not in caplog.text


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


def test_exec_refused_by_a_full_controller_pool_is_retried(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(iris_backend, "EXEC_SHED_BACKOFF", ExponentialBackoff(initial=0.001, maximum=0.001))
    rpc = RefusingRpc(Code.RESOURCE_EXHAUSTED, refusals=2)
    machine, _ = local_machine(tmp_path, rpc)

    result = asyncio.run(machine.run(Command(("echo", "ran"))))

    assert result.stdout == b"ran\n"
    assert rpc.sent("echo ran") == 3


def test_exec_failing_after_the_controller_accepted_it_is_not_repeated(tmp_path: Path) -> None:
    rpc = RefusingRpc(Code.UNAVAILABLE, refusals=1)
    machine, _ = local_machine(tmp_path, rpc)

    with pytest.raises(ConnectError):
        asyncio.run(machine.run(Command(("echo", "once"))))
    assert rpc.sent("echo once") == 1


@pytest.mark.parametrize(
    ("network", "egress"),
    [
        (NetworkPolicy.ALLOW, job_pb2.EGRESS_POLICY_INTERNET),
        (NetworkPolicy.DENY, job_pb2.EGRESS_POLICY_NONE),
    ],
)
def test_network_policy_selects_the_egress_policy(monkeypatch, network, egress):
    client = RecordingClient()
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: LocalEndpoint())
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: client)
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: LocalRpc())
    factory = IrisMachineFactory(controller_url="http://controller")

    with pytest.raises(SubmissionRecorded):
        asyncio.run(factory.create(MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir="/tmp", network=network)))

    assert client.submitted["container_profile"] == job_pb2.CONTAINER_PROFILE_SANDBOX
    assert client.submitted["egress_policy"] == egress
