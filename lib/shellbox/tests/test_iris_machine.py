# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Iris backend with a local subprocess exec provider and a scripted controller."""

import asyncio
import json
import os
import pwd
import re
import shlex
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
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
from shellbox.machine import (
    Command,
    ExecSetupFailed,
    ExitReason,
    MachineCancelled,
    MachineSpec,
    MachineTerminated,
    MachineUnavailable,
    NetworkPolicy,
    UnsupportedMachineSpec,
)

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
    def __init__(self, task: LocalTask | None = None):
        self.job_id = JobName.from_wire("/user/shellbox")
        self.cancelled = False
        self.killed = threading.Event()
        self._task = task

    def tasks(self):
        return [] if self._task is None else [self._task]

    def task(self, index):
        return self.tasks()[index]

    def cancel(self):
        self.cancelled = True
        self.killed.set()


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


class TaskClient(LocalClient):
    def __init__(self, job: LocalJob):
        self.job = job

    def submit(self, **kwargs):
        return self.job


class LocalEndpoint:
    url = "http://controller"
    credentials = None

    def close(self):
        pass


def local_machine(
    tmp_path: Path, rpc=None, task: LocalTask | None = None, job: LocalJob | None = None
) -> tuple[IrisMachine, LocalJob]:
    job = job or LocalJob()
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
        assert job.cancelled

    asyncio.run(scenario())


@pytest.mark.parametrize("private_credentials", [False, True])
@pytest.mark.parametrize("has_timeout", [False, True])
def test_factory_uses_typed_iris_states_and_cancels_the_job(
    tmp_path, monkeypatch, caplog, private_credentials, has_timeout
):
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

    class ProbeRpc(LocalRpc):
        def exec_in_container(self, request, timeout_ms):
            if request.command == ["sh", "-c", "command -v timeout"]:
                return SimpleNamespace(exit_code=0 if has_timeout else 1, stdout="", stderr="", error="")
            return super().exec_in_container(request, timeout_ms)

    monkeypatch.setattr("shellbox.backends.iris.machine.ControllerServiceClientSync", lambda **kwargs: ProbeRpc())
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
            assert machine._timeout_available is has_timeout
            assert secret not in json.dumps(asdict(machine.spec), default=str)
            result = await machine.run(Command(("sh", "-c", "printf ready")))
            assert result.stdout == b"ready"
        finally:
            await machine.close()

    asyncio.run(scenario())
    assert job.cancelled
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


def test_dead_container_exec_error_overrides_stale_running_status(tmp_path: Path) -> None:
    machine, _ = local_machine(tmp_path, FailingRpc(), LocalTask(TaskState.RUNNING))

    with pytest.raises(RuntimeError, match="Iris exec failed") as raised:
        asyncio.run(machine.run(Command(("true",))))
    assert isinstance(raised.value, MachineTerminated)


@pytest.mark.parametrize("code", list(Code))
def test_exec_rpc_failure_is_retried(tmp_path: Path, monkeypatch, code) -> None:
    monkeypatch.setattr(iris_backend, "EXEC_SHED_BACKOFF", ExponentialBackoff(initial=0.001, maximum=0.001))
    rpc = RefusingRpc(code, refusals=2)
    machine, _ = local_machine(tmp_path, rpc)

    result = asyncio.run(machine.run(Command(("echo", "ran"))))

    assert result.stdout == b"ran\n"
    assert rpc.sent("echo ran") == 3


def test_exec_programming_error_is_not_transport_unavailability() -> None:
    error = ValueError("malformed local request")
    call = Mock(side_effect=error)
    with pytest.raises(ValueError) as caught:
        iris_backend._controller_call(call, exec_call=True)
    assert caught.value is error
    assert call.call_count == 1


class ClosingRpc(RefusingRpc):
    """Refuses every exec for a full pool; during the first refusal another thread closes the machine."""

    def __init__(self):
        super().__init__(Code.RESOURCE_EXHAUSTED, refusals=1_000)
        self.machine: IrisMachine | None = None
        self.closer: threading.Thread | None = None

    def exec_in_container(self, request, timeout_ms):
        if self.closer is None:
            machine = self.machine
            assert machine is not None
            closing = machine.close()
            self.closer = threading.Thread(target=lambda: asyncio.run(closing))
            self.closer.start()
            assert machine.job.killed.wait(timeout=10)  # type: ignore[attr-defined]
        return super().exec_in_container(request, timeout_ms)


def test_exec_retry_stops_once_the_machine_is_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(iris_backend, "EXEC_SHED_BACKOFF", ExponentialBackoff(initial=0.001, maximum=0.001))
    rpc = ClosingRpc()
    machine, job = local_machine(tmp_path, rpc)
    rpc.machine = machine

    with pytest.raises(RuntimeError, match="Machine is closed"):
        asyncio.run(machine.run(Command(("echo", "late"))))
    assert rpc.closer is not None
    rpc.closer.join(timeout=10)
    assert not rpc.closer.is_alive()
    assert job.cancelled
    assert rpc.sent("echo late") == 1


class HeldRpc:
    """An exec the controller admitted, still running until ``release`` is set."""

    def __init__(self, release: threading.Event):
        self.release = release
        self.admitted = threading.Event()
        self.returned = threading.Event()

    def exec_in_container(self, request, timeout_ms):
        del request, timeout_ms
        self.admitted.set()
        assert self.release.wait(timeout=10)
        self.returned.set()
        return SimpleNamespace(exit_code=0, stdout="", stderr="", error="")


def test_close_returns_only_after_admitted_execs_end(tmp_path: Path, caplog) -> None:
    caplog.set_level("INFO")
    job = LocalJob()
    # Terminating the sandbox ends the exec the controller already admitted.
    rpc = HeldRpc(release=job.killed)
    machine, _ = local_machine(tmp_path, rpc, job=job)

    with ThreadPoolExecutor(max_workers=1) as pool:
        exec_future = pool.submit(machine._exec_sync, ["true"])
        assert rpc.admitted.wait(timeout=10)

        asyncio.run(machine.close())

        assert rpc.returned.is_set()
        exec_future.result(timeout=10)
    assert "Iris kill job=/user/shellbox result=acknowledged returned_state=not_provided" in caplog.text


def test_close_raises_when_an_admitted_exec_outlives_the_sandbox(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(iris_backend, "CLOSE_DRAIN_SECONDS", 0.01)
    rpc = HeldRpc(release=threading.Event())
    machine, job = local_machine(tmp_path, rpc)

    with ThreadPoolExecutor(max_workers=1) as pool:
        exec_future = pool.submit(machine._exec_sync, ["true"])
        assert rpc.admitted.wait(timeout=10)

        with pytest.raises(RuntimeError, match=r"1 Iris exec\(s\) still in flight .*: true$"):
            asyncio.run(machine.close())

        assert job.cancelled
        rpc.release.set()
        exec_future.result(timeout=10)


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


@pytest.mark.parametrize("task_missing", [False, True])
def test_factory_waits_for_a_pending_task_without_holding_an_executor_thread(monkeypatch, tmp_path, task_missing):
    task = LocalTask(TaskState.PENDING)
    current_status = task.status

    def status():
        nonlocal task_missing
        if task_missing:
            task_missing = False
            raise ConnectError(Code.NOT_FOUND, "Task is not registered yet")
        return current_status()

    task.status = status
    job = LocalJob(task)
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: LocalEndpoint())
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: TaskClient(job))
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: LocalRpc())
    factory = IrisMachineFactory(controller_url="http://controller")
    workdir = tmp_path / "workspace"
    spec = MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(workdir))
    existing, _ = local_machine(tmp_path)

    async def scenario():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        creation = asyncio.create_task(factory.create(spec))
        command = asyncio.create_task(existing.run(Command(("printf", "ready"))))
        done, _ = await asyncio.wait((command,), timeout=5)
        task.state = TaskState.RUNNING
        machine = await creation
        try:
            assert command in done and command.result().stdout == b"ready"
            assert workdir.is_dir()
        finally:
            await command
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "state,reason,error_type",
    [
        (TaskState.FAILED, "container exited", MachineTerminated),
        (TaskState.FAILED, "PodDeleted: pod was deleted while the attempt was starting", MachineTerminated),
        (TaskState.PREEMPTED, "WorkloadEvictedDueToPreempted", MachineTerminated),
        (TaskState.KILLED, "Parent job terminated", MachineCancelled),
        (TaskState.PENDING, "Controller is shutting down", MachineUnavailable),
    ],
)
def test_factory_cancels_the_job_when_the_task_fails(monkeypatch, tmp_path, state, reason, error_type):
    task = LocalTask(state)
    # Pilots k/l: PodDeleted; m: parent killed; n/o: gap plus failed cleanup.
    unavailable = state is TaskState.PENDING
    task.status = (
        Mock(side_effect=ConnectError(Code.UNAVAILABLE, reason))
        if unavailable
        else Mock(return_value=SimpleNamespace(state=state, error_message=reason))
    )
    job = LocalJob(task)
    job.cancel = Mock(wraps=job.cancel, side_effect=RuntimeError("cancel unavailable") if unavailable else None)
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: LocalEndpoint())
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: TaskClient(job))
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: LocalRpc())
    factory = IrisMachineFactory(controller_url="http://controller")
    spec = MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(tmp_path / "workspace"))

    with pytest.raises(error_type, match=reason.lower() if state is TaskState.KILLED else reason) as caught:
        asyncio.run(factory.create(spec))
    assert job.cancel.called
    if unavailable:
        assert caught.value.__cause__ is task.status.side_effect
        assert any("cancel unavailable" in note for note in caught.value.__notes__)


def test_commands_may_name_the_container_user_but_no_other(tmp_path: Path) -> None:
    async def scenario() -> None:
        machine, _ = local_machine(tmp_path)
        try:
            by_uid = await machine.run(Command(("id", "-u"), user=str(os.getuid())))
            by_name = await machine.run(Command(("id", "-u"), user=pwd.getpwuid(os.getuid()).pw_name))
            assert by_uid.stdout.strip() == by_name.stdout.strip() == str(os.getuid()).encode()
            with pytest.raises(UnsupportedMachineSpec, match="cannot run as 65534"):
                await machine.run(Command(("id", "-u"), user="65534"))
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("exit_code", [124, 137])
@pytest.mark.parametrize("elapsed", [10, 30])
def test_command_timeout_keeps_output_and_machine_usable(tmp_path, monkeypatch, exit_code, elapsed):
    machine, job = local_machine(tmp_path)
    machine._timeout_available = True
    scripts = []
    deadlines = []
    now = [0.0]
    monkeypatch.setattr(iris_backend, "time", SimpleNamespace(monotonic=lambda: now[0]))

    async def script(text, timeout=None):
        scripts.append(text)
        deadlines.append(timeout)
        now[0] += elapsed if "timeout -k" in text else 0
        return iris_backend.ExecOutput(exit_code if "timeout -k" in text else 0, "", "")

    async def download(path, limit=None, *, known_file=False):
        return (b"partial output" if path.endswith("-out") else b"diagnostic"), False

    monkeypatch.setattr(machine, "_script", script)
    monkeypatch.setattr(machine, "_download_bytes", download)

    async def scenario():
        command = Command(("sh", "-c", "printf 'partial output'; sleep 90"), timeout=30)
        result = await machine.run(command)
        assert result.reason == (ExitReason.EXITED if exit_code == 137 and elapsed < 30 else ExitReason.TIMED_OUT)
        assert result.exit_code == exit_code
        assert result.stdout == b"partial output" and result.stderr == b"diagnostic"
        assert not job.cancelled
        assert (await machine.run(Command(("true",)))).reason == ExitReason.EXITED
        assert not job.cancelled
        assert deadlines[0] == 30 + iris_backend.RPC_PADDING_SECONDS
        timed = shlex.split(scripts[0].splitlines()[-1])
        assert timed[:6] == ["timeout", "-k", "5", "30s", "sh", "-c"]
        assert shlex.split(timed[6]) == list(command.argv)
        await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("has_timeout", [False, True])
def test_missing_timeout_or_rpc_backstop_closes_the_machine(tmp_path, monkeypatch, has_timeout):
    machine, job = local_machine(tmp_path)
    machine._timeout_available = has_timeout
    monkeypatch.setattr(iris_backend, "RPC_PADDING_SECONDS", 0.01)
    deadlines = []
    wait_for = asyncio.wait_for

    async def record_wait(awaitable, timeout):
        deadlines.append(timeout)
        return await wait_for(awaitable, timeout=timeout)

    async def stuck_script(text, timeout=None):
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio, "wait_for", record_wait)
    monkeypatch.setattr(machine, "_script", stuck_script)
    result = asyncio.run(machine.run(Command(("true",), timeout=0.01)))
    assert result.reason == ExitReason.TIMED_OUT
    assert job.cancelled
    assert deadlines == [0.02 if has_timeout else 0.01]


def test_missing_download_is_a_file_error(tmp_path):
    machine, _ = local_machine(tmp_path)
    missing = str(tmp_path / "missing file.py")
    with pytest.raises(FileNotFoundError, match=r"missing file\.py"):
        asyncio.run(machine.download(missing, tmp_path / "download"))


@pytest.mark.parametrize("memory_mb", [None, 4096])
def test_oom_termination_reports_effective_memory_request(tmp_path, memory_mb):
    task = LocalTask(TaskState.FAILED)
    task.status = lambda: SimpleNamespace(state=TaskState.FAILED, error_message="OOMKilled")
    machine, _ = local_machine(tmp_path, FailingRpc(), task)
    machine.spec = replace(machine.spec, memory_mb=memory_mb)
    with pytest.raises(MachineTerminated, match=f"memory request: {memory_mb or 2048} MiB"):
        asyncio.run(machine.run(Command(("true",))))


@pytest.mark.parametrize("timeout", [None, 180.1])
def test_exec_rpc_sends_explicit_deadline_instead_of_controller_default(tmp_path, timeout):
    machine, _ = local_machine(tmp_path)
    requests = []

    class Rpc:
        def exec_in_container(self, request, timeout_ms):
            requests.append((request.timeout_seconds, timeout_ms))
            return SimpleNamespace(exit_code=0, stdout="ok", stderr="", error="")

    machine.rpc = Rpc()
    assert machine._exec_sync(["true"], timeout).stdout == "ok"
    expected = iris_backend.DEFAULT_JOB_TTL if timeout is None else 181
    assert requests == [(expected, (expected + iris_backend.RPC_PADDING_SECONDS) * 1000)]
    asyncio.run(machine.close())


def test_ranged_file_reads_transfer_only_slice_and_reject_directories(tmp_path):
    class RecordingRpc(LocalRpc):
        def __init__(self):
            self.scripts = []

        def exec_in_container(self, request, timeout_ms):
            self.scripts.append(request.command[-1])
            return super().exec_in_container(request, timeout_ms)

    rpc = RecordingRpc()
    machine, _ = local_machine(tmp_path, rpc=rpc)
    path = tmp_path / "large.bin"
    path.write_bytes(b"a" * 1_000_000 + b"slice" + b"z" * 1_000_000)

    async def scenario():
        assert await machine.read_file(str(path), offset=1_000_000, limit=5) == b"slice"
        transfers = [script for script in rpc.scripts if "tail -c" in script]
        assert len(transfers) == 1 and "tail -c +1000001" in transfers[0] and "head -c 5" in transfers[0]
        assert await machine.read_file(str(path), offset=3_000_000, limit=5) == b""
        rpc.scripts.clear()
        with pytest.raises(IsADirectoryError):
            await machine.read_file(str(tmp_path))
        assert len(rpc.scripts) == 1 and "test -d" in rpc.scripts[0]
        assert not any("tar " in script or "tail -c" in script for script in rpc.scripts)
        with pytest.raises(FileNotFoundError):
            await machine.read_file(str(tmp_path / "absent"), limit=5)
        await machine.close()

    asyncio.run(scenario())


def test_process_cleanup_preserves_supervisor_and_current_exec_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # PID 2 was recycled: matching its old PID alone must not preserve the leaked process.
    # The active Python reset (92), timeout wrapper (91), and exec shell (90)
    # are all ancestors of the cleanup command and must survive it.
    processes = {1: (0, "100"), 2: (1, "999"), 3: (1, "300"), 90: (1, "900"), 91: (90, "910"), 92: (91, "920")}
    killed = []

    def stat(path: Path, *args: object, **kwargs: object) -> str:
        pid = int(path.parent.name)
        parent, started = processes[pid]
        fields = ["S", str(parent), *(["0"] * 17), started]
        return f"{pid} (command with spaces) " + " ".join(fields)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "glob", lambda path, pattern: iter(Path(f"/proc/{pid}/stat") for pid in processes))
        patch.setattr(Path, "read_text", stat)
        patch.setattr(os, "getpid", lambda: 92)
        patch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
        patch.setattr(sys, "argv", ["reset", json.dumps([{"1": ["0", "100"], "2": ["1", "200"]}, []])])
        exec(iris_backend.RESET.split("task_roots =", 1)[0], {})
    assert killed == [(2, signal.SIGKILL), (3, signal.SIGKILL)]


def test_reset_preserves_mounts_and_capture_files(tmp_path, monkeypatch):
    # Rewrite Python literals before run() shell-quotes the script.
    reset = iris_backend.RESET.replace("Path('/proc')", f"Path({str(tmp_path / 'proc')!r})")
    reset = reset.replace("os.listdir('/')", f"os.listdir({str(tmp_path)!r})")
    reset = reset.replace("roots.update('/' + name", f"roots.update({str(tmp_path)!r} + '/' + name")
    monkeypatch.setattr(iris_backend, "RESET", reset)

    class SandboxRpc(LocalRpc):
        def exec_in_container(self, request, timeout_ms):
            # Hide paths already under tmp_path (itself under /tmp on Linux) from the rewrite.
            root, marker = str(tmp_path), "\0tmp_path\0"
            command = [
                re.sub(
                    r"/(tmp|workspace|tests|app)(?=/|[\s'\"]|$)",
                    lambda m: str(tmp_path / m[1]),
                    part.replace(root, marker),
                ).replace(marker, root)
                for part in request.command
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            return SimpleNamespace(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr, error="")

    for name in ("tmp", "workspace", "tests", "app", "proc"):
        (tmp_path / name).mkdir()
    inodes = {name: (tmp_path / name).stat().st_ino for name in ("tmp", "workspace", "tests", "app")}
    runtime = tmp_path / "tmp/.shellbox-state"
    runtime.write_text("keep")
    machine, job = local_machine(tmp_path, rpc=SandboxRpc())
    machine.spec = replace(machine.spec, workdir="/app")

    async def scenario():
        try:
            await machine.reset()
            for name in inodes:
                (tmp_path / name / "residue").write_text("old candidate")
            (tmp_path / "new-file").write_text("old candidate")
            for _ in range(2):
                await machine.reset()
                assert (await machine.run(Command(("printf", "after reset")))).stdout == b"after reset"
                assert all((tmp_path / name).stat().st_ino == inode for name, inode in inodes.items())
                assert all(not (tmp_path / name / "residue").exists() for name in inodes)
                assert not (tmp_path / "new-file").exists()
                assert runtime.read_text() == "keep"
                assert not job.cancelled
        finally:
            await machine.close()

    asyncio.run(scenario())


# Pilots k/k2/l: old controllers wrap the server's exec failure in ValueError text.
@pytest.mark.parametrize(
    "message,kind,detail",
    [
        ("Websocket setup failed (status 0): handshake refused", ExecSetupFailed, None),
        ("'NoneType' object has no attribute 'decode'", ExecSetupFailed, None),
        ("Non-integer exec status: invalid literal for int()", RuntimeError, None),
        ("invalid literal for int() with base 10: malformed envelope", RuntimeError, None),
        ("unexpected adapter bug", RuntimeError, None),
        (
            "invalid literal for int(): error executing command in container: container not found",
            RuntimeError,
            None,
        ),
        # The controller truncates long exec errors mid-repr; the transport marker must still classify them.
        (
            "invalid literal for int() with base 10: 'Internal error occurred: error executing command in"
            ' container: failed to start exec "5281c76e": OCI runtime exec f',
            MachineUnavailable,
            "Internal error occurred: error executing command in container: failed to start exec"
            ' "5281c76e": OCI runtime exec f',
        ),
    ]
    + [
        (
            prefix + "invalid literal for int() with base 10: " + repr(detail),
            MachineUnavailable if detail.startswith("Internal error occurred") else RuntimeError,
            detail,
        )
        for prefix in ("", "Non-integer exec status: ")
        for detail in (
            'Internal error occurred: failed to start exec "dead"',
            "Internal error: container's runtime failed\nsecond line",
        )
    ],
)
def test_controller_wire_errors(tmp_path, message, kind, detail):
    for field in (("error", "stderr") if kind is MachineUnavailable else ("error",)):
        rpc = SimpleNamespace(
            exec_in_container=Mock(
                return_value=SimpleNamespace(
                    **{"exit_code": int(field == "stderr"), "stdout": "", "stderr": "", "error": "", field: message}
                )
            )
        )
        machine, _ = local_machine(tmp_path, rpc, LocalTask(TaskState.RUNNING))
        with pytest.raises(kind) as caught:
            machine._exec_sync(["true"])
        assert type(caught.value) is kind
        assert str(caught.value) == "Iris exec failed: " + (message if detail is None else detail)
        if kind is MachineUnavailable:
            assert str(caught.value.__cause__) == str(caught.value)


@pytest.mark.parametrize("script,count", [("true", 4), ("printf out", 5), ("printf out; printf err >&2", 6)])
def test_command_capture_exec_rpc_count(tmp_path: Path, script: str, count: int) -> None:
    class CountingRpc(LocalRpc):
        calls = 0

        def exec_in_container(self, request, timeout_ms):
            self.calls += 1
            return super().exec_in_container(request, timeout_ms)

    rpc = CountingRpc()
    machine, _ = local_machine(tmp_path, rpc)
    result = asyncio.run(machine.run(Command(("sh", "-c", script))))
    assert result.exit_code == 0
    assert rpc.calls == count


@pytest.mark.parametrize(
    "code,message,cancelled",
    [
        (
            Code.FAILED_PRECONDITION,
            "Cannot submit job: parent job /user/pilot has terminated (state=JOB_STATE_KILLED)",
            True,
        ),
        (Code.FAILED_PRECONDITION, "Cannot submit job: parent job /user/pilot is absent from the database", False),
        (Code.INTERNAL, "Cannot submit job: parent job /user/pilot has terminated (state=JOB_STATE_KILLED)", False),
    ],
)
# Pilot m-w4: only the parent-terminated submit response is cancellation.
def test_parent_termination_at_submit_is_typed_after_cleanup(monkeypatch, code, message, cancelled):
    error = ConnectError(code, message)
    cleaned = []
    client = RecordingClient()
    endpoint = LocalEndpoint()

    client.submit = Mock(side_effect=error)
    client.shutdown = lambda: cleaned.append("client")
    endpoint.close = lambda: cleaned.append("endpoint")
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: endpoint)
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: client)
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: LocalRpc())
    factory = IrisMachineFactory(controller_url="http://controller")
    with pytest.raises(MachineCancelled if cancelled else ConnectError) as caught:
        asyncio.run(factory.create(MachineSpec(source=RegistryImage("ubuntu:24.04"))))
    assert cleaned == ["client", "endpoint"]
    if cancelled:
        assert str(caught.value) == "parent job terminated"
        assert caught.value.__cause__ is error
    else:
        assert caught.value is error


# Pilot n-w5: existing submit/status retry must not acquire a second wait budget.
def test_sandbox_collision_retry_mints_a_fresh_id(monkeypatch):
    client, names = RecordingClient(), []

    def submit(**kwargs):
        names.append(kwargs["name"])
        raise ConnectError(Code.ALREADY_EXISTS, "Job already exists and is still running")

    client.submit = submit
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: LocalEndpoint())
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: client)
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: LocalRpc())
    factory = IrisMachineFactory(controller_url="http://controller")
    for _ in range(2):
        with pytest.raises(MachineUnavailable, match="fresh sandbox id"):
            asyncio.run(factory.create(MachineSpec(source=RegistryImage("ubuntu:24.04"))))
    assert len(set(names)) == 2


@pytest.mark.parametrize("exec_call", [False, True])
@pytest.mark.parametrize("attempts", [2, 240])
@pytest.mark.parametrize(
    "error",
    [
        ConnectError(Code.UNAVAILABLE, "Controller is shutting down"),
        ConnectionRefusedError("refused"),
        ConnectError(
            Code.INTERNAL,
            "RPC ExecInContainer: QueuePool limit of size 4 overflow 4 reached, connection timed out, timeout 30.00",
        ),
        ConnectError(Code.DEADLINE_EXCEEDED, "RPC timed out"),
        httpx.ReadTimeout("connection timed out"),
        httpx.RemoteProtocolError("connection closed"),
    ],
)
def test_controller_wait_budget(monkeypatch, exec_call, attempts, error):
    clock = [0.0]
    monkeypatch.setattr(iris_backend, "DEFAULT_RETRY_MAX_ATTEMPTS", attempts)
    monkeypatch.setattr(iris_backend.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(iris_backend.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(iris_backend, "DEFAULT_RETRY_MAX_ELAPSED", 0.003)
    monkeypatch.setattr(iris_backend, "EXEC_SHED_BACKOFF", ExponentialBackoff(initial=0.002, maximum=0.002, jitter=0))
    call = Mock(side_effect=error)
    if not exec_call and not iris_backend._controller_unavailable(error):
        with pytest.raises(type(error)) as caught:
            iris_backend._controller_call(call, exec_call=False)
        assert caught.value is error
        assert call.call_count == 1
        assert clock[0] == 0
        return
    with pytest.raises(MachineUnavailable) as caught:
        iris_backend._controller_call(call, exec_call=exec_call)
    assert caught.value.__cause__ is error
    assert call.call_count == (min(attempts, 3) if exec_call else 1)
    assert clock[0] == ((0.002 if attempts == 2 else 0.003) if exec_call else 0)
