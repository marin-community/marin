# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Iris backend with a local subprocess exec provider and a scripted controller."""

import asyncio
import json
import os
import pwd
import re
import shutil
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from errno import EACCES, EISDIR, ENAMETOOLONG, ENOENT, ENOTDIR
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
from iris.rpc.async_adapter import BoundedThreadExecutor
from rigging.timing import ExponentialBackoff
from shellbox.backends.iris import machine as iris_backend
from shellbox.backends.iris.machine import IrisMachine, IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import (
    Command,
    CommandOutcomeUnknown,
    ExecSetupFailed,
    GuestHelperFailed,
    GuestPathError,
    MachineCancelled,
    MachineSpec,
    MachineTerminated,
    MachineUnavailable,
    NetworkPolicy,
    UnsupportedMachineSpec,
)

# Linux MAX_ARG_STRLEN: the worker passes the exec command as argv to `docker exec` or `kubectl exec`.
LINUX_ARGUMENT_LIMIT_BYTES = 128 * 1024
# Guest images' /bin/sh is a POSIX shell such as dash; macOS's sh is bash, which tolerates more.
GUEST_SH = shutil.which("dash") or "sh"


class LocalRpc:
    def exec_in_container(self, request, timeout_ms):
        del timeout_ms
        if max(len(argument.encode()) + 1 for argument in request.command) > LINUX_ARGUMENT_LIMIT_BYTES:
            return SimpleNamespace(exit_code=0, stdout="", stderr="", error="[Errno 7] Argument list too long: 'docker'")
        command = [GUEST_SH if request.command[0] == "sh" else request.command[0], *request.command[1:]]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return SimpleNamespace(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr, error="")


class FailingRpc:
    def exec_in_container(self, request, timeout_ms):
        del request, timeout_ms
        return SimpleNamespace(exit_code=0, stdout="", stderr="", error="Task /user/shellbox/0 is not running")


class RefusingRpc(LocalRpc):
    """Fails the first ``refusals`` execs with ``code`` before running anything, then runs them locally."""

    def __init__(self, code: Code, refusals: int, cause: Exception | None = None, message: str = "refused"):
        self.code = code
        self.refusals = refusals
        self.cause = cause
        self.message = message
        self.scripts: list[str] = []

    def exec_in_container(self, request, timeout_ms):
        self.scripts.append(request.command[-1])
        if len(self.scripts) <= self.refusals:
            raise ConnectError(self.code, self.message) from self.cause
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


def test_paths_the_caller_chose_fail_as_os_errors_and_transport_failures_do_not(tmp_path: Path) -> None:
    async def scenario() -> None:
        machine, _ = local_machine(tmp_path)
        source = tmp_path / "input.txt"
        source.write_text("payload")
        (tmp_path / "directory").mkdir()
        locked = tmp_path / "locked"
        locked.mkdir()
        (locked / "secret.txt").write_text("hidden")
        unsearchable = tmp_path / "unsearchable"
        unsearchable.mkdir()
        (unsearchable / "file.txt").write_text("hidden")
        for path, mode in ((locked / "secret.txt", 0), (locked, 0o500), (unsearchable, 0o600)):
            path.chmod(mode)

        async def guest_errno(operation) -> int | None:
            with pytest.raises(GuestPathError) as raised:
                await operation
            return raised.value.errno

        try:
            assert await guest_errno(machine.download(str(tmp_path / "missing.txt"), tmp_path / "out.txt")) == ENOENT
            assert await guest_errno(machine.upload(source, str(tmp_path / "directory"))) == EISDIR
            assert await guest_errno(machine.upload(source, str(tmp_path / "input.txt" / "child"))) == ENOTDIR
            assert (
                await guest_errno(machine.download(str(tmp_path / "input.txt" / "child"), tmp_path / "out")) == ENOTDIR
            )
            if os.geteuid() != 0:
                assert await guest_errno(machine.upload(source, str(locked / "new" / "file.txt"))) == EACCES
                assert await guest_errno(machine.download(str(locked / "secret.txt"), tmp_path / "out.txt")) == EACCES
                for operation in (
                    machine.download(str(unsearchable / "file.txt"), tmp_path / "out.txt"),
                    machine.upload(source, str(unsearchable / "file.txt")),
                ):
                    assert await guest_errno(operation) == EACCES
        finally:
            for path in (locked, unsearchable):
                path.chmod(0o700)
            (locked / "secret.txt").chmod(0o600)
            await machine.close()
        unreachable, _ = local_machine(tmp_path, FailingRpc())
        try:
            with pytest.raises(RuntimeError) as failure:
                await unreachable.download(str(source), tmp_path / "out.txt")
        finally:
            await unreachable.close()
        assert not isinstance(failure.value, OSError)

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


def test_exec_error_on_a_running_sandbox_is_an_unknown_outcome(tmp_path: Path) -> None:
    rpc = SimpleNamespace(
        exec_in_container=lambda request, timeout_ms: SimpleNamespace(
            exit_code=0, stdout="", stderr="", error="unexpected adapter bug"
        )
    )
    machine, _ = local_machine(tmp_path, rpc, LocalTask(TaskState.RUNNING))

    with pytest.raises(CommandOutcomeUnknown, match="unexpected adapter bug"):
        asyncio.run(machine.run(Command(("true",))))


class FailingCleanupRpc(LocalRpc):
    def exec_in_container(self, request, timeout_ms):
        if request.command[-1].startswith("rm -f /tmp/.shellbox-"):
            return SimpleNamespace(exit_code=1, stdout="", stderr="rm: Operation not permitted", error="")
        return super().exec_in_container(request, timeout_ms)


def test_failed_capture_cleanup_is_logged_and_keeps_the_result(tmp_path: Path, caplog) -> None:
    machine, _ = local_machine(tmp_path, FailingCleanupRpc())

    result = asyncio.run(machine.run(Command(("echo", "kept"))))

    assert (result.exit_code, result.stdout) == (0, b"kept\n")
    # The guest's own diagnostic reaches the operator; the result is unchanged.
    assert any(
        record.levelname == "WARNING" and "rm: Operation not permitted" in record.getMessage()
        for record in caplog.records
    )


def test_dead_container_exec_error_overrides_stale_running_status(tmp_path: Path) -> None:
    machine, _ = local_machine(tmp_path, FailingRpc(), LocalTask(TaskState.RUNNING))

    with pytest.raises(MachineTerminated, match="Iris exec failed"):
        asyncio.run(machine.run(Command(("true",))))


def pool_refusal() -> ConnectError:
    """The refusal lib/iris's bounded exec pool raises when it is full, from the real executor."""
    executor = BoundedThreadExecutor(max_workers=1, max_pending=0, thread_name_prefix="test-exec")
    release = threading.Event()

    async def refused() -> ConnectError:
        busy = asyncio.ensure_future(executor.run(release.wait))
        await asyncio.sleep(0)
        try:
            await executor.run(lambda: None)
        except ConnectError as error:
            return error
        finally:
            release.set()
            await busy
            executor.shutdown()
        raise AssertionError("A full exec pool admitted another call")

    return asyncio.run(refused())


def test_exec_refused_by_a_full_controller_pool_is_retried(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(iris_backend, "EXEC_SHED_BACKOFF", ExponentialBackoff(initial=0.001, maximum=0.001))
    refusal = pool_refusal()
    rpc = RefusingRpc(refusal.code, refusals=2, message=refusal.message)
    machine, _ = local_machine(tmp_path, rpc)

    result = asyncio.run(machine.run(Command(("echo", "ran"))))

    assert result.stdout == b"ran\n"
    assert rpc.sent("echo ran") == 3


# Review 2026-11-20 (follow-up 3): the generated client raises RESOURCE_EXHAUSTED for a reply above its read limit,
# after the exec ran, and that reran the command in place.
def test_resource_exhausted_other_than_the_pool_refusal_is_not_retried(tmp_path: Path) -> None:
    rpc = RefusingRpc(Code.RESOURCE_EXHAUSTED, refusals=1, message="message is larger than configured max 24")
    machine, _ = local_machine(tmp_path, rpc)

    with pytest.raises(CommandOutcomeUnknown, match="larger than configured max"):
        asyncio.run(machine.run(Command(("echo", "once"))))
    assert rpc.sent("echo once") == 1


@pytest.mark.parametrize("connected", [True, False])
def test_exec_failing_after_the_controller_accepted_it_is_not_repeated(tmp_path: Path, connected: bool) -> None:
    rpc = RefusingRpc(Code.UNAVAILABLE, refusals=1, cause=None if connected else httpx.ConnectError("refused"))
    machine, _ = local_machine(tmp_path, rpc)

    with pytest.raises(MachineUnavailable) as raised:
        asyncio.run(machine.run(Command(("echo", "once"))))
    # Only a request that never opened a connection is known not to have run.
    assert isinstance(raised.value, CommandOutcomeUnknown) is connected
    assert rpc.sent("echo once") == 1


class ClosingRpc(RefusingRpc):
    """Refuses every exec for a full pool; during the first refusal another thread closes the machine."""

    def __init__(self):
        refusal = pool_refusal()
        super().__init__(refusal.code, refusals=1_000, message=refusal.message)
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


def test_close_returns_only_after_admitted_execs_end(tmp_path: Path) -> None:
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
        (TaskState.PREEMPTED, "WorkloadEvictedDueToPreempted", MachineTerminated),
        (TaskState.KILLED, "Parent job terminated", MachineCancelled),
        (TaskState.PENDING, "Controller is shutting down", MachineUnavailable),
    ],
)
def test_factory_cancels_the_job_when_the_task_fails(monkeypatch, tmp_path, state, reason, error_type):
    task = LocalTask(state)
    unreachable = state is TaskState.PENDING
    task.status = (
        Mock(side_effect=ConnectError(Code.UNAVAILABLE, reason))
        if unreachable
        else Mock(return_value=SimpleNamespace(state=state, error_message=reason))
    )
    job = LocalJob(task)
    # A cancel that fails while the controller is down must not replace the startup error.
    job.cancel = Mock(wraps=job.cancel, side_effect=RuntimeError("cancel unavailable") if unreachable else None)
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: LocalEndpoint())
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: TaskClient(job))
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: LocalRpc())
    factory = IrisMachineFactory(controller_url="http://controller")
    spec = MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(tmp_path / "workspace"))

    with pytest.raises(error_type, match=reason) as caught:
        asyncio.run(factory.create(spec))
    assert job.cancel.called
    if unreachable:
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


@pytest.mark.parametrize("memory_mb", [None, 4096])
def test_oom_termination_reports_effective_memory_request(tmp_path, memory_mb):
    task = LocalTask(TaskState.FAILED)
    task.status = lambda: SimpleNamespace(state=TaskState.FAILED, error_message="OOMKilled")
    machine, _ = local_machine(tmp_path, FailingRpc(), task)
    machine.spec = replace(machine.spec, memory_mb=memory_mb)
    with pytest.raises(MachineTerminated, match=f"memory request: {memory_mb or 2048} MiB"):
        asyncio.run(machine.run(Command(("true",))))


def test_command_exec_carries_the_command_deadline(tmp_path):
    machine, _ = local_machine(tmp_path)
    requests = []

    class Rpc(LocalRpc):
        def exec_in_container(self, request, timeout_ms):
            requests.append((request.command[-1], request.timeout_seconds, timeout_ms))
            return super().exec_in_container(request, timeout_ms)

    machine.rpc = Rpc()
    assert asyncio.run(machine.run(Command(("printf", "ok"), timeout=180.1))).stdout == b"ok"
    command = [request for request in requests if "printf ok" in request[0]]
    assert [request[1:] for request in command] == [(181, (181 + iris_backend.RPC_PADDING_SECONDS) * 1000)]
    asyncio.run(machine.close())


@pytest.mark.requires_linux_procfs
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
            child = subprocess.Popen(("sleep", "60"))
            proc_stat = tmp_path / "proc" / str(child.pid) / "stat"
            try:
                proc_stat.parent.mkdir()
                proc_stat.write_text(Path(f"/proc/{child.pid}/stat").read_text())
                await machine.reset()
                assert child.wait(timeout=5) == -signal.SIGKILL
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
                proc_stat.unlink(missing_ok=True)
                proc_stat.parent.rmdir()
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


# The controller reports an exec without an integer status as exit 1 with the client's ValueError text,
# and may truncate it mid-quote.
@pytest.mark.parametrize(
    "field,message,kind,detail",
    [
        ("error", "'NoneType' object has no attribute 'decode'", ExecSetupFailed, None),
        ("error", "unexpected adapter bug", RuntimeError, None),
        ("stderr", "invalid literal for int() with base 10: malformed envelope", RuntimeError, "malformed envelope"),
        (
            "stderr",
            "invalid literal for int() with base 10: 'Internal error occurred: error executing command in"
            ' container: failed to start exec "5281c76e": OCI runtime exec f',
            MachineUnavailable,
            'Internal error occurred: error executing command in container: failed to start exec "5281c76e": OCI'
            " runtime exec f",
        ),
        (
            "stderr",
            "invalid literal for int() with base 10: " + repr("Internal error: container's runtime failed\nsecond line"),
            RuntimeError,
            "Internal error: container's runtime failed\nsecond line",
        ),
    ],
)
def test_controller_wire_errors(tmp_path, field, message, kind, detail):
    response = {"exit_code": int(field == "stderr"), "stdout": "", "stderr": "", "error": "", field: message}
    rpc = SimpleNamespace(exec_in_container=Mock(return_value=SimpleNamespace(**response)))
    machine, _ = local_machine(tmp_path, rpc, LocalTask(TaskState.RUNNING))
    with pytest.raises(kind) as caught:
        machine._exec_sync(["true"])
    assert type(caught.value) is kind
    assert str(caught.value) == "Iris exec failed: " + (message if detail is None else detail)


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


class VanishingRpc(LocalRpc):
    """Removes ``path`` just before the first chunk read of it, after its size was measured."""

    def __init__(self, path: Path):
        self.path = path

    def exec_in_container(self, request, timeout_ms):
        if request.command[-1].startswith("tail -c +1 ") and str(self.path) in request.command[-1]:
            self.path.unlink(missing_ok=True)
        return super().exec_in_container(request, timeout_ms)


# Review 2026-11-20: a file removed between its size probe and its first chunk downloaded as zero bytes.
def test_download_of_a_file_that_vanishes_mid_read_fails_while_an_empty_file_succeeds(tmp_path: Path) -> None:
    vanishing, empty = tmp_path / "vanishing.txt", tmp_path / "empty.txt"
    vanishing.write_bytes(b"sixbyt")
    empty.write_bytes(b"")
    machine, _ = local_machine(tmp_path, VanishingRpc(vanishing))

    async def scenario() -> None:
        try:
            with pytest.raises(GuestPathError) as raised:
                await machine.download(str(vanishing), tmp_path / "vanishing.out")
            assert raised.value.errno == ENOENT
            await machine.download(str(empty), tmp_path / "empty.out")
        finally:
            await machine.close()

    asyncio.run(scenario())
    assert not (tmp_path / "vanishing.out").exists()
    assert (tmp_path / "empty.out").read_bytes() == b""


class CountingRpc(LocalRpc):
    """Runs each exec locally after ``prelude`` and records its script."""

    def __init__(self, prelude: str = ""):
        self.prelude = prelude
        self.scripts: list[str] = []

    def exec_in_container(self, request, timeout_ms):
        *shell, script = request.command
        self.scripts.append(script)
        return super().exec_in_container(SimpleNamespace(command=[*shell, self.prelude + script]), timeout_ms)


# Review 2026-11-20 (follow-up 1): a directory read was archived over five RPCs; writes through a link cycle, a
# dangling link with a missing target parent, or an overlong name raised runtime errors; a failing `dirname`
# looped forever. Each path fault is classified in the one exec that attempted the operation.
@pytest.mark.parametrize(
    "operation,name,code",
    [
        ("read", "directory", EISDIR),
        ("read", "loop", ENOENT),
        ("read", "absent/deeper/file", ENOENT),
        ("read", "n" * 256, ENAMETOOLONG),
        ("write", "loop", ENOENT),
        ("write", "dangling", ENOENT),
        ("write", "parent-link/file", ENOENT),
        ("write", "n" * 256, ENAMETOOLONG),
        ("write", "directory", EISDIR),
        ("write", "regular/child", ENOTDIR),
    ],
)
def test_guest_path_faults_are_classified_in_one_exec(tmp_path: Path, operation: str, name: str, code: int) -> None:
    (tmp_path / "directory").mkdir()
    (tmp_path / "regular").write_text("file")
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    (tmp_path / "dangling").symlink_to(tmp_path / "absent" / "file")
    (tmp_path / "parent-link").symlink_to(tmp_path / "absent")
    source = tmp_path / "input.txt"
    source.write_text("payload")
    rpc = CountingRpc(prelude="dirname() { return 1; }; ")
    machine, _ = local_machine(tmp_path, rpc)

    async def scenario() -> None:
        try:
            with pytest.raises(GuestPathError) as raised:
                if operation == "read":
                    await machine.download_file(str(tmp_path / name), tmp_path / "out")
                else:
                    await machine.upload(source, str(tmp_path / name))
            assert (raised.value.errno, raised.value.filename) == (code, str(tmp_path / name))
        finally:
            await machine.close()

    asyncio.run(scenario())
    assert len(rpc.scripts) == 1


def test_directory_download_stays_recursive_and_helper_failures_are_typed(tmp_path: Path) -> None:
    (tmp_path / "tree" / "nested").mkdir(parents=True)
    (tmp_path / "tree" / "nested" / "leaf.txt").write_text("leaf")
    (tmp_path / "present.txt").write_text("present")

    async def scenario() -> None:
        machine, _ = local_machine(tmp_path)
        broken, _ = local_machine(tmp_path, CountingRpc(prelude="wc() { echo 'wc: not found' >&2; return 127; }; "))
        try:
            await machine.download(str(tmp_path / "tree"), tmp_path / "copy")
            assert (tmp_path / "copy" / "nested" / "leaf.txt").read_text() == "leaf"
            with pytest.raises(GuestHelperFailed, match="wc: not found"):
                await broken.download(str(tmp_path / "present.txt"), tmp_path / "out")
        finally:
            await machine.close()
            await broken.close()

    asyncio.run(scenario())
