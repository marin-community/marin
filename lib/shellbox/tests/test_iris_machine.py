# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Iris backend with a local subprocess exec provider and a scripted controller."""

import asyncio
import json
import os
import pwd
import subprocess
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("iris")

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.client.workload_codec import task_status_from_proto
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from iris.resources.state import TaskState
from iris.rpc import job_pb2
from rigging.timing import ExponentialBackoff
from shellbox.backends.iris import machine as iris_backend
from shellbox.backends.iris.machine import IrisMachine, IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, MachineTerminated, NetworkPolicy, UnsupportedMachineSpec
from shellbox.service import ReadinessProbe, ServiceEndpoint, ServiceGroupSpec, ServiceSpec

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
        self.submitted: dict = {}

    def submit(self, **kwargs):
        self.submitted = kwargs
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


def test_directory_upload_does_not_restore_host_ownership(tmp_path, monkeypatch):
    original_metadata = tarfile.TarFile.gettarinfo

    def foreign_owner(archive, *args, **kwargs):
        metadata = original_metadata(archive, *args, **kwargs)
        metadata.uid = os.getuid() + 1
        metadata.gid = os.getgid() + 1
        metadata.uname = metadata.gname = ""
        return metadata

    # Force the ownership-restoration behavior GNU tar uses for root, while
    # exercising real extraction as an unprivileged test user.
    monkeypatch.setenv("TAR_OPTIONS", "--same-owner")
    monkeypatch.setattr(tarfile.TarFile, "gettarinfo", foreign_owner)
    source = tmp_path / "input"
    source.mkdir()
    (source / "solution.sh").write_text("#!/bin/sh\nexit 0\n")
    (source / "solution.sh").chmod(0o755)
    target = tmp_path / "remote"

    async def run():
        machine, _ = local_machine(tmp_path)
        try:
            await machine.upload(source, str(target))
            assert (target / "solution.sh").read_bytes() == (source / "solution.sh").read_bytes()
            assert (target / "solution.sh").stat().st_uid == os.getuid()
            assert (target / "solution.sh").stat().st_mode & 0o777 == 0o755
        finally:
            await machine.close()

    asyncio.run(run())


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
@pytest.mark.parametrize("grouped", [False, True])
def test_factory_waits_for_a_pending_task_without_holding_an_executor_thread(
    monkeypatch, tmp_path, task_missing, grouped
):
    task = LocalTask(TaskState.PENDING)
    current_status = task.status
    job = LocalJob(task)
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: LocalEndpoint())
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: TaskClient(job))
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: GroupRpc())
    factory = IrisMachineFactory(controller_url="http://controller")
    workdir = tmp_path / "workspace"
    spec = MachineSpec(source=RegistryImage("example.test/prepared@sha256:" + "a" * 64), workdir=str(workdir))
    existing, _ = local_machine(tmp_path)

    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        pending = asyncio.Event()

        def status():
            nonlocal task_missing
            loop.call_soon_threadsafe(pending.set)
            if task_missing:
                task_missing = False
                raise ConnectError(Code.NOT_FOUND, "Task is not registered yet")
            return current_status()

        task.status = status
        creation = asyncio.create_task(
            factory.create_group(ServiceGroupSpec(spec, (ServiceSpec("runtime", spec),)))
            if grouped
            else factory.create(spec)
        )
        await asyncio.wait_for(pending.wait(), timeout=5)
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


@pytest.mark.parametrize("grouped", [False, True])
def test_cancelled_factory_wait_reclaims_the_submitted_job(monkeypatch, tmp_path, grouped):
    task = LocalTask(TaskState.PENDING)
    job = LocalJob(task)
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: LocalEndpoint())
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: TaskClient(job))
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: GroupRpc())
    factory = IrisMachineFactory(controller_url="http://controller")
    spec = MachineSpec(source=RegistryImage("example.test/prepared@sha256:" + "a" * 64), workdir=str(tmp_path))

    async def scenario():
        loop = asyncio.get_running_loop()
        pending = asyncio.Event()
        current_status = task.status

        def status():
            loop.call_soon_threadsafe(pending.set)
            return current_status()

        task.status = status
        creation = asyncio.create_task(
            factory.create_group(ServiceGroupSpec(spec, (ServiceSpec("runtime", spec),)))
            if grouped
            else factory.create(spec)
        )
        await asyncio.wait_for(pending.wait(), timeout=5)
        creation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await creation
        assert job.cancelled

    asyncio.run(scenario())


def test_nested_factory_uses_task_identity_instead_of_host_credentials(monkeypatch):
    info = JobInfo(
        task_id=JobName.from_wire("/user/driver/0"),
        controller_address="http://controller",
        task_token="task-scoped-token",
    )
    client = RecordingClient()
    attached_headers: dict[str, str] = {}

    def remote(*args, credentials, **kwargs):
        attached_headers.update(credentials.headers())
        return client

    monkeypatch.setattr(iris_backend, "get_job_info", lambda: info)
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: pytest.fail("Read host cluster credentials"))
    monkeypatch.setattr(iris_backend.IrisClient, "remote", remote)
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: LocalRpc())
    factory = IrisMachineFactory(controller_url="http://controller")

    with pytest.raises(SubmissionRecorded):
        asyncio.run(factory.create(MachineSpec(source=RegistryImage("ubuntu:24.04"))))

    assert attached_headers == {"authorization": "Bearer task-scoped-token"}
    assert client.submitted["container_profile"] == job_pb2.CONTAINER_PROFILE_SANDBOX


class GroupRpc(LocalRpc):
    def __init__(self):
        self.targets: set[str] = set()

    def list_backends(self, request):
        return SimpleNamespace(backends=[SimpleNamespace(capabilities=["sandbox-service-groups"])])

    def exec_in_container(self, request, timeout_ms):
        self.targets.add(request.container_name)
        return super().exec_in_container(request, timeout_ms)


def test_private_group_routes_service_exec_reserves_resources_and_closes_together(monkeypatch, tmp_path):
    job = LocalJob(LocalTask(TaskState.RUNNING))
    client = TaskClient(job)
    rpc = GroupRpc()
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: LocalEndpoint())
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: client)
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: rpc)
    image = RegistryImage("example.test/prepared@sha256:" + "a" * 64)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    probe = Command(
        ("sh", "-c", "n=$(cat attempts 2>/dev/null || echo 0); n=$((n+1)); echo $n > attempts; test $n -ge 2")
    )
    spec = ServiceGroupSpec(
        task=MachineSpec(image, workdir=str(tmp_path / "task"), cpus=2, memory_mb=256),
        services=(
            ServiceSpec(
                name="runtime",
                machine=MachineSpec(image, workdir=str(runtime), cpus=1, memory_mb=128),
                endpoints=(ServiceEndpoint("http", 8000),),
                readiness=ReadinessProbe(probe, timeout=5, interval=0.001, attempts=3),
            ),
        ),
    )

    async def run():
        group = await IrisMachineFactory(controller_url="http://controller").create_group(spec)
        assert group.endpoint("runtime", "http") == "runtime:8000"
        await group.wait_ready()
        assert (runtime / "attempts").read_text().strip() == "2"
        assert (await group.services["runtime"].run(Command(("printf", "runtime")))).stdout == b"runtime"
        await group.services["runtime"].close()
        assert job.cancelled
        with pytest.raises(RuntimeError, match="closed"):
            await group.task.run(Command(("true",)))
        await group.close()

    asyncio.run(run())
    assert rpc.targets == {"", "runtime"}
    assert client.submitted["resources"].cpu == 3
    assert client.submitted["resources"].memory == 384 * 1024 * 1024
    environment = client.submitted["environment"].to_sandbox_proto()
    assert not environment.secrets
    assert environment.sandbox_services[0].name == "runtime"
    assert not environment.sandbox_services[0].HasField("entrypoint")
    assert not environment.sandbox_services[0].HasField("arguments")


def test_readiness_timeout_retries_without_stopping_runtime_or_leaving_probe_children(monkeypatch, tmp_path):
    job = LocalJob(LocalTask(TaskState.RUNNING))
    client = TaskClient(job)
    rpc = GroupRpc()
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: LocalEndpoint())
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: client)
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: rpc)
    image = RegistryImage("example.test/prepared@sha256:" + "a" * 64)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    probe_program = """\
import fcntl
import os
from pathlib import Path
import subprocess
import sys

assert os.getcwd() == os.environ['EXPECTED_CWD']
assert sys.stdin.buffer.read() == b'probe input'
counter = Path('attempts')
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt))
if attempt == 1:
    subprocess.run([sys.executable, '-c',
        "import fcntl,time; lock=open('probe.lock','w'); "
        "fcntl.flock(lock,fcntl.LOCK_EX); time.sleep(3)"], check=True)
else:
    assert Path('probe.lock').exists()
    with open('probe.lock') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Path('healthy').write_text(str(attempt))
"""
    probe = Command(
        ("python3", "-c", probe_program),
        cwd=str(runtime),
        env={"EXPECTED_CWD": str(runtime)},
        stdin=b"probe input",
        timeout=1,
        output_limit_bytes=8,
        user=str(os.getuid()),
    )
    spec = ServiceGroupSpec(
        task=MachineSpec(image, workdir=str(tmp_path / "task")),
        services=(
            ServiceSpec(
                name="runtime",
                machine=MachineSpec(image, workdir=str(tmp_path)),
                readiness=ReadinessProbe(probe, timeout=8, interval=0.001, attempts=2),
            ),
        ),
    )

    async def run():
        group = await IrisMachineFactory(controller_url="http://controller").create_group(spec)
        try:
            await group.wait_ready()
            assert (runtime / "healthy").read_text() == "2"
            assert not job.cancelled
            assert (await group.services["runtime"].run(Command(("printf", "running")))).stdout == b"running"
        finally:
            await group.close()
        assert job.cancelled

    asyncio.run(run())


def test_factory_cancels_the_job_when_the_task_fails(monkeypatch, tmp_path):
    job = LocalJob(LocalTask(TaskState.FAILED))
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: LocalEndpoint())
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: TaskClient(job))
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: LocalRpc())
    factory = IrisMachineFactory(controller_url="http://controller")
    spec = MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(tmp_path / "workspace"))

    with pytest.raises(RuntimeError, match="container exited"):
        asyncio.run(factory.create(spec))
    assert job.cancelled


@pytest.mark.parametrize("advertise_support", [True, False])
def test_group_failure_cannot_leave_running_members(monkeypatch, tmp_path, advertise_support):
    job = LocalJob(LocalTask(TaskState.RUNNING))
    client = TaskClient(job)
    rpc = GroupRpc()
    if not advertise_support:
        monkeypatch.setattr(rpc, "list_backends", lambda _: SimpleNamespace(backends=[]))
    monkeypatch.setattr(iris_backend, "connect_controller", lambda **_: LocalEndpoint())
    monkeypatch.setattr(iris_backend.IrisClient, "remote", lambda *_, **__: client)
    monkeypatch.setattr(iris_backend, "ControllerServiceClientSync", lambda **_: rpc)
    image = RegistryImage("example.test/prepared@sha256:" + "a" * 64)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    spec = ServiceGroupSpec(
        task=MachineSpec(image, workdir=str(tmp_path / "task")),
        services=(
            ServiceSpec(
                name="runtime",
                machine=MachineSpec(image, workdir=str(runtime)),
                readiness=ReadinessProbe(
                    Command(("sh", "-c", "echo unhealthy >&2; exit 1")), timeout=5, interval=0.001, attempts=2
                ),
            ),
        ),
    )

    async def run():
        factory = IrisMachineFactory(controller_url="http://controller")
        if not advertise_support:
            with pytest.raises(UnsupportedMachineSpec):
                await factory.create_group(spec)
            assert not client.submitted
            return
        group = await factory.create_group(spec)
        with pytest.raises(RuntimeError):
            await group.wait_ready()
        assert job.cancelled
        with pytest.raises(RuntimeError):
            await group.services["runtime"].run(Command(("true",)))
        await asyncio.gather(group.close(), group.close())

    asyncio.run(run())


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
