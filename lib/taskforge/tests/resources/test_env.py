# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import os
import shutil
import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import override

import pytest
from lib.taskforge.tests.resource_doubles import FakeCommand, FakeEnv
from lib.taskforge.tests.resources.iris_fakes import LocalController, local_iris_machine
from rolloutengine.test_support import machine_runtime
from shellbox.backends.iris.machine import IrisMachine
from shellbox.machine import (
    Command,
    CommandOutcomeUnknown,
    ExecSetupFailed,
    ExitReason,
    GuestPathError,
    Machine,
    MachineCancelled,
    MachineOutputLimitExceeded,
    MachineResourceExhausted,
    MachineTerminated,
    MachineUnavailable,
    PackageSpec,
    Result,
)
from shellsim import SimulationError
from taskcompendium.models import CommandSemantics, EnvironmentRequirements

from taskforge.effects.events import Event, Note, Retrying
from taskforge.effects.issues import Hold, OutputBudgetExhausted, Restart, Transient
from taskforge.resources import env as env_module
from taskforge.resources.env import (
    CommandResult,
    Env,
    ReadFile,
    RunCommand,
    ShellboxEnv,
    ShellboxEnvFactory,
    WriteFile,
    read_file,
    run_command,
    write_file,
)
from taskforge.resources.sandbox.factories import shellsim_factory

COMMAND = "printf changed > /workspace/output; printf hello; printf err >&2; exit 7"
OUTPUT = PurePosixPath("/workspace/output")
VALUE = CommandResult(7, "hello", "err", 0.1, False)


class MemoryMachine(Machine):
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.closed = False
        self.failure: Exception | None = None
        self.calls = 0

    @override
    async def run(self, command: Command) -> Result:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        if self.closed:
            raise MachineTerminated("closed")
        assert command.argv == ("sh", "-c", COMMAND)
        self.files[str(OUTPUT)] = b"changed"
        return Result(7, b"hello", b"err", False, False, ExitReason.EXITED)

    @override
    async def upload(self, source: Path, target: str) -> None:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        self.files[target] = source.read_bytes()

    @override
    async def download(self, source: str, target: Path) -> None:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        target.write_bytes(self.files[source])

    @override
    async def close(self) -> None:
        self.closed = True


class Notes:
    def __init__(self) -> None:
        self.events: list[Note] = []

    def __call__(self, event: Event[None]) -> None:
        assert isinstance(event, Note)
        self.events.append(event)


@pytest.fixture(
    params=[
        "fake",
        "shellbox-memory",
        "shellsim",
    ]
)
async def create_env(request: pytest.FixtureRequest) -> AsyncIterator[Callable[[], Awaitable[Env]]]:
    envs: list[Env] = []

    async def create() -> Env:
        match request.param:
            case "fake":
                env = FakeEnv("test-image", {COMMAND: FakeCommand(VALUE, {OUTPUT: b"changed"})})
            case "shellbox-memory":
                machine = MemoryMachine()
                owned = AsyncExitStack()
                owned.push_async_callback(machine.close)
                env = ShellboxEnv(machine, "test-image", owned)
            case _:
                try:
                    factory = shellsim_factory()
                except ImportError:
                    pytest.skip("Optional sandbox dependency is not installed")
                semantics = CommandSemantics.SHELL_SIMULATOR
                requirements = EnvironmentRequirements(command_semantics=semantics, package_spec=PackageSpec())
                env = await ShellboxEnvFactory(factory, machine_runtime(backend=request.param), 5).create(requirements)
        envs.append(env)
        return env

    yield create
    for env in envs:
        await env.close()


async def test_environment_writes_binary_files_and_fingerprints_command_history(
    create_env: Callable[[], Awaitable[Env]],
) -> None:
    original = await create_env()
    initial = original.fingerprint()
    path = PurePosixPath("/workspace/binary")
    data = b"\x00\xff\nquoted ' bytes"
    await original.write(path, data)
    after_write = original.fingerprint()
    assert after_write != initial
    assert await original.read(path) == data
    assert original.fingerprint() == after_write
    command = RunCommand(COMMAND, 5)
    result = await original.run(command.cmd, timeout=command.timeout)
    assert (result.exit_code, result.stdout, result.stderr) == (7, "hello", "err")
    assert await original.read(OUTPUT) == b"changed"
    assert original.fingerprint() != after_write
    await original.close()
    with pytest.raises(Restart):
        await original.run(COMMAND, timeout=5)


@pytest.mark.parametrize(
    "failure,decision",
    [
        (MachineTerminated("gone"), Restart),
        (MachineOutputLimitExceeded("ShellSim terminated: output_limit_exceeded"), OutputBudgetExhausted),
        (MachineCancelled("parent job terminated"), Transient),
        (MachineResourceExhausted("Iris sandbox OOMKilled"), Hold),
        (MachineTerminated("Iris sandbox is preempted: WorkloadEvictedDueToPreempted"), Restart),
        (MachineTerminated("Machine is closed"), Restart),
        (MachineResourceExhausted("ShellSim terminated: cpu_exhausted"), Hold),
        (TimeoutError("busy"), Transient),
        (MachineUnavailable("Controller is shutting down"), Transient),
        (CommandOutcomeUnknown("reply lost"), Restart),
    ],
)
async def test_shellbox_failure_preserves_unresolved_history(
    failure: RuntimeError | TimeoutError, decision: type[Exception]
) -> None:
    machine = MemoryMachine()
    machine.failure = failure
    env = ShellboxEnv(machine, "image", AsyncExitStack())
    before = env.fingerprint()
    with pytest.raises(decision):
        await env.run(COMMAND, timeout=5)
    assert env.fingerprint() == before


async def test_shellbox_unexpected_runtime_error_propagates_without_reclassification() -> None:
    machine = MemoryMachine()
    bug = RuntimeError("unexpected adapter bug")
    machine.failure = bug
    env = ShellboxEnv(machine, "image", AsyncExitStack())
    with pytest.raises(RuntimeError) as caught:
        await env.run(COMMAND, timeout=5)
    assert caught.value is bug


@pytest.mark.parametrize("operation", ["read", "write", "run"])
@pytest.mark.parametrize(
    "failure",
    [
        SimulationError("No such file or directory: /workspace/output"),
        GuestPathError(errno.ENOENT, "No such file", str(OUTPUT)),
    ],
)
async def test_missing_file_is_tool_feedback_without_retry(operation: str, failure: Exception) -> None:
    machine = MemoryMachine()
    machine.failure = failure
    env = ShellboxEnv(machine, "image", AsyncExitStack())
    before = env.fingerprint()
    with pytest.raises(FileNotFoundError) as caught:
        if operation == "read":
            await read_file(ReadFile(OUTPUT), env, notify=Notes())
        elif operation == "write":
            await write_file(WriteFile(OUTPUT, b"data"), env, notify=Notes())
        else:
            await run_command(RunCommand(COMMAND, 5), env, notify=Notes())
    assert str(OUTPUT) in str(caught.value)
    if operation != "run":
        assert caught.value.filename == str(OUTPUT)
    assert machine.calls == 1
    assert env.fingerprint() == before


async def test_cpu_exhaustion_holds_with_reason_without_retry() -> None:
    machine = MemoryMachine()
    machine.failure = MachineResourceExhausted("ShellSim terminated: cpu_exhausted")
    env = ShellboxEnv(machine, "image", AsyncExitStack())
    with pytest.raises(Hold, match="cpu_exhausted"):
        await run_command(RunCommand(COMMAND, 5), env, notify=Notes())
    assert machine.calls == 1


@pytest.mark.parametrize("workdir", ["/workspace", "/custom/work"])
async def test_relative_paths_use_environment_workdir(workdir: str) -> None:
    machine = MemoryMachine()
    env = ShellboxEnv(machine, "image", AsyncExitStack(), workdir)
    await env.write(PurePosixPath("release_notes.py"), b"notes")
    assert machine.files == {f"{workdir}/release_notes.py": b"notes"}
    assert await env.read(PurePosixPath("release_notes.py")) == b"notes"
    await env.write(PurePosixPath("/absolute.py"), b"absolute")
    assert machine.files["/absolute.py"] == b"absolute"


async def test_websocket_setup_failure_retries_only_three_times() -> None:
    machine = MemoryMachine()
    machine.failure = ExecSetupFailed("handshake refused")
    env = ShellboxEnv(machine, "image", AsyncExitStack())
    events: list[Event[None]] = []
    with pytest.raises(Restart, match="exhausted its retry budget"):
        await run_command(RunCommand(COMMAND, 5), env, notify=events.append)
    assert machine.calls == 3
    assert len(events) == 2 and all(isinstance(event, Retrying) for event in events)


@pytest.mark.parametrize("failure", [ExecSetupFailed("handshake refused"), MachineCancelled("parent job terminated")])
async def test_machine_startup_expected_failure_is_transient(
    monkeypatch: pytest.MonkeyPatch, failure: RuntimeError
) -> None:

    @asynccontextmanager
    async def failed_start(*args: object, **kwargs: object) -> AsyncIterator[Machine]:
        raise failure
        yield MemoryMachine()

    monkeypatch.setattr(env_module, "prepare_machine", failed_start)
    factory = ShellboxEnvFactory(shellsim_factory(), machine_runtime(backend="shellsim"), 5)
    with pytest.raises(Transient) as caught:
        await factory.create(EnvironmentRequirements(command_semantics=CommandSemantics.SHELL_SIMULATOR))
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    "failure",
    [
        SimulationError("Is a directory: /app"),
        GuestPathError(errno.EISDIR, "Is a directory", "/workspace/app"),
        GuestPathError(errno.EISDIR, "Is a directory", "/tmp/download/file"),
    ],
)
async def test_directory_read_names_requested_path_without_retry(failure: Exception) -> None:
    machine = MemoryMachine()
    machine.failure = failure
    env = ShellboxEnv(machine, "image", AsyncExitStack())
    with pytest.raises(IsADirectoryError) as caught:
        await read_file(ReadFile(PurePosixPath("/app")), env, notify=Notes())
    assert caught.value.filename == "/app"
    assert machine.calls == 1


async def test_noninteger_exec_status_requires_new_machine_without_retry() -> None:
    detail = "container no longer accepts exec"
    machine = MemoryMachine()
    machine.failure = MachineTerminated(detail)
    env = ShellboxEnv(machine, "image", AsyncExitStack())
    events: list[Event[None]] = []
    with pytest.raises(Restart) as caught:
        await write_file(WriteFile(OUTPUT, b"data"), env, notify=events.append)
    assert machine.calls == 1
    assert not events
    assert detail in str(caught.value)


async def test_read_ranges_flow_through_executor(create_env: Callable[[], Awaitable[Env]]) -> None:
    env = await create_env()
    await env.write(OUTPUT, b"0123456789")
    assert await read_file(ReadFile(OUTPUT, offset=3, limit=4), env, notify=Notes()) == b"3456"
    assert await read_file(ReadFile(OUTPUT, offset=8), env, notify=Notes()) == b"89"
    assert await read_file(ReadFile(OUTPUT, limit=0), env, notify=Notes()) == b""


class TimedIrisMachine(IrisMachine):
    """An Iris machine whose wrapped command ends with ``exit_code`` after ``elapsed`` seconds."""

    def __init__(self, exit_code: int, elapsed: float) -> None:
        self.exit_code, self.elapsed = exit_code, elapsed
        self.commands: list[Command] = []

    @override
    async def run(self, command: Command) -> Result:
        self.commands.append(command)
        await asyncio.sleep(self.elapsed)
        return Result(self.exit_code, b"partial", b"", False, False, ExitReason.EXITED)


@pytest.mark.parametrize("exit_code,elapsed,reported", [(124, 0.2, 124), (137, 0.2, 124), (137, 0, 137), (7, 0, 7)])
async def test_iris_command_is_bounded_inside_the_container(exit_code: int, elapsed: float, reported: int) -> None:
    machine = TimedIrisMachine(exit_code, elapsed)
    env = ShellboxEnv(machine, "image", AsyncExitStack())
    result = await env.run(COMMAND, timeout=0.1)
    assert (result.exit_code, result.stdout) == (reported, "partial")
    (command,) = machine.commands
    assert command.argv == ("sh", "-c", env_module.CONTAINED_TIMEOUT, "forge-timeout", "0.1s", COMMAND)
    assert command.timeout == 0.1 + env_module.BACKSTOP


@pytest.mark.parametrize(
    "script,timeout,code,stdout", [('printf "a  b"; exit 3', "5s", 3, b"a  b"), ("sleep 5", "0.2s", 124, b"")]
)
def test_contained_timeout_preserves_the_command(script: str, timeout: str, code: int, stdout: bytes) -> None:
    if code == 124 and shutil.which("timeout") is None:
        pytest.skip("this host has no timeout command")
    argv = ["sh", "-c", env_module.CONTAINED_TIMEOUT, "forge-timeout", timeout, script]
    result = subprocess.run(argv, capture_output=True, timeout=10)
    assert (result.returncode, result.stdout) == (code, stdout)


@pytest.mark.parametrize("code", [errno.ENOTDIR, errno.EACCES, errno.ENOENT, errno.ELOOP])
@pytest.mark.parametrize("simulated", [False, True])
async def test_write_backend_path_failures_are_normalized(code: int, simulated: bool) -> None:
    expected = type(OSError(code, os.strerror(code)))
    machine = MemoryMachine()
    # Iris raises errno-typed errors against its remote path; ShellSim raises "<strerror>: <path>" text.
    remote = "/remote/workspace/output"
    machine.failure = (
        SimulationError(f"{os.strerror(code)}: {remote}")
        if simulated
        else GuestPathError(code, os.strerror(code), remote)
    )
    env = ShellboxEnv(machine, "image", AsyncExitStack())
    with pytest.raises(expected) as caught:
        await env.write(OUTPUT, b"data")
    assert caught.value.filename == str(OUTPUT)


# Review 2026-11-20: a lost Iris reply after the exec ran made the handler run the command again and succeed.
@pytest.mark.parametrize("lost", ["reply", "capture"])
async def test_command_with_a_lost_outcome_runs_once_and_restarts_the_stage(tmp_path: Path, lost: str) -> None:
    counter = tmp_path / "counter"
    command = f"echo ran >> {counter}"

    def lose_reply(script: str) -> bool:
        return command in script if lost == "reply" else script.startswith("wc -c") and script.endswith("-out")

    env = ShellboxEnv(local_iris_machine(tmp_path, LocalController(lose_reply)), "image", AsyncExitStack())
    events: list[Event[None]] = []
    with pytest.raises(Restart, match="outcome unknown"):
        await run_command(RunCommand(command, 5), env, notify=events.append)
    assert counter.read_text() == "ran\n"
    assert not events


async def test_host_staging_failure_holds_with_its_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    def full_disk(self: Path, data: bytes) -> int:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(self))

    machine = MemoryMachine()
    env = ShellboxEnv(machine, "image", AsyncExitStack())
    monkeypatch.setattr(Path, "write_bytes", full_disk)
    with pytest.raises(Hold, match="No space left on device") as caught:
        await write_file(WriteFile(OUTPUT, b"data"), env, notify=Notes())
    assert not isinstance(caught.value, OSError) and machine.calls == 0


async def test_failing_guest_transfer_helper_is_not_a_path_failure(tmp_path: Path) -> None:
    (tmp_path / "present.txt").write_text("present")
    controller = LocalController(prelude="wc() { echo 'wc: not found' >&2; return 127; }; ")
    env = ShellboxEnv(local_iris_machine(tmp_path, controller), "image", AsyncExitStack())
    with pytest.raises(RuntimeError, match="wc: not found") as caught:
        await read_file(ReadFile(PurePosixPath(tmp_path / "present.txt")), env, notify=Notes())
    assert not isinstance(caught.value, OSError)
