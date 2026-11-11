# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import override

import pytest
from rolloutengine.test_support import machine_runtime
from shellbox.backends.local.machine import SandboxUnavailable
from shellbox.machine import (
    Command,
    ExecSetupFailed,
    ExitReason,
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

from forge.effects.events import Event, Note, Resolved, Retrying
from forge.effects.issues import Hold, OutputBudgetExhausted, Restart, Transient
from forge.resources import env as env_module
from forge.resources.env import (
    CommandResult,
    Env,
    FakeCommand,
    FakeEnv,
    Mutation,
    ReadFile,
    RunCommand,
    ShellboxEnv,
    ShellboxEnvFactory,
    WriteFile,
    read_file,
    run_command,
    write_file,
)
from forge.resources.sandbox.factories import shellsim_factory

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
                env = FakeEnv("test-image", {COMMAND: FakeCommand(VALUE, {OUTPUT: b"changed"})}, Notes())
            case "shellbox-memory":
                machine = MemoryMachine()
                owned = AsyncExitStack()
                owned.push_async_callback(machine.close)
                env = ShellboxEnv(machine, "test-image", Notes(), owned)
            case _:
                try:
                    factory = shellsim_factory()
                except ImportError:
                    pytest.skip("Optional sandbox dependency is not installed")
                semantics = CommandSemantics.SHELL_SIMULATOR
                requirements = EnvironmentRequirements(command_semantics=semantics, package_spec=PackageSpec())
                env = await ShellboxEnvFactory(factory, machine_runtime(backend=request.param), 5, Notes()).create(
                    requirements
                )
        envs.append(env)
        return env

    yield create
    for env in envs:
        await env.close()


async def test_environment_restores_binary_writes_and_command_history(
    create_env: Callable[[], Awaitable[Env]],
) -> None:
    original = await create_env()
    initial = original.fingerprint()
    path = PurePosixPath("/workspace/binary")
    data = b"\x00\xff\nquoted ' bytes"
    writes = WriteFile(path, data)
    await original.write(path, data)
    after_write = original.fingerprint()
    assert after_write != initial
    assert await original.read(path) == data
    assert original.fingerprint() == after_write
    command = RunCommand(COMMAND, 5)
    result = await original.run(command.cmd, timeout=command.timeout)
    assert (result.exit_code, result.stdout, result.stderr) == (7, "hello", "err")
    assert await original.read(OUTPUT) == b"changed"
    records: tuple[Mutation, ...] = (Resolved(writes, (initial,), None), Resolved(command, (after_write,), result))
    restored = await create_env()
    await restored.restore(records)
    assert restored.fingerprint() == original.fingerprint()
    assert await restored.read(path) == data
    assert await restored.read(OUTPUT) == b"changed"
    await restored.close()
    with pytest.raises(Restart):
        await restored.run(COMMAND, timeout=5)


async def test_restore_warns_on_output_divergence_and_keeps_logged_position() -> None:
    notes = Notes()
    source = FakeEnv("image", {COMMAND: FakeCommand(VALUE, {})}, notes)
    first = source.fingerprint()
    await source.run(COMMAND, timeout=5)
    second = source.fingerprint()
    await source.write(OUTPUT, b"after")
    replay = FakeEnv("image", {COMMAND: FakeCommand(replace(VALUE, stdout="different", wall=9), {})}, notes)
    await replay.restore(
        (Resolved(RunCommand(COMMAND, 5), (first,), VALUE), Resolved(WriteFile(OUTPUT, b"after"), (second,), None))
    )
    assert len(notes.events) == 1
    assert replay.fingerprint() == source.fingerprint()
    assert await replay.read(OUTPUT) == b"after"


async def test_restore_refuses_wrong_image_before_mutating_files() -> None:
    notes = Notes()
    expected = FakeEnv("image-a", {}, notes).fingerprint()
    env = FakeEnv("image-b", {}, notes)
    before = env.fingerprint()
    with pytest.raises(Hold):
        await env.restore((Resolved(WriteFile(OUTPUT, b"data"), (expected,), None),))
    assert env.fingerprint() == before


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
    ],
)
async def test_shellbox_failure_preserves_unresolved_history(
    failure: RuntimeError | TimeoutError, decision: type[Exception]
) -> None:
    machine = MemoryMachine()
    machine.failure = failure
    env = ShellboxEnv(machine, "image", Notes(), AsyncExitStack())
    before = env.fingerprint()
    with pytest.raises(decision):
        await env.run(COMMAND, timeout=5)
    assert env.fingerprint() == before


async def test_shellbox_unexpected_runtime_error_propagates_without_reclassification() -> None:
    machine = MemoryMachine()
    bug = RuntimeError("unexpected adapter bug")
    machine.failure = bug
    env = ShellboxEnv(machine, "image", Notes(), AsyncExitStack())
    with pytest.raises(RuntimeError) as caught:
        await env.run(COMMAND, timeout=5)
    assert caught.value is bug


@pytest.mark.parametrize("operation", ["read", "write", "run"])
@pytest.mark.parametrize(
    "failure",
    [
        SimulationError("No such file or directory: /workspace/output"),
        FileNotFoundError(2, "No such file", str(OUTPUT)),
    ],
)
async def test_missing_file_is_tool_feedback_without_retry(operation: str, failure: Exception) -> None:
    machine = MemoryMachine()
    machine.failure = failure
    env = ShellboxEnv(machine, "image", Notes(), AsyncExitStack())
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
    env = ShellboxEnv(machine, "image", Notes(), AsyncExitStack())
    with pytest.raises(Hold, match="cpu_exhausted"):
        await run_command(RunCommand(COMMAND, 5), env, notify=Notes())
    assert machine.calls == 1


@pytest.mark.parametrize("workdir", ["/workspace", "/custom/work"])
async def test_relative_paths_use_environment_workdir(workdir: str) -> None:
    machine = MemoryMachine()
    env = ShellboxEnv(machine, "image", Notes(), AsyncExitStack(), workdir)
    await env.write(PurePosixPath("release_notes.py"), b"notes")
    assert machine.files == {f"{workdir}/release_notes.py": b"notes"}
    assert await env.read(PurePosixPath("release_notes.py")) == b"notes"
    await env.write(PurePosixPath("/absolute.py"), b"absolute")
    assert machine.files["/absolute.py"] == b"absolute"


async def test_websocket_setup_failure_retries_only_three_times() -> None:
    machine = MemoryMachine()
    machine.failure = ExecSetupFailed("handshake refused")
    env = ShellboxEnv(machine, "image", Notes(), AsyncExitStack())
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
    factory = ShellboxEnvFactory(shellsim_factory(), machine_runtime(backend="shellsim"), 5, Notes())
    with pytest.raises(Transient) as caught:
        await factory.create(EnvironmentRequirements(command_semantics=CommandSemantics.SHELL_SIMULATOR))
    assert caught.value.__cause__ is failure


async def test_sandbox_startup_failure_restarts_the_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = SandboxUnavailable("Persistent bubblewrap keeper failed: setting up uid map")

    @asynccontextmanager
    async def failed_start(*args: object, **kwargs: object) -> AsyncIterator[Machine]:
        raise failure
        yield MemoryMachine()

    monkeypatch.setattr(env_module, "prepare_machine", failed_start)
    factory = ShellboxEnvFactory(shellsim_factory(), machine_runtime(backend="shellsim"), 5, Notes())
    with pytest.raises(Restart, match="keeper failed") as caught:
        await factory.create(EnvironmentRequirements(command_semantics=CommandSemantics.SHELL_SIMULATOR))
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    "failure",
    [
        SimulationError("Is a directory: /app"),
        RuntimeError("Iris command failed: Is a directory"),
        IsADirectoryError("temporary download directory"),
    ],
)
async def test_directory_read_names_requested_path_without_retry(failure: Exception) -> None:
    machine = MemoryMachine()
    machine.failure = failure
    env = ShellboxEnv(machine, "image", Notes(), AsyncExitStack())
    with pytest.raises(IsADirectoryError) as caught:
        await read_file(ReadFile(PurePosixPath("/app")), env, notify=Notes())
    assert caught.value.filename == "/app"
    assert machine.calls == 1


async def test_noninteger_exec_status_requires_new_machine_without_retry() -> None:
    detail = "container no longer accepts exec"
    machine = MemoryMachine()
    machine.failure = MachineTerminated(detail)
    env = ShellboxEnv(machine, "image", Notes(), AsyncExitStack())
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


class RangedMachine(MemoryMachine):
    async def read_file(self, path: str, *, offset: int = 0, limit: int | None = None) -> bytes:
        assert (path, offset, limit) == (str(OUTPUT), 2, 3)
        if self.failure is not None:
            raise self.failure
        return b"234"

    @override
    async def download(self, source: str, target: Path) -> None:
        raise AssertionError("FileReader must bypass downloads, including on directory errors")


@pytest.mark.parametrize("directory", [False, True])
async def test_native_read_preflight_and_range_bypass_download(directory: bool) -> None:
    machine = RangedMachine()
    env = ShellboxEnv(machine, "image", Notes(), AsyncExitStack())
    if directory:
        machine.failure = IsADirectoryError(str(OUTPUT))
        with pytest.raises(IsADirectoryError):
            await read_file(ReadFile(OUTPUT, offset=2, limit=3), env, notify=Notes())
    else:
        assert await read_file(ReadFile(OUTPUT, offset=2, limit=3), env, notify=Notes()) == b"234"


@pytest.mark.parametrize(
    "detail,expected",
    [
        ("Not a directory", NotADirectoryError),
        ("Permission denied", PermissionError),
        ("No such file or directory", FileNotFoundError),
    ],
)
async def test_write_backend_path_failures_are_normalized(detail: str, expected: type[OSError]) -> None:
    machine = MemoryMachine()
    machine.failure = RuntimeError(f"Iris command failed: {detail}")
    env = ShellboxEnv(machine, "image", Notes(), AsyncExitStack())
    with pytest.raises(expected) as caught:
        await env.write(OUTPUT, b"data")
    assert caught.value.filename == str(OUTPUT)
