# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import os
import re
import shutil
import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import override

import pytest
from lib.taskforge.tests.resources.iris_fakes import (
    RPC_SITES,
    Fault,
    Injection,
    LocalController,
    injections,
    local_iris_machine,
)
from rigging.timing import ExponentialBackoff
from rolloutengine.test_support import machine_runtime
from shellbox.backends.iris import machine as iris_backend
from shellbox.backends.iris.machine import IrisMachine
from shellbox.machine import (
    Command,
    CommandOutcomeUnknown,
    ExecSetupFailed,
    ExitReason,
    GuestHelperFailed,
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
    FakeCommand,
    FakeEnv,
    ReadFile,
    RunCommand,
    ShellboxEnv,
    ShellboxEnvFactory,
    WriteFile,
    WriteFiles,
    read_file,
    run_command,
    write_file,
    write_files,
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
        GuestPathError(errno.ENOENT, "No such file", str(OUTPUT)),
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
    env = ShellboxEnv(machine, "image", Notes(), AsyncExitStack())
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
    env = ShellboxEnv(machine, "image", Notes(), AsyncExitStack())
    with pytest.raises(expected) as caught:
        await env.write(OUTPUT, b"data")
    assert caught.value.filename == str(OUTPUT)


def _iris_env(tmp_path: Path, controller: LocalController) -> ShellboxEnv:
    return ShellboxEnv(local_iris_machine(tmp_path, controller), "image", Notes(), AsyncExitStack())


def _counted(tmp_path: Path) -> RunCommand:
    """A command that appends one line to ``tmp_path / "counter"`` each time it runs."""
    return RunCommand(f"echo ran >> {tmp_path / 'counter'}; printf done; printf oops >&2", 5)


def _injected(names: Sequence[str]) -> list[Injection]:
    """Parse ``site:fault:ran`` or ``site:fault:unsent`` names into fixture injections."""
    return [(site, Fault(fault), ran == "ran") for site, fault, ran in (name.split(":") for name in names)]


def _appends(tmp_path: Path) -> int:
    counter = tmp_path / "counter"
    return len(counter.read_text().splitlines()) if counter.exists() else 0


@pytest.fixture
def fast_refusal_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(iris_backend, "EXEC_SHED_BACKOFF", ExponentialBackoff(initial=0.001, maximum=0.001))


# Review 2026-11-20: a lost Iris reply after the exec ran made the handler run the command again and succeed.
# Follow-up 1: so did a capture or cleanup RPC that failed before it was sent, and a cleanup failure after a lost
# reply replaced the primary failure. Follow-up 2: a lost exec stream whose diagnostic status lookup was refused
# retried in place, and one with a healthy status escaped raw. Follow-up 3: the WebSocket setup-failure text after
# a lost upgrade reply, and a reply over the client's size limit, both reran an executed command in place.
# Only a connection refused before sending, or the exec pool's admission refusal, may be retried.
@pytest.mark.parametrize(
    "faults,outcome",
    [
        (("command:lose-reply:ran",), "restart"),
        (("command:stream-lost:ran",), "restart"),
        (("command:stream-lost-raised:ran",), "restart"),
        (("command:setup-failed:ran",), "restart"),
        (("command:reply-too-large:ran",), "restart"),
        (("command:stream-lost:ran", "status:connect-refused:unsent"), "restart"),
        (("command:stream-lost:ran", "status:lose-reply:unsent"), "restart"),
        (("stdout-size:lose-reply:ran",), "restart"),
        (("stdout-size:setup-failed:unsent",), "restart"),
        (("cleanup:setup-failed:unsent",), "result"),
        (("cleanup:lose-reply:ran",), "result"),
        (("command:lose-reply:ran", "cleanup:connect-refused:unsent"), "restart"),
        (("command:connect-refused:unsent",), "retried"),
        (("command:admission-refused:unsent",), "admitted"),
    ],
)
async def test_command_runs_once_whichever_iris_rpc_fails(
    tmp_path: Path, fast_refusal_backoff: None, faults: tuple[str, ...], outcome: str
) -> None:
    controller = LocalController(_injected(faults))
    events: list[Event[None]] = []
    if outcome == "restart":
        with pytest.raises(Restart, match="outcome unknown"):
            await run_command(_counted(tmp_path), _iris_env(tmp_path, controller), notify=events.append)
    else:
        result = await run_command(_counted(tmp_path), _iris_env(tmp_path, controller), notify=events.append)
        assert (result.exit_code, result.stdout, result.stderr) == (0, "done", "oops")
    assert _appends(tmp_path) == 1
    # A refused connection is retried by Taskforge; an admission refusal by the machine's own send.
    assert controller.sent("command") == 1 + (outcome in ("retried", "admitted"))
    assert sum(isinstance(event, Retrying) for event in events) == (outcome == "retried")


# Review 2026-11-20 (follow-up 3): a missing `wc` or an OOM-killed sandbox during capture restarted on a fresh
# sandbox, three appends before the hold; each held after one at 47749c2e21.
@pytest.mark.parametrize("cause", [GuestHelperFailed, MachineResourceExhausted])
async def test_capture_failure_with_a_known_cause_holds_after_one_run(tmp_path: Path, cause: type[Exception]) -> None:
    if cause is GuestHelperFailed:
        controller = LocalController(prelude="wc() { echo 'wc: not found' >&2; return 127; }; ")
    else:
        controller = LocalController([("stdout-size", Fault.OOM_KILLED, True)])
    events: list[Event[None]] = []
    with pytest.raises(Hold, match="wc: not found" if cause is GuestHelperFailed else "OOMKilled") as caught:
        await run_command(_counted(tmp_path), _iris_env(tmp_path, controller), notify=events.append)
    assert isinstance(caught.value.__cause__, cause)
    assert _appends(tmp_path) == 1 and controller.sent("command") == 1
    assert not events


RETRIED_IN_PLACE = (Fault.CONNECT_REFUSED, Fault.ADMISSION_REFUSED)
REPLY_LOSSES = (Fault.LOSE_REPLY, Fault.STREAM_LOST, Fault.STREAM_LOST_RAISED)
# Every RPC ``IrisMachine.run`` makes, failed in every physically possible way, before and after the controller ran
# the exec: the command's own dispatch, capture and cleanup after a reply, and the status diagnostic and cleanup
# after each way the reply can be lost. ``NEVER_AFTER_RUN`` names the faults that cannot follow an executed exec.
FAULT_PLANS = [
    *((injection,) for fault in Fault for injection in injections("command", fault)),
    *(
        (injection,)
        for site in RPC_SITES
        if site not in ("command", "status")
        for fault in Fault
        for injection in injections(site, fault)
    ),
    *(
        (("command", loss, True), injection)
        for loss in REPLY_LOSSES
        for site in ("status", "cleanup")
        for fault in Fault
        for injection in injections(site, fault)
    ),
]


@pytest.mark.parametrize(
    "faults",
    FAULT_PLANS,
    ids=lambda faults: "+".join(f"{site}:{fault}:{'ran' if ran else 'unsent'}" for site, fault, ran in faults),
)
async def test_no_failure_after_dispatch_reruns_the_command_or_escapes(
    tmp_path: Path, fast_refusal_backoff: None, faults: tuple[Injection, ...]
) -> None:
    controller = LocalController(faults)
    events: list[Event[None]] = []
    try:
        result = await run_command(_counted(tmp_path), _iris_env(tmp_path, controller), notify=events.append)
    except Restart as error:
        assert "outcome unknown" in str(error) or "environment terminated" in str(error)
    except Hold as error:
        assert isinstance(error.__cause__, MachineResourceExhausted)
    else:
        assert (result.exit_code, result.stdout, result.stderr) == (0, "done", "oops")
    assert not controller.faults, "every injected fault must fire"
    (site, fault, ran), *_ = faults
    retried = site == "command" and fault in RETRIED_IN_PLACE
    # Only the two never-sent dispatch failures resend the command, and then it runs once.
    assert controller.sent("command") == 1 + retried
    assert _appends(tmp_path) == (0 if site == "command" and not ran and not retried else 1)
    assert sum(isinstance(event, Retrying) for event in events) == (retried and fault is Fault.CONNECT_REFUSED)


LATE = Fault.SETUP_FAILED_LATE
FILES = tuple((PurePosixPath(f"dir/f{index}.txt"), f"payload {index}\n".encode()) for index in range(4))


def _sandbox(root: Path, faults: Sequence[Injection] = ()) -> tuple[ShellboxEnv, LocalController]:
    root.mkdir()
    controller = LocalController(faults)
    return ShellboxEnv(local_iris_machine(root, controller), "image", Notes(), AsyncExitStack(), str(root)), controller


async def _write(env: Env, count: int, notify: Callable[[Event[None]], object]) -> None:
    """Write the first ``count`` files as the write tool (one file) or stage hydration (several) does."""
    if count == 1:
        await write_file(WriteFile(*FILES[0]), env, notify=notify)
    else:
        await write_files(WriteFiles(FILES[:count]), env, notify=notify)


# Review 2026-11-21 (follow-up 4): a lost upgrade reply for the first upload chunk retried the write in the same
# sandbox, and the original append landed after the retry truncated the file: `File written` with 14 bytes on disk
# for a 7-byte write. Two-file hydration corrupted its first file the same way.
@pytest.mark.parametrize("count", [1, 2], ids=["write", "hydrate-2"])
async def test_lost_upload_reply_restarts_on_a_fresh_sandbox_with_exact_bytes(tmp_path: Path, count: int) -> None:
    first, controller = _sandbox(tmp_path / "sandbox-0", [("append", LATE, True)])
    events: list[Event[None]] = []
    with pytest.raises(Restart, match="write outcome unknown"):
        await _write(first, count, events.append)
    assert controller.sent("truncate") == 1 and not events, "the write must not be resent to this sandbox"
    fresh, _ = _sandbox(tmp_path / "sandbox-1")
    await _write(fresh, count, events.append)
    for path, data in FILES[:count]:
        assert await read_file(ReadFile(path), fresh, notify=events.append) == data


# Every guest mutation Taskforge drives through ShellboxEnv, with one helper's upgrade reply lost while that helper is
# still running. Only a resend to a fresh guest name, the archive of four-file hydration, is safe on the same sandbox;
# a read-only helper keeps its in-place retry, and so does a write that was positively never sent.
@pytest.mark.parametrize(
    "operation,fault,resent",
    [
        ("write", "truncate:setup-failed-late:ran", False),
        ("write", "append:setup-failed-late:ran", False),
        ("hydrate-2", "append:setup-failed-late:ran", False),
        ("hydrate-4", "append:setup-failed-late:ran", True),
        ("hydrate-4", "command:setup-failed-late:ran", False),
        ("run", "command:setup-failed-late:ran", False),
        ("read", "read-chunk:setup-failed-late:ran", True),
        ("write", "append:connect-refused:unsent", True),
        ("write", "append:admission-refused:unsent", True),
    ],
)
async def test_lost_helper_reply_resends_only_a_read_or_a_fresh_name_in_place(
    tmp_path: Path, fast_refusal_backoff: None, operation: str, fault: str, resent: bool
) -> None:
    ((site, kind, _),) = injection = _injected([fault])
    env, controller = _sandbox(tmp_path / "sandbox", injection)
    events: list[Event[None]] = []
    count = int(operation.removeprefix("hydrate-")) if operation.startswith("hydrate") else 1
    if operation == "run":
        action = run_command(
            RunCommand("mkdir -p dir && printf 'payload 0\\n' > dir/f0.txt", 5), env, notify=events.append
        )
    elif operation == "read":
        (tmp_path / "sandbox" / "dir").mkdir()
        (tmp_path / "sandbox" / FILES[0][0]).write_bytes(FILES[0][1])
        action = read_file(ReadFile(FILES[0][0]), env, notify=events.append)
    else:
        action = _write(env, count, events.append)
    try:
        if not resent:
            with pytest.raises(Restart):
                await action
        else:
            assert await action in (None, FILES[0][1])
            assert [(tmp_path / "sandbox" / path).read_bytes() for path, _ in FILES[:count]] == [
                data for _, data in FILES[:count]
            ]
        assert controller.sent(site) == 1 + resent
        # The exec pool's admission refusal is retried inside the machine's own send.
        assert sum(isinstance(event, Retrying) for event in events) == (resent and kind is not Fault.ADMISSION_REFUSED)
    finally:
        for script in controller.scripts:
            for archive in re.findall(r"/tmp/forge-hydration-\w+\.tar", script):
                Path(archive).unlink(missing_ok=True)


async def test_host_staging_failure_holds_with_its_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    def full_disk(self: Path, data: bytes) -> int:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(self))

    machine = MemoryMachine()
    env = ShellboxEnv(machine, "image", Notes(), AsyncExitStack())
    monkeypatch.setattr(Path, "write_bytes", full_disk)
    with pytest.raises(Hold, match="No space left on device") as caught:
        await write_file(WriteFile(OUTPUT, b"data"), env, notify=Notes())
    assert not isinstance(caught.value, OSError) and machine.calls == 0


# Review 2026-11-20 (follow-up 1): a directory read was archived and extracted, then held as host I/O; writes
# through a link cycle, to a dangling link whose target parent is missing, or with an overlong name escaped raw.
async def test_ordinary_iris_path_mistakes_are_tool_feedback_before_any_transfer(tmp_path: Path) -> None:
    (tmp_path / "directory").mkdir()
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    (tmp_path / "dangling").symlink_to(tmp_path / "absent" / "file")
    (tmp_path / "parent-link").symlink_to(tmp_path / "absent")
    controller = LocalController()
    env = ShellboxEnv(local_iris_machine(tmp_path, controller), "image", Notes(), AsyncExitStack())
    cases = [
        (env.read(PurePosixPath(tmp_path / "directory")), errno.EISDIR, "directory"),
        (env.write(PurePosixPath(tmp_path / "loop"), b"x"), errno.ENOENT, "loop"),
        (env.write(PurePosixPath(tmp_path / "dangling"), b"x"), errno.ENOENT, "dangling"),
        (env.write(PurePosixPath(tmp_path / "parent-link" / "file"), b"x"), errno.ENOENT, "parent-link/file"),
        (env.write(PurePosixPath(tmp_path / ("n" * 300)), b"x"), errno.ENAMETOOLONG, "n" * 300),
    ]
    for operation, code, name in cases:
        controller.scripts.clear()
        with pytest.raises(OSError) as caught:
            await operation
        assert (caught.value.errno, caught.value.filename) == (code, str(tmp_path / name))
        assert not isinstance(caught.value, GuestPathError)
        assert len(controller.scripts) == 1


# Review 2026-11-20 (follow-up 1): a missing `wc` became a bug hold, and a missing `dirname` looped forever.
@pytest.mark.parametrize(
    "helper,operation,expected",
    [
        ("wc", "read", Hold),
        ("mkdir", "write", Hold),
        ("dirname", "read", FileNotFoundError),
        ("dirname", "write", None),
    ],
)
async def test_broken_guest_helpers_hold_with_their_cause_in_one_bounded_probe(
    tmp_path: Path, helper: str, operation: str, expected: type[Exception] | None
) -> None:
    (tmp_path / "present.txt").write_text("present")
    controller = LocalController(prelude=f"{helper}() {{ echo '{helper}: not found' >&2; return 127; }}; ")
    env = ShellboxEnv(local_iris_machine(tmp_path, controller), "image", Notes(), AsyncExitStack())
    if operation == "read":
        path = tmp_path / ("present.txt" if helper == "wc" else "absent/deeper/file.txt")
        action = read_file(ReadFile(PurePosixPath(path)), env, notify=Notes())
    else:
        action = write_file(WriteFile(PurePosixPath(tmp_path / "new" / "file.txt"), b"data"), env, notify=Notes())
    if expected is None:
        await action
        assert (tmp_path / "new" / "file.txt").read_bytes() == b"data"
        return
    with pytest.raises(expected, match=f"{helper}: not found" if expected is Hold else None) as caught:
        await action
    assert len(controller.scripts) == 1
    if expected is Hold:
        assert isinstance(caught.value.__cause__, GuestHelperFailed)
