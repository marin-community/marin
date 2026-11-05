# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import override

import pytest
from rolloutengine.test_support import machine_runtime
from shellbox.machine import Command, ExitReason, Machine, MachineTerminated, PackageSpec, Result
from taskcompendium.models import CommandSemantics, EnvironmentRequirements

from forge.effects.events import Note, Resolved
from forge.effects.issues import Hold, Restart, Transient
from forge.resources.env import (
    CommandResult,
    Env,
    FakeCommand,
    FakeEnv,
    Mutation,
    RunCommand,
    ShellboxEnv,
    ShellboxEnvFactory,
    WriteFile,
)
from forge.resources.sandbox.factories import local_factory, shellsim_factory

COMMAND = "printf changed > /workspace/output; printf hello; printf err >&2; exit 7"
OUTPUT = PurePosixPath("/workspace/output")
VALUE = CommandResult(7, "hello", "err", 0.1, False)


class MemoryMachine(Machine):
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.closed = False
        self.failure: RuntimeError | TimeoutError | None = None

    @override
    async def run(self, command: Command) -> Result:
        if self.failure is not None:
            raise self.failure
        if self.closed:
            raise MachineTerminated("closed")
        assert command.argv == ("sh", "-c", COMMAND)
        self.files[str(OUTPUT)] = b"changed"
        return Result(7, b"hello", b"err", False, False, ExitReason.EXITED)

    @override
    async def upload(self, source: Path, target: str) -> None:
        self.files[target] = source.read_bytes()

    @override
    async def download(self, source: str, target: Path) -> None:
        target.write_bytes(self.files[source])

    @override
    async def close(self) -> None:
        self.closed = True


class Notes:
    def __init__(self) -> None:
        self.events: list[Note] = []

    async def __call__(self, event: Note) -> None:
        self.events.append(event)


@pytest.fixture(
    params=[
        "fake",
        "shellbox-memory",
        pytest.param("local", marks=pytest.mark.cluster),
        pytest.param("shellsim", marks=pytest.mark.cluster),
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
                    factory = local_factory() if request.param == "local" else shellsim_factory()
                except ImportError:
                    pytest.skip("Optional sandbox dependency is not installed")
                semantics = (
                    CommandSemantics.LINUX_PROCESS if request.param == "local" else CommandSemantics.SHELL_SIMULATOR
                )
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
        (RuntimeError("Machine is closed"), Restart),
        (TimeoutError("busy"), Transient),
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
