# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run bounded commands in pinned sandbox namespaces with owned cleanup."""

import asyncio
import errno
import json
import os
import pwd
import resource
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import BinaryIO, Protocol

from shellbox.backends.local import launch, sandbox
from shellbox.backends.local.cgroup import MachineCgroup
from shellbox.backends.local.disk import LocalDisk
from shellbox.backends.local.disk_lifecycle import MEBIBYTE
from shellbox.backends.local.images import PreparedLocalImage
from shellbox.machine import (
    Command,
    ExitReason,
    MachineDiskFull,
    MachineSpec,
    Result,
    UnsupportedMachineSpec,
)
from shellbox.process_session import finish_owned, own_cleanup

RUN_SCRIPT = 'cd "$1" || exit; shift; exec "$@"'

NPROC_HEADROOM = 256

FILE_SIZE_LIMIT = 1 << 30


def _resource_limits(file_size_limit: int = FILE_SIZE_LIMIT) -> list[tuple[int, int]]:
    host_tasks = int(Path("/proc/loadavg").read_text().split()[3].split("/")[1])
    return [
        (resource.RLIMIT_NPROC, host_tasks + NPROC_HEADROOM),
        (resource.RLIMIT_FSIZE, file_size_limit),
        (resource.RLIMIT_CORE, 0),
    ]


def _command_account(user: str | None) -> pwd.struct_passwd | None:
    """The host account a command switches to, or None to run as this process's user."""
    uid = os.geteuid()
    if user is None or user == str(uid):
        return None
    try:
        account = pwd.getpwuid(int(user)) if user.isdigit() else pwd.getpwnam(user)
    except KeyError as error:
        raise UnsupportedMachineSpec(f"The local backend found no host account for user {user}") from error
    if account.pw_uid == uid:
        return None
    if uid != 0:
        raise UnsupportedMachineSpec(f"The local backend runs as uid {uid}; only root can run a command as {user}")
    return account


class CommandScope(Protocol):
    group: MachineCgroup
    _closed: bool
    _commands: dict[MachineCgroup, asyncio.Task[asyncio.subprocess.Process]]


class CommandHost(Protocol):
    spec: MachineSpec
    image: PreparedLocalImage | None
    _disk: LocalDisk | None
    _root_fd: int | None
    _namespaces: tuple[int, ...]
    _closed: bool
    _commands: set[asyncio.subprocess.Process]
    _run_commands: dict[MachineCgroup, asyncio.Task[asyncio.subprocess.Process]]
    _command_cgroups: set[MachineCgroup]
    _background_cgroups: set[MachineCgroup]

    def _new_command_cgroup(self) -> MachineCgroup | None: ...
    async def _stop_command(self, process: asyncio.subprocess.Process, group: MachineCgroup) -> None: ...
    def _check_open(self) -> None: ...
    async def close(self) -> None: ...


@dataclass(frozen=True)
class CommandLaunch:
    namespaces: tuple[int, ...]
    root: int | None
    limits: list[tuple[int, int]]
    account: tuple[int, int] | None
    script: str
    cwd: str
    argv: tuple[str, ...]
    environment: dict[str, str]
    cgroup: str | None
    status: int | None


@dataclass(frozen=True)
class CommandStreams:
    status: BinaryIO
    status_writer: BinaryIO
    stdout: BinaryIO
    stderr: BinaryIO
    stdin: BinaryIO


@dataclass(frozen=True)
class CommandCompletion:
    process: asyncio.subprocess.Process
    reason: ExitReason


def command_config(machine: CommandHost, command: Command, environment: dict[str, str]) -> CommandLaunch:
    machine._check_open()
    if not command.argv:
        raise ValueError("Command argv is empty")
    if command.output_limit_bytes < 0:
        raise ValueError("Output limit must be nonnegative")
    if machine.image is not None:
        if command.user not in (None, "0", "0:0"):
            raise UnsupportedMachineSpec("Prepared local images currently support only root USER")
        account = None
    else:
        selected = _command_account(command.user)
        account = None if selected is None else (selected.pw_uid, selected.pw_gid)
    config = CommandLaunch(
        namespaces=machine._namespaces,
        root=machine._root_fd,
        limits=_resource_limits(
            machine.spec.storage_mb * MEBIBYTE if machine.spec.storage_mb is not None else FILE_SIZE_LIMIT
        ),
        account=account,
        script=RUN_SCRIPT,
        cwd=command.cwd or machine.spec.workdir or "/",
        argv=command.argv,
        environment=environment,
        cgroup=None,
        status=None,
    )
    if any("\0" in argument for argument in (*command.argv, *environment, *environment.values())):
        raise ValueError("Sandbox arguments cannot contain NUL bytes")
    return config


def _stage_stdin(stream: BinaryIO, payload: bytes, disk: LocalDisk | None) -> None:
    try:
        written = 0
        while written < len(payload):
            size = stream.write(payload[written:])
            assert size > 0
            written += size
        stream.seek(0)
    except OSError as error:
        if disk is not None and error.errno in (errno.ENOSPC, errno.EDQUOT):
            raise MachineDiskFull("LOCAL cannot allocate command stdin") from error
        raise


def _command_result(code: int, stdout: bytes, stderr: bytes, output_limit_bytes: int) -> Result:
    return Result(
        128 - code if code < 0 else code,
        stdout[:output_limit_bytes],
        stderr[:output_limit_bytes],
        len(stdout) > output_limit_bytes,
        len(stderr) > output_limit_bytes,
        ExitReason.EXITED,
    )


async def start_process(
    config: CommandLaunch,
    *,
    stdin: BinaryIO | int,
    stdout: BinaryIO | int,
    stderr: BinaryIO | int,
) -> asyncio.subprocess.Process:
    """Keep launch descriptors private and alive until subprocess acquisition finishes."""
    assert config.root is not None
    with tempfile.TemporaryFile() as configuration:
        configuration.write(json.dumps(asdict(config)).encode())
        configuration.seek(0)
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-S",
            "-c",
            sandbox.LAUNCH_SOURCE,
            str(configuration.fileno()),
            pass_fds=(
                configuration.fileno(),
                *((config.status,) if config.status is not None else ()),
                config.root,
                *dict.fromkeys(config.namespaces),
            ),
            env=sandbox.CONTROL_ENVIRONMENT,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )


@contextmanager
def command_streams(disk: LocalDisk | None, stdin: bytes) -> Iterator[CommandStreams]:
    """Own bounded anonymous spools until supervision and output decoding finish."""
    status_read, status_write = os.pipe()
    with (
        os.fdopen(status_read, "rb", buffering=0) as status,
        os.fdopen(status_write, "wb", buffering=0) as status_writer,
        disk.spool() if disk is not None else tempfile.TemporaryFile() as stdout_file,
        disk.spool() if disk is not None else tempfile.TemporaryFile() as stderr_file,
        disk.spool() if disk is not None else tempfile.TemporaryFile() as stdin_file,
    ):
        # The launcher writes before exit. Nonblocking reads cannot be held open
        # by a descendant that accidentally inherited another pipe end.
        os.set_blocking(status_read, False)
        _stage_stdin(stdin_file, stdin, disk)
        yield CommandStreams(status, status_writer, stdout_file, stderr_file, stdin_file)


async def _supervise_command(
    machine: CommandHost,
    acquisition: asyncio.Task[asyncio.subprocess.Process],
    group: MachineCgroup | None,
    scope: CommandScope | None,
    streams: CommandStreams,
    timeout: float | None,
) -> CommandCompletion:
    process = None
    try:
        async with asyncio.timeout(timeout):
            process = await asyncio.shield(acquisition)
            machine._commands.add(process)
            streams.status_writer.close()
            await process.wait()
    except BaseException as interruption:

        async def cleanup() -> None:
            nonlocal process
            if process is None:
                # A child may already have forked before asyncio hands
                # back its handle. Own that handle before closing its
                # namespace or the private configuration descriptors.
                process = await finish_owned(acquisition)
                machine._commands.add(process)
            streams.status_writer.close()
            if group is not None:
                if scope is not None:
                    scope._commands.pop(group, None)
                await machine._stop_command(process, group)
            elif not machine._closed:
                # Teardown unmounts the quota filesystem after killing
                # the child. Release our spool references before then.
                streams.stdout.close()
                streams.stderr.close()
                streams.stdin.close()
                await machine.close()
            await process.wait()

        try:
            await finish_owned(own_cleanup(cleanup()))
        except BaseException as cleanup_error:
            if isinstance(interruption, TimeoutError):
                raise RuntimeError("Cannot confirm timed-out command cleanup") from cleanup_error
            interruption.add_note(f"Command cleanup failed ({type(cleanup_error).__name__})")
        if isinstance(interruption, TimeoutError):
            assert process is not None
            return CommandCompletion(process, ExitReason.TIMED_OUT)
        raise
    finally:
        if process is not None:
            machine._commands.discard(process)
        if scope is not None and group is not None and group in scope._commands and not machine._closed:
            if not group.populated:
                await group.close()
                scope._commands.pop(group, None)
        if scope is None and group is not None and not machine._closed and group in machine._command_cgroups:
            if group.populated:
                machine._background_cgroups.add(group)
            else:
                await group.close()
                machine._command_cgroups.discard(group)
                machine._run_commands.pop(group, None)
    assert process is not None
    return CommandCompletion(process, ExitReason.EXITED)


def _read_result(process: asyncio.subprocess.Process, streams: CommandStreams, limit: int) -> Result:
    if error := streams.status.read(launch.SETUP_ERROR_LIMIT_BYTES):
        raise RuntimeError(f"Cannot start the command in the machine's sandbox: {error.decode(errors='replace')}")
    streams.stdout.seek(0)
    streams.stderr.seek(0)
    stdout = streams.stdout.read(limit + 1)
    stderr = streams.stderr.read(limit + 1)
    assert process.returncode is not None
    return _command_result(process.returncode, stdout, stderr, limit)


async def execute_command(
    machine: CommandHost, command: Command, environment: dict[str, str], *, scope: CommandScope | None = None
) -> Result:
    if scope is not None and scope._closed:
        raise RuntimeError("Process scope is closed")
    config = command_config(machine, command, environment)
    with command_streams(machine._disk, command.stdin) as streams:
        group = scope.group.create_child() if scope is not None else machine._new_command_cgroup()
        config = replace(
            config,
            status=streams.status_writer.fileno(),
            cgroup=None if group is None else str(group.path / "cgroup.procs"),
        )
        acquisition = asyncio.create_task(
            start_process(config, stdin=streams.stdin, stdout=streams.stdout, stderr=streams.stderr)
        )
        if group is not None:
            if scope is None:
                machine._run_commands[group] = acquisition
            else:
                scope._commands[group] = acquisition
        completion = await _supervise_command(machine, acquisition, group, scope, streams, command.timeout)
        if completion.reason is ExitReason.TIMED_OUT:
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        machine._check_open()
        return _read_result(completion.process, streams, command.output_limit_bytes)
