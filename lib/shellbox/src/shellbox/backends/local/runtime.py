# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own a local sandbox and its persistent processes, descriptors, and filesystem."""

import asyncio
import logging
import os
import signal
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath

from shellbox.backends.local import sandbox
from shellbox.backends.local.cgroup import MachineCgroup
from shellbox.backends.local.commands import command_config, execute_command, start_process
from shellbox.backends.local.disk import LocalDisk
from shellbox.backends.local.disk_lifecycle import SHARED_MEMORY_DIRECTORY
from shellbox.backends.local.images import PreparedLocalImage
from shellbox.backends.local.keeper import allocate_keeper
from shellbox.backends.local.transfers import (
    ARCHIVE_LIMIT_BYTES,
    ARCHIVE_SCRIPT,
    EXTRACT_SCRIPT,
    _archive,
    unpack_local_archive,
)
from shellbox.machine import (
    DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    Command,
    MachineSpec,
    MachineTerminated,
    Mount,
    NetworkPolicy,
    Result,
    UnsupportedMachineSpec,
)
from shellbox.process_session import SubprocessSession, acquire_owned, await_cleanup, finish_owned, own_cleanup

logger = logging.getLogger(__name__)


class LocalProcessScope:
    """Agent command scope contained by a worker-owned cgroup."""

    def __init__(self, machine: "LocalMachine", group: MachineCgroup):
        self.machine = machine
        self.group = group
        self._closed = False
        self._active: set[asyncio.Task[Result]] = set()
        self._commands: dict[MachineCgroup, asyncio.Task[asyncio.subprocess.Process]] = {}
        self._closing: asyncio.Task[None] | None = None

    async def run(self, command: Command) -> Result:
        if self._closed:
            raise RuntimeError("Process scope is closed")
        self.machine._check_open()
        environment = {**self.machine._environment, **self.machine.spec.env, **command.env}
        task = asyncio.create_task(execute_command(self.machine, command, environment, scope=self))
        self._active.add(task)
        try:
            return await task
        finally:
            self._active.discard(task)

    async def terminate(self) -> None:
        if self._closing is None:
            self._closed = True
            self._closing = own_cleanup(self._terminate())
        await await_cleanup(self._closing)

    async def _terminate(self) -> None:
        async def stop(group: MachineCgroup, acquisition: asyncio.Task[asyncio.subprocess.Process]) -> None:
            self._commands.pop(group, None)
            process = await finish_owned(acquisition)
            await self.machine._stop_command(process, group)

        await asyncio.gather(*(stop(group, acquisition) for group, acquisition in tuple(self._commands.items())))
        if self._active:
            await asyncio.gather(*tuple(self._active), return_exceptions=True)
        await self.group.close()
        self.machine._scopes.discard(self)


class LocalMachine:
    """A persistent bubblewrap member, with commands entering pinned namespaces."""

    def __init__(
        self,
        spec: MachineSpec,
        *,
        bwrap: Path,
        root: Path,
        layers: Path | None,
        mounts: tuple[Mount, ...],
        environment: dict[str, str],
        image: PreparedLocalImage | None = None,
        bundle: Path | None = None,
        cgroup_parent: Path | None = None,
        disk: LocalDisk | None = None,
    ):
        self.spec = spec
        self.root = root
        self.layers = layers
        self.mounts = mounts
        self.image = image
        self._bundle = bundle
        self._disk = disk
        self._cgroup_parent = cgroup_parent
        self._cgroup: MachineCgroup | None = None
        self._command_cgroups: set[MachineCgroup] = set()
        self._background_cgroups: set[MachineCgroup] = set()
        self._run_commands: dict[MachineCgroup, asyncio.Task[asyncio.subprocess.Process]] = {}
        self._scopes: set[LocalProcessScope] = set()
        self._keeper_cgroup: MachineCgroup | None = None
        self._bwrap = bwrap
        self._environment = environment
        self._closed = False
        self._closing: asyncio.Task[None] | None = None
        self._task_close: Callable[[], Awaitable[None]] | None = None
        self._keeper: asyncio.subprocess.Process | None = None
        self._keeper_pidfd: int | None = None
        self._init_pidfd: int | None = None
        self._namespaces: tuple[int, ...] = ()
        self._owned_namespaces: tuple[int, ...] = ()
        self._network_namespaces: tuple[int, ...] = ()
        self._root_fd: int | None = None
        self._commands: set[asyncio.subprocess.Process] = set()
        self._sessions: set[SubprocessSession] = set()
        self._watcher: asyncio.Task[None] | None = None

    def _check_open(self) -> None:
        if self._closed:
            raise MachineTerminated("Machine is closed")

    def _prepare_keeper_cgroup(self) -> None:
        if self._cgroup_parent is not None:
            self._cgroup = MachineCgroup.create(self._cgroup_parent, cpus=self.spec.cpus, memory_mb=self.spec.memory_mb)
            self._keeper_cgroup = self._cgroup.create_child()
        elif self.spec.cpus is not None or self.spec.memory_mb is not None:
            logger.warning(
                "Local CPU/memory limits cannot be applied without cgroup delegation (cpus=%s, memory_mb=%s)",
                self.spec.cpus,
                self.spec.memory_mb,
            )

    def _keeper_arguments(self, peer: "LocalMachine | None") -> list[str]:
        if self.image is None:
            arguments = sandbox._sandbox_argv(
                self._bwrap,
                root=self.root,
                layers=self.layers,
                mounts=self.mounts,
                network=self.spec.network if peer is None else NetworkPolicy.ALLOW,
            )
        else:
            arguments = [
                str(self._bwrap),
                "--unshare-user",
                "--uid",
                "0",
                "--gid",
                "0",
                "--unshare-ipc",
                "--unshare-pid",
                "--unshare-uts",
                "--unshare-cgroup-try",
                "--as-pid-1",
                "--new-session",
                "--die-with-parent",
                "--cap-drop",
                "ALL",
                "--bind",
                str(self.root),
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/dev/shm",
            ]
            if peer is None and self.spec.network is NetworkPolicy.DENY:
                arguments.append("--unshare-net")
        if self._disk is not None:
            arguments += [
                "--bind",
                str(self._disk.data / SHARED_MEMORY_DIRECTORY),
                sandbox.SHARED_MEMORY,
                "--remount-ro",
                "/dev",
            ]
        # A new user namespace is needed for each independent mount namespace,
        # including members that inherit a peer's network namespace.
        if os.geteuid() != 0 and "--unshare-user" not in arguments:
            arguments.insert(1, "--unshare-user")
        if self.image is None and os.geteuid() != 0:
            arguments[1:1] = ["--uid", str(os.geteuid()), "--gid", str(os.getegid())]
        return arguments

    async def _start_keeper(self, peer: "LocalMachine | None" = None) -> None:
        self._prepare_keeper_cgroup()
        arguments = self._keeper_arguments(peer)
        block_read, block_write = os.pipe()
        allocation = asyncio.create_task(allocate_keeper(self, arguments, peer, block_read, block_write))
        try:
            try:
                await asyncio.shield(allocation)
            except asyncio.CancelledError:
                # Allocation still consumes the startup descriptors until it
                # finishes, even if another cancellation arrives while waiting.
                while not allocation.done():
                    try:
                        await asyncio.shield(allocation)
                    except asyncio.CancelledError:
                        pass
                allocation.result()
                raise
            finally:
                # An unpinned init may be waiting for this barrier and holding
                # stderr open. Withhold its start token and release the barrier
                # before cleanup waits for stderr EOF.
                if self._keeper is not None and self._keeper.stdin is not None:
                    self._keeper.stdin.close()
                os.close(block_read)
                os.close(block_write)
        except BaseException:
            await asyncio.shield(self._close_local())
            raise
        assert self._keeper_pidfd is not None
        self._watcher = asyncio.create_task(self._watch_keeper(os.dup(self._keeper_pidfd)))
        self._watcher.add_done_callback(_log_task_failure)

    async def _watch_keeper(self, descriptor: int) -> None:
        # Process.wait also waits for stderr EOF. Watch process death directly:
        # an orphaned init can keep that pipe open until our cleanup kills it.
        loop = asyncio.get_running_loop()
        exited = loop.create_future()

        def ready() -> None:
            loop.remove_reader(descriptor)
            exited.set_result(None)

        loop.add_reader(descriptor, ready)
        try:
            await exited
        finally:
            loop.remove_reader(descriptor)
            os.close(descriptor)
        if not self._closed:
            await self.close()

    async def run(self, command: Command) -> Result:
        return await execute_command(self, command, {**self._environment, **self.spec.env, **command.env})

    async def open_process_scope(self) -> LocalProcessScope:
        """Contain scoped commands and detached descendants outside the guest."""
        self._check_open()
        if self._cgroup is None:
            raise UnsupportedMachineSpec("Process scopes require delegated cgroups")
        group = self._cgroup.create_child()
        scope = LocalProcessScope(self, group)
        self._scopes.add(scope)
        return scope

    async def open_process(self, command: Command) -> SubprocessSession:
        """Start a foreground process in the same pinned namespaces as run."""
        if command.stdin or command.timeout is not None:
            raise ValueError("Process sessions own stdin and their lifetime")
        self._check_open()
        environment = {**self._environment, **self.spec.env, **command.env}
        config = command_config(self, command, environment)

        async def acquire() -> SubprocessSession:
            group = None
            # Keep all private descriptors alive until asyncio returns the child
            # handle. The launcher closes them before executing guest code.
            try:
                group = self._new_command_cgroup()
                launch_config = replace(config, cgroup=None if group is None else str(group.path / "cgroup.procs"))
                process = await start_process(
                    launch_config,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except BaseException:
                if group is not None:
                    await finish_owned(own_cleanup(group.close()))
                    self._command_cgroups.discard(group)
                raise
            self._commands.add(process)

            async def stop() -> None:
                if group is not None:
                    try:
                        await self._stop_command(process, group)
                        await session.drain_output()
                    finally:
                        self._commands.discard(process)
                        self._sessions.discard(session)
                    return
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    async with asyncio.timeout(sandbox.PROCESS_STOP_TIMEOUT):
                        await process.wait()
                        await session.drain_output()
                except TimeoutError:
                    # A command can detach from its launcher's process group
                    # and retain the pipes. Reclaim its entire task namespace.
                    await self.close()
                finally:
                    self._commands.discard(process)
                    self._sessions.discard(session)

            session = SubprocessSession(process, stop=stop, stderr_limit=command.output_limit_bytes)
            self._sessions.add(session)
            return session

        return await acquire_owned(asyncio.create_task(acquire()))

    async def _control(
        self,
        script: str,
        path: PurePosixPath,
        *,
        stdin: bytes = b"",
        output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    ) -> Result:
        return await execute_command(
            self,
            Command(
                ("/bin/sh", "-c", script, "sh", str(path)), cwd="/", stdin=stdin, output_limit_bytes=output_limit_bytes
            ),
            {**sandbox.CONTROL_ENVIRONMENT, "PATH": self._environment["PATH"]},
        )

    def _new_command_cgroup(self) -> MachineCgroup | None:
        if self._cgroup is None:
            return None
        for group in tuple(self._background_cgroups):
            if group.remove_if_empty():
                self._background_cgroups.discard(group)
                self._command_cgroups.discard(group)
                self._run_commands.pop(group, None)
        group = self._cgroup.create_child()
        self._command_cgroups.add(group)
        return group

    async def _stop_command(self, process: asyncio.subprocess.Process, group: MachineCgroup) -> None:
        # Stop the trusted launcher first: it may not yet have joined its cgroup
        # when subprocess acquisition returns. It must never fork after cleanup.
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await group.close()
        self._command_cgroups.discard(group)
        self._background_cgroups.discard(group)
        self._run_commands.pop(group, None)
        await process.wait()

    async def quiesce_commands(self) -> None:
        """Stop all run commands and their background work, preserving process sessions.

        Requires delegated cgroups. Call after stopping agent command submission,
        before collecting outputs or starting verification in this machine.
        """
        self._check_open()
        if self._cgroup is None:
            raise UnsupportedMachineSpec("Command quiescence requires delegated cgroups")

        async def stop(group: MachineCgroup, acquisition: asyncio.Task[asyncio.subprocess.Process]) -> None:
            process = await finish_owned(acquisition)
            if group in self._command_cgroups:
                await self._stop_command(process, group)

        async def cleanup() -> None:
            async with asyncio.TaskGroup() as pending:
                for group, acquisition in tuple(self._run_commands.items()):
                    pending.create_task(stop(group, acquisition))

        await await_cleanup(own_cleanup(cleanup()))

    async def upload(self, source: Path, target: str) -> None:
        self._check_open()
        path = sandbox._absolute_path(target)
        if self.image is None:
            writable_system = self.layers is not None and sandbox._within(path, sandbox.OVERLAID_DIRECTORIES)
            if (sandbox._within(path, sandbox.SYSTEM_DIRECTORIES) and not writable_system) or any(
                path.is_relative_to(mount.guest) for mount in self.mounts
            ):
                raise UnsupportedMachineSpec(f"The local backend's {target} is a read-only host path")
        name = "." if source.is_dir() else path.name
        stream = await asyncio.to_thread(_archive, source, name)
        result = await self._control(EXTRACT_SCRIPT, path if source.is_dir() else path.parent, stdin=stream)
        if result.exit_code != 0:
            raise RuntimeError(f"Cannot upload to {target}: {result.stderr.decode(errors='replace').strip()}")

    async def download(self, source: str, target: Path) -> None:
        self._check_open()
        path = sandbox._absolute_path(source)
        if self.image is None and path == PurePosixPath("/"):
            raise UnsupportedMachineSpec("A host-program machine cannot download its entire system root")
        result = await self._control(ARCHIVE_SCRIPT, path, output_limit_bytes=ARCHIVE_LIMIT_BYTES)
        if result.exit_code != 0:
            raise RuntimeError(result.stderr.decode(errors="replace").strip())
        if result.stdout_truncated:
            raise RuntimeError(f"{source} exceeds the {ARCHIVE_LIMIT_BYTES}-byte download limit")
        await asyncio.to_thread(unpack_local_archive, result.stdout, target, source_name=path.name)

    async def close(self) -> None:
        if self._task_close is not None:
            await self._task_close()
            return
        await self._close_local()

    async def _close_local(self) -> None:
        if self._closing is None:
            self._closed = True
            self._closing = asyncio.create_task(self._destroy())
        await asyncio.shield(self._closing)

    async def _destroy(self) -> None:
        await asyncio.gather(*(scope.terminate() for scope in tuple(self._scopes)))
        if self._cgroup is not None:
            self._cgroup.kill()
        # Kill the verified, pinned guest init before its host launcher. An
        # immediate close can otherwise race bwrap's parent-death setup and
        # leave the init holding the launcher's stderr pipe open.
        for descriptor in (self._init_pidfd, self._keeper_pidfd):
            if descriptor is None:
                continue
            try:
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            except ProcessLookupError:
                pass

        # Process.wait also waits for buffered pipe output. Read through the
        # session lock so shutdown can run alongside a caller or session.close.
        # Calling session.close here would recurse when its stop escalates here.
        async with asyncio.TaskGroup() as pending:
            for session in tuple(self._sessions):
                pending.create_task(session.drain_output())
            if self._keeper is not None:
                pending.create_task(self._keeper.wait())
            for process in tuple(self._commands):
                pending.create_task(process.wait())
        if self._cgroup is not None:
            await asyncio.gather(*(group.close() for group in tuple(self._command_cgroups)))
            if self._keeper_cgroup is not None:
                await self._keeper_cgroup.close()
            await self._cgroup.remove()
        if self._keeper_pidfd is not None:
            os.close(self._keeper_pidfd)
        if self._init_pidfd is not None:
            os.close(self._init_pidfd)
        # Prefix descriptors belong to the first member and are closed by it.
        for descriptor in self._owned_namespaces:
            os.close(descriptor)
        if self._root_fd is not None:
            os.close(self._root_fd)
        if self._disk is not None:
            await asyncio.to_thread(self._disk.close)
        else:
            await asyncio.to_thread(sandbox._remove, self._bundle or self.root)
            if self.layers is not None:
                await asyncio.to_thread(sandbox._remove, self.layers)


def _log_task_failure(task: asyncio.Task) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        if isinstance(error, MachineTerminated):
            return  # Closing a task interrupts its still-running companion monitors.
        logger.error("Local sandbox lifecycle failed", exc_info=(type(error), error, error.__traceback__))
