# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local process transport for testing remote Shellbox adapter boundaries."""

import asyncio
import os
import shutil
import signal
import socket
import tempfile
from pathlib import Path

from shellbox.machine import Command, ExitReason, MachineSpec, Result
from shellbox.process_session import SubprocessSession, acquire_owned


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class ProcessMachine:
    """Local subprocess transport for the external machine boundary in this regression."""

    def __init__(self, root: Path, spec: MachineSpec):
        self.root, self.spec = root, spec
        self.closed = False
        self.sessions: set[SubprocessSession] = set()
        self.commands: set[asyncio.subprocess.Process] = set()
        self.path(spec.workdir).mkdir(parents=True, exist_ok=True)

    def path(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def argument(self, value: str) -> str:
        for prefix in (
            "/logs",
            "/workspace",
            "/tests",
            "/setup_files",
        ):
            value = value.replace(prefix, str(self.path(prefix)))
        return value.replace("tau3-runtime", "127.0.0.1")

    async def open_process(self, command: Command) -> SubprocessSession:
        async def acquire() -> SubprocessSession:
            process = await asyncio.create_subprocess_exec(
                *(self.argument(value) for value in command.argv),
                cwd=self.path(command.cwd or self.spec.workdir),
                env={
                    key: self.argument(value) if key == "PATH" else value
                    for key, value in {
                        "PATH": os.environ["PATH"],
                        **self.spec.env,
                        **command.env,
                    }.items()
                },
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )

            async def stop() -> None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            session = SubprocessSession(
                process, stop=stop, stderr_limit=command.output_limit_bytes
            )
            self.sessions.add(session)
            return session

        return await acquire_owned(asyncio.create_task(acquire()))

    async def run(self, command: Command) -> Result:
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = await asyncio.create_subprocess_exec(
                *(self.argument(value) for value in command.argv),
                cwd=self.path(command.cwd or self.spec.workdir),
                env={
                    key: self.argument(value) if key == "PATH" else value
                    for key, value in {
                        "PATH": os.environ["PATH"],
                        **self.spec.env,
                        **command.env,
                    }.items()
                },
                stdin=asyncio.subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            self.commands.add(process)
            try:
                async with asyncio.timeout(command.timeout):
                    if command.stdin:
                        assert process.stdin is not None
                        process.stdin.write(command.stdin)
                        await process.stdin.drain()
                    if process.stdin is not None:
                        process.stdin.close()
                    await process.wait()
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
                self.commands.discard(process)
                raise
            stdout_file.seek(0)
            stderr_file.seek(0)
            return Result(
                process.returncode,
                stdout_file.read(),
                stderr_file.read(),
                False,
                False,
                ExitReason.EXITED,
            )

    async def quiesce_commands(self) -> None:
        for process in self.commands:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                # A completed command's group ID can be reused by an unrelated
                # process before this test transport closes.
                pass
            await process.wait()

    async def upload(self, source: Path, target: str) -> None:
        destination = self.path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, destination)

    async def download(self, source: str, target: Path) -> None:
        if self.path(source).is_dir():
            shutil.copytree(self.path(source), target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.path(source), target)

    async def close(self) -> None:
        self.closed = True
        await self.quiesce_commands()
        await asyncio.gather(*(session.close() for session in self.sessions))
