# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local process transport for testing remote Shellbox adapter boundaries."""

import asyncio
import os
import shutil
import signal
import socket
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
        self.path(spec.workdir).mkdir(parents=True)

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
                env={"PATH": os.environ["PATH"], **self.spec.env, **command.env},
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

            session = SubprocessSession(process, stop=stop, stderr_limit=command.output_limit_bytes)
            self.sessions.add(session)
            return session

        return await acquire_owned(asyncio.create_task(acquire()))

    async def run(self, command: Command) -> Result:
        process = await asyncio.create_subprocess_exec(
            *(self.argument(value) for value in command.argv),
            cwd=self.path(command.cwd or self.spec.workdir),
            env={"PATH": os.environ["PATH"], **self.spec.env, **command.env},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(command.stdin)
        return Result(
            process.returncode, stdout, stderr, False, False, ExitReason.EXITED
        )

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
        await asyncio.gather(*(session.close() for session in self.sessions))
