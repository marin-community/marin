# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host subprocess machines for grading behavior tests across runtimes."""

import asyncio
import os
import sys
import tarfile
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.machine import Backend, Command, ExitReason, Result
from verifyit.spec import ScriptSpec, render_spec, spec_from_table

from taskcompendium.runtime.grading import SPEC_PATH, STAGING_ARCHIVE

EXITED = Result(0, b"", b"", False, False, ExitReason.EXITED)


@dataclass(kw_only=True)
class LocalGradingMachine:
    """Run grading commands as host subprocesses under ``root``, translating absolute container paths."""

    root: Path
    workdir: str
    env: dict[str, str] = field(default_factory=dict)
    remove_exit_code: int = 0
    closed: bool = False

    def local(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def local_arguments(self, arguments) -> list[str]:
        return [str(self.local(value)) if value.startswith("/") else value for value in arguments]

    async def upload(self, source: Path, target: str):
        path = self.local(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())

    async def download(self, source: str, target: Path):
        target.write_bytes(self.local(source).read_bytes())

    async def close(self):
        self.closed = True

    async def run(self, command: Command):
        program = command.argv[0]
        if command.argv[:2] == ("rm", "-f"):
            if self.remove_exit_code and STAGING_ARCHIVE in command.argv:
                return Result(self.remove_exit_code, b"", b"cleanup failed", False, False, ExitReason.EXITED)
            for path in command.argv[2:]:
                self.local(path).unlink(missing_ok=True)
            return EXITED
        if command.argv[:2] == ("mkdir", "-p"):
            process = await asyncio.create_subprocess_exec(
                "mkdir",
                "-p",
                *(str(self.local(path)) if path else path for path in command.argv[2:]),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return Result(process.returncode, stdout, stderr, False, False, ExitReason.EXITED)
        if command.argv[:2] == ("tar", "-xf"):
            with tarfile.open(self.local(command.argv[2])) as archive:
                archive.extractall(self.root, filter="data")
            return EXITED
        argv = self.local_arguments(command.argv)
        env = {**os.environ, **self.env, **command.env}
        if program == "python3":
            argv[0] = sys.executable
            env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
            if SPEC_PATH in command.argv:
                spec_path = self.local(SPEC_PATH)
                spec = spec_from_table(tomllib.loads(spec_path.read_text()))
                if isinstance(spec, ScriptSpec):
                    spec_path.write_text(render_spec(replace(spec, workspace=str(self.local("/app")))))
                argv.extend(("--logs-dir", str(self.local("/logs/verifier"))))
        else:
            assert Path(program).name in {"bash", "sh", "cat"}, command.argv
            argv[0] = program
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.local(command.cwd or self.workdir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return Result(process.returncode, stdout, stderr, False, False, ExitReason.EXITED)


@dataclass
class LocalGradingMachines(DockerMachineFactory):
    root: Path
    remove_exit_code: int = 0
    machines: list[LocalGradingMachine] = field(default_factory=list)
    backend = Backend.DOCKER

    async def create(self, spec):
        (self.root / spec.workdir.lstrip("/")).mkdir(parents=True, exist_ok=True)
        machine = LocalGradingMachine(
            root=self.root, workdir=spec.workdir, env=dict(spec.env), remove_exit_code=self.remove_exit_code
        )
        self.machines.append(machine)
        return machine
