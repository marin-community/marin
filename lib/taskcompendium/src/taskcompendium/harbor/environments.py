# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor environments for direct chat and persistent ShellSim sessions."""

import asyncio
import json
import shlex
from pathlib import Path, PurePosixPath

import msgspec
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment

from taskcompendium.harbor.snapshot import download_snapshot
from taskcompendium.models import DockerEnvironment as DockerRequirement
from taskcompendium.models import ExecutionConfig, FinalState, image_digest
from taskcompendium.models import ShellSimEnvironment as ShellSimRequirement
from taskcompendium.serialization import from_json, protocol_from_json
from taskcompendium.shellsim import ShellSimLimits, ShellSimSession


class NoToolEnvironment(BaseEnvironment):
    """A chat trial with host-owned diagnostics and no agent filesystem or shell."""

    @staticmethod
    def type() -> str:
        return "taskcompendium-chat"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    def _validate_definition(self) -> None:
        if (self.environment_dir / "inputs").exists():
            raise ValueError("No-tool environments cannot contain filesystem inputs")

    async def start(self, force_build: bool) -> None:
        pass

    async def stop(self, delete: bool) -> None:
        pass

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResult:
        raise ValueError("Chat does not provide shell execution")

    async def upload_file(self, source_path, target_path) -> None:
        raise ValueError("Chat does not provide filesystem uploads")

    async def upload_dir(self, source_dir, target_dir) -> None:
        raise ValueError("Chat does not provide filesystem uploads")

    async def download_file(self, source_path, target_path) -> None:
        raise ValueError("Chat does not provide filesystem downloads")

    async def download_dir(self, source_dir, target_dir) -> None:
        # Harbor requests diagnostics after every agent run. Direct-chat agents
        # already write their logs on the host; there is no remote copy to merge.
        if source_dir not in {"/logs/agent", "/logs/artifacts"}:
            raise ValueError("Chat does not provide filesystem downloads")


class ShellSimEnvironment(BaseEnvironment):
    """Run every agent shell action in one bounded, persistent simulated world."""

    def __init__(self, *args, bridge_path="taskcompendium-shellsim", limits=None, **kwargs):
        self.bridge_path = bridge_path
        self.limits = ShellSimLimits(**limits) if limits is not None else ShellSimLimits()
        self.session: ShellSimSession | None = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "taskcompendium-shellsim"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    def _validate_definition(self) -> None:
        if not self.environment_dir.is_dir():
            raise ValueError("Missing ShellSim environment directory")

    def _session(self) -> ShellSimSession:
        if self.session is None:
            raise RuntimeError("ShellSim environment is not running")
        return self.session

    async def start(self, force_build: bool) -> None:
        self.session = ShellSimSession(self.bridge_path, limits=self.limits)
        workdir = self.task_env_config.workdir or "/app"
        await asyncio.to_thread(self.session.mkdir, workdir)
        await asyncio.to_thread(self.session.mkdir, "/logs/agent")
        await asyncio.to_thread(self.session.mkdir, "/logs/artifacts")
        await asyncio.to_thread(self.session.run, f"cd {shlex.quote(workdir)}")
        inputs = self.environment_dir / "inputs"
        if inputs.exists():
            await self.upload_dir(inputs, workdir)
        specification = from_json((self.environment_dir.parent / "specification.json").read_bytes())
        requirement = specification.environment
        if isinstance(requirement, ShellSimRequirement):
            for directory in requirement.additional_directories:
                await asyncio.to_thread(self.session.mkdir, directory)
            for command in requirement.setup_commands:
                result = await self.exec(command, cwd="/")
                if result.return_code != 0:
                    raise RuntimeError(f"ShellSim setup failed: {result.stderr or result.stdout}")
            await asyncio.to_thread(self.session.run, f"cd {shlex.quote(workdir)}")

    async def stop(self, delete: bool) -> None:
        if self.session is not None:
            await asyncio.to_thread(self.session.close)
            self.session = None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResult:
        result = await asyncio.to_thread(self._session().run, command, cwd=cwd, env=self._merge_env(env))
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.return_code)

    async def upload_file(self, source_path, target_path) -> None:
        session = self._session()
        await asyncio.to_thread(session.mkdir, str(PurePosixPath(target_path).parent))
        await asyncio.to_thread(session.write_file, target_path, Path(source_path).read_bytes())

    async def upload_dir(self, source_dir, target_dir) -> None:
        source = Path(source_dir)
        await asyncio.to_thread(self._session().mkdir, target_dir)
        for path in sorted(source.rglob("*")):
            target = str(PurePosixPath(target_dir) / path.relative_to(source).as_posix())
            if path.is_symlink():
                raise ValueError(f"ShellSim input must not contain symlinks: {path}")
            if path.is_dir():
                await asyncio.to_thread(self._session().mkdir, target)
            else:
                await self.upload_file(path, target)

    async def download_file(self, source_path, target_path) -> None:
        data = await asyncio.to_thread(self._session().read_file, source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def download_dir(self, source_dir, target_dir) -> None:
        paths = await asyncio.to_thread(self._session().list_files, source_dir)
        for path in paths:
            relative = PurePosixPath(path).relative_to(source_dir)
            await self.download_file(path, Path(target_dir) / str(relative))

    async def is_dir(self, path, user=None) -> bool:
        return await asyncio.to_thread(self._session().is_dir, path)

    async def is_file(self, path, user=None) -> bool:
        return await asyncio.to_thread(self._session().is_file, path)


class TaskDockerEnvironment(DockerEnvironment):
    """Native Harbor Docker lifecycle with only declared public inputs uploaded."""

    def _validate_definition(self) -> None:
        super()._validate_definition()
        if self.task_env_config.docker_image is None:
            raise ValueError("TaskCompendium requires a pinned container image")
        image_digest(self.task_env_config.docker_image)

    async def _upload_environment_dir_after_start(self) -> None:
        manifest = json.loads((self.environment_dir.parent / "manifest.json").read_text())
        execution = msgspec.convert(manifest["execution"], type=ExecutionConfig)
        environment = execution.environment
        if not isinstance(environment, DockerRequirement):
            raise ValueError("Docker bootstrap requires a Docker execution environment")
        workdir = self.task_env_config.workdir
        if not workdir:
            raise ValueError("TaskCompendium Docker tasks require an explicit workdir")
        result = await self.exec(f"mkdir -p {shlex.quote(workdir)}", cwd="/")
        if result.return_code != 0:
            raise RuntimeError(f"Cannot create task workspace: {result.stderr}")
        inputs = self.environment_dir / "inputs"
        if inputs.is_dir():
            await self.upload_dir(inputs, workdir)
        for directory in environment.additional_directories:
            result = await self.exec(f"mkdir -p {shlex.quote(directory)}", cwd="/")
            if result.return_code != 0:
                raise RuntimeError(f"Cannot create declared filesystem directory: {result.stderr}")
        for command in environment.setup_commands:
            result = await self.exec(command, cwd="/", user="root")
            if result.return_code != 0:
                raise RuntimeError(f"Task environment setup failed: {result.stderr or result.stdout}")

    async def download_dir(self, source_dir, target_dir) -> None:
        if PurePosixPath(source_dir) != PurePosixPath(self.task_env_config.workdir or "/app"):
            await super().download_dir(source_dir, target_dir)
            return
        protocol = protocol_from_json((self.environment_dir.parent / "protocol.json").read_bytes())
        exclusions = protocol.submission.excluded_paths if isinstance(protocol.submission, FinalState) else ()
        result = await self._run_docker_compose_command(["ps", "-q", "main"])
        container_id = (result.stdout or "").strip()
        if not container_id or len(container_id.splitlines()) != 1:
            raise RuntimeError("Expected one running Docker workspace container")
        await download_snapshot(container_id, str(source_dir), Path(target_dir), exclusions)

    async def stop(self, delete: bool) -> None:
        # Pinned input images are shared by trials and must survive cleanup.
        try:
            await self.prepare_logs_for_host()
            await self._run_docker_compose_command(["down", "--volumes", "--remove-orphans"])
        finally:
            self._cleanup_mounts_compose_file()
            self._cleanup_resources_compose_file()
