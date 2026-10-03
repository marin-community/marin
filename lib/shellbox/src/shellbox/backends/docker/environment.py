# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor commands and diagnostic transfer over a Shellbox Docker machine."""

import shlex
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities
from harbor.models.task.config import TaskOS
from harbor.models.trial.config import ServiceVolumeConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.docker.terminal import DockerControlPlane
from shellbox.machine import (
    HARBOR_EXEC_OUTPUT_LIMIT_BYTES,
    Command,
    DockerImage,
    ExitReason,
    Machine,
    MachineFactory,
    MachineSpec,
    NetworkPolicy,
)

LOG_PATHS = EnvironmentPaths()
UNSUPPORTED_DEFINITION_FILES = ("Dockerfile", "docker-compose.yaml", "docker-compose.yml")


def public_log_mounts(mounts: list[ServiceVolumeConfig] | None, trial_paths: TrialPaths) -> list[ServiceVolumeConfig]:
    """Validate Harbor's bookkeeping mounts and return only public log paths."""
    allowed = {
        str(LOG_PATHS.agent_dir): trial_paths.agent_dir,
        str(LOG_PATHS.artifacts_dir): trial_paths.artifacts_dir,
        str(LOG_PATHS.verifier_dir): trial_paths.verifier_dir,
    }
    for mount in mounts or []:
        if (
            mount["target"] not in allowed
            or Path(mount["source"]).resolve() != Path(str(allowed[mount["target"]])).resolve()
        ):
            raise ValueError("Shellbox Docker cannot expose caller-selected host paths")
    return [mount for mount in mounts or [] if mount["target"] != str(LOG_PATHS.verifier_dir)]


class DockerEnvironment(BaseEnvironment):
    """Own one Shellbox Docker machine for the complete Harbor trial."""

    def __init__(
        self,
        *args,
        trial_paths: TrialPaths,
        archive_socket: str | None = None,
        archive_api_version: str | None = None,
        machine_factory: MachineFactory | None = None,
        mounts: list[ServiceVolumeConfig] | None = None,
        **kwargs,
    ):
        if machine_factory is None:
            if archive_socket is None or archive_api_version is None:
                raise ValueError("Docker requires an explicit archive socket and API version")
            self.machine_factory: MachineFactory = DockerMachineFactory(
                control=DockerControlPlane(archive_socket, archive_api_version)
            )
        else:
            if archive_socket is not None or archive_api_version is not None:
                raise ValueError("Supply a machine factory or Docker control-plane inputs, not both")
            self.machine_factory = machine_factory
        self.machine: Machine | None = None
        super().__init__(*args, trial_paths=trial_paths, mounts=public_log_mounts(mounts, trial_paths), **kwargs)

    @staticmethod
    def type() -> str:
        return "shellbox-docker"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    def _validate_definition(self) -> None:
        if self.task_env_config.docker_image is None or not self.task_env_config.workdir:
            raise ValueError("Shellbox Docker requires an image and explicit workdir")
        if self.os is not TaskOS.LINUX:
            raise NotImplementedError("Shellbox Docker supports Linux tasks only")
        if any((self.environment_dir / name).exists() for name in UNSUPPORTED_DEFINITION_FILES):
            raise ValueError("Shellbox Docker uses the selected local image without build or compose definitions")

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError("Shellbox Docker uses the selected local image without building")
        self.machine = await self.machine_factory.create(
            MachineSpec(
                source=DockerImage(self.task_env_config.docker_image or ""),
                workdir=self.task_env_config.workdir or "",
                network=NetworkPolicy.DENY,
                cpus=self._effective_cpus,
                memory_mb=self._effective_memory_mb,
                storage_mb=self._effective_storage_mb,
                startup_timeout=self.task_env_config.build_timeout_sec,
            )
        )
        try:
            workdir = self.task_env_config.workdir or ""
            quoted = shlex.quote(workdir)
            self._require_success(
                await self.exec(f"test ! -L {quoted} && mkdir -p {quoted} && test -d {quoted}", cwd="/", user="root")
            )
            # Resolve the image's default user before Harbor applies agent user overrides.
            identity = await self.exec("id -u; id -g", cwd="/")
            identifiers = (identity.stdout or "").splitlines()
            if identity.return_code != 0 or len(identifiers) != 2 or not all(value.isdecimal() for value in identifiers):
                raise RuntimeError("Cannot resolve the Docker worker's numeric UID and GID")
            owner = ":".join(identifiers)
            await self._upload_environment_dir_after_start()
            modes: dict[int, list[str]] = {0o755: [workdir]}
            for source in sorted(self.environment_dir.rglob("*")):
                target = str(Path(workdir) / source.relative_to(self.environment_dir))
                mode = 0o755 if source.is_dir() else source.stat().st_mode & 0o7777
                modes.setdefault(mode, []).append(target)
            paths = " ".join(shlex.quote(path) for group in modes.values() for path in group)
            self._require_success(await self.exec(f"chown -- {owner} {paths}", cwd="/", user="root"))
            for mode, group in modes.items():
                paths = " ".join(shlex.quote(path) for path in group)
                self._require_success(await self.exec(f"chmod {mode:o} {paths}", cwd="/", user="root"))
            diagnostics = await self.ensure_dirs([LOG_PATHS.agent_dir, LOG_PATHS.artifacts_dir])
            assert diagnostics is not None
            self._require_success(diagnostics)
        except BaseException:
            await self.stop(True)
            raise

    @staticmethod
    def _require_success(result: ExecResult) -> None:
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout)

    async def stop(self, delete: bool) -> None:
        if self.machine is not None:
            await self.machine.close()
            self.machine = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if self.machine is None:
            raise RuntimeError("Shellbox Docker environment is not running")
        effective_user = self._resolve_user(user)
        result = await self.machine.run(
            Command(
                argv=("/bin/bash", "-c", command),
                cwd=cwd,
                env=self._merge_env(env) or {},
                timeout=timeout_sec,
                output_limit_bytes=HARBOR_EXEC_OUTPUT_LIMIT_BYTES,
                user=str(effective_user) if effective_user is not None else None,
            )
        )
        if result.reason is ExitReason.TIMED_OUT:
            raise TimeoutError("Docker command timed out")
        assert result.exit_code is not None
        return ExecResult(
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
            return_code=result.exit_code,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        if self.machine is None:
            raise RuntimeError("Shellbox Docker environment is not running")
        await self.machine.upload(Path(source_path), target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self.upload_file(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        if self.machine is None:
            raise RuntimeError("Shellbox Docker environment is not running")
        await self.machine.download(source_path, Path(target_path))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        destination = Path(target_dir)
        destination.mkdir(parents=True, exist_ok=True)
        await self.download_file(source_dir, destination)
