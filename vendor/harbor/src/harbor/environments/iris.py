# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a prebuilt Harbor task in a credential-free Iris sandbox."""

import asyncio
import re
import tempfile
from pathlib import Path, PurePosixPath

from harbor.environments.base import EXEC_OUTPUT_LIMIT_BYTES, BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.paths import EnvironmentPaths
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.storage_path import StoragePath

from shellbox.backends.iris.machine import DEFAULT_JOB_TTL, DEFAULT_SCHEDULING_TIMEOUT, IrisMachine, IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, ExitReason, MachineSpec, NetworkPolicy

COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")
PINNED_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
BASE_ENVIRONMENT_ARGUMENTS = frozenset(
    {
        "environment_dir",
        "environment_name",
        "session_id",
        "trial_paths",
        "task_env_config",
        "logger",
        "override_cpus",
        "override_memory_mb",
        "override_storage_mb",
        "override_gpus",
        "override_tpu",
        "cpu_enforcement_policy",
        "memory_enforcement_policy",
        "persistent_env",
        "mounts",
        "extra_docker_compose",
    }
)


def validate_iris_task(task_env_config: EnvironmentConfig, environment_dir: Path) -> None:
    """Reject task requirements the offline single-container runtime cannot preserve."""
    image = task_env_config.docker_image
    if image is None or PINNED_IMAGE.fullmatch(image) is None:
        raise ValueError("Iris sandbox requires a prebuilt docker_image pinned with @sha256")
    if (environment_dir / "Dockerfile").exists():
        raise ValueError("Iris sandbox does not build Dockerfiles; prepare a pinned image first")
    if any((environment_dir / name).exists() for name in COMPOSE_FILENAMES):
        raise ValueError("Iris sandbox does not support Compose tasks")
    if task_env_config.allow_internet:
        raise ValueError("Iris sandbox requires allow_internet=false")
    if not task_env_config.workdir or not PurePosixPath(task_env_config.workdir).is_absolute():
        raise ValueError("Iris sandbox requires an explicit absolute workdir matching the prepared image")
    if any("${" in value for value in task_env_config.env.values()):
        raise ValueError("Iris sandbox task environment must contain literal values only")


class IrisEnvironment(BaseEnvironment):
    """Execute one task in gVisor with Iris's NONE egress policy.

    Images must be built and pinned before evaluation. Model requests and result
    uploads belong to the external driver. Kubernetes platform DNS remains available.
    """

    def __init__(
        self,
        *args,
        cluster: str | None = None,
        controller_url: str | None = None,
        scheduling_timeout: int = DEFAULT_SCHEDULING_TIMEOUT,
        job_ttl: int = DEFAULT_JOB_TTL,
        **kwargs,
    ):
        unknown = kwargs.keys() - BASE_ENVIRONMENT_ARGUMENTS
        if unknown:
            raise ValueError(f"Unsupported Iris sandbox arguments: {sorted(unknown)}")
        if kwargs.get("persistent_env"):
            raise ValueError("Iris sandbox does not accept driver environment variables")
        if cluster is None and controller_url is None:
            info = get_job_info()
            controller_url = info.controller_address if info is not None else None
        self.factory = IrisMachineFactory(
            cluster=cluster,
            controller_url=controller_url,
            scheduling_timeout=scheduling_timeout,
            job_ttl=job_ttl,
        )
        self.machine: IrisMachine | None = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "marin-iris"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_request=True, memory_request=True)

    def _maybe_resolve_task_env(self) -> None:
        # BaseEnvironment otherwise substitutes task-controlled references from os.environ.
        if any("${" in value for value in self.task_env_config.env.values()):
            raise ValueError("Iris sandbox task environment must contain literal values only")
        self._persistent_env = {**self.task_env_config.env, **self._persistent_env}

    def _validate_definition(self) -> None:
        validate_iris_task(self.task_env_config, self.environment_dir)
        paths = EnvironmentPaths()
        log_targets = {
            str(path) for path in (paths.agent_dir, paths.verifier_dir, paths.user_agent_dir, paths.artifacts_dir)
        }
        if any(str(mount["target"]) not in log_targets for mount in self._mounts):
            raise ValueError("Iris sandbox does not support custom mounts")

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError("Iris sandbox uses prebuilt images; force_build is unsupported")
        image = self.task_env_config.docker_image
        workdir = self.task_env_config.workdir
        assert image is not None
        assert workdir is not None
        spec = MachineSpec(
            source=RegistryImage(image),
            workdir=workdir,
            network=NetworkPolicy.DENY,
            cpus=self._resource_request_value("cpu", auto_mode=ResourceMode.REQUEST),
            memory_mb=self._resource_request_value("memory", auto_mode=ResourceMode.REQUEST),
            storage_mb=self.task_env_config.storage_mb,
        )
        creation = asyncio.create_task(self.factory.create(spec))
        try:
            self.machine = await asyncio.shield(creation)
        except asyncio.CancelledError:
            # Iris submission runs in a thread; cancelling its waiter does not stop
            # the job being created. Reclaim it once submission has completed.
            machine = await creation
            await asyncio.shield(machine.close())
            raise
        try:
            targets = self._mount_targets(writable_only=True)
            if targets:
                result = await self.exec(self._ensure_dirs_command(targets, chmod=True), cwd="/")
                if result.return_code:
                    raise RuntimeError(f"Failed to prepare Iris sandbox paths: {result.stderr or result.stdout}")
            await self._upload_environment_dir_after_start()
        except BaseException:
            await asyncio.shield(self.stop(True))
            raise

    async def stop(self, delete: bool) -> None:
        if self.machine is not None:
            machine, self.machine = self.machine, None
            await machine.close()

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if self.machine is None:
            raise RuntimeError("Iris sandbox is not running")
        effective_user = self._resolve_user(user)
        result = await self.machine.run(
            Command(
                argv=("sh", "-c", command),
                cwd=cwd,
                env=self._merge_env(env) or {},
                timeout=timeout_sec,
                user=str(effective_user) if effective_user is not None else None,
                output_limit_bytes=EXEC_OUTPUT_LIMIT_BYTES,
            )
        )
        if result.reason is ExitReason.TIMED_OUT:
            raise TimeoutError("Iris sandbox command timed out and the sandbox was terminated")
        return ExecResult(
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
            return_code=result.exit_code,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        if self.machine is None:
            raise RuntimeError("Iris sandbox is not running")
        await self.machine.upload(Path(source_path), target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self.upload_file(source_dir, target_dir)

    async def _download(self, source: str, target: Path | str, *, directory: bool) -> None:
        if self.machine is None:
            raise RuntimeError("Iris sandbox is not running")
        destination = StoragePath(str(target))
        if destination.is_local:
            local = Path("/" + destination.key) if destination.scheme == "file" else Path(str(destination))
            await self.machine.download(source, local)
            return
        with tempfile.TemporaryDirectory() as staging:
            local = Path(staging) / "download"
            await self.machine.download(source, local)
            await asyncio.to_thread(destination.upload_from, str(local), recursive=directory)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._download(source_path, target_path, directory=False)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._download(source_dir, target_dir, directory=True)
