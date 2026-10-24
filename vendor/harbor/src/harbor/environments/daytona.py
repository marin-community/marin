# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor's single-container adapter for Shellbox Daytona machines."""

import asyncio
import os
import tempfile
from contextlib import AsyncExitStack
from pathlib import Path

from daytona import AsyncDaytona, DaytonaConfig
from rigging.filesystem.storage_path import StoragePath

from harbor.environments.base import (
    EXEC_OUTPUT_LIMIT_BYTES,
    BaseEnvironment,
    ExecResult,
)
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.iris_compose import COMPOSE_FILENAMES
from harbor.environments.model_relay import MODEL_RELAY_PORT, ModelRelay
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.paths import EnvironmentPaths
from shellbox.backends.daytona.machine import (
    DEFAULT_SANDBOX_TTL_MINUTES,
    DaytonaMachine,
    DaytonaMachineFactory,
    DaytonaNetworkMode,
    DaytonaNetworkPolicy,
)
from shellbox.http_service import serve_http
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import Command, ExitReason, MachineSpec, NetworkPolicy

_BASE_ARGUMENTS = frozenset(
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


class DaytonaEnvironment(BaseEnvironment):
    """Allocate, execute and transfer exclusively through Shellbox's Daytona backend."""

    def __init__(
        self,
        *args,
        api_key_env: str | None = None,
        network_policy: dict[str, str] | None = None,
        ttl_minutes: int = DEFAULT_SANDBOX_TTL_MINUTES,
        create_timeout: float = 600,
        model_endpoint: str | None = None,
        model_name: str | None = None,
        verifier_model_endpoint: str | None = None,
        **kwargs,
    ):
        unknown = kwargs.keys() - _BASE_ARGUMENTS
        if unknown:
            raise ValueError(
                f"Unsupported Shellbox Daytona arguments: {sorted(unknown)}"
            )
        policy = None
        if network_policy is not None:
            if network_policy.keys() - {"mode", "value"}:
                raise ValueError("Unknown Daytona network policy fields")
            policy = DaytonaNetworkPolicy(
                DaytonaNetworkMode(network_policy["mode"]), network_policy.get("value")
            )
        client_factory = (
            AsyncDaytona
            if api_key_env is None
            else lambda: AsyncDaytona(DaytonaConfig(api_key=os.environ[api_key_env]))
        )
        self.factory = DaytonaMachineFactory(
            client_factory,
            ttl_minutes=ttl_minutes,
            create_timeout=create_timeout,
            network_policy=policy,
        )
        self.model_endpoint, self.model_name = model_endpoint, model_name
        self.verifier_model_endpoint = verifier_model_endpoint
        self.machine: DaytonaMachine | None = None
        self._relay = AsyncExitStack()
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DAYTONA

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_request=True, memory_request=True)

    def _validate_definition(self) -> None:
        if self.extra_docker_compose_paths or any(
            (self.environment_dir / name).exists() for name in COMPOSE_FILENAMES
        ):
            raise ValueError(
                "Shellbox Daytona supports single-container tasks; use Iris for Compose service groups"
            )
        if (
            not self.task_env_config.docker_image
            and not (self.environment_dir / "Dockerfile").is_file()
        ):
            raise ValueError("Daytona task requires a registry image or Dockerfile")
        paths = EnvironmentPaths()
        allowed = {
            str(path)
            for path in (
                paths.agent_dir,
                paths.verifier_dir,
                paths.user_agent_dir,
                paths.artifacts_dir,
            )
        }
        if any(str(mount["target"]) not in allowed for mount in self._mounts):
            raise ValueError("Shellbox Daytona does not support custom host mounts")

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError(
                "Shellbox snapshots are immutable; change the prepared image or build inputs"
            )
        source = (
            RegistryImage(self.task_env_config.docker_image)
            if self.task_env_config.docker_image
            else DockerfileSource(
                self.environment_dir, self.environment_dir / "Dockerfile"
            )
        )
        spec = MachineSpec(
            source=source,
            workdir=self.task_env_config.workdir or "",
            env=self._merge_env(None) or {},
            network=(
                NetworkPolicy.ALLOW
                if self.task_env_config.allow_internet
                else NetworkPolicy.DENY
            ),
            cpus=self._effective_cpus,
            memory_mb=self._effective_memory_mb,
            storage_mb=self._effective_storage_mb,
        )
        if self.verifier_model_endpoint is not None and (
            self.model_endpoint is None or self.model_name is None
        ):
            raise ValueError(
                "Daytona model capability requires a host endpoint and model"
            )
        creation = asyncio.create_task(self.factory.create(spec))
        try:
            self.machine = await asyncio.shield(creation)
        except asyncio.CancelledError:
            machine = await creation
            await asyncio.shield(machine.close())
            raise
        try:
            if self.model_endpoint is not None:
                assert self.model_name is not None
                await self._relay.enter_async_context(
                    serve_http(
                        self.machine,
                        ModelRelay(self.model_endpoint, self.model_name).handle,
                        port=MODEL_RELAY_PORT,
                    )
                )
            targets = self._mount_targets(writable_only=True)
            if targets:
                result = await self.exec(
                    self._ensure_dirs_command(targets, chmod=True), cwd="/"
                )
                if result.return_code:
                    raise RuntimeError("Failed to prepare Daytona artifact paths")
            await self._upload_environment_dir_after_start()
        except BaseException:
            await asyncio.shield(self.stop(True))
            raise

    async def stop(self, delete: bool) -> None:
        try:
            await self._relay.aclose()
        finally:
            machine, self.machine = self.machine, None
            if machine is not None:
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
            raise RuntimeError("Daytona environment is not running")
        effective_user = self._resolve_user(user)
        result = await self.machine.run(
            Command(
                ("sh", "-c", command),
                cwd=cwd,
                env=self._merge_env(env) or {},
                timeout=timeout_sec,
                user=str(effective_user) if effective_user is not None else None,
                output_limit_bytes=EXEC_OUTPUT_LIMIT_BYTES,
            )
        )
        if result.reason is ExitReason.TIMED_OUT:
            raise TimeoutError("Daytona command timed out")
        return ExecResult(
            return_code=result.exit_code,
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        if self.machine is None:
            raise RuntimeError("Daytona environment is not running")
        await self.machine.upload(Path(source_path), target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self.upload_file(source_dir, target_dir)

    async def _download(
        self, source: str, target: Path | str, *, directory: bool
    ) -> None:
        if self.machine is None:
            raise RuntimeError("Daytona environment is not running")
        destination = StoragePath(str(target))
        if destination.is_local:
            local = (
                Path("/" + destination.key)
                if destination.scheme == "file"
                else Path(str(destination))
            )
            await self.machine.download(source, local)
            return
        with tempfile.TemporaryDirectory() as staging:
            local = Path(staging) / "download"
            await self.machine.download(source, local)
            await asyncio.to_thread(
                destination.upload_from,
                str(local) + "/" if directory else str(local),
                recursive=directory,
            )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._download(source_path, target_path, directory=False)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._download(source_dir, target_dir, directory=True)
