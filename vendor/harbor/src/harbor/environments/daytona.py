# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor's single-container adapter for Shellbox Daytona machines."""

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from pathlib import Path

from daytona import AsyncDaytona

from harbor.environments.base import (
    BASE_ENVIRONMENT_ARGUMENTS,
    BaseEnvironment,
    ExecResult,
)
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.docker_translate import COMPOSE_FILENAMES
from harbor.environments.artifacts import download_artifact
from harbor.environments.execution import execute_shellbox_command
from harbor.host_capabilities import HostCapabilities
from harbor.environments.model_relay import (
    JudgeAccess,
    ModelRelay,
    start_model_relays,
    task_model_relays,
    verifier_model_access,
)
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.paths import EnvironmentPaths
from shellbox.backends.daytona.machine import (
    DEFAULT_SANDBOX_TTL_MINUTES,
    DaytonaMachine,
    DaytonaMachineFactory,
    DaytonaNetworkPolicy,
)
from shellbox.backends.daytona.snapshot import (
    SnapshotRequest,
    snapshot_request_for_spec,
    validate_source,
)
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.backends.daytona.provider import PROVIDER_REQUEST_TIMEOUT, ProviderWaitPolicy, retry_provider_call
from functools import partial
from shellbox.machine import MachineSpec, NetworkPolicy, UnsupportedMachineSpec
from shellbox.process_session import (
    acquire_owned,
    await_cleanup,
    finish_owned,
    own_cleanup,
)


def task_snapshot_source(docker_image: str | None, environment_dir: Path) -> RegistryImage | DockerfileSource:
    """Select a Harbor task's snapshot source from its image or environment."""
    if docker_image:
        return RegistryImage(docker_image)
    dockerfile = environment_dir / "Dockerfile"
    if not dockerfile.is_file():
        raise UnsupportedMachineSpec("Daytona task requires a registry image or Dockerfile")
    return validate_source(DockerfileSource(environment_dir, dockerfile))


def machine_spec(task_env_config, environment_dir: Path, env: dict[str, str], *, cpus: int | None, memory_mb: int | None, storage_mb: int | None, gpus: int = 0) -> MachineSpec:
    """The shellbox spec for one task environment.

    Pure projection from the task's environment config, so measurement code
    can derive snapshot identities from the same inputs production uses.
    Source validation happens once per boundary: environment construction
    rejects unusable definitions, and snapshot preparation validates what it
    is about to build.
    """
    source = (
        RegistryImage(task_env_config.docker_image)
        if task_env_config.docker_image
        else DockerfileSource(environment_dir, environment_dir / "Dockerfile")
    )
    return MachineSpec(
        source=source,
        workdir=task_env_config.workdir or "",
        env=env or {},
        network=NetworkPolicy.ALLOW if task_env_config.allow_internet else NetworkPolicy.DENY,
        cpus=cpus,
        memory_mb=memory_mb,
        storage_mb=storage_mb,
        gpus=gpus,
    )


class DaytonaEnvironment(BaseEnvironment):
    """Allocate, execute and transfer exclusively through Shellbox's Daytona backend."""

    def __init__(
        self,
        *args,
        client_factory: Callable[[], AsyncDaytona] | None = None,
        network_policy: dict[str, str] | None = None,
        ttl_minutes: int = DEFAULT_SANDBOX_TTL_MINUTES,
        create_timeout: float = 600,
        model_endpoint: str | None = None,
        model_name: str | None = None,
        judge_access: JudgeAccess = JudgeAccess.VERIFIER,
        candidate_model_endpoint: str | None = None,
        candidate_model_name: str | None = None,
        **kwargs,
    ):
        unknown = kwargs.keys() - BASE_ENVIRONMENT_ARGUMENTS
        if unknown:
            raise ValueError(
                f"Unsupported Shellbox Daytona arguments: {sorted(unknown)}"
            )
        policy = None
        if network_policy is not None:
            policy = DaytonaNetworkPolicy.from_config(network_policy)
        if client_factory is None:
            capabilities: HostCapabilities = kwargs.get(
                "host_capabilities", HostCapabilities()
            )
            connection = capabilities.daytona
            # An absent connection keeps the environment constructible for
            # callers that only inspect the task definition, such as snapshot
            # preflight; the factory then fails when a sandbox is requested.
            client_factory = (
                connection.client_factory() if connection is not None else None
            )
        self.factory = DaytonaMachineFactory(
            client_factory,
            ttl_minutes=ttl_minutes,
            create_timeout=create_timeout,
            provider_wait_policy=ProviderWaitPolicy.JOB,
            network_policy=policy,
        )
        self.model_endpoint, self.model_name = model_endpoint, model_name
        self.judge_access = JudgeAccess(judge_access)
        self._judge: ModelRelay | None = None
        self.candidate_model_endpoint, self.candidate_model_name = (
            candidate_model_endpoint,
            candidate_model_name,
        )
        self.machine: DaytonaMachine | None = None
        self._relay = AsyncExitStack()
        self._stopping: asyncio.Task[None] | None = None
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
        if any((self.environment_dir / name).exists() for name in COMPOSE_FILENAMES):
            raise ValueError(
                "Shellbox Daytona supports single-container tasks; use local for Compose service groups"
            )
        task_snapshot_source(self.task_env_config.docker_image, self.environment_dir)
        paths = EnvironmentPaths()
        allowed = {
            str(path)
            for path in (
                paths.agent_dir,
                paths.verifier_dir,
                paths.artifacts_dir,
            )
        }
        if any(str(mount["target"]) not in allowed for mount in self._mounts):
            raise ValueError("Shellbox Daytona does not support custom host mounts")

    def _machine_spec(self) -> MachineSpec:
        return machine_spec(
            self.task_env_config,
            self.environment_dir,
            self._merge_env(None),
            cpus=self._effective_cpus,
            memory_mb=self._effective_memory_mb,
            storage_mb=self._effective_storage_mb,
            gpus=self._effective_gpus,
        )

    def snapshot_request(self) -> SnapshotRequest:
        """The snapshot that this task's sandbox starts from."""
        return snapshot_request_for_spec(self._machine_spec())

    async def start(
        self, force_build: bool, *, service_tasks: asyncio.TaskGroup
    ) -> None:
        if force_build:
            raise ValueError(
                "Shellbox snapshots are immutable; change the prepared image or build inputs"
            )
        spec = self._machine_spec()
        judge, candidate = task_model_relays(
            model_endpoint=self.model_endpoint,
            model_name=self.model_name,
            candidate_model_endpoint=self.candidate_model_endpoint,
            candidate_model_name=self.candidate_model_name,
            authentication=self.host_capabilities.model_authentication,
            judge_access=self.judge_access,
            environments=(spec.env,),
        )
        self._judge = judge
        self._stopping = None
        self.machine = await acquire_owned(
            asyncio.create_task(self.factory.create(spec))
        )
        try:
            await start_model_relays(
                self.machine,
                self._relay,
                service_tasks,
                judge=judge if self.judge_access is JudgeAccess.RUNTIME else None,
                candidate=candidate,
            )
            targets = self._mount_targets(writable_only=True)
            if targets:
                result = await self.exec_control(
                    self._ensure_dirs_command(targets, chmod=True), cwd="/", timeout_sec=PROVIDER_REQUEST_TIMEOUT, user="root"
                )
                if result.return_code:
                    raise RuntimeError("Failed to prepare Daytona artifact paths")
            await self._upload_environment_dir_after_start()
        except BaseException as error:
            try:
                await finish_owned(own_cleanup(self.stop(True)))
            except BaseException as cleanup_error:
                error.add_note(
                    f"Daytona sandbox cleanup failed ({type(cleanup_error).__name__})"
                )
            raise

    @property
    def requires_agent_quiescence_for_verifier(self) -> bool:
        return self._judge is not None and self.judge_access is JudgeAccess.VERIFIER

    def verifier_model_access(
        self, service_tasks: asyncio.TaskGroup
    ) -> AbstractAsyncContextManager[None]:
        if self.machine is None:
            raise RuntimeError("Daytona environment is not running")
        return verifier_model_access(
            self.machine, self._judge, self.judge_access, service_tasks
        )

    async def stop(self, delete: bool) -> None:
        if self._stopping is None:
            self._stopping = own_cleanup(self._stop())
        await await_cleanup(self._stopping)

    async def _stop(self) -> None:
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
        return await execute_shellbox_command(
            self.machine,
            command,
            cwd=cwd,
            env=self._merge_env(env) or {},
            timeout_sec=timeout_sec,
            user=str(effective_user) if effective_user is not None else None,
        )

    async def exec_control(self, command: str, *, timeout_sec: int | None = None, user: str | int | None = None, cwd: str | None = None) -> ExecResult:
        return await retry_provider_call(
            "sandbox.control_command", partial(self.exec, command, timeout_sec=PROVIDER_REQUEST_TIMEOUT, user=user, cwd=cwd),
            wait_policy=ProviderWaitPolicy.JOB,
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
        await download_artifact(self.machine, source, str(target), directory=directory)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._download(source_path, target_path, directory=False)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._download(source_dir, target_dir, directory=True)
