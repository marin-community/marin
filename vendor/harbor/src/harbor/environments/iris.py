# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a prebuilt Harbor task in a credential-free Iris sandbox."""

import asyncio
import tempfile
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path


from harbor.environments.base import (
    EXEC_OUTPUT_LIMIT_BYTES,
    BaseEnvironment,
    ExecResult,
)
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.iris_compose import import_iris_task
from harbor.environments.model_relay import MODEL_RELAY_PORT, ModelRelay
from harbor.models.task.config import EnvironmentConfig
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.paths import EnvironmentPaths
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.storage_path import StoragePath
from harbor.utils.env import resolve_env_vars

from shellbox.backends.iris.machine import (
    DEFAULT_JOB_TTL,
    DEFAULT_SCHEDULING_TIMEOUT,
    IrisMachine,
    IrisMachineFactory,
)
from shellbox.machine import Command, ExitReason
from shellbox.service import ServiceGroup
from shellbox.http_service import serve_http

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


def validate_iris_task(
    task_env_config: EnvironmentConfig, environment_dir: Path
) -> None:
    """Validate prepared images and translate task services before allocating a sandbox."""
    import_iris_task(task_env_config, environment_dir)


class IrisEnvironment(BaseEnvironment):
    """Execute one task in gVisor with Iris's NONE egress policy.

    Images must be built and pinned before evaluation. Model requests and result
    uploads belong to the external driver. Service groups allow only pod-local networking.
    """

    def __init__(
        self,
        *args,
        cluster: str | None = None,
        controller_url: str | None = None,
        scheduling_timeout: int = DEFAULT_SCHEDULING_TIMEOUT,
        job_ttl: int = DEFAULT_JOB_TTL,
        model_endpoint: str | None = None,
        model_name: str | None = None,
        verifier_model_endpoint: str | None = None,
        **kwargs,
    ):
        unknown = kwargs.keys() - BASE_ENVIRONMENT_ARGUMENTS
        if unknown:
            raise ValueError(f"Unsupported Iris sandbox arguments: {sorted(unknown)}")
        if kwargs.get("persistent_env"):
            raise ValueError(
                "Iris sandbox does not accept driver environment variables"
            )
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
        self.group: ServiceGroup | None = None
        self._relay = AsyncExitStack()
        self._service_log_paths: tuple[tuple[str, str], ...] = ()
        self.model_endpoint = model_endpoint
        self.model_name = model_name
        self.verifier_model_endpoint = verifier_model_endpoint
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.IRIS

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_request=True, memory_request=True)

    def _maybe_resolve_task_env(self) -> None:
        self._persistent_env = {
            **resolve_env_vars(self.task_env_config.env),
            **self._persistent_env,
        }

    def _validate_definition(self) -> None:
        validate_iris_task(self.task_env_config, self.environment_dir)
        paths = EnvironmentPaths()
        log_targets = {
            str(path)
            for path in (
                paths.agent_dir,
                paths.verifier_dir,
                paths.user_agent_dir,
                paths.artifacts_dir,
            )
        }
        if any(str(mount["target"]) not in log_targets for mount in self._mounts):
            raise ValueError("Iris sandbox does not support custom mounts")

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError(
                "Iris sandbox uses prebuilt images; force_build is unsupported"
            )
        plan = import_iris_task(self.task_env_config, self.environment_dir)
        task_spec = replace(
            plan.group.task,
            cpus=self._resource_request_value("cpu", auto_mode=ResourceMode.REQUEST),
            memory_mb=self._resource_request_value(
                "memory", auto_mode=ResourceMode.REQUEST
            ),
        )
        group_spec = replace(plan.group, task=task_spec)
        needs_model = any(
            "OPENAI_BASE_URL" in machine.env
            for machine in (task_spec, *(item.machine for item in group_spec.services))
        )
        if (needs_model or self.verifier_model_endpoint is not None) and (
            self.model_endpoint is None or self.model_name is None
        ):
            raise ValueError(
                "Task model calls require an explicit hosted judge (--judge-model) or standalone host model endpoint"
            )
        creation = asyncio.create_task(
            self.factory.create_group(group_spec)
            if group_spec.services
            else self.factory.create(task_spec)
        )
        try:
            created = await asyncio.shield(creation)
        except asyncio.CancelledError:
            # Iris submission runs in a thread; cancelling its waiter does not stop
            # the job being created. Reclaim it once submission has completed.
            machine = await creation
            await asyncio.shield(machine.close())
            raise
        if isinstance(created, ServiceGroup):
            self.group = created
            self.machine = created.task
        else:
            self.machine = created
        try:
            if needs_model or self.verifier_model_endpoint is not None:
                await self._relay.enter_async_context(
                    serve_http(
                        self.machine,
                        ModelRelay(self.model_endpoint, self.model_name).handle,
                        port=MODEL_RELAY_PORT,
                    )
                )
            if self.group is not None:
                await self.group.wait_ready()
                for name, path in plan.service_log_paths:
                    result = await self.group.services[name].run(
                        Command(("mkdir", "-p", path), cwd="/", user="root")
                    )
                    if result.exit_code:
                        raise RuntimeError(
                            "Failed to prepare service artifact directory"
                        )
            self._service_log_paths = plan.service_log_paths
            targets = self._mount_targets(writable_only=True)
            if targets:
                result = await self.exec(
                    self._ensure_dirs_command(targets, chmod=True), cwd="/"
                )
                if result.return_code:
                    raise RuntimeError(
                        f"Failed to prepare Iris sandbox paths: {result.stderr or result.stdout}"
                    )
            await self._upload_environment_dir_after_start()
        except BaseException:
            await asyncio.shield(self.stop(True))
            raise

    async def stop(self, delete: bool) -> None:
        try:
            await self._relay.aclose()
        finally:
            group, machine = self.group, self.machine
            self.group = self.machine = None
            self._service_log_paths = ()
            if group is not None:
                await group.close()
            elif machine is not None:
                await machine.close()

    async def _sync_service_logs(self) -> None:
        if self.group is None or self.machine is None:
            return
        for name, target in self._service_log_paths:
            with tempfile.TemporaryDirectory() as staging:
                local = Path(staging) / "logs"
                await self.group.services[name].download(target, local)
                await self.machine.upload(local, target)

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
        await self._sync_service_logs()
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
            raise TimeoutError(
                "Iris sandbox command timed out and the sandbox was terminated"
            )
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

    async def _download(
        self, source: str, target: Path | str, *, directory: bool
    ) -> None:
        if self.machine is None:
            raise RuntimeError("Iris sandbox is not running")
        await self._sync_service_logs()
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
            upload_source = str(local) + "/" if directory else str(local)
            await asyncio.to_thread(
                destination.upload_from, upload_source, recursive=directory
            )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._download(source_path, target_path, directory=False)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._download(source_dir, target_dir, directory=True)
