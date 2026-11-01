# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute prepared Harbor tasks in persistent private bubblewrap sandboxes."""

import asyncio
import tempfile
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path

from harbor.environments.artifacts import download_artifact
from harbor.environments.base import BASE_ENVIRONMENT_ARGUMENTS, BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities
from harbor.environments.execution import execute_shellbox_command
from harbor.environments.model_relay import MODEL_RELAY_PORT, ModelRelay
from harbor.environments.prepared_compose import import_prepared_task, validate_prepared_task
from harbor.environments.resource_policies import validate_local_resources
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.env import resolve_env_vars
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.http_service import serve_http
from shellbox.machine import Command, Machine
from shellbox.process_session import acquire_owned, await_cleanup, finish_owned, own_cleanup
from shellbox.task import Task


class LocalEnvironment(BaseEnvironment):
    """Use separate root filesystems and a private shared loopback namespace."""

    def __init__(
        self,
        *args,
        bwrap: str | None = None,
        umoci: str = "umoci",
        image_cache: str | None = None,
        registry_authfile: str | None = None,
        signature_policy: str | None = None,
        cgroup_parent: str | None = None,
        ttl: float = 3600,
        model_endpoint: str | None = None,
        model_name: str | None = None,
        verifier_model_endpoint: str | None = None,
        **kwargs,
    ):
        self.cgroup_parent = Path(cgroup_parent) if cgroup_parent is not None else None
        unknown = kwargs.keys() - BASE_ENVIRONMENT_ARGUMENTS
        if unknown:
            raise ValueError(f"Unsupported prepared sandbox arguments: {sorted(unknown)}")
        if kwargs.get("persistent_env"):
            raise ValueError("Prepared sandbox does not accept driver environment variables")
        self.machine: Machine | None = None
        self.execution: Task | None = None
        self._relay = AsyncExitStack()
        self._stopping: asyncio.Task[None] | None = None
        self._service_log_paths: tuple[tuple[str, str], ...] = ()
        self.model_endpoint = model_endpoint
        self.model_name = model_name
        self.verifier_model_endpoint = verifier_model_endpoint
        super().__init__(*args, **kwargs)
        self.factory = LocalMachineFactory(
            bwrap=Path(bwrap) if bwrap is not None else None,
            umoci=Path(umoci),
            image_cache=Path(image_cache) if image_cache is not None else None,
            authfile=Path(registry_authfile) if registry_authfile is not None else None,
            policy=Path(signature_policy) if signature_policy is not None else None,
            cgroup_parent=self.cgroup_parent,
            ttl=ttl,
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.LOCAL

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    def _validate_definition(self) -> None:
        validate_prepared_task(self.task_env_config, self.environment_dir)
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
            raise ValueError("Prepared sandbox does not support custom mounts")
        validate_local_resources(
            self.task_env_config,
            cpu_enforcement_policy=self._cpu_resource_mode,
            memory_enforcement_policy=self._memory_resource_mode,
            cgroup_parent=self.cgroup_parent,
        )

    def _command_user(self, user: str | int | None) -> str:
        selected = self._resolve_user(user)
        if selected is None or str(selected) in {"root", "0", "0:0"}:
            return "0"
        raise ValueError("Local prepared sandboxes support only root commands")

    def _maybe_resolve_task_env(self) -> None:
        self._persistent_env = {
            **resolve_env_vars(self.task_env_config.env),
            **self._persistent_env,
        }

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError("Prepared sandbox uses prebuilt images; force_build is unsupported")
        plan = import_prepared_task(self.task_env_config, self.environment_dir)
        task_spec = replace(
            plan.task,
            containers=tuple(
                replace(
                    container,
                    machine=replace(
                        container.machine,
                        cpus=self._resource_limit_value("cpu", auto_mode=ResourceMode.LIMIT),
                        memory_mb=self._resource_limit_value("memory", auto_mode=ResourceMode.LIMIT),
                    ),
                )
                for container in plan.task.containers
            ),
        )
        needs_model = any("OPENAI_BASE_URL" in container.machine.env for container in task_spec.containers)
        if (needs_model or self.verifier_model_endpoint is not None) and (
            self.model_endpoint is None or self.model_name is None
        ):
            raise ValueError(
                "Task model calls require an explicit hosted judge (--judge-model) or standalone host model endpoint"
            )
        if self.verifier_model_endpoint is not None and self.verifier_model_endpoint != self.model_endpoint:
            raise ValueError("Prepared sandbox supports a single host model endpoint for runtime and verifier calls")
        self._stopping = None
        execution = await acquire_owned(asyncio.create_task(self.factory.create_task(task_spec)))
        self.execution = execution
        self.machine = execution.primary
        try:
            if needs_model or self.verifier_model_endpoint is not None:
                await self._relay.enter_async_context(
                    serve_http(
                        self.machine,
                        ModelRelay(self.model_endpoint, self.model_name).handle,
                        port=MODEL_RELAY_PORT,
                    )
                )
            await execution.wait_ready()
            for name, path in plan.service_log_paths:
                result = await execution.containers[name].run(
                    Command(("mkdir", "-p", path), cwd="/", user=self._command_user("root"))
                )
                if result.exit_code:
                    raise RuntimeError("Failed to prepare service artifact directory")
            self._service_log_paths = plan.service_log_paths
            targets = self._mount_targets(writable_only=True)
            if targets:
                result = await self.exec(self._ensure_dirs_command(targets, chmod=True), cwd="/")
                if result.return_code:
                    raise RuntimeError(f"Failed to prepare sandbox paths: {result.stderr or result.stdout}")
            await self._upload_environment_dir_after_start()
        except BaseException as error:
            try:
                await finish_owned(own_cleanup(self.stop(True)))
            except BaseException as cleanup_error:
                error.add_note(f"Prepared sandbox cleanup failed ({type(cleanup_error).__name__})")
            raise

    async def stop(self, delete: bool) -> None:
        if self._stopping is None:
            self._stopping = own_cleanup(self._stop())
        await await_cleanup(self._stopping)

    async def _stop(self) -> None:
        try:
            await self._relay.aclose()
        finally:
            execution = self.execution
            self.execution = self.machine = None
            self._service_log_paths = ()
            if execution is not None:
                await execution.close()

    async def _sync_service_logs(self) -> None:
        if self.execution is None or self.machine is None:
            return
        for name, target in self._service_log_paths:
            with tempfile.TemporaryDirectory() as staging:
                local = Path(staging) / "logs"
                await self.execution.containers[name].download(target, local)
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
            raise RuntimeError("Prepared sandbox is not running")
        await self._sync_service_logs()
        return await execute_shellbox_command(
            self.machine,
            command,
            cwd=cwd,
            env=self._merge_env(env) or {},
            timeout_sec=timeout_sec,
            user=self._command_user(user),
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        if self.machine is None:
            raise RuntimeError("Prepared sandbox is not running")
        await self.machine.upload(Path(source_path), target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self.upload_file(source_dir, target_dir)

    async def _download(self, source: str, target: Path | str, *, directory: bool) -> None:
        if self.machine is None:
            raise RuntimeError("Prepared sandbox is not running")
        await self._sync_service_logs()
        await download_artifact(self.machine, source, str(target), directory=directory)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._download(source_path, target_path, directory=False)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._download(source_dir, target_dir, directory=True)
