# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Harbor execution over prepared Shellbox machines and service groups."""

import asyncio
import tempfile
from contextlib import AsyncExitStack
from abc import abstractmethod
from dataclasses import replace
from pathlib import Path


from harbor.environments.base import (
    BASE_ENVIRONMENT_ARGUMENTS,
    BaseEnvironment,
    ExecResult,
)
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
)
from harbor.environments.prepared_compose import import_prepared_task, validate_prepared_task
from harbor.environments.artifacts import download_artifact
from harbor.environments.execution import execute_shellbox_command
from harbor.environments.model_relay import MODEL_RELAY_PORT, ModelRelay
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.env import resolve_env_vars

from shellbox.machine import Command, Machine, MachineSpec
from shellbox.service import ServiceGroup, ServiceGroupFactory
from shellbox.http_service import serve_http


class PreparedEnvironment(BaseEnvironment):
    """Execute prepared tasks through a credential-free Shellbox factory."""

    factory: ServiceGroupFactory

    def __init__(
        self,
        *args,
        model_endpoint: str | None = None,
        model_name: str | None = None,
        verifier_model_endpoint: str | None = None,
        **kwargs,
    ):
        unknown = kwargs.keys() - BASE_ENVIRONMENT_ARGUMENTS
        if unknown:
            raise ValueError(f"Unsupported prepared sandbox arguments: {sorted(unknown)}")
        if kwargs.get("persistent_env"):
            raise ValueError("Prepared sandbox does not accept driver environment variables")
        self.machine: Machine | None = None
        self.group: ServiceGroup | None = None
        self._relay = AsyncExitStack()
        self._service_log_paths: tuple[tuple[str, str], ...] = ()
        self.model_endpoint = model_endpoint
        self.model_name = model_name
        self.verifier_model_endpoint = verifier_model_endpoint
        super().__init__(*args, **kwargs)

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    @abstractmethod
    def _resource_spec(self, spec: MachineSpec) -> MachineSpec:
        """Apply this backend's declared resource policy to a group member."""
        ...

    def _command_user(self, user: str | int | None) -> str | None:
        selected = self._resolve_user(user)
        return str(selected) if selected is not None else None

    def _maybe_resolve_task_env(self) -> None:
        self._persistent_env = {
            **resolve_env_vars(self.task_env_config.env),
            **self._persistent_env,
        }

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

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError("Prepared sandbox uses prebuilt images; force_build is unsupported")
        plan = import_prepared_task(self.task_env_config, self.environment_dir)
        task_spec = self._resource_spec(plan.group.task)
        group_spec = replace(
            plan.group,
            task=task_spec,
            services=tuple(
                replace(service, machine=self._resource_spec(service.machine)) for service in plan.group.services
            ),
        )
        needs_model = any(
            "OPENAI_BASE_URL" in machine.env for machine in (task_spec, *(item.machine for item in group_spec.services))
        )
        if (needs_model or self.verifier_model_endpoint is not None) and (
            self.model_endpoint is None or self.model_name is None
        ):
            raise ValueError(
                "Task model calls require an explicit hosted judge (--judge-model) or standalone host model endpoint"
            )
        creation = asyncio.create_task(
            self.factory.create_group(group_spec) if group_spec.services else self.factory.create(task_spec)
        )
        try:
            created = await asyncio.shield(creation)
        except asyncio.CancelledError:
            # A backend may still acquire resources after cancellation. Own and
            # reclaim the late machine or group before leaving this operation.
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
            raise RuntimeError("Prepared sandbox is not running")
        await self._sync_service_logs()
        return await execute_shellbox_command(
            self.machine,
            ("sh", "-c", command),
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
