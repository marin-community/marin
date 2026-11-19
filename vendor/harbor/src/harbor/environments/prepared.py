# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared import, execution and artifact lifecycle for prepared task images."""

import asyncio
import tempfile
from abc import abstractmethod
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass, replace
from pathlib import Path

from harbor.environments.artifacts import download_artifact
from harbor.environments.base import (
    BASE_ENVIRONMENT_ARGUMENTS,
    BaseEnvironment,
    ExecResult,
)
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
)
from harbor.environments.execution import execute_shellbox_command, root_command_user
from harbor.environments.model_relay import (
    JudgeAccess,
    ModelRelay,
    start_model_relays,
    task_model_relays,
    verifier_model_access,
)
from harbor.environments.docker_translate import import_prepared_task
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.paths import EnvironmentPaths
from shellbox.machine import Command, CommandSemantics, Machine
from shellbox.process_session import (
    acquire_owned,
    await_cleanup,
    finish_owned,
    own_cleanup,
)
from shellbox.task import Task, TaskFactory


@dataclass(frozen=True)
class _TrialExecutionRequirements:
    command_semantics: CommandSemantics = CommandSemantics.LINUX_PROCESS
    capabilities: tuple[str, ...] = ("retained_command_cancellation",)
    docker_image: str | None = None


_TRIAL_EXECUTION_REQUIREMENTS = _TrialExecutionRequirements()


class PreparedEnvironment(BaseEnvironment):
    """Keep task files available across agent execution and verification."""

    def __init__(
        self,
        *args,
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
                f"Unsupported prepared sandbox arguments: {sorted(unknown)}"
            )
        if kwargs.get("persistent_env"):
            raise ValueError(
                "Prepared sandbox does not accept driver environment variables"
            )
        self.machine: Machine | None = None
        self.execution: Task | None = None
        self._relay = AsyncExitStack()
        self._stopping: asyncio.Task[None] | None = None
        self._service_log_paths: tuple[tuple[str, str], ...] = ()
        self.model_endpoint = model_endpoint
        self.model_name = model_name
        self.judge_access = JudgeAccess(judge_access)
        self._judge: ModelRelay | None = None
        self.candidate_model_endpoint = candidate_model_endpoint
        self.candidate_model_name = candidate_model_name
        super().__init__(*args, **kwargs)
        self.factory: TaskFactory

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    @abstractmethod
    def _validate_task_resources(self) -> None: ...

    def _validate_definition(self) -> None:
        self._prepared_task = import_prepared_task(
            self.task_env_config,
            self.environment_dir,
            guest_environment=self.guest_environment,
            protected_values=self.host_capabilities.protected_values,
        )
        paths = EnvironmentPaths()
        log_targets = {
            str(path)
            for path in (
                paths.agent_dir,
                paths.verifier_dir,
                paths.artifacts_dir,
            )
        }
        if any(str(mount["target"]) not in log_targets for mount in self._mounts):
            raise ValueError("Prepared sandbox does not support custom mounts")
        self._validate_task_resources()

    def _command_user(self, user: str | int | None) -> str:
        return root_command_user(self._resolve_user(user))

    def _maybe_resolve_task_env(self) -> None:
        # Task and Compose values belong to the imported machine specification.
        # Reapplying task values per command would override Compose precedence.
        pass

    async def start(
        self, force_build: bool, *, service_tasks: asyncio.TaskGroup
    ) -> None:
        if force_build:
            raise ValueError(
                "Prepared sandbox uses prebuilt images; force_build is unsupported"
            )
        plan = self._prepared_task
        task_spec = replace(
            plan.task,
            containers=tuple(
                replace(
                    container,
                    machine=replace(
                        container.machine,
                        cpus=self._resource_limit_value(
                            "cpu", auto_mode=ResourceMode.LIMIT
                        ),
                        memory_mb=self._resource_limit_value(
                            "memory", auto_mode=ResourceMode.LIMIT
                        ),
                    ),
                )
                for container in plan.task.containers
            ),
        )
        judge, candidate = task_model_relays(
            model_endpoint=self.model_endpoint,
            model_name=self.model_name,
            candidate_model_endpoint=self.candidate_model_endpoint,
            candidate_model_name=self.candidate_model_name,
            authentication=self.host_capabilities.model_authentication,
            judge_access=self.judge_access,
            environments=(container.machine.env for container in task_spec.containers),
        )
        self._judge = judge
        self._stopping = None
        self.factory.check_environment(
            _TRIAL_EXECUTION_REQUIREMENTS,
            task_spec.primary_container.machine,
        )
        execution = await acquire_owned(
            asyncio.create_task(self.factory.create_task(task_spec))
        )
        self.execution = execution
        self.machine = execution.primary
        try:
            await start_model_relays(
                self.machine,
                self._relay,
                service_tasks,
                judge=judge if self.judge_access is JudgeAccess.RUNTIME else None,
                candidate=candidate,
            )
            await execution.wait_ready()
            for name, path in plan.service_log_paths:
                result = await execution.containers[name].run(
                    Command(
                        ("mkdir", "-p", path), cwd="/", user=self._command_user("root")
                    )
                )
                if result.exit_code:
                    raise RuntimeError("Failed to prepare service artifact directory")
            self._service_log_paths = plan.service_log_paths
            targets = self._mount_targets(writable_only=True)
            if targets:
                result = await self.exec(
                    self._ensure_dirs_command(targets, chmod=True), cwd="/"
                )
                if result.return_code:
                    raise RuntimeError(
                        f"Failed to prepare sandbox paths: {result.stderr or result.stdout}"
                    )
            await self._upload_environment_dir_after_start()
        except BaseException as error:
            try:
                await finish_owned(self._stop_task(error))
            except BaseException as cleanup_error:
                error.add_note(
                    f"Prepared sandbox cleanup failed ({type(cleanup_error).__name__})"
                )
            raise

    async def stop(self, delete: bool) -> None:
        await await_cleanup(self._stop_task())

    def _stop_task(self, primary: BaseException | None = None) -> asyncio.Task[None]:
        if self._stopping is None:
            self._stopping = own_cleanup(self._stop(primary))
        return self._stopping

    async def _stop(self, primary: BaseException | None = None) -> None:
        try:
            await self._relay.__aexit__(
                type(primary) if primary is not None else None,
                primary,
                primary.__traceback__ if primary is not None else None,
            )
        finally:
            execution = self.execution
            self.execution = self.machine = None
            self._service_log_paths = ()
            if execution is not None:
                await execution.close()

    async def collect_runtime_logs(self) -> None:
        if self.execution is None or self.machine is None:
            return
        for name, target in self._service_log_paths:
            with tempfile.TemporaryDirectory() as staging:
                local = Path(staging) / "logs"
                await self.execution.containers[name].download(target, local)
                await self.machine.upload(local, target)

    @property
    def requires_agent_quiescence_for_verifier(self) -> bool:
        return self._judge is not None and self.judge_access is JudgeAccess.VERIFIER

    def verifier_model_access(
        self, service_tasks: asyncio.TaskGroup
    ) -> AbstractAsyncContextManager[None]:
        if self.machine is None:
            raise RuntimeError("Prepared sandbox is not running")
        return verifier_model_access(
            self.machine, self._judge, self.judge_access, service_tasks
        )

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

    async def _download(
        self, source: str, target: Path | str, *, directory: bool
    ) -> None:
        if self.machine is None:
            raise RuntimeError("Prepared sandbox is not running")
        await download_artifact(self.machine, source, str(target), directory=directory)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._download(source_path, target_path, directory=False)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._download(source_dir, target_dir, directory=True)
