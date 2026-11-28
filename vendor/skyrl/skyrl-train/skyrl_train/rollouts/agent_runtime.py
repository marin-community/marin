"""Installed Harbor agents within the shared task rollout worker."""

import asyncio
import json
import shutil
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.model_relay import JudgeAccess
from harbor.literal.native_api import NativeAPILimits
from harbor.literal.proxy import RecordProxy
from harbor.models.task.config import EnvironmentConfig, TaskConfig as PackageTaskConfig
from harbor.host_capabilities import (
    HostCapabilities,
    judge_model,
    model_api_url,
    model_guest_environment,
    task_model_environment,
)
from harbor.models.trial.config import AgentConfig, EnvironmentConfig as TrialEnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from shellbox.backends.daytona.connection import DaytonaConnection
from marinskyrl.harbor_environments import (
    HarborTaskBinding,
    apply_task_binding,
)
from marinskyrl.resource_locator import join_resource_path
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.inference_clock import remote_execution_watchdog, current_inference_clock
from rolloutengine.models import ModelTurn, Prompt
from rolloutengine.models import TaskRolloutSpec
from shellbox.machine import Machine, NetworkPolicy
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TaskSpec
from taskcompendium.submission import conversation_messages
from taskcompendium.runtime.resources import resource_bytes

from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from skyrl_train.io import io
from skyrl_train.dataset.harbor_import import restore_harbor_trial_task
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_endpoint import registered_agent_endpoint
from skyrl_train.rollouts.agent_environment import BorrowedMachineEnvironment
from skyrl_train.rollouts.agent_service import serve_agent_app
from skyrl_train.rollouts.installed_agent import run_installed_agent
from skyrl_train.rollouts.harbor_trial_session import HarborTrialHarness, run_harbor_trial
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.harbor_settings import EnvironmentType


async def publish_attempt(evidence: Path, destination: str) -> None:
    """Publish attempt evidence and retain the local copy if publication fails."""
    if io.is_cloud_path(destination):
        await asyncio.to_thread(io.upload_directory, str(evidence), destination)
    else:
        await asyncio.to_thread(shutil.copytree, evidence, destination)
    shutil.rmtree(evidence)



def package_config(task: TaskSpec) -> PackageTaskConfig:
    """The original Harbor package configuration carried in the task's verifier resources."""
    return PackageTaskConfig.model_validate_toml(
        resource_bytes(next(resource for resource in task.resources.verifier if resource.path == "task.toml")).decode()
    )


def package_environment_requirements(task: TaskSpec) -> EnvironmentRequirements:
    """The Harbor package's declared [environment], as TaskSpec environment requirements.

    Shared-source packages (SWE-rebench V2.1@repaired and the like) resolve their prebuilt
    image once and deliberately declare the SAME docker image, command semantics and env
    vars in both the package task.toml and the TaskSpec, so the catalog carries them. The
    recording engine derives the trial environment from the run settings and this package
    config instead; the TaskSpec copies are declarations, not second sources of truth.
    """
    environment = package_config(task).environment
    return EnvironmentRequirements(
        capabilities=task.environment_requirements.capabilities,
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=environment.docker_image,
        environment_variables=dict(environment.env),
    )


def harbor_trial_engine_requirements(task: TaskSpec) -> EnvironmentRequirements:
    """Engine-visible environment requirements for a Harbor-trial task.

    Accepts the TaskSpec's declared requirements when they are IDENTICAL to the Harbor
    package's [environment] (setup commands and the working directory keep their own
    override paths); fails closed on any mismatch or extra requirement so a package and
    its TaskSpec can never disagree about what runs.
    """
    engine = task.environment_requirements.model_copy(update={"setup_commands": (), "working_directory": None})
    declared = package_environment_requirements(task)
    if engine != declared:
        raise ValueError(
            "TaskSpec environment requirements disagree with the Harbor package's declared "
            "[environment]; shared-source packages must declare the identical docker image, "
            "command semantics and env vars in both places "
            f"(TaskSpec: {engine.model_dump(exclude_none=True)}; "
            f"package: {declared.model_dump(exclude_none=True)})"
        )
    return engine


def trial_setup_commands(task: TaskSpec) -> tuple[str, ...]:
    """Setup the engine runs after Harbor's own: none when the package declares the identical setup.

    Harbor runs a package's ``[environment.setup]`` before ENVIRONMENT_READY, so repeating the
    same TaskSpec setup would run it twice (SWE-rebench shared-source packages re-clone and
    re-bootstrap their protected root). Any other TaskSpec setup still runs.
    """
    declared = task.environment_requirements.setup_commands
    setup = package_config(task).environment.setup
    return () if setup is not None and declared == (setup.command,) else declared

class InstalledAgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: AgentConfig
    output_path: str
    setup_timeout: float
    run_timeout: float
    cleanup_timeout: float
    service_startup_timeout: float
    service_shutdown_timeout: float
    task_bindings: dict[str, HarborTaskBinding] = Field(default_factory=dict)
    trial_reward_key: str = "reward"

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_snapshots(cls, values: object) -> object:
        if isinstance(values, Mapping) and "environment_snapshots" in values:
            raise ValueError(
                "environment_snapshots is retired; prepare task docker_image references "
                "and remove the legacy snapshot mapping before launching"
            )
        return values


@dataclass(frozen=True)
class OriginalTrialBinding:
    environment: TrialEnvironmentConfig
    host_capabilities: HostCapabilities
    verifier_environment: dict[str, str]
    guest_environment: dict[str, str]


class InstalledAgentRuntime:
    def __init__(
        self,
        settings: InstalledAgentSettings,
        backend: AgentInferenceBackend,
        max_output_tokens: int,
        *,
        daytona_connection: DaytonaConnection,
        eval_daytona_connection: DaytonaConnection,
    ):
        self.settings = settings
        self.backend = backend
        self.daytona_connection = daytona_connection
        self.eval_daytona_connection = eval_daytona_connection
        self.protected_values = frozenset(
            value for value in (daytona_connection.api_key, eval_daytona_connection.api_key) if value
        )
        # The runner budget covers the complete trajectory. The installed
        # agent's declared output limit bounds each individual model turn.
        turn_limit = settings.agent.kwargs.get("model_info", {}).get("max_output_tokens", max_output_tokens)
        self.limits = NativeAPILimits(backend.model_name, backend.max_model_len, min(max_output_tokens, turn_limit))
        self.resources = AsyncExitStack()
        self.port: int | None = None

    async def startup(self) -> None:
        set_global_state(self.backend, None)
        try:
            self.port = await self.resources.enter_async_context(
                serve_agent_app(
                    create_app(),
                    host="127.0.0.1",
                    startup_timeout=self.settings.service_startup_timeout,
                    shutdown_timeout=self.settings.service_shutdown_timeout,
                )
            )
        except BaseException:
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        try:
            await self.resources.aclose()
        finally:
            self.port = None
            set_global_state(None, None)

    def serving_agent_config(
        self, source: AgentConfig, api_base: str, *, transport_watchdog: float | None = None
    ) -> AgentConfig:
        """Bind an installed agent to the model relay without widening its turn limit."""
        return source.model_copy(
            update={
                "model_name": source.model_name or f"hosted_vllm/{self.backend.model_name}",
                "model_alias": f"hosted_vllm/{self.backend.model_name}",
                "kwargs": {
                    **source.kwargs,
                    "api_base": api_base,
                    "api_key": "EMPTY",
                    **({"http_idle_timeout": transport_watchdog} if source.name == "pi" and transport_watchdog is not None else {}),
                    "collect_rollout_details": True,
                    "model_info": {
                        **source.kwargs.get("model_info", {}),
                        "max_input_tokens": self.backend.max_model_len,
                        "max_output_tokens": self.limits.max_output_tokens,
                    },
                },
            }
        )

    def validate_agent(self) -> dict:
        """Construct the serving agent on CPU without starting a sandbox or relay."""
        config = self.serving_agent_config(self.settings.agent, "http://127.0.0.1:1/v1")
        with tempfile.TemporaryDirectory(prefix="skyrl-agent-preflight-") as logs:
            agent = AgentFactory.create_agent_from_config(config, logs_dir=Path(logs))
            if not isinstance(agent, BaseInstalledAgent) or agent.rollout_correlation_id is None:
                raise ValueError("Harbor training requires an installed agent with rollout capture")
        return {"agent": config.name, "model_info": config.kwargs["model_info"]}

    async def run(
        self,
        machine: Machine | None,
        complete: Callable[[Prompt], Awaitable[ModelTurn]],
        *,
        lowered: TaskRolloutSpec,
        session_id: str,
    ) -> None:
        if self.port is None or machine is None:
            raise RuntimeError("Installed agents require a started HTTP service and task machine")
        messages = conversation_messages(lowered.task.context.events)
        if len(messages) != 1 or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str):
            raise ValueError("Installed Harbor agents require one text task instruction")
        attempt_id = uuid4().hex
        destination = join_resource_path(self.settings.output_path, attempt_id)
        trial_dir = Path(tempfile.mkdtemp(prefix="skyrl-agent-"))
        (trial_dir / "task.json").write_text(lowered.model_dump_json(indent=2))
        (trial_dir / "identity.json").write_text(
            json.dumps({"session_id": session_id, "attempt_id": attempt_id, "task_id": lowered.task.id})
        )
        try:
            await self._run_trial(machine, complete, lowered, trial_dir, messages[0]["content"], attempt_id)
        finally:
            await publish_attempt(trial_dir, destination)

    def original_trial_binding(
        self, lowered: TaskRolloutSpec, settings: HarborTaskSettings, phase: str, verifier_environment: dict[str, str]
    ) -> OriginalTrialBinding:
        """Bind phase credentials and project declared verifier calls onto the host relay."""
        kwargs = dict(settings.environment.kwargs)
        kwargs.pop("api_key_env", None)
        kwargs.pop("eval_api_key_env", None)
        connection = self.eval_daytona_connection if phase == "eval" else self.daytona_connection
        environment_fields = settings.environment.model_dump(exclude={"eval_image_build", "eval_override_storage_mb"})
        if phase == "eval":
            if settings.environment.eval_image_build is not None:
                kwargs["image_build"] = settings.environment.eval_image_build
            if settings.environment.eval_override_storage_mb is not None:
                environment_fields["override_storage_mb"] = settings.environment.eval_override_storage_mb
        package = package_config(lowered.task)
        verifier_environment = {**package.verifier.env, **verifier_environment}
        authentication = {}
        protected = self.protected_values
        guest_environment = {}
        model = judge_model(verifier_environment)
        if model is not None:
            judge = lowered.judge_connection
            if judge is None:
                raise ValueError("A declared Harbor verifier model requires a remote judge connection")
            kwargs.update(model_endpoint=judge.base_url, model_name=model)
            authentication[judge.base_url] = judge.api_key
            protected = protected | frozenset({judge.api_key})
            verifier_environment = model_guest_environment(verifier_environment, model)
            if model_api_url(package.environment.env) is not None:
                kwargs["judge_access"] = JudgeAccess.RUNTIME.value
                guest_environment = task_model_environment(model)
        return OriginalTrialBinding(
            TrialEnvironmentConfig.model_validate({**environment_fields, "kwargs": kwargs}),
            HostCapabilities(daytona=connection, model_authentication=authentication, protected_values=protected),
            verifier_environment,
            guest_environment,
        )

    def original_harness(
        self, lowered: TaskRolloutSpec, session_id: str, settings: HarborTaskSettings, phase: str
    ) -> HarborTrialHarness:
        """Bind a full Harbor trial to the phase's deadlines and environment settings."""
        if lowered.limits.attempt_timeout is None:
            raise ValueError("Full Harbor trials require a finite attempt timeout for the inference capability")
        if settings.environment.type is not EnvironmentType.DAYTONA:
            raise ValueError(
                "Original Harbor trials require Shellbox Daytona; engine-owned Docker "
                "machines use the borrowed-machine harness"
            )
        agent = self.settings.agent.model_copy(
            update={
                "override_timeout_sec": lowered.limits.total_turn_timeout or self.settings.run_timeout,
            }
        )
        config = TrialConfig(
            task=TaskConfig(path="runtime-package"),
            agent=agent,
            environment=settings.environment.model_dump(exclude={"eval_image_build", "eval_override_storage_mb"}),
            verifier={**settings.verifier.model_dump(), "override_timeout_sec": lowered.limits.verifier_timeout},
            trial_attempt_timeout_sec=lowered.limits.attempt_timeout,
            trial_cleanup_grace_sec=lowered.limits.cleanup_timeout,
        )
        config = apply_task_binding(config, self.settings.task_bindings.get(lowered.task.id))
        binding = self.original_trial_binding(lowered, settings, phase, config.verifier.env)
        config = config.model_copy(
            update={
                "environment": binding.environment,
                "verifier": config.verifier.model_copy(update={"env": binding.verifier_environment}),
            }
        )

        async def run(complete: Callable[[Prompt], Awaitable[ModelTurn]]) -> TrialResult:
            return await self.run_original_trial(
                complete,
                lowered=lowered,
                session_id=session_id,
                config=config,
                host_capabilities=binding.host_capabilities,
                guest_environment=binding.guest_environment,
            )

        return HarborTrialHarness(run, self.settings.trial_reward_key)

    async def run_original_trial(
        self,
        complete: Callable[[Prompt], Awaitable[ModelTurn]],
        *,
        lowered: TaskRolloutSpec,
        session_id: str,
        config: TrialConfig,
        host_capabilities: HostCapabilities,
        guest_environment: dict[str, str],
    ) -> TrialResult:
        """Execute the original package with Harbor-owned environments and verification."""
        if self.port is None:
            raise RuntimeError("Harbor trials require a started HTTP service")
        attempt_id = uuid4().hex
        destination = join_resource_path(self.settings.output_path, attempt_id)
        evidence = Path(tempfile.mkdtemp(prefix="skyrl-harbor-"))
        (evidence / "identity.json").write_text(
            json.dumps({"session_id": session_id, "attempt_id": attempt_id, "task_id": lowered.task.id})
        )
        try:
            task_dir = evidence / "task"
            restore_harbor_trial_task(lowered.task, task_dir)
            binding = self.settings.task_bindings.get(lowered.task.id)
            environment = config.environment
            if binding is not None:
                environment = environment.model_copy(update=binding.environment_resources.model_dump(exclude_none=True))
            run_timeout = lowered.limits.total_turn_timeout or self.settings.run_timeout
            assert config.trial_attempt_timeout_sec is not None
            capability_budget = remote_execution_watchdog(config.trial_attempt_timeout_sec)
            assert capability_budget is not None
            async with self._served_agent(
                evidence,
                run_timeout,
                capability_lifetime=capability_budget + lowered.limits.cleanup_timeout,
                agent=config.agent,
            ) as agent_config:
                trial_config = TrialConfig.model_validate(
                    {
                        **config.model_dump(),
                        "task": TaskConfig(path=task_dir),
                        "trials_dir": str(evidence / "trials"),
                        "trial_name": attempt_id,
                        "agent": agent_config,
                        "environment": environment,
                        # model_dump() runs the sensitive-env redactor even in python
                        # mode, turning the relay placeholder OPENAI_API_KEY="EMPTY"
                        # into the literal "****". Harbor's task-environment boundary
                        # then rejects that literal before the verifier runs, so every
                        # judged trial fails as unscored infrastructure. Carry the
                        # in-memory verifier config through the rebuild instead.
                        "verifier": config.verifier,
                    }
                )
                trial = await Trial.create(
                    trial_config, host_capabilities=host_capabilities, guest_environment=guest_environment
                )
                result = await run_harbor_trial(
                    trial,
                    self.backend,
                    complete,
                    evidence / "engine",
                    setup_commands=trial_setup_commands(lowered.task),
                    setup_cwd=lowered.task.environment_requirements.working_directory,
                    setup_timeout=self.settings.setup_timeout,
                )
                return result.model_copy(update={"trial_uri": join_resource_path(destination, "trials", attempt_id)})
        finally:
            await publish_attempt(evidence, destination)

    @asynccontextmanager
    async def _served_agent(
        self, trial_dir: Path, run_timeout: float, *, capability_lifetime: float, agent: AgentConfig | None = None
    ) -> AsyncIterator[AgentConfig]:
        transport_watchdog = remote_execution_watchdog(run_timeout)
        assert transport_watchdog is not None
        proxy = RecordProxy(
            f"http://127.0.0.1:{self.port}", trial_dir / "engine" / "literal.jsonl", timeout=transport_watchdog
        )
        async with (
            aclosing(proxy),
            serve_agent_app(
                proxy.app(self.limits),
                host="0.0.0.0",
                startup_timeout=self.settings.service_startup_timeout,
                shutdown_timeout=self.settings.service_shutdown_timeout,
            ) as port,
            registered_agent_endpoint(port, request_timeout=transport_watchdog) as endpoint,
        ):
            api_base = await endpoint.api_base(required_lifetime=capability_lifetime)
            source = self.settings.agent if agent is None else agent
            yield self.serving_agent_config(
                source, api_base, transport_watchdog=transport_watchdog if current_inference_clock() is not None else None
            )

    async def _run_trial(
        self,
        machine: Machine,
        complete: Callable[[Prompt], Awaitable[ModelTurn]],
        lowered: TaskRolloutSpec,
        trial_dir: Path,
        instruction: str,
        attempt_id: str,
    ) -> None:
        run_timeout = lowered.limits.total_turn_timeout or self.settings.run_timeout
        capability_budget = remote_execution_watchdog(run_timeout)
        assert capability_budget is not None
        async with self._served_agent(
            trial_dir,
            run_timeout,
            capability_lifetime=self.settings.setup_timeout + capability_budget + self.settings.cleanup_timeout,
        ) as config:
            host_capabilities = HostCapabilities(protected_values=self.protected_values)
            agent = AgentFactory.create_agent_from_config(
                config, logs_dir=trial_dir / "agent", host_capabilities=host_capabilities, guest_environment={}
            )
            if not isinstance(agent, BaseInstalledAgent):
                raise ValueError("The selected agent must be an installed Harbor agent")
            environment = BorrowedMachineEnvironment(
                machine,
                trial_dir,
                attempt_id,
                EnvironmentConfig(allow_internet=lowered.task_machine.network == NetworkPolicy.ALLOW),
                host_capabilities=host_capabilities,
            )
            with environment.with_default_user(lowered.task_machine.user):
                await run_installed_agent(
                    agent,
                    environment,
                    self.backend,
                    complete,
                    instruction=instruction,
                    setup_timeout=self.settings.setup_timeout,
                    run_timeout=run_timeout,
                    cleanup_timeout=self.settings.cleanup_timeout,
                )
