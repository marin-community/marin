"""Installed Harbor agents within the shared task rollout worker."""

import asyncio
import json
import shutil
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from functools import partial
from pathlib import Path
from uuid import uuid4

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.literal.native_api import NativeAPILimits
from harbor.literal.proxy import RecordProxy
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig as TrialEnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from marinskyrl.resource_locator import join_resource_path
from pydantic import BaseModel, ConfigDict, Field
from rolloutengine.models import ModelTurn, Prompt
from rolloutengine.models import TaskRolloutSpec
from shellbox.machine import Machine, NetworkPolicy
from shellbox.image import DockerfileSource, RegistryImage, image_source_key
from taskcompendium.submission import conversation_messages

from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from skyrl_train.io import io
from skyrl_train.dataset.harbor_import import TaskConfig as PackageConfig, restore_harbor_trial_task
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_endpoint import registered_agent_endpoint
from skyrl_train.rollouts.agent_environment import BorrowedMachineEnvironment
from skyrl_train.rollouts.agent_service import serve_agent_app
from skyrl_train.rollouts.installed_agent import run_installed_agent
from skyrl_train.rollouts.harbor_trial_session import HarborTrialHarness, run_harbor_trial
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings


async def publish_attempt(evidence: Path, destination: str) -> None:
    """Publish attempt evidence and retain the local copy if publication fails."""
    if io.is_cloud_path(destination):
        await asyncio.to_thread(io.upload_directory, str(evidence), destination)
    else:
        await asyncio.to_thread(shutil.copytree, evidence, destination)
    shutil.rmtree(evidence)


class InstalledAgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: AgentConfig
    output_path: str
    setup_timeout: float
    run_timeout: float
    cleanup_timeout: float
    service_startup_timeout: float
    service_shutdown_timeout: float
    environment_snapshots: dict[str, str] = Field(default_factory=dict)
    trial_reward_key: str = "reward"


class InstalledAgentRuntime:
    def __init__(self, settings: InstalledAgentSettings, backend: AgentInferenceBackend, max_output_tokens: int):
        self.settings = settings
        self.backend = backend
        self.limits = NativeAPILimits(backend.model_name, backend.max_model_len, max_output_tokens)
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

    def original_harness(
        self, lowered: TaskRolloutSpec, session_id: str, settings: HarborTaskSettings, phase: str
    ) -> HarborTrialHarness:
        """Bind a full Harbor trial to the phase's deadlines and environment settings."""
        if lowered.limits.attempt_timeout is None:
            raise ValueError("Full Harbor trials require a finite attempt timeout for the inference capability")
        kwargs = dict(settings.environment.kwargs)
        key_env = kwargs.pop("eval_api_key_env" if phase == "eval" else "api_key_env", None)
        kwargs.pop("api_key_env", None)
        kwargs.pop("eval_api_key_env", None)
        if key_env is not None:
            kwargs["api_key_env"] = key_env
        environment = TrialEnvironmentConfig.model_validate({**settings.environment.model_dump(), "kwargs": kwargs})
        agent = self.settings.agent.model_copy(
            update={
                "override_timeout_sec": lowered.limits.total_turn_timeout or self.settings.run_timeout,
            }
        )
        config = TrialConfig(
            task=TaskConfig(path="runtime-package"),
            agent=agent,
            environment=environment,
            verifier={**settings.verifier.model_dump(), "override_timeout_sec": lowered.limits.verifier_timeout},
            trial_attempt_timeout_sec=lowered.limits.attempt_timeout,
            trial_cleanup_grace_sec=lowered.limits.cleanup_timeout,
        )
        return HarborTrialHarness(
            partial(self.run_original_trial, lowered=lowered, session_id=session_id, config=config),
            self.settings.trial_reward_key,
        )

    async def run_original_trial(
        self,
        complete: Callable[[Prompt], Awaitable[ModelTurn]],
        *,
        lowered: TaskRolloutSpec,
        session_id: str,
        config: TrialConfig,
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
            package = PackageConfig.model_validate_toml((task_dir / "task.toml").read_text())
            context = task_dir / "environment"
            source = (
                RegistryImage(package.environment.docker_image)
                if package.environment.docker_image
                else DockerfileSource(context, context / "Dockerfile")
            )
            key = image_source_key(source)
            if key not in self.settings.environment_snapshots:
                raise ValueError(f"Harbor environment has no approved snapshot: {key}")
            environment = config.environment.model_copy(
                update={
                    "force_build": False,
                    "kwargs": {
                        **config.environment.kwargs,
                        "snapshot_template_name": self.settings.environment_snapshots[key],
                        "auto_snapshot": False,
                        "assume_global_snapshot": True,
                    },
                }
            )
            run_timeout = lowered.limits.total_turn_timeout or self.settings.run_timeout
            assert config.trial_attempt_timeout_sec is not None
            async with self._served_agent(
                evidence,
                run_timeout,
                capability_lifetime=config.trial_attempt_timeout_sec + lowered.limits.cleanup_timeout,
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
                    }
                )
                trial = await Trial.create(trial_config)
                result = await run_harbor_trial(trial, self.backend, complete, evidence / "engine")
                return result.model_copy(update={"trial_uri": join_resource_path(destination, "trials", attempt_id)})
        finally:
            await publish_attempt(evidence, destination)

    @asynccontextmanager
    async def _served_agent(
        self, trial_dir: Path, run_timeout: float, *, capability_lifetime: float, agent: AgentConfig | None = None
    ) -> AsyncIterator[AgentConfig]:
        proxy = RecordProxy(
            f"http://127.0.0.1:{self.port}", trial_dir / "engine" / "literal.jsonl", timeout=run_timeout
        )
        async with (
            aclosing(proxy),
            serve_agent_app(
                proxy.app(self.limits),
                host="0.0.0.0",
                startup_timeout=self.settings.service_startup_timeout,
                shutdown_timeout=self.settings.service_shutdown_timeout,
            ) as port,
            registered_agent_endpoint(port, request_timeout=run_timeout) as endpoint,
        ):
            api_base = await endpoint.api_base(required_lifetime=capability_lifetime)
            source = self.settings.agent if agent is None else agent
            yield source.model_copy(
                update={
                    "model_name": source.model_name or f"hosted_vllm/{self.backend.model_name}",
                    "model_alias": f"hosted_vllm/{self.backend.model_name}",
                    "kwargs": {
                        **source.kwargs,
                        "api_base": api_base,
                        "api_key": "capability-url",
                        "collect_rollout_details": True,
                        "model_info": {
                            "max_input_tokens": self.backend.max_model_len,
                            "max_output_tokens": self.limits.max_output_tokens,
                        },
                    },
                }
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
        async with self._served_agent(
            trial_dir,
            run_timeout,
            capability_lifetime=self.settings.setup_timeout + run_timeout + self.settings.cleanup_timeout,
        ) as config:
            agent = AgentFactory.create_agent_from_config(config, logs_dir=trial_dir / "agent")
            if not isinstance(agent, BaseInstalledAgent):
                raise ValueError("The selected agent must be an installed Harbor agent")
            environment = BorrowedMachineEnvironment(
                machine,
                trial_dir,
                attempt_id,
                EnvironmentConfig(allow_internet=lowered.task_machine.network == NetworkPolicy.ALLOW),
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
