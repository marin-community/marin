"""Installed Harbor agents within the shared task rollout worker."""

import asyncio
import json
import shutil
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, aclosing, asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.literal.native_api import NativeAPILimits
from harbor.literal.proxy import RecordProxy
from harbor.models.agent.pi_trajectory import PiCollatedTrajectory
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig as TrialEnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rolloutengine.models import ModelTurn, Prompt, TaskRolloutSpec
from shellbox.machine import Machine, NetworkPolicy
from shellbox.image import DockerfileSource, RegistryImage, image_source_key
from taskcompendium.submission import conversation_messages

from marinskyrl.resource_locator import join_resource_path
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from skyrl_train.io import io
from skyrl_train.dataset.harbor_import import TaskConfig as PackageConfig, restore_harbor_trial_task
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_endpoint import registered_agent_endpoint
from skyrl_train.rollouts.agent_environment import BorrowedMachineEnvironment
from skyrl_train.rollouts.agent_service import serve_agent_app
from skyrl_train.rollouts.diagnostic_spool import DiagnosticSpool
from skyrl_train.rollouts.installed_agent import run_installed_agent
from skyrl_train.rollouts.pi_teacher_binding import PiTeacherBinding
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
    teacher: PiTeacherBinding | None = None
    output_path: str
    setup_timeout: float
    run_timeout: float
    cleanup_timeout: float
    service_startup_timeout: float
    service_shutdown_timeout: float
    environment_snapshots: dict[str, str] = Field(default_factory=dict)
    trial_reward_key: str = "reward"

    @model_validator(mode="after")
    def require_supported_pi_contract(self) -> "InstalledAgentSettings":
        """Reject unattributed teacher execution and unimplemented actor plans."""
        pi = self.agent.pi
        if pi is None or pi.mode == "literal":
            if self.teacher is not None:
                raise ValueError("Pi teacher binding cannot be used for student-only execution")
            return self
        if pi.mode != "teacher_trace" or self.teacher is None:
            raise ValueError("Pi teacher modes require per-trial teacher routing and hydrated evidence")
        if self.agent.model_name != f"hosted_vllm/{self.teacher.model}":
            raise ValueError("Pi teacher trace must address the bound independent teacher model")
        return self


def trial_agent_config(
    source: AgentConfig,
    *,
    api_base: str,
    model_name: str,
    max_input_tokens: int,
    max_output_tokens: int,
    trial_id: str,
    policy_step: int,
) -> AgentConfig:
    """Bind one installed Pi agent and immutable identity to the leased policy."""
    if not trial_id or type(policy_step) is not int or policy_step < 0:
        raise ValueError("Installed trial requires a valid identity and leased policy step")
    pi = source.pi
    if pi is not None:
        if (pi.trial_id is not None and pi.trial_id != trial_id) or (
            pi.policy_step is not None and pi.policy_step != policy_step
        ):
            raise ValueError("Pi trial identity differs from the policy-step lease")
        pi = pi.model_copy(update={"trial_id": trial_id, "policy_step": policy_step})
    config = source.model_copy(
        update={
            "pi": pi,
            "model_name": source.model_name or f"hosted_vllm/{model_name}",
            "model_alias": f"hosted_vllm/{model_name}",
            "kwargs": {
                **source.kwargs,
                "api_base": api_base,
                "api_key": "capability-url",
                **({"collect_rollout_details": True} if source.pi is None else {}),
                "model_info": {"max_input_tokens": max_input_tokens, "max_output_tokens": max_output_tokens},
            },
        }
    )
    return AgentConfig.model_validate(config.model_dump())


class InstalledAgentRuntime:
    def __init__(
        self,
        settings: InstalledAgentSettings,
        backend: AgentInferenceBackend,
        max_output_tokens: int,
        *,
        owned_teacher_transport: Any | None = None,
        endpoint_factory: Callable[..., AbstractAsyncContextManager[Any]] | None = None,
    ):
        self.settings = settings
        if settings.teacher is not None and (
            backend.model_name != settings.teacher.model
            or backend.model_revision != settings.teacher.revision
            or backend.source_kind != settings.teacher.source
            or backend.model_endpoint != settings.teacher.endpoint
        ):
            raise ValueError(
                "Pi teacher backend source, model or revision differs from the independent teacher binding"
            )
        self.backend = backend
        self.endpoint_factory = registered_agent_endpoint if endpoint_factory is None else endpoint_factory
        self.limits = NativeAPILimits(backend.model_name, backend.max_model_len, max_output_tokens)
        self.completed_pi_trajectories: dict[str, dict[str, Any]] = {}
        self.resources = AsyncExitStack()
        if owned_teacher_transport is not None:
            self.resources.push_async_callback(owned_teacher_transport.close)
        self.port: int | None = None

    @property
    def student_only(self) -> bool:
        pi = self.settings.agent.pi
        return pi is not None and pi.mode == "literal"

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

    def take_pi_trajectory(self, session_id: str) -> dict[str, Any] | None:
        """Transfer one validated in-memory capture to the rollout worker."""
        return self.completed_pi_trajectories.pop(session_id, None)

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
        policy_step: int = 0,
    ) -> None:
        if self.port is None or machine is None:
            raise RuntimeError("Installed agents require a started HTTP service and task machine")
        messages = conversation_messages(lowered.task.context.events)
        if len(messages) != 1 or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str):
            raise ValueError("Installed Harbor agents require one text task instruction")
        self.completed_pi_trajectories.pop(session_id, None)
        attempt_id = uuid4().hex
        destination = join_resource_path(self.settings.output_path, attempt_id)
        trial_dir = Path(tempfile.mkdtemp(prefix="skyrl-agent-"))
        (trial_dir / "task.json").write_text(lowered.model_dump_json(indent=2))
        (trial_dir / "identity.json").write_text(
            json.dumps({"session_id": session_id, "attempt_id": attempt_id, "task_id": lowered.task.id})
        )
        spool = (
            DiagnosticSpool(
                trial_dir / "engine" / "diagnostics.jsonl",
                join_resource_path(destination, "engine", "diagnostics.jsonl"),
            )
            if io.is_cloud_path(destination)
            else None
        )
        stop_spool = asyncio.Event()
        spool_task = asyncio.create_task(spool.run(stop_spool)) if spool is not None else None
        trial_task = asyncio.create_task(
            self._run_trial(
                machine,
                complete,
                lowered,
                trial_dir,
                messages[0]["content"],
                attempt_id,
                session_id,
                policy_step,
            )
        )
        primary_error: BaseException | None = None
        trial_observed = False
        try:
            if spool_task is not None:
                done, _ = await asyncio.wait({trial_task, spool_task}, return_when=asyncio.FIRST_COMPLETED)
                if trial_task not in done:
                    await spool_task  # Fail the attempt if its durable diagnostics cannot be published.
                    raise RuntimeError("Diagnostic spool stopped before the installed-agent attempt")
            trial_observed = True
            trajectory = await trial_task
            if trajectory is not None:
                validated = PiCollatedTrajectory.model_validate(trajectory)
                if validated.trial_id != session_id or validated.policy_step != policy_step:
                    raise ValueError("Pi capture differs from the leased trial identity")
                self.completed_pi_trajectories[session_id] = validated.model_dump(mode="json")
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cancelled_trial_error: BaseException | None = None
            if not trial_observed:
                if not trial_task.done():
                    trial_task.cancel()
                try:
                    await trial_task
                except asyncio.CancelledError:
                    pass  # Intentional cancellation after spool failure or caller cancellation.
                except Exception as error:  # noqa: BLE001 - re-raised after publishing failed-attempt artifacts
                    cancelled_trial_error = error
            spool_error: BaseException | None = None
            if spool_task is not None:
                stop_spool.set()
                try:
                    await spool_task
                except Exception as error:  # noqa: BLE001 - re-raised after final artifact upload
                    spool_error = error
            # Pi's fixed-schema phase journal is streamed separately. Never persist
            # raw sandbox logs or unchecked worker artifacts on an ordinary Pi run.
            if self.settings.agent.pi is None:
                await publish_attempt(trial_dir, destination)
            else:
                shutil.rmtree(trial_dir)
            if cancelled_trial_error is not None:
                if primary_error is not None:
                    primary_error.add_note(f"Cancelled trial also failed: {type(cancelled_trial_error).__name__}")
                    raise primary_error from cancelled_trial_error
                raise cancelled_trial_error
            if spool_error is not None and spool_error is not primary_error:
                if primary_error is not None:
                    primary_error.add_note(f"Diagnostic stream also failed: {type(spool_error).__name__}")
                    raise primary_error from spool_error
                raise spool_error

    def original_harness(
        self, lowered: TaskRolloutSpec, session_id: str, settings: HarborTaskSettings, phase: str, policy_step: int
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
            partial(
                self.run_original_trial, lowered=lowered, session_id=session_id, config=config, policy_step=policy_step
            ),
            self.settings.trial_reward_key,
        )

    async def run_original_trial(
        self,
        complete: Callable[[Prompt], Awaitable[ModelTurn]],
        *,
        lowered: TaskRolloutSpec,
        session_id: str,
        config: TrialConfig,
        policy_step: int,
    ) -> TrialResult:
        """Execute the original package with Harbor-owned environments and verification."""
        if self.port is None:
            raise RuntimeError("Harbor trials require a started HTTP service")
        self.completed_pi_trajectories.pop(session_id, None)
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
                trial_id=session_id,
                policy_step=policy_step,
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
                if agent_config.pi is not None:
                    trajectory = (
                        (result.agent_result.metadata or {}).get("pi_hydrated_trajectory")
                        if result.agent_result
                        else None
                    )
                    if trajectory is not None:
                        validated = PiCollatedTrajectory.model_validate(trajectory)
                        if validated.trial_id != session_id or validated.policy_step != policy_step:
                            raise ValueError("Pi capture differs from the leased trial identity")
                        self.completed_pi_trajectories[session_id] = validated.model_dump(mode="json")
                return result.model_copy(update={"trial_uri": join_resource_path(destination, "trials", attempt_id)})
        finally:
            if self.settings.agent.pi is None:
                await publish_attempt(evidence, destination)
            else:
                shutil.rmtree(evidence)

    @asynccontextmanager
    async def _served_agent(
        self,
        trial_dir: Path,
        run_timeout: float,
        *,
        capability_lifetime: float,
        agent: AgentConfig | None = None,
        trial_id: str | None = None,
        policy_step: int = 0,
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
            self.endpoint_factory(port, request_timeout=run_timeout) as endpoint,
        ):
            api_base = await endpoint.api_base(required_lifetime=capability_lifetime)
            source = self.settings.agent if agent is None else agent
            if trial_id is not None:
                yield trial_agent_config(
                    source,
                    api_base=api_base,
                    model_name=self.backend.model_name,
                    max_input_tokens=self.backend.max_model_len,
                    max_output_tokens=self.limits.max_output_tokens,
                    trial_id=trial_id,
                    policy_step=policy_step,
                )
                return
            if source.pi is not None and source.pi.mode == "teacher_trace":
                raise ValueError("Pi teacher trials require a bound policy-step identity")
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
        session_id: str,
        policy_step: int,
    ) -> dict[str, Any] | None:
        run_timeout = lowered.limits.total_turn_timeout or self.settings.run_timeout
        async with self._served_agent(
            trial_dir,
            run_timeout,
            capability_lifetime=self.settings.setup_timeout + run_timeout + self.settings.cleanup_timeout,
            trial_id=session_id,
            policy_step=policy_step,
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
                return await run_installed_agent(
                    agent,
                    environment,
                    self.backend,
                    complete,
                    instruction=instruction,
                    setup_timeout=self.settings.setup_timeout,
                    run_timeout=run_timeout,
                    cleanup_timeout=self.settings.cleanup_timeout,
                )
