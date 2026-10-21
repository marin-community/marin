"""TaskCompendium execution and projection into SkyRL training batches."""

import asyncio
import math
import os
import shutil
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, fields, replace
from functools import partial
from typing import Any, Protocol
from uuid import uuid4

from jinja2 import TemplateError
from daytona import AsyncDaytona, DaytonaConfig, DaytonaError
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, Machine, MachineFactory, NetworkPolicy
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.models import NoGrader
from verifyit.modes.grade_judge import JudgeConnection
from rolloutengine.harness.core import Harness
from rolloutengine.harness.external import AgentHarness
from rolloutengine.models import (
    GenerationLimitReached,
    GradedRollout,
    GradedRolloutInterrupted,
    ModelRequest,
    ModelTurn,
    RolloutContractError,
    RolloutFailure,
    RolloutOperation,
    Prompt,
    TaskRolloutSpec,
)
from rolloutengine.engine import RolloutEngine
from skyrl_gym.rollout_specs import is_shell_harness, shell_harness
from skyrl_train.config.utils import generation_context_limit
from skyrl_train.dataset.harbor_import import HARBOR_TRIAL_SESSION
from skyrl_gym.task_factories import EpisodeFactory, episode_factories, episode_harnesses
from skyrl_gym.task_records import fold_grades, grade_skipped, skipped_grade
from skyrl_gym.verification import VERIFIER_RUNTIME_ERROR

from skyrl_train.inference_engines.base import ChatContinuation, InferenceEngineInterface, InferenceEngineInput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.rollouts.buffer import RolloutTask, RolloutWriter, write_trajectory_batch
from skyrl_train.rollouts.group_grading import GROUP_GRADERS, GroupGrader, grade_groups
from skyrl_train.rollouts.harbor_tasks import (
    HARBOR_MACHINE_BACKEND,
    HARBOR_EVAL_MACHINE_BACKEND,
    HarborTaskSettings,
    harbor_grading_failure,
    shape_harbor_rollouts,
)
from skyrl_train.rollouts.harbor_settings import EnvironmentType
from skyrl_train.rollouts.machines import OwnedMachineFactory, TaskMachineError
from skyrl_train.rollouts.pi_scenario_etl import PiScenario, scenario_rows_from_config, training_rows_to_batch
from skyrl_train.rollouts.workers import WorkerShard, detached_config
from skyrl_train.rollouts.finalization import finalize_trajectory_batch, propagate_data_sources
from skyrl_train.rollout_observability import rollout_phase, rollout_wait
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink
from skyrl_train.trajectory_runners.model_clients import (
    DirectModelClient,
    GenerationBudgetExceededError,
    ModelClient,
    ModelServerError,
)
from skyrl_train.rollouts.task_projections import (
    StepTaskProjection,
    WholeTaskProjection,
    logprobs_requested,
)
from skyrl_train.rollouts.training_record import TrainingRollout, training_rollout
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig
from skyrl_train.trajectory_runners.types import (
    RESPONSE_POLICY_STEPS_KEY,
    ROUTED_EXPERTS_KEY,
    TokenProvenance,
    TrajectoryBatch,
    TrajectoryID,
    TaskRequestBatch,
)

DAYTONA_DIAGNOSTIC_CALLS = frozenset(
    {
        ("daytona._async.daytona", "create"),
        ("daytona._async.daytona", "_create"),
        ("daytona._async.daytona", "delete"),
        ("daytona._async.snapshot", "get"),
        ("daytona._async.snapshot", "create"),
        ("daytona._async.sandbox", "wait_for_sandbox_start"),
        ("daytona._async.process", "exec"),
        ("daytona._async.filesystem", "upload_file_stream"),
        ("daytona._async.filesystem", "download_file"),
        ("daytona_api_client_async.api.sandbox_api", "create_sandbox"),
        ("daytona_api_client_async.api.sandbox_api", "get_toolbox_proxy_url"),
        ("daytona_api_client_async.api.sandbox_api", "get_sandbox"),
    }
)
DAYTONA_RATE_LIMIT_HEADERS = frozenset(
    f"{prefix}-{throttler}"
    for prefix in ("x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset", "retry-after")
    for throttler in ("anonymous", "authenticated", "sandbox-create", "sandbox-lifecycle")
) | {"retry-after"}


class InstalledAgentWorker(Protocol):
    @property
    def student_only(self) -> bool: ...

    def take_pi_trajectory(self, session_id: str) -> dict[str, Any] | None: ...

    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    def original_harness(
        self, lowered: TaskRolloutSpec, session_id: str, settings: HarborTaskSettings, phase: str, policy_step: int
    ) -> Harness: ...

    async def run(
        self,
        machine: Machine | None,
        complete: Callable[[Prompt], Awaitable[ModelTurn]],
        *,
        lowered: TaskRolloutSpec,
        session_id: str,
        policy_step: int,
    ) -> None: ...


@dataclass(frozen=True)
class PiTrainingRollout(TrainingRollout):
    """A graded rollout with a separately validated Pi in-memory capture."""

    pi_trajectory: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""


SKYRL_AGENT_HARNESS = "skyrl_agent"


def _pi_training_payload(payload: dict[str, Any] | None, result: TrainingRollout) -> dict[str, Any]:
    """Reject failed trials without conflating failed grading and absent Pi capture."""
    if result.failure is not None:
        failure_type = result.failure.exception_type
        if not failure_type.isidentifier() or len(failure_type) > 80:
            failure_type = "InvalidFailureType"
        raise ValueError(
            f"Installed Pi trial failed before admission: {failure_type} "
            f"(grade_status={result.grade.status.value}, capture_present={payload is not None})"
        )
    if payload is None:
        raise ValueError("Installed Pi trial did not return a validated in-memory trajectory")
    return payload


async def _model_turn(
    client: ModelClient,
    request: ModelRequest,
    *,
    sampling_params: dict[str, Any],
    max_context_length: int,
    max_prompt_length: int | None,
    chat_template_kwargs: dict[str, Any],
    session_id: str,
) -> ModelTurn:
    """Convert an inference response to the common exact-token record."""
    options = dict(request.options)
    template_kwargs = {**chat_template_kwargs, **options.get("chat_template_kwargs", {})}
    if template_kwargs and request.prompt_token_ids is None:
        options["chat_template_kwargs"] = template_kwargs
    continuation = None
    if request.continuation is not None:
        continuation = ChatContinuation(
            served_prefix_token_ids=list(request.continuation.served_token_ids),
            assistant_message_index=request.continuation.assistant_message_index,
        )
    try:
        with rollout_wait("model_client_await"):
            if request.prompt_token_ids is not None:
                prompt_ids = list(request.prompt_token_ids)
                remaining = max_context_length - len(prompt_ids)
                if remaining <= 0 or (max_prompt_length is not None and len(prompt_ids) > max_prompt_length):
                    raise GenerationLimitReached(tuple(prompt_ids))
                raw_sampling = {**sampling_params, **options}
                raw_sampling["max_tokens"] = min(int(raw_sampling.get("max_tokens", remaining)), remaining)
                model_input = InferenceEngineInput(
                    prompt_token_ids=[prompt_ids],
                    session_ids=[session_id],
                    sampling_params=raw_sampling,
                )
            else:
                model_input = InferenceEngineInput(
                    prompts=[list(request.messages)],
                    chat_completion_params=[options],
                    chat_continuations=[continuation],
                    session_ids=[session_id],
                    sampling_params=sampling_params,
                    max_context_length=max_context_length,
                    max_prompt_length=max_prompt_length,
                )
            output = await client.generate(model_input)
    except GenerationBudgetExceededError as limit:
        raise GenerationLimitReached(limit.prompt_token_ids) from limit
    if output["token_provenance"] != TokenProvenance.ENGINE:
        raise RolloutContractError("The canonical rollout engine requires exact model tokens")
    logprobs = output.get("response_logprobs")
    if logprobs is not None and any(not math.isfinite(value) for value in logprobs[0]):
        raise RolloutContractError("Behavior logprobs must be finite")
    message = (
        {"role": "assistant", "content": output["responses"][0]}
        if request.prompt_token_ids is not None
        else output["assistant_messages"][0]
    )
    return ModelTurn(
        message=message,
        prompt_token_ids=request.prompt_token_ids
        if request.prompt_token_ids is not None
        else tuple(output["prompt_ids"][0]),
        response_token_ids=tuple(output["response_ids"][0]),
        logprobs=None if logprobs is None else tuple(logprobs[0]),
        stop_reason=output["stop_reasons"][0],
        text=output.get("responses", [message.get("content") or ""])[0],
        metadata={
            **{
                name: output[name][0]
                for name in (
                    "student_topk_indices",
                    "behavior_topk_logprobs",
                    ROUTED_EXPERTS_KEY,
                    RESPONSE_POLICY_STEPS_KEY,
                )
                if output.get(name) is not None
            },
            "generation_token_budget": (output.get("generation_token_budgets") or [None])[0],
        },
    )


def _failed_rollout(
    interruption: GradedRolloutInterrupted,
    lowered: TaskRolloutSpec,
) -> TrainingRollout:
    """Retain verified turns and safe diagnostics for the training policy."""
    task = lowered.task
    error = interruption.__cause__
    assert isinstance(error, Exception)
    if isinstance(error, (TypeError, AttributeError, AssertionError, ImportError)):
        raise error
    if interruption.operation == RolloutOperation.MODEL and not isinstance(
        error, (ModelServerError, TimeoutError, ConnectionError, TemplateError)
    ):
        raise error
    exception_type = type(error).__name__
    diagnostics = {"operation": interruption.operation.value, "timed_out": isinstance(error, TimeoutError)}
    if isinstance(error, TaskMachineError) and error.__cause__ is not None:
        diagnostics["cause_error_type"] = type(error.__cause__).__name__
        diagnostics["machine_operation"] = error.operation.value
        provider_error = error.__cause__
        if isinstance(provider_error, DaytonaError):
            if type(provider_error.status_code) is int and 100 <= provider_error.status_code <= 599:
                diagnostics["provider_status_code"] = provider_error.status_code
            rate_limits = {
                name.lower(): int(value)
                for name, value in (provider_error.headers or {}).items()
                if name.lower() in DAYTONA_RATE_LIMIT_HEADERS
                and isinstance(value, str)
                and value.isascii()
                and value.isdecimal()
                and len(value) <= 12
            }
            if rate_limits:
                diagnostics["provider_rate_limits"] = rate_limits
            # This identifies an SDK call, not the HTTP phase of a compound operation.
            diagnostics["provider_sdk_call"] = "unknown"
            nested_error: BaseException | None = provider_error
            seen = set()
            while nested_error is not None and id(nested_error) not in seen:
                seen.add(id(nested_error))
                traceback = nested_error.__traceback__
                while traceback is not None:
                    frame = traceback.tb_frame
                    call = (frame.f_globals.get("__name__"), frame.f_code.co_name)
                    if call in DAYTONA_DIAGNOSTIC_CALLS:
                        diagnostics["provider_sdk_call"] = ".".join(call)
                    traceback = traceback.tb_next
                # The SDK translates HTTP errors with `raise ... from None`.
                nested_error = nested_error.__cause__ or nested_error.__context__
    if isinstance(error, TimeoutError):
        exception_type = {
            RolloutOperation.ATTEMPT: "TrialTimeoutError",
            RolloutOperation.START: "EnvironmentStartTimeoutError",
            RolloutOperation.PREPARE: "AgentSetupTimeoutError",
            RolloutOperation.GRADE: "VerifierTimeoutError",
        }.get(interruption.operation, "AgentTimeoutError")
    elif isinstance(error, TemplateError):
        exception_type = "TemplateError"
        diagnostics["cause_error_type"] = type(error).__name__
    elif isinstance(error, TaskMachineError):
        exception_type = "TaskMachineError"
    elif interruption.operation in {RolloutOperation.START, RolloutOperation.PREPARE}:
        exception_type = VERIFIER_RUNTIME_ERROR
        diagnostics.setdefault("cause_error_type", type(error).__name__)
    elif isinstance(error, ModelServerError):
        if error.category == "context_overflow":
            exception_type = "ContextLengthExceededError"
        diagnostics.update(error_category=error.category, request_id=error.request_id, status_code=error.status_code)
    if interruption.grade.failure == GradingFailure.TIMEOUT:
        # The engine may attempt verification after a model failure. The verifier
        # timeout determines whether pass-through can retain a scored rollout.
        exception_type = "VerifierTimeoutError"
    rollout = training_rollout(GradedRollout(interruption.rollout, interruption.grade), lowered)
    if (
        interruption.operation == RolloutOperation.ADVANCE
        and not is_shell_harness(lowered.harness)
        and rollout.steps
        and rollout.steps[-1].transition.metrics.get("advance_incomplete")
    ):
        # A native grader failure cannot supply credit for its incomplete turn.
        steps = rollout.steps[:-1]
        end = steps[-1].response_end + 1 if steps else 0
        rollout = replace(
            rollout,
            response_token_ids=rollout.response_token_ids[:end],
            loss_mask=rollout.loss_mask[:end],
            logprobs=None if rollout.logprobs is None else rollout.logprobs[:end],
            messages=steps[-1].messages if steps else (),
            steps=steps,
        )
    grade = rollout.grade
    if grade.status == Outcome.UNAVAILABLE and interruption.operation != RolloutOperation.GRADE and rollout.steps:
        grade = fold_grades([step.transition.grade for step in rollout.steps if step.transition.grade is not None])
    if cleanup_errors := interruption.grade.diagnostics.get("cleanup_errors"):
        grade = replace(grade, diagnostics={**grade.diagnostics, "cleanup_errors": cleanup_errors})
    logger.warning("Task {} interrupted during {}: {}", task.id, interruption.operation, exception_type)
    return replace(
        rollout,
        grade=grade,
        stop_reason="error",
        failure=RolloutFailure(exception_type, diagnostics),
    )


@dataclass(frozen=True)
class RemoteJudgeSettings:
    base_url: str
    api_key_env: str

    def connection(self) -> JudgeConnection:
        """Resolve the judge credential on the rollout worker, outside task data."""
        return JudgeConnection(self.base_url, os.environ[self.api_key_env])


class TaskRolloutWorker:
    """Run canonical task rollouts and submit complete groups to the buffer."""

    def __init__(
        self,
        trajectory_runner_cfg: DictConfig,
        projection: WholeTaskProjection | StepTaskProjection,
        model_client: ModelClient,
        factories: Mapping[str, MachineFactory],
        max_verifier_workers: int = 0,
        harbor: HarborTaskSettings | None = None,
        concurrent_tasks: int | None = None,
        concurrent_harbor_tasks: int | None = None,
        retry_wait: Callable[[float], Awaitable[None]] = asyncio.sleep,
        group_graders: Mapping[str, GroupGrader] = GROUP_GRADERS,
        sessions: Mapping[str, EpisodeFactory] | None = None,
        *,
        shutdown_timeout: float,
        installed_agents: InstalledAgentWorker | None = None,
        judge_connection: JudgeConnection | None = None,
        pi_scenario: Mapping[str, Any] | None = None,
    ):
        self.trajectory_runner_cfg = trajectory_runner_cfg
        self.model_client = model_client
        self.judge_connection = judge_connection
        self.factories = {name: OwnedMachineFactory(factory, name) for name, factory in factories.items()}
        self.active_requests: set[asyncio.Task] = set()
        self.closing = False
        self.shutdown_timeout = shutdown_timeout
        self.chat_template_kwargs = dict(trajectory_runner_cfg.get("chat_template_kwargs", {}))
        self.harbor = harbor
        self.installed_agents = installed_agents
        self.pi_scenario = None if pi_scenario is None else PiScenario.from_config(pi_scenario)
        self.retry_wait = retry_wait
        self.group_graders = group_graders
        self.task_slots = asyncio.Semaphore(concurrent_tasks) if concurrent_tasks is not None else nullcontext()
        self.harbor_task_slots = (
            nullcontext()
            if harbor is None
            else asyncio.Semaphore(
                harbor.concurrent_trials if concurrent_harbor_tasks is None else concurrent_harbor_tasks
            )
        )
        self.harbor_eval_slots = nullcontext() if harbor is None else asyncio.Semaphore(harbor.concurrent_trials)
        self.error_handling = ErrorHandlingConfig.from_mapping(trajectory_runner_cfg.get("error_handling", {}))
        self.projection = replace(
            projection,
            error_handling=self.error_handling,
            harbor_error_handling=None if harbor is None else harbor.error_handling,
        )
        self.verifier_executor = (
            ThreadPoolExecutor(max_workers=max_verifier_workers, thread_name_prefix="task-verifier")
            if max_verifier_workers > 0
            else None
        )
        self.sessions = episode_factories(executor=self.verifier_executor)
        if sessions is not None:
            self.sessions.update(sessions)
        self.trajectory_sink: RetentionSink | None = None

    @contextmanager
    def _active_request(self):
        if self.closing:
            raise RuntimeError("The rollout worker is closed")
        task = asyncio.current_task()
        assert task is not None
        if task in self.active_requests:
            yield
            return
        self.active_requests.add(task)
        try:
            yield
        finally:
            self.active_requests.discard(task)

    def _configured_task_and_harnesses(
        self, lowered: TaskRolloutSpec, harbor: HarborTaskSettings | None, session_id: str, phase: str, policy_step: int
    ) -> tuple[TaskRolloutSpec, Mapping[str, Harness]]:
        harnesses = episode_harnesses(lowered, self.sessions)
        if is_shell_harness(lowered.harness):
            harnesses[lowered.harness] = shell_harness(lowered.harness)
        if lowered.harness == HARBOR_TRIAL_SESSION:
            if harbor is None or self.installed_agents is None:
                raise ValueError("Full Harbor trials require Harbor and installed-agent settings")
            harnesses[HARBOR_TRIAL_SESSION] = self.installed_agents.original_harness(
                lowered, session_id, harbor, phase, policy_step
            )
        elif harbor is not None and self.installed_agents is not None:
            if not is_shell_harness(lowered.harness):
                raise ValueError("SkyRL agents require a Shellbox task grader")
            lowered = lowered.model_copy(update={"harness": SKYRL_AGENT_HARNESS})

            async def agent(context, complete):
                await self.installed_agents.run(
                    context.machine, complete, lowered=lowered, session_id=session_id, policy_step=policy_step
                )

            harnesses[SKYRL_AGENT_HARNESS] = AgentHarness(agent)
        return lowered, harnesses

    async def generate(self, request: TaskRequestBatch) -> list[TrainingRollout]:
        with self._active_request():
            return await self._generate(request)

    async def _generate(self, request: TaskRequestBatch) -> list[TrainingRollout]:
        lowered_tasks = [row.spec for row in request.rows]
        group_graders = [row.group_grader for row in request.rows]
        metadata = request.batch_metadata
        phase = "train" if metadata is None else metadata.training_phase
        if self.pi_scenario is not None and metadata is None:
            raise ValueError("Pi scenario requires a leased policy step before sampling")
        if self.pi_scenario is not None and phase == "eval":
            if self.installed_agents is None or not self.installed_agents.student_only:
                raise ValueError("Pi evaluation requires a student-only literal agent before sampling")
        harbor_settings = [self.harbor if "harbor" in item.task.tags else None for item in lowered_tasks]
        policies = [self.error_handling if harbor is None else harbor.error_handling for harbor in harbor_settings]
        lowered_tasks = [
            item if harbor is None else harbor.configured_spec(item, phase=phase)
            for item, harbor in zip(lowered_tasks, harbor_settings, strict=True)
        ]
        if self.pi_scenario is not None and any(
            item.task_machine is not None and item.task_machine.network is NetworkPolicy.DENY for item in lowered_tasks
        ):
            raise ValueError("Harbor Pi installation needs sandbox internet access for apt, nvm and npm")
        if self.pi_scenario is not None and (self.installed_agents is None or any(h is None for h in harbor_settings)):
            raise ValueError("Pi scenario requires Harbor-installed agents for every task")
        trajectory_ids = request.trajectory_ids
        session_ids = [
            trajectory_ids[index].to_string() if trajectory_ids is not None else uuid4().hex
            for index in range(len(lowered_tasks))
        ]
        selected_harnesses: list[Mapping[str, Harness]] = []
        for index, lowered in enumerate(lowered_tasks):
            lowered, harnesses = self._configured_task_and_harnesses(
                lowered,
                harbor_settings[index],
                session_ids[index],
                phase,
                0 if metadata is None else metadata.global_step,
            )
            lowered_tasks[index] = lowered.model_copy(update={"judge_connection": self.judge_connection})
            selected_harnesses.append(harnesses)
        engines = [RolloutEngine(self.factories, harnesses) for harnesses in selected_harnesses]
        for engine, lowered in zip(engines, lowered_tasks, strict=True):
            engine.check(lowered)
        tasks = [item.task for item in lowered_tasks]
        sampling = get_sampling_params_for_backend(
            self.trajectory_runner_cfg.backend, self.trajectory_runner_cfg.sampling_params
        )
        sampling.update(request.sampling_params or {})
        require_logprobs = logprobs_requested(request, self.trajectory_runner_cfg)
        max_input_length = int(self.trajectory_runner_cfg.max_input_length)
        max_context_length = generation_context_limit(self.trajectory_runner_cfg)

        async def run(index, lowered):
            task = lowered.task
            harbor = harbor_settings[index]
            session_id = session_ids[index]

            async def model(request):
                return await _model_turn(
                    self.model_client,
                    request,
                    sampling_params=sampling,
                    max_context_length=max_context_length,
                    max_prompt_length=max_input_length,
                    chat_template_kwargs=self.chat_template_kwargs,
                    session_id=session_id,
                )

            engine = engines[index]
            harbor_slots = (
                nullcontext()
                if harbor is None
                else self.harbor_eval_slots
                if phase == "eval"
                else self.harbor_task_slots
            )
            retries = 0
            while True:
                try:
                    async with harbor_slots, self.task_slots:
                        result = training_rollout(await engine.run_task(lowered, model, run_id=session_id), lowered)
                except GradedRolloutInterrupted as interruption:
                    result = _failed_rollout(interruption, lowered)
                if (
                    isinstance(task.grader, NoGrader)
                    and result.grade.status == Outcome.UNAVAILABLE
                    and result.failure is None
                    and result.steps
                    and not any(step.transition.grade is not None for step in result.steps)
                ):
                    result = replace(
                        result, grade=skipped_grade(task.grader.reason, diagnostics=result.grade.diagnostics)
                    )
                if harbor is None:
                    return result
                result = harbor_grading_failure(result)
                delay = harbor.retry_delay(result.failure, retries)
                if delay is None:
                    if self.pi_scenario is not None and phase != "eval":
                        assert self.installed_agents is not None
                        payload = _pi_training_payload(self.installed_agents.take_pi_trajectory(session_id), result)
                        result = PiTrainingRollout(
                            **{item.name: getattr(result, item.name) for item in fields(TrainingRollout)},
                            pi_trajectory=payload,
                            session_id=session_id,
                        )
                    return replace(result, metrics={**result.metrics, "rollout_retries": float(retries)})
                if self.pi_scenario is not None:
                    assert self.installed_agents is not None
                    self.installed_agents.take_pi_trajectory(session_id)
                logger.info("Task {} retry {} after {} seconds", task.id, retries + 1, delay)
                with rollout_wait("retry_backoff"):
                    await self.retry_wait(delay)
                retries += 1

        with rollout_phase("collect"):
            async with asyncio.TaskGroup() as group:
                pending = [group.create_task(run(index, item)) for index, item in enumerate(lowered_tasks)]
            rollouts = await grade_groups(
                tasks,
                group_graders,
                [task.result() for task in pending],
                [item.instance_id for item in trajectory_ids]
                if trajectory_ids is not None
                else [task.id for task in tasks],
                self.group_graders,
                phase,
                policies,
                logprobs_required=require_logprobs,
            )
            if self.harbor is not None:
                rollouts = shape_harbor_rollouts(
                    rollouts,
                    request,
                    self.harbor,
                    int(self.trajectory_runner_cfg.sampling_params.max_generate_length),
                    self.projection.tokenizer,
                    harbor_tasks=[harbor is not None for harbor in harbor_settings],
                )
        return rollouts

    async def run(self, input_batch: TaskRequestBatch) -> TrajectoryBatch:
        with self._active_request():
            outputs = await self.generate(input_batch)
            return await self.training_batch(input_batch, outputs)

    def etl_pi_trial(self, payload: Mapping[str, Any], *, trial_id: str, policy_step: int) -> TrajectoryBatch:
        """Project a complete Pi trial without reading model or artifact logs."""
        if self.pi_scenario is None:
            raise ValueError("Pi ETL requires a reviewed scenario.preset")
        rows = scenario_rows_from_config(
            {"preset": self.pi_scenario.value},
            payload,
            self.projection.tokenizer,
            expected_trial_id=trial_id,
            expected_policy_step=policy_step,
        )
        return training_rows_to_batch(rows)

    async def training_batch(self, request: TaskRequestBatch, outputs: Sequence[TrainingRollout]) -> TrajectoryBatch:
        with rollout_phase("assemble"):
            metadata = request.batch_metadata
            if self.pi_scenario is None or (metadata is not None and metadata.training_phase == "eval"):
                batch = self.projection.project(outputs, request)
            else:
                if metadata is None or not outputs:
                    raise ValueError("Pi training requires a policy-step lease and a completed trial")
                batches = []
                rejected_teacher_trials = 0
                identities = request.trajectory_ids
                if self.pi_scenario in (PiScenario.VERIFIER_GRPO, PiScenario.DISTRIBUTION) and identities is None:
                    raise ValueError("Pi per-turn student training requires physical trial IDs")
                expanded_ids = []
                for index, rollout in enumerate(outputs):
                    if not isinstance(rollout, PiTrainingRollout) or rollout.failure is not None:
                        raise ValueError("Pi training requires the validated, collated trial result")
                    if grade_skipped(rollout.grade):
                        if self.pi_scenario is not PiScenario.TEACHER_TRACE:
                            raise ValueError("Only unfiltered Pi teacher SFT can skip verification")
                        status = "skipped"
                    elif rollout.grade.status in (Outcome.GRADED, Outcome.SUBMISSION_FAILURE):
                        status = rollout.grade.status.value
                    else:
                        raise ValueError("Pi trial has no usable verifier outcome")
                    if status != "skipped" and rollout.grade.reward is None:
                        raise ValueError("Pi trial has no verifier score")
                    if self.pi_scenario is PiScenario.TEACHER_TRACE_VERIFIED and (
                        status != "graded" or rollout.grade.passed is None
                    ):
                        raise ValueError("Verified Pi teacher SFT needs an explicit graded pass/fail verdict")
                    payload = {
                        **rollout.pi_trajectory,
                        "judge": None,
                        "verifier_status": status,
                        "verifier_reward": None if status == "skipped" else rollout.grade.reward,
                        "verifier_passed": None if status == "skipped" else rollout.grade.passed,
                    }
                    segment = self.etl_pi_trial(payload, trial_id=rollout.session_id, policy_step=metadata.global_step)
                    if self.pi_scenario is PiScenario.TEACHER_TRACE_VERIFIED and rollout.grade.passed is False:
                        rejected_teacher_trials += 1
                    batches.append(segment)
                    if identities is not None:
                        if self.pi_scenario is PiScenario.DISTRIBUTION:
                            physical = identities[index]
                            expanded_ids.extend(
                                TrajectoryID(physical.instance_id, physical.repetition_id, turn_index=turn)
                                for turn in range(len(segment["response_ids"]))
                            )
                        else:
                            expanded_ids.extend([identities[index]] * len(segment["response_ids"]))
                if any(
                    (segment["rollout_logprobs"] is None) != (batches[0]["rollout_logprobs"] is None)
                    for segment in batches
                ):
                    raise ValueError("Pi training cannot mix student behavior and teacher SFT evidence")
                batch = (
                    batches[0]
                    if len(batches) == 1
                    else concatenate_trajectory_batches(batches, tis_lcs_alert_threshold=0.005)
                )
                if identities is not None:
                    batch["trajectory_ids"] = expanded_ids
                if self.pi_scenario is PiScenario.TEACHER_TRACE_VERIFIED:
                    batch["rollout_metrics"] = {
                        **(batch.get("rollout_metrics") or {}),
                        "pi/teacher_verified_trials": float(len(outputs)),
                        "pi/teacher_rejected_trials": float(rejected_teacher_trials),
                    }
            propagate_data_sources(request, batch)
        return await finalize_trajectory_batch(request, batch, self.trajectory_runner_cfg, self.trajectory_sink)

    def set_trajectory_sink(self, sink: RetentionSink) -> None:
        sink.bind_runner(type(self).__name__)
        self.trajectory_sink = sink

    async def startup(self) -> None:
        if self.installed_agents is not None:
            await self.installed_agents.startup()

    async def shutdown(self) -> None:
        """Cancel rollouts and release their machines before actor termination."""
        self.closing = True
        deadline = asyncio.get_running_loop().time() + self.shutdown_timeout
        for factory in self.factories.values():
            factory.stop_creating()
        pending = tuple(self.active_requests)
        for task in pending:
            task.cancel()
        failures = []
        try:
            if pending:
                _, remaining = await asyncio.wait(pending, timeout=self.shutdown_timeout / 2)
                if remaining:
                    logger.error("Task worker shutdown deadline expired with {} active rollouts", len(remaining))
                    failures.append(TimeoutError("Task rollouts did not stop before the shutdown deadline"))
            results = await asyncio.gather(
                *(factory.close(deadline) for factory in self.factories.values()), return_exceptions=True
            )
            failures.extend(result for result in results if isinstance(result, Exception))
        finally:
            if self.verifier_executor is not None:
                self.verifier_executor.shutdown(wait=False, cancel_futures=True)
            if self.installed_agents is not None:
                await self.installed_agents.shutdown()
        if failures:
            raise ExceptionGroup("Task worker shutdown failed", failures)

    async def start_eval_session(self, *, run_name: str, eval_step: int, val_set_name: str | None) -> None:
        pass

    async def stop_eval_session(self) -> None:
        pass

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        with self._active_request():
            rollouts = await self.generate(task.request)
            batch = await self.training_batch(task.request, rollouts)
            return await write_trajectory_batch(task, writer, batch)


@dataclass(frozen=True)
class TaskRolloutWorkerSpec:
    config: DictConfig
    engines: list[InferenceEngineInterface]
    sessions: dict[str, EpisodeFactory] = field(default_factory=dict)
    harbor_config: DictConfig | None = None
    teacher_engines: list[InferenceEngineInterface] | None = None
    teacher_tokenizer: Any | None = None

    @classmethod
    def from_config(
        cls,
        config,
        engines,
        *,
        harbor_config: DictConfig | None = None,
        sessions: Mapping[str, EpisodeFactory] | None = None,
        teacher_engines: Sequence[InferenceEngineInterface] | None = None,
        teacher_tokenizer: Any | None = None,
    ):
        return cls(
            detached_config(config),
            list(engines),
            {} if sessions is None else dict(sessions),
            None if harbor_config is None else detached_config(harbor_config),
            None if teacher_engines is None else list(teacher_engines),
            teacher_tokenizer,
        )

    def build(self, tokenizer, shard: WorkerShard) -> TaskRolloutWorker:
        judge_config = self.config.environment.task_sessions.get("remote_judge")
        judge = None if judge_config is None else RemoteJudgeSettings(**dict(judge_config)).connection()
        harbor = None if self.harbor_config is None else HarborTaskSettings.from_config(self.harbor_config)
        runner_config = self.config.generator
        client_config = OmegaConf.merge(self.config, {"generator": {"enable_http_endpoint": False}})
        inference = InferenceEngineClient(self.engines, tokenizer, client_config)
        client = DirectModelClient(inference)
        installed_agents = None
        if self.harbor_config is not None and self.harbor_config.get("installed_agent") is not None:
            # Full Harbor is an optional dependency for installed-agent runs.
            from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
            from skyrl_train.rollouts.agent_backend import AgentInferenceBackend

            settings = InstalledAgentSettings.model_validate(
                OmegaConf.to_container(self.harbor_config.installed_agent, resolve=True)
            )
            teacher = settings.teacher
            owned_teacher_transport = None
            if teacher is not None:
                from skyrl_train.inference_engines.pi_external_vllm import ExternalPiTeacherEngine

                if teacher.student_model != self.config.trainer.policy.model.path:
                    raise ValueError("Pi teacher binding must identify the configured student policy")
                if teacher.source == "skyrl_managed":
                    if not self.teacher_engines or self.teacher_tokenizer is None:
                        raise ValueError("SkyRL-managed Pi teacher needs independent teacher engines and tokenizer")
                    if any(
                        teacher_engine is student_engine
                        for teacher_engine in self.teacher_engines
                        for student_engine in self.engines
                    ):
                        raise ValueError("Pi teacher cannot reuse a student inference engine")
                    if any(engine.max_model_len != teacher.max_context_tokens for engine in self.teacher_engines):
                        raise ValueError("Pi teacher engine context limit differs from the bound model")
                    teacher_engines = self.teacher_engines
                    teacher_tokenizer = self.teacher_tokenizer
                else:
                    if self.teacher_engines is not None:
                        raise ValueError("External Pi teacher must not reuse a managed engine")
                    if self.teacher_tokenizer is None:
                        raise ValueError("External Pi teacher needs its own pinned tokenizer")
                    assert teacher.endpoint is not None and teacher.auth is not None
                    owned_teacher_transport = ExternalPiTeacherEngine(
                        endpoint=teacher.endpoint,
                        model=teacher.model,
                        auth=teacher.auth,
                        max_model_len=teacher.max_context_tokens,
                    )
                    teacher_engines = [owned_teacher_transport]
                    teacher_tokenizer = self.teacher_tokenizer
                teacher_config = OmegaConf.merge(
                    client_config, {"generator": {"engine_init_kwargs": {"served_model_name": teacher.model}}}
                )
                teacher_inference = InferenceEngineClient(teacher_engines, teacher_tokenizer, teacher_config)
                client = DirectModelClient(teacher_inference)
                backend = AgentInferenceBackend(
                    teacher_inference.model_name,
                    teacher.max_context_tokens,
                    teacher_inference.tokenize,
                    teacher_tokenizer,
                    model_revision=teacher.revision,
                    source_kind=teacher.source,
                    model_endpoint=teacher.endpoint,
                )
            else:
                backend = AgentInferenceBackend(
                    inference.model_name, generation_context_limit(runner_config), inference.tokenize, tokenizer
                )
            installed_agents = InstalledAgentRuntime(
                settings,
                backend,
                int(runner_config.sampling_params.max_generate_length),
                owned_teacher_transport=owned_teacher_transport,
            )
        factories: dict[str, MachineFactory] = {"shellsim": ShellSimMachineFactory()}
        daytona = self.config.environment.task_sessions.get("daytona")
        if daytona is not None:
            factories[Backend.DAYTONA] = DaytonaMachineFactory(
                client_factory=partial(AsyncDaytona, DaytonaConfig(api_key=os.environ[daytona.api_key_env])),
                ttl_minutes=int(daytona.ttl_minutes),
            )
        if shutil.which("docker") is not None:
            factories[Backend.DOCKER] = DockerMachineFactory()
        if harbor is not None:
            if harbor.environment.type != EnvironmentType.DOCKER:
                factories[HARBOR_MACHINE_BACKEND] = harbor.machine_factory()
                if harbor.environment.kwargs.get("eval_api_key_env"):
                    factories[HARBOR_EVAL_MACHINE_BACKEND] = harbor.machine_factory(phase="eval")
            elif Backend.DOCKER in factories:
                factories[HARBOR_MACHINE_BACKEND] = factories[Backend.DOCKER]
        return TaskRolloutWorker(
            runner_config,
            (
                StepTaskProjection(runner_config, tokenizer)
                if self.config.trainer.step_wise_training
                else WholeTaskProjection(runner_config, tokenizer)
            ),
            client,
            factories,
            max_verifier_workers=int(self.config.environment.task_sessions.max_verifier_workers),
            harbor=harbor,
            concurrent_tasks=int(self.config.trajectory_runner.max_concurrent_tasks),
            concurrent_harbor_tasks=None if harbor is None else max(1, harbor.concurrent_trials // shard.count),
            sessions=self.sessions,
            shutdown_timeout=float(self.config.trajectory_runner.shutdown_timeout),
            installed_agents=installed_agents,
            judge_connection=judge,
            pi_scenario=None
            if self.config.get("scenario") is None
            else OmegaConf.to_container(self.config.scenario, resolve=True),
        )
