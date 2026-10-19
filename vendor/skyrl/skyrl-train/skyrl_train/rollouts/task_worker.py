"""TaskCompendium execution and projection into SkyRL training batches."""

import asyncio
import math
import os
import shutil
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
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
from shellbox.machine import Backend, Machine, MachineFactory
from taskcompendium.grading_result import Outcome
from taskcompendium.models import NoGrader
from verifyit.modes.grade_judge import JudgeConnection
from rolloutengine.agent_session import AgentTaskSession
from rolloutengine.contracts import (
    GenerationLimitReached,
    ModelRequest,
    ModelTurn,
    RolloutContractError,
    RolloutData,
    RolloutFailure,
    RolloutInterrupted,
    RolloutOperation,
    TaskSession,
    TaskSessionContext,
    TaskSessionFactory,
    SessionStart,
)
from rolloutengine.engine import RolloutEngine
from rolloutengine.spec import LoweredTaskSpec
from rolloutengine.lowering import SHELLBOX_SESSION, validate_lowered_task
from skyrl_train.config.utils import generation_context_limit
from skyrl_train.dataset.harbor_import import HARBOR_TRIAL_SESSION
from skyrl_gym.task_factories import adapt_self_grading_session, session_factories
from skyrl_gym.task_records import fold_grades, skipped_grade
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
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig
from skyrl_train.trajectory_runners.types import (
    RESPONSE_POLICY_STEPS_KEY,
    ROUTED_EXPERTS_KEY,
    TokenProvenance,
    TrajectoryBatch,
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
    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    def original_session(
        self, lowered: LoweredTaskSpec, session_id: str, settings: HarborTaskSettings, phase: str
    ) -> TaskSession: ...

    async def run(
        self,
        machine: Machine | None,
        complete: Callable[[SessionStart], Awaitable[ModelTurn]],
        *,
        lowered: LoweredTaskSpec,
        session_id: str,
    ) -> None: ...


SKYRL_AGENT_SESSION = "skyrl_agent"


def _agent_session_factory(
    runtime: InstalledAgentWorker,
    session_id: str,
    context: TaskSessionContext,
) -> TaskSession:
    runner = partial(runtime.run, context.machine, lowered=context.lowered, session_id=session_id)
    return AgentTaskSession(runner, context.require_task_grader())


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
    if request.assistant_message_index is not None:
        continuation = ChatContinuation(
            served_prefix_token_ids=list(request.prefix_token_ids),
            assistant_message_index=request.assistant_message_index,
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
    interruption: RolloutInterrupted,
    lowered: LoweredTaskSpec,
) -> RolloutData:
    """Retain verified turns and safe diagnostics for the training policy."""
    task = lowered.task
    error = interruption.__cause__
    assert isinstance(error, Exception)
    if isinstance(error, (TypeError, AttributeError, AssertionError)):
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
            RolloutOperation.CLEANUP: "CleanupTimeoutError",
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
    rollout = interruption.rollout
    if (
        interruption.operation == RolloutOperation.ADVANCE
        and lowered.session.task_session != SHELLBOX_SESSION
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
    if cleanup_errors := interruption.rollout.grade.diagnostics.get("cleanup_errors"):
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
        sessions: Mapping[str, Callable[[LoweredTaskSpec, Machine | None], TaskSession]] | None = None,
        *,
        shutdown_timeout: float,
        installed_agents: InstalledAgentWorker | None = None,
        judge_connection: JudgeConnection | None = None,
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
        self.sessions = session_factories(executor=self.verifier_executor)
        if sessions is not None:
            self.sessions.update({name: adapt_self_grading_session(factory) for name, factory in sessions.items()})
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

    def _configured_task_and_sessions(
        self, lowered: LoweredTaskSpec, harbor: HarborTaskSettings | None, session_id: str, phase: str
    ) -> tuple[LoweredTaskSpec, Mapping[str, TaskSessionFactory]]:
        sessions = self.sessions
        if lowered.session.task_session == HARBOR_TRIAL_SESSION:
            if harbor is None or self.installed_agents is None:
                raise ValueError("Full Harbor trials require Harbor and installed-agent settings")
            sessions = {
                **self.sessions,
                HARBOR_TRIAL_SESSION: lambda context: self.installed_agents.original_session(
                    context.lowered, session_id, harbor, phase
                ),
            }
        elif harbor is not None and self.installed_agents is not None:
            if lowered.session.task_session != SHELLBOX_SESSION:
                raise ValueError("SkyRL agents require a Shellbox task grader")
            lowered = lowered.model_copy(
                update={"session": lowered.session.model_copy(update={"task_session": SKYRL_AGENT_SESSION})}
            )
            sessions = {
                **self.sessions,
                SKYRL_AGENT_SESSION: partial(_agent_session_factory, self.installed_agents, session_id),
            }
        validate_lowered_task(lowered, factories=self.factories, sessions=sessions, providers={})
        return lowered, sessions

    async def generate(self, request: TaskRequestBatch) -> list[RolloutData]:
        with self._active_request():
            return await self._generate(request)

    async def _generate(self, request: TaskRequestBatch) -> list[RolloutData]:
        lowered_tasks = [row.lowered for row in request.rows]
        group_graders = [row.group_grader for row in request.rows]
        metadata = request.batch_metadata
        phase = "train" if metadata is None else metadata.training_phase
        harbor_settings = [self.harbor if "harbor" in item.task.tags else None for item in lowered_tasks]
        policies = [self.error_handling if harbor is None else harbor.error_handling for harbor in harbor_settings]
        lowered_tasks = [
            item if harbor is None else harbor.lowered(item, phase=phase)
            for item, harbor in zip(lowered_tasks, harbor_settings, strict=True)
        ]
        trajectory_ids = request.trajectory_ids
        session_ids = [
            trajectory_ids[index].to_string() if trajectory_ids is not None else uuid4().hex
            for index in range(len(lowered_tasks))
        ]
        selected_sessions: list[Mapping[str, TaskSessionFactory]] = []
        for index, lowered in enumerate(lowered_tasks):
            lowered, sessions = self._configured_task_and_sessions(
                lowered, harbor_settings[index], session_ids[index], phase
            )
            lowered_tasks[index] = lowered
            selected_sessions.append(sessions)
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

            engine = RolloutEngine(
                model,
                self.factories,
                sessions=selected_sessions[index],
                judge_connection=self.judge_connection,
            )
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
                        result = await engine.run(lowered)
                except RolloutInterrupted as interruption:
                    result = _failed_rollout(interruption, lowered)
                if (
                    isinstance(task.grader, NoGrader)
                    and result.grade.status == Outcome.UNAVAILABLE
                    and result.failure is None
                    and result.steps
                ):
                    result = replace(
                        result, grade=skipped_grade(task.grader.reason, diagnostics=result.grade.diagnostics)
                    )
                if harbor is None:
                    return result
                result = harbor_grading_failure(result)
                delay = harbor.retry_delay(result.failure, retries)
                if delay is None:
                    return replace(result, metrics={**result.metrics, "rollout_retries": float(retries)})
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

    async def training_batch(self, request: TaskRequestBatch, outputs: Sequence[RolloutData]) -> TrajectoryBatch:
        with rollout_phase("assemble"):
            batch = self.projection.project(outputs, request)
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
    sessions: dict[str, Callable[[LoweredTaskSpec, Machine | None], TaskSession]] = field(default_factory=dict)
    harbor_config: DictConfig | None = None

    @classmethod
    def from_config(
        cls,
        config,
        engines,
        *,
        harbor_config: DictConfig | None = None,
        sessions: Mapping[str, Callable[[LoweredTaskSpec, Machine | None], TaskSession]] | None = None,
    ):
        return cls(
            detached_config(config),
            list(engines),
            {} if sessions is None else dict(sessions),
            None if harbor_config is None else detached_config(harbor_config),
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
            installed_agents = InstalledAgentRuntime(
                settings,
                AgentInferenceBackend(
                    inference.model_name, generation_context_limit(runner_config), inference.tokenize, tokenizer
                ),
                int(runner_config.sampling_params.max_generate_length),
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
        )
