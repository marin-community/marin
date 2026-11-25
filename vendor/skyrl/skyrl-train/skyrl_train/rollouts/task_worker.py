"""TaskCompendium execution and projection into SkyRL training batches."""

import asyncio
import heapq
import math
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Any, Protocol
from uuid import uuid4

from jinja2 import TemplateError
from daytona import DaytonaError
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rigging.redaction import redact_string
from shellbox.machine import Backend, Machine, MachineFactory
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.models import EnvironmentRequirements, NoGrader, ReasoningMarkupPolicy
from verifyit.modes.grade_judge import JudgeConnection
from rolloutengine.harness.core import Harness
from rolloutengine.harness.external import AgentHarness
from rolloutengine.models import (
    GenerationLimitReached,
    ContextRejected,
    ContextWindowExceeded,
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
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS, AssistantPrefixHarness
from skyrl_gym.rollout_specs import is_shell_harness, shell_harness
from marinskyrl.task_machine_factories import task_machine_factories
from skyrl_train.config.utils import generation_context_limit
from skyrl_train.dataset.harbor_import import HARBOR_TRIAL_SESSION
from skyrl_gym.task_factories import EpisodeFactory, episode_factories, episode_harnesses
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
from skyrl_train.rollouts.harbor_trial_session import HarborTrialHarness
from skyrl_train.rollouts.machines import OwnedMachineFactory, TaskMachineError
from skyrl_train.rollouts.workers import WorkerShard, detached_config
from skyrl_train.rollouts.finalization import finalize_trajectory_batch, propagate_data_sources
from skyrl_train.rollout_observability import rollout_phase, rollout_wait
from skyrl_train.telemetry import record_event
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink
from skyrl_train.trajectory_runners.model_clients import (
    ContextLengthExceededError,
    REASONING_VIOLATION_TOKENS_KEY,
    REASONING_VIOLATIONS_KEY,
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
from skyrl_train.rollouts.training_record import (
    REASONING_VIOLATION_TOKEN_METADATA_KEY,
    TrainingRollout,
    reject_malformed_reasoning,
    training_rollout,
)
from skyrl_train.utils.harbor_errors import AGENT_TIMEOUT_ERROR, ErrorHandlingConfig, inherited_harbor_error_handling
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

    def original_harness(
        self, lowered: TaskRolloutSpec, session_id: str, settings: HarborTaskSettings, phase: str
    ) -> Harness: ...

    async def run(
        self,
        machine: Machine | None,
        complete: Callable[[Prompt], Awaitable[ModelTurn]],
        *,
        lowered: TaskRolloutSpec,
        session_id: str,
    ) -> None: ...


SKYRL_AGENT_HARNESS = "skyrl_agent"


async def _model_turn(
    client: ModelClient,
    request: ModelRequest,
    *,
    sampling_params: dict[str, Any],
    max_context_length: int,
    max_prompt_length: int | None,
    chat_template_kwargs: dict[str, Any],
    session_id: str,
    priority: int = 0,
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
            model_input["priority"] = priority
            output = await client.generate(model_input)
    except GenerationBudgetExceededError as error:
        raise GenerationLimitReached(error.prompt_token_ids) from error
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
            REASONING_VIOLATION_TOKEN_METADATA_KEY: (output.get(REASONING_VIOLATION_TOKENS_KEY) or [None])[0],
        },
        reasoning_violation=(output.get(REASONING_VIOLATIONS_KEY) or [None])[0],
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
    elif isinstance(error, ContextWindowExceeded):
        exception_type = "ContextLengthExceededError"
        diagnostics["error_category"] = "context_overflow"
    elif interruption.operation in {RolloutOperation.START, RolloutOperation.PREPARE}:
        exception_type = VERIFIER_RUNTIME_ERROR
        diagnostics.setdefault("cause_error_type", type(error).__name__)
    elif isinstance(error, ModelServerError):
        if error.category == "context_overflow":
            exception_type = "ContextLengthExceededError"
        diagnostics.update(
            error_category=error.category,
            request_id=error.request_id,
            status_code=error.status_code,
            message=str(error),
        )
    if interruption.grade.failure == GradingFailure.TIMEOUT:
        # The engine may attempt verification after a model failure. The verifier
        # timeout determines whether pass-through can retain a scored rollout.
        exception_type = "VerifierTimeoutError"
    rollout = training_rollout(GradedRollout(interruption.rollout, interruption.grade), lowered)
    if (
        interruption.operation == RolloutOperation.ADVANCE
        and not is_shell_harness(lowered.harness)
        and not (
            exception_type == AGENT_TIMEOUT_ERROR
            and rollout.grade.status in {Outcome.GRADED, Outcome.SUBMISSION_FAILURE}
        )
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
    detail = exception_type
    if isinstance(error, TaskMachineError) and error.__cause__ is not None:
        # Diagnostics carry only allowlisted fields. The cause's text can contain credentials, so log it redacted.
        detail = f"{exception_type} ({type(error.__cause__).__name__}: {redact_string(str(error.__cause__))})"
    logger.warning("Task {} interrupted during {}: {}", task.id, interruption.operation, detail)
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


class HarborSlots:
    """Grant available Harbor capacity to the oldest waiting lease batch first."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("Harbor slot capacity must be positive")
        self._available = capacity
        self._sequence = 0
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []

    def _grant(self) -> None:
        while self._available and self._waiters:
            _, _, future = heapq.heappop(self._waiters)
            if future.cancelled():
                continue
            self._available -= 1
            future.set_result(None)

    def _release(self) -> None:
        self._available += 1
        self._grant()

    @asynccontextmanager
    async def admit(self, batch_id: int) -> AsyncIterator[None]:
        future = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (batch_id, self._sequence, future))
        self._sequence += 1
        self._grant()
        blocked = not future.done()
        started = time.monotonic()
        fields = {"gate": "harbor_trials", "batch_id": batch_id}
        try:
            if blocked:
                record_event("rollout_admission_blocked", fields)
            await future
        except BaseException:
            if future.done() and not future.cancelled():
                self._release()
            else:
                future.cancel()
                self._grant()
            raise
        try:
            if blocked:
                record_event("rollout_admission_resumed", {**fields, "wait_seconds": time.monotonic() - started})
            yield
        finally:
            self._release()


@asynccontextmanager
async def admission_slot(gate: str, slots: asyncio.Semaphore | None) -> AsyncIterator[None]:
    """Record a saturated trajectory resource while preserving semaphore ownership."""
    if slots is None:
        yield
        return
    blocked = slots.locked()
    started = time.monotonic()
    if blocked:
        record_event("rollout_admission_blocked", {"gate": gate})
    async with slots:
        if blocked:
            record_event("rollout_admission_resumed", {"gate": gate, "wait_seconds": time.monotonic() - started})
        yield


@dataclass(frozen=True)
class PreparedTask:
    spec: TaskRolloutSpec
    harbor: HarborTaskSettings | None
    engine: RolloutEngine


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
        retrieval_service: str | None = None,
        eval_factories: Mapping[str, MachineFactory] | None = None,
        evaluation_priority: int = 0,
    ):
        self.trajectory_runner_cfg = trajectory_runner_cfg
        self.evaluation_priority = evaluation_priority
        self.model_client = model_client
        self.judge_connection = judge_connection
        self.retrieval_service = retrieval_service
        self.factories = {name: OwnedMachineFactory(factory, name) for name, factory in factories.items()}
        self.eval_factories = {
            name: OwnedMachineFactory(factory, name) for name, factory in (eval_factories or {}).items()
        }
        self.active_requests: set[asyncio.Task] = set()
        self.closing = False
        self.shutdown_timeout = shutdown_timeout
        self.chat_template_kwargs = dict(trajectory_runner_cfg.get("chat_template_kwargs", {}))
        self.harbor = harbor
        self.installed_agents = installed_agents
        self.retry_wait = retry_wait
        self.group_graders = group_graders
        self.task_slots = asyncio.Semaphore(concurrent_tasks) if concurrent_tasks is not None else None
        self.harbor_task_slots = (
            None
            if harbor is None
            else HarborSlots(
                harbor.concurrent_trials if concurrent_harbor_tasks is None else concurrent_harbor_tasks
            )
        )
        self.error_handling = ErrorHandlingConfig.from_mapping(trajectory_runner_cfg.get("error_handling", {}))
        # Harbor rows inherit the generator's declared classification policy for
        # fields its config omits, so a Harbor-tagged failure is scored under the
        # shared taxonomy instead of falling back to the unclassified defaults.
        self.harbor_error_handling = (
            None
            if harbor is None
            else inherited_harbor_error_handling(
                harbor.error_handling, harbor.declared_error_handling_fields, self.error_handling
            )
        )
        self.projection = replace(
            projection,
            error_handling=self.error_handling,
            harbor_error_handling=self.harbor_error_handling,
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
        self, lowered: TaskRolloutSpec, harbor: HarborTaskSettings | None, session_id: str, phase: str
    ) -> tuple[TaskRolloutSpec, Mapping[str, Harness]]:
        lowered = lowered.model_copy(update={"judge_connection": self.judge_connection})
        harnesses = episode_harnesses(lowered, self.sessions)
        if (
            lowered.harness == ASSISTANT_PREFIX_HARNESS
            and lowered.task.reasoning_markup == ReasoningMarkupPolicy.REJECT
        ):
            # The assistant-prefix transport records no reasoning violation, so rejection would never fire.
            raise ValueError("Assistant-prefix tasks cannot reject malformed reasoning spans")
        if lowered.harness == ASSISTANT_PREFIX_HARNESS:
            harnesses[lowered.harness] = AssistantPrefixHarness()
        elif is_shell_harness(lowered.harness):
            harnesses[lowered.harness] = shell_harness(lowered.harness, retrieval_service=self.retrieval_service)
        if lowered.harness == HARBOR_TRIAL_SESSION:
            if harbor is None or self.installed_agents is None:
                raise ValueError("Full Harbor trials require Harbor and installed-agent settings")
            engine_requirements = lowered.task.environment_requirements.model_copy(
                update={"setup_commands": (), "working_directory": None}
            )
            if engine_requirements != EnvironmentRequirements():
                raise ValueError(
                    "Declare environment requirements in the Harbor package; "
                    "TaskSpec overrides support only setup_commands and their working_directory"
                )
            harnesses[HARBOR_TRIAL_SESSION] = self.installed_agents.original_harness(lowered, session_id, harbor, phase)
            # The bound Harbor trial retains and executes the original setup.
            # The recording engine must not provision a second task machine for it.
            engine_task = lowered.task.model_copy(update={"environment_requirements": engine_requirements})
            # Harbor owns the phase deadlines and must still run its verifier
            # after an agent timeout. The outer attempt deadline covers the full trial.
            lowered = lowered.model_copy(
                update={
                    "task": engine_task,
                    "limits": lowered.limits.model_copy(
                        update={
                            "model_turn_timeout": None,
                            "tool_turn_timeout": None,
                            "total_turn_timeout": None,
                        }
                    ),
                }
            )
        elif harbor is not None and self.installed_agents is not None:
            if not is_shell_harness(lowered.harness):
                raise ValueError("SkyRL agents require a Shellbox task grader")
            lowered = lowered.model_copy(update={"harness": SKYRL_AGENT_HARNESS})

            async def agent(context, complete):
                await self.installed_agents.run(context.machine, complete, lowered=lowered, session_id=session_id)

            harnesses[SKYRL_AGENT_HARNESS] = AgentHarness(agent)
        return lowered, harnesses

    def prepare_task(self, lowered: TaskRolloutSpec, *, phase: str, session_id: str) -> PreparedTask:
        """Check the actual runtime machine and harness before any task allocation."""
        harbor = self.harbor if "harbor" in lowered.task.tags else None
        if harbor is not None:
            lowered = harbor.configured_spec(lowered, phase=phase)
        lowered, harnesses = self._configured_task_and_harnesses(lowered, harbor, session_id, phase)
        factories = self.factories if phase == "train" else self.factories | self.eval_factories
        engine = RolloutEngine(factories, harnesses)
        try:
            engine.check(lowered)
        except (ValueError, NotImplementedError) as error:
            raise ValueError(
                f"Invalid rollout task {lowered.task.source.dataset}/{lowered.task.id}: {error}"
            ) from error
        return PreparedTask(lowered, harbor, engine)

    async def generate(self, request: TaskRequestBatch, *, batch_id: int = 0) -> list[TrainingRollout]:
        with self._active_request():
            return await self._generate(request, batch_id=batch_id)

    async def _generate(self, request: TaskRequestBatch, *, batch_id: int) -> list[TrainingRollout]:
        lowered_tasks = [row.spec for row in request.rows]
        group_graders = [row.group_grader for row in request.rows]
        metadata = request.batch_metadata
        phase = "train" if metadata is None else metadata.training_phase
        trajectory_ids = request.trajectory_ids
        session_ids = [
            trajectory_ids[index].to_string() if trajectory_ids is not None else uuid4().hex
            for index in range(len(lowered_tasks))
        ]
        prepared = [
            self.prepare_task(item, phase=phase, session_id=session_ids[index])
            for index, item in enumerate(lowered_tasks)
        ]
        lowered_tasks = [item.spec for item in prepared]
        harbor_settings = [item.harbor for item in prepared]
        # Per-task Harbor settings are either None or self.harbor, so the
        # inherited policy covers every harbor-tagged row here.
        policies = [
            self.error_handling if harbor is None else self.harbor_error_handling for harbor in harbor_settings
        ]
        engines = [item.engine for item in prepared]
        tasks = [item.task for item in lowered_tasks]
        sampling = get_sampling_params_for_backend(
            self.trajectory_runner_cfg.backend, self.trajectory_runner_cfg.sampling_params
        )
        sampling.update(request.sampling_params or {})
        priority = self.evaluation_priority if phase == "eval" else 0
        require_logprobs = logprobs_requested(request, self.trajectory_runner_cfg)
        max_input_length = int(self.trajectory_runner_cfg.max_input_length)
        max_context_length = generation_context_limit(self.trajectory_runner_cfg)

        async def run(index, lowered):
            task = lowered.task
            harbor = harbor_settings[index]
            session_id = session_ids[index]

            async def model(request):
                try:
                    return await _model_turn(
                        self.model_client,
                        request,
                        sampling_params=sampling,
                        max_context_length=max_context_length,
                        max_prompt_length=max_input_length,
                        chat_template_kwargs=self.chat_template_kwargs,
                        session_id=session_id,
                        priority=priority,
                    )
                except (GenerationLimitReached, ContextLengthExceededError) as error:
                    raise ContextRejected(
                        error,
                        f"This request exceeds the maximum context length of {max_context_length} tokens. "
                        "Compact the conversation and retry.",
                    ) from error

            engine = engines[index]
            harbor_slots = None if harbor is None else self.harbor_task_slots
            retries = 0
            while True:
                try:
                    harbor_slot = nullcontext() if harbor_slots is None else harbor_slots.admit(batch_id)
                    async with harbor_slot, admission_slot("trajectory_tasks", self.task_slots):
                        result = training_rollout(await engine.run_task(lowered, model, run_id=session_id), lowered)
                except GradedRolloutInterrupted as interruption:
                    result = _failed_rollout(interruption, lowered)
                harness = engine.harnesses[lowered.harness]
                if isinstance(harness, HarborTrialHarness):
                    result = harness.with_verifier_result(result)
                if task.reasoning_markup == ReasoningMarkupPolicy.REJECT:
                    # Before group grading, so a malformed rollout never enters a GenRM comparison.
                    result = reject_malformed_reasoning(result)
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

    async def training_batch(self, request: TaskRequestBatch, outputs: Sequence[TrainingRollout]) -> TrajectoryBatch:
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
        for factory in (*self.factories.values(), *self.eval_factories.values()):
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
                *(factory.close(deadline) for factory in (*self.factories.values(), *self.eval_factories.values())),
                return_exceptions=True,
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
            started = time.perf_counter()
            rollouts = await self.generate(task.request, batch_id=task.lease.batch_id)
            batch = await self.training_batch(task.request, rollouts)
            record_event(
                "task_group_completed",
                {
                    "uid": task.prompt["uid"],
                    "duration_seconds": time.perf_counter() - started,
                    "response_tokens": sum(len(tokens) for tokens in batch["response_ids"]),
                    "loss_tokens": sum(sum(mask) for mask in batch["loss_masks"]),
                    **(batch.get("rollout_metrics") or {}),
                },
                attributes={
                    "source": task.request.metadata[0].data_source or "_unknown",
                    "step": str(task.lease.policy_step),
                },
            )
            return await write_trajectory_batch(task, writer, batch)


@dataclass(frozen=True)
class TaskRolloutWorkerSpec:
    config: DictConfig
    engines: list[InferenceEngineInterface]
    sessions: dict[str, EpisodeFactory] = field(default_factory=dict)
    harbor_config: DictConfig | None = None

    @classmethod
    def from_config(
        cls,
        config,
        engines,
        *,
        harbor_config: DictConfig | None = None,
        sessions: Mapping[str, EpisodeFactory] | None = None,
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
            from shellbox.backends.daytona.connection import DEFAULT_DAYTONA_API_URL, DaytonaConnection
            from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
            from skyrl_train.rollouts.agent_backend import AgentInferenceBackend

            settings = InstalledAgentSettings.model_validate(
                OmegaConf.to_container(self.harbor_config.installed_agent, resolve=True)
            )
            assert harbor is not None
            daytona_options = harbor.environment.kwargs
            daytona_connection = DaytonaConnection(
                api_key=os.environ.get(daytona_options.get("api_key_env") or "DAYTONA_API_KEY"),
                api_url=os.environ.get("DAYTONA_API_URL", DEFAULT_DAYTONA_API_URL),
                target=os.environ.get("DAYTONA_TARGET"),
            )
            installed_agents = InstalledAgentRuntime(
                settings,
                AgentInferenceBackend(
                    inference.model_name, generation_context_limit(runner_config), inference.tokenize, tokenizer
                ),
                int(runner_config.sampling_params.max_generate_length),
                daytona_connection=daytona_connection,
                eval_daytona_connection=replace(
                    daytona_connection,
                    # Inherit the training credential when eval has no separate binding.
                    api_key=(
                        os.environ.get(daytona_options["eval_api_key_env"])
                        if daytona_options.get("eval_api_key_env")
                        else daytona_connection.api_key
                    ),
                ),
            )
        factories = task_machine_factories(self.config.environment.task_sessions)
        eval_factories = task_machine_factories(self.config.environment.task_sessions, phase="eval")
        if harbor is not None:
            if harbor.environment.type != EnvironmentType.DOCKER:
                factories[HARBOR_MACHINE_BACKEND] = harbor.machine_factory()
                # Evaluation-only declarative provisioning selects the eval factory even
                # when it shares the training credential.
                factories[HARBOR_EVAL_MACHINE_BACKEND] = harbor.machine_factory(phase="eval")
            elif Backend.DOCKER in factories:
                factories[HARBOR_MACHINE_BACKEND] = factories[Backend.DOCKER]
                factories[HARBOR_EVAL_MACHINE_BACKEND] = factories[Backend.DOCKER]
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
            retrieval_service=self.config.environment.task_sessions.get("retrieval_service"),
            eval_factories=eval_factories,
            evaluation_priority=-1 if self.config.trainer.rollout_buffer.eval_drain == "priority" else 0,
        )
