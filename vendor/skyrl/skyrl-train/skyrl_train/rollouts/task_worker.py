"""TaskCompendium execution and projection into SkyRL training batches."""

import asyncio
import math
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, TypedDict
from uuid import uuid4

from jinja2 import TemplateError
from daytona import DaytonaError
from finelog.client import LogClient
from finestore.rl.strata import TaskStrata
from finestore.rl.task_health import AttemptReceipt
from iris.runtime import telemetry as runtime_telemetry
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rigging.redaction import redact_string
from rigging.task_errors import ErrorCategory, error_category, setup_exit_is_task_failure
from shellbox.machine import ExitReason, MachineFactory
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.models import NoGrader, ReasoningMarkupPolicy
from verifyit.modes.grade_judge import JudgeConnection
from rolloutengine.agents.model_relay import PromptTokenizationError
from rolloutengine.harness.core import Harness
from rolloutengine.machines import MachineSetupError
from rolloutengine.models import (
    GenerationLimitReached,
    ContextRejected,
    ContextWindowExceeded,
    GradedRollout,
    GradedRolloutInterrupted,
    Model,
    ModelRequest,
    ModelTurn,
    RolloutContractError,
    RolloutFailure,
    RolloutOperation,
    TaskRolloutSpec,
)
from rolloutengine.engine import RolloutEngine
from rolloutengine.finelog_observer import FinelogObserver, FinelogRolloutLog, RolloutScope
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS, AssistantPrefixHarness
from skyrl_gym.rollout_specs import is_shell_harness, shell_harness
from marinskyrl.agent_harnesses import AgentHarnessSettings, TaskAgentHarnesses
from marinskyrl.task_machine_factories import task_machine_factories
from skyrl_train.config.utils import generation_context_limit
from skyrl_gym.task_factories import EpisodeFactory, episode_factories, episode_harnesses
from skyrl_gym.task_records import fold_grades, skipped_grade
from skyrl_gym.verification import VERIFIER_RUNTIME_ERROR

from skyrl_train.inference_engines.base import ChatContinuation, InferenceEngineInterface, InferenceEngineInput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.rollouts.buffer import RolloutTask, RolloutWriter, write_trajectory_batch
from skyrl_train.rollouts.group_grading import GROUP_GRADERS, GroupGrader, grade_groups
from skyrl_train.rollouts.machines import OwnedMachineFactory, TaskMachineError
from skyrl_train.rollouts.workers import WorkerShard, detached_config
from skyrl_train.rollouts.finalization import finalize_trajectory_batch, propagate_data_sources
from skyrl_train.rollouts.task_health import attempt_receipt, write_attempt_receipts
from skyrl_train.rollout_observability import rollout_phase, rollout_wait
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
from skyrl_train.task_error_handling import ErrorHandlingConfig
from skyrl_train.task_errors import ERROR_CATEGORIES
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


AGENT_RETRY_BACKOFF = 1.0


async def _tokenize(
    inference: InferenceEngineClient, template_kwargs: Mapping[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """Tokenize agent prompts on the same serving actors that sample replies."""
    body = {**payload, "model": inference.model_name}
    if "messages" in body and template_kwargs:
        body["chat_template_kwargs"] = {**template_kwargs, **body.get("chat_template_kwargs", {})}
    result = await inference.tokenize({"json": body, "headers": {}})
    if "error" in result:
        code = result["error"].get("code")
        raise PromptTokenizationError(code if isinstance(code, int) else None)
    return result


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


class _DaytonaDiagnostics(TypedDict, total=False):
    provider_sdk_call: str
    provider_status_code: int
    provider_rate_limits: dict[str, int]


def _daytona_diagnostics(error: DaytonaError) -> _DaytonaDiagnostics:
    diagnostics: _DaytonaDiagnostics = {"provider_sdk_call": "unknown"}
    status = error.status_code
    if isinstance(status, int) and 100 <= status <= 599:
        diagnostics["provider_status_code"] = status
    rate_limits = {
        name.lower(): int(value)
        for name, value in (error.headers or {}).items()
        if name.lower() in DAYTONA_RATE_LIMIT_HEADERS
        and isinstance(value, str)
        and value.isascii()
        and value.isdecimal()
        and len(value) <= 12
    }
    if rate_limits:
        diagnostics["provider_rate_limits"] = rate_limits
    # A compound provider operation can span several SDK calls. The SDK may suppress the cause's traceback.
    nested_error: BaseException | None = error
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
        nested_error = nested_error.__cause__ or nested_error.__context__
    return diagnostics


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
    if isinstance(error, MachineSetupError):
        diagnostics.update(
            setup_exit_code=error.exit_code,
            setup_reason=error.reason.value,
            setup_stdout=redact_string(error.stdout),
            setup_stderr=redact_string(error.stderr),
        )
    if isinstance(error, PromptTokenizationError):
        diagnostics["status_code"] = error.status_code
    if isinstance(error, TaskMachineError) and error.__cause__ is not None:
        diagnostics["cause_error_type"] = type(error.__cause__).__name__
        diagnostics["machine_operation"] = error.operation.value
        if isinstance(error.__cause__, DaytonaError):
            diagnostics.update(_daytona_diagnostics(error.__cause__))
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
    elif isinstance(error, ContextWindowExceeded):
        exception_type = "ContextLengthExceededError"
        diagnostics["error_category"] = "context_overflow"
    elif interruption.operation in {RolloutOperation.START, RolloutOperation.PREPARE} and not isinstance(
        error, (TaskMachineError, PromptTokenizationError)
    ):
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
    grade = replace(grade, diagnostics={**interruption.grade.diagnostics, **grade.diagnostics})
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


async def _run_task_attempt(
    engine: RolloutEngine, spec: TaskRolloutSpec, model: Model, rollout_id: str
) -> TrainingRollout:
    try:
        result = training_rollout(await engine.run_task(spec, model, run_id=rollout_id), spec)
    except GradedRolloutInterrupted as interruption:
        result = _failed_rollout(interruption, spec)
    task = spec.task
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
        return replace(result, grade=skipped_grade(task.grader.reason, diagnostics=result.grade.diagnostics))
    return result


def _reject_unconfirmed_cleanup(result: TrainingRollout) -> TrainingRollout:
    if not result.grade.diagnostics.get("cleanup_errors"):
        return result
    grade = replace(
        result.grade,
        status=Outcome.INFRA_ERROR,
        reward=None,
        passed=None,
        error="Could not confirm agent task cleanup",
        failure=GradingFailure.EXECUTION,
    )
    failure = result.failure or RolloutFailure("TaskMachineError", {"machine_operation": "close"})
    return replace(result, grade=grade, failure=failure)


def _retryable_agent_attempt(result: TrainingRollout) -> bool:
    """Retry unscored infrastructure failures after confirmed cleanup."""
    failure = result.failure
    if failure is None or result.grade.status in {Outcome.GRADED, Outcome.SUBMISSION_FAILURE}:
        return False
    if result.grade.diagnostics.get("cleanup_errors"):
        return False
    if failure.diagnostics.get("setup_reason") == ExitReason.TIMED_OUT:
        return True
    setup_exit = failure.diagnostics.get("setup_exit_code")
    if isinstance(setup_exit, int):
        return setup_exit != 0 and not setup_exit_is_task_failure(setup_exit)
    status = failure.diagnostics.get("provider_status_code", failure.diagnostics.get("status_code"))
    if isinstance(status, int):
        return status in (409, 429) or 500 <= status < 600
    return error_category(failure.exception_type, categories=ERROR_CATEGORIES) is ErrorCategory.INFRASTRUCTURE


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
        concurrent_tasks: int | None = None,
        retry_wait: Callable[[float], Awaitable[None]] = asyncio.sleep,
        group_graders: Mapping[str, GroupGrader] = GROUP_GRADERS,
        sessions: Mapping[str, EpisodeFactory] | None = None,
        *,
        shutdown_timeout: float,
        concurrent_training_machine_tasks: int | None = None,
        agents: TaskAgentHarnesses | None = None,
        judge_connection: JudgeConnection | None = None,
        task_health_export: str | None = None,
        finelog: runtime_telemetry.FinelogRuntime | None = None,
    ):
        self.trajectory_runner_cfg = trajectory_runner_cfg
        self.model_client = model_client
        self.judge_connection = judge_connection
        self.factories = {name: OwnedMachineFactory(factory, name) for name, factory in factories.items()}
        self.active_requests: set[asyncio.Task] = set()
        self.closing = False
        self.shutdown_timeout = shutdown_timeout
        self.chat_template_kwargs = dict(trajectory_runner_cfg.get("chat_template_kwargs", {}))
        self.agents = agents
        self.task_health_export = task_health_export
        self.finelog_client = (
            None if finelog is None else LogClient.connect(finelog.endpoint, resolver=finelog.resolver)
        )
        self.rollout_log = None if self.finelog_client is None else FinelogRolloutLog(self.finelog_client)
        self.rollout_scope = (
            None if finelog is None else RolloutScope(finelog.attributes["run_id"], finelog.attributes["execution_uid"])
        )
        self.retry_wait = retry_wait
        self.group_graders = group_graders
        self.task_slots = asyncio.Semaphore(concurrent_tasks) if concurrent_tasks is not None else nullcontext()
        self.training_machine_slots = (
            asyncio.Semaphore(concurrent_training_machine_tasks)
            if concurrent_training_machine_tasks is not None
            else nullcontext()
        )
        self.error_handling = ErrorHandlingConfig.from_mapping(trajectory_runner_cfg.get("error_handling", {}))
        self.projection = replace(projection, error_handling=self.error_handling)
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

    def _harnesses(self, lowered: TaskRolloutSpec, agents: Mapping[str, Harness]) -> dict[str, Harness]:
        if lowered.harness in agents:
            return {lowered.harness: agents[lowered.harness]}
        harnesses = episode_harnesses(lowered, self.sessions)
        if lowered.harness == ASSISTANT_PREFIX_HARNESS and lowered.task.reasoning_markup == ReasoningMarkupPolicy.REJECT:
            # The assistant-prefix transport records no reasoning violation, so rejection would never fire.
            raise ValueError("Assistant-prefix tasks cannot reject malformed reasoning spans")
        if lowered.harness == ASSISTANT_PREFIX_HARNESS:
            harnesses[lowered.harness] = AssistantPrefixHarness()
        elif is_shell_harness(lowered.harness):
            harnesses[lowered.harness] = shell_harness(lowered.harness)
        return harnesses

    async def generate(self, request: TaskRequestBatch) -> list[TrainingRollout]:
        with self._active_request():
            return await self._generate(request)

    async def _generate(self, request: TaskRequestBatch) -> list[TrainingRollout]:
        receipt_identities: list[tuple[str, str]] = []
        if self.task_health_export is not None:
            for row in request.rows:
                identity = row.metadata.evaluation_identity
                if identity is None or not identity.task_digest:
                    raise ValueError(f"Task {row.spec.task.id} has no stored task digest for its attempt receipt")
                receipt_identities.append((identity.task_digest, identity.strata.source))
        lowered_tasks = [row.spec.model_copy(update={"judge_connection": self.judge_connection}) for row in request.rows]
        group_graders = [row.group_grader for row in request.rows]
        metadata = request.batch_metadata
        phase = "train" if metadata is None else metadata.training_phase
        trajectory_ids = request.trajectory_ids
        session_ids = [
            trajectory_ids[index].to_string() if trajectory_ids is not None else uuid4().hex
            for index in range(len(lowered_tasks))
        ]
        sampling = get_sampling_params_for_backend(
            self.trajectory_runner_cfg.backend, self.trajectory_runner_cfg.sampling_params
        )
        sampling.update(request.sampling_params or {})
        require_logprobs = logprobs_requested(request, self.trajectory_runner_cfg)
        max_input_length = int(self.trajectory_runner_cfg.max_input_length)
        max_context_length = generation_context_limit(self.trajectory_runner_cfg)
        agents = {} if self.agents is None else self.agents.bind(max_context_length, int(sampling["max_tokens"]))
        observer = None
        if self.rollout_scope is not None and self.rollout_log is not None:
            scope = replace(self.rollout_scope, phase=phase, step=None if metadata is None else metadata.global_step)
            observer = FinelogObserver(self.rollout_log, scope, content_fraction=1.0)
        engines = [
            RolloutEngine(self.factories, self._harnesses(spec, agents), observer=observer) for spec in lowered_tasks
        ]
        for engine, lowered in zip(engines, lowered_tasks, strict=True):
            engine.check(lowered)
        tasks = [item.task for item in lowered_tasks]
        evidence: dict[str, AttemptReceipt] = {}

        def record_attempt(index: int, attempt_id: str, rollout_id: str, result: TrainingRollout | BaseException):
            if self.task_health_export is None:
                return
            task_digest, source_id = receipt_identities[index]
            evidence[rollout_id] = attempt_receipt(
                attempt_id,
                lowered_tasks[index],
                result,
                rollout_id=rollout_id,
                source_id=None if source_id == TaskStrata().source else source_id,
                task_digest=task_digest,
            )

        async def run(index: int, lowered: TaskRolloutSpec) -> tuple[TrainingRollout, str, str]:
            task = lowered.task
            session_id = session_ids[index]
            attempt_id = session_id
            external_agent = lowered.harness in agents

            async def model(request):
                try:
                    return await _model_turn(
                        self.model_client,
                        request,
                        sampling_params=sampling,
                        max_context_length=max_context_length,
                        max_prompt_length=None if external_agent else max_input_length,
                        chat_template_kwargs=self.chat_template_kwargs,
                        session_id=attempt_id,
                    )
                except (GenerationLimitReached, ContextLengthExceededError) as error:
                    raise ContextRejected(
                        error,
                        f"This request exceeds the maximum context length of {max_context_length} tokens. "
                        "Compact the conversation and retry.",
                    ) from error

            engine = engines[index]
            retries = 0
            machine_slots = (
                self.training_machine_slots if phase == "train" and lowered.task_machine is not None else nullcontext()
            )
            while True:
                async with machine_slots, self.task_slots:
                    rollout_id = uuid4().hex
                    observed: TrainingRollout | BaseException
                    try:
                        result = await _run_task_attempt(engine, lowered, model, rollout_id)
                        if external_agent:
                            result = _reject_unconfirmed_cleanup(result)
                        observed = result
                    except BaseException as error:
                        observed = error
                        raise
                    finally:
                        record_attempt(index, attempt_id, rollout_id, observed)
                if not external_agent:
                    return result, attempt_id, rollout_id
                if retries or not _retryable_agent_attempt(result):
                    result = replace(result, metrics={**result.metrics, "rollout_retries": float(retries)})
                    return result, attempt_id, rollout_id
                logger.info("Retrying task {} after an infrastructure failure", task.id)
                with rollout_wait("retry_backoff"):
                    await self.retry_wait(AGENT_RETRY_BACKOFF)
                retries += 1
                attempt_id = f"{session_id}:retry-{retries}"

        with rollout_phase("collect"):
            batch_failure = None
            try:
                async with asyncio.TaskGroup() as group:
                    pending = [group.create_task(run(index, item)) for index, item in enumerate(lowered_tasks)]
                attempts = [done.result() for done in pending]
                rollouts = await grade_groups(
                    tasks,
                    group_graders,
                    [result for result, _, _ in attempts],
                    [item.instance_id for item in trajectory_ids]
                    if trajectory_ids is not None
                    else [task.id for task in tasks],
                    self.group_graders,
                    phase,
                    self.error_handling,
                    logprobs_required=require_logprobs,
                )
                for index, (rollout, (result, attempt_id, rollout_id)) in enumerate(zip(rollouts, attempts, strict=True)):
                    if rollout is not result:
                        record_attempt(index, attempt_id, rollout_id, rollout)
            except BaseException as error:
                batch_failure = error
                raise
            finally:
                if self.task_health_export is not None and evidence:
                    try:
                        await asyncio.to_thread(
                            write_attempt_receipts, self.task_health_export, list(evidence.values()), phase=phase
                        )
                    except Exception as error:
                        if batch_failure is None:
                            raise
                        batch_failure.add_note(f"Could not store task receipts ({type(error).__name__})")
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
        pass

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
            if self.finelog_client is not None:
                try:
                    async with asyncio.timeout_at(deadline):
                        await asyncio.to_thread(self._close_finelog)
                except TimeoutError:
                    logger.warning("Task worker shutdown deadline expired while flushing Finelog")
                except Exception as error:
                    logger.warning("Finelog shutdown failed: {}", type(error).__name__)
        if failures:
            raise ExceptionGroup("Task worker shutdown failed", failures)

    def _close_finelog(self) -> None:
        assert self.rollout_log is not None and self.finelog_client is not None
        try:
            self.rollout_log.flush(timeout=5)
        finally:
            self.finelog_client.close()

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

    @classmethod
    def from_config(
        cls,
        config,
        engines,
        *,
        sessions: Mapping[str, EpisodeFactory] | None = None,
    ):
        return cls(
            detached_config(config),
            list(engines),
            {} if sessions is None else dict(sessions),
        )

    def build(self, tokenizer, shard: WorkerShard) -> TaskRolloutWorker:
        task_sessions = self.config.environment.task_sessions
        judge_config = task_sessions.get("remote_judge")
        judge = None if judge_config is None else RemoteJudgeSettings(**dict(judge_config)).connection()
        runner_config = self.config.generator
        client_config = OmegaConf.merge(self.config, {"generator": {"enable_http_endpoint": False}})
        inference = InferenceEngineClient(self.engines, tokenizer, client_config)
        client = DirectModelClient(inference)
        agents = None
        if configured_agents := task_sessions.get("agent_harnesses"):
            agents = TaskAgentHarnesses(
                AgentHarnessSettings.model_validate(OmegaConf.to_container(configured_agents, resolve=True)),
                inference.model_name,
                partial(_tokenize, inference, dict(runner_config.get("chat_template_kwargs", {}))),
            )
        factories = task_machine_factories(task_sessions)
        machine_task_limit = self.config.trajectory_runner.get("max_concurrent_training_machine_tasks")
        runtime = self.config.get("runtime", {})
        finelog = runtime_telemetry.resolve(run_id=runtime.get("run_id"), execution_uid=runtime.get("attempt_id"))
        return TaskRolloutWorker(
            runner_config,
            (
                StepTaskProjection(runner_config, tokenizer)
                if self.config.trainer.step_wise_training
                else WholeTaskProjection(runner_config, tokenizer)
            ),
            client,
            factories,
            max_verifier_workers=int(task_sessions.max_verifier_workers),
            concurrent_tasks=int(self.config.trajectory_runner.max_concurrent_tasks),
            sessions=self.sessions,
            shutdown_timeout=float(self.config.trajectory_runner.shutdown_timeout),
            concurrent_training_machine_tasks=None if machine_task_limit is None else max(1, machine_task_limit // shard.count),
            agents=agents,
            judge_connection=judge,
            task_health_export=str(self.config.trainer.export_path),
            finelog=finelog,
        )
