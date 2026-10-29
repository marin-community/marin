# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute published TaskSpec evaluation corpora and save raw, resumable FineStore results."""

import asyncio
import dataclasses
import json
from collections.abc import Mapping
from enum import StrEnum
from functools import partial
from typing import Any

from finestore.eval import (
    ARCHIVE_ROLLOUTS_TABLE,
    ARCHIVE_SAMPLES_TABLE,
    Choice,
    EvalSample,
    EvaluationStore,
    Grading,
    Message,
    SampleKind,
    sample_to_archive_row,
)
from finestore.reader import ReadView
from finestore.schema import arrow_schema
from finestore.store import DataStore
from openai import APIError, APIStatusError
from pydantic import BaseModel, ConfigDict, Field
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.exact_chat import EXACT_CHAT_HARNESS, ExactChatHarness
from rolloutengine.models import (
    GradedRolloutInterrupted,
    MachineRuntimeSpec,
    Model,
    ModelRequest,
    ModelResponseError,
    ModelTurn,
    Rollout,
    RolloutLimits,
    TaskRolloutSpec,
    TraceMode,
)
from shellbox.machine import MachineFactory, MachineTerminated
from taskcompendium.evaluation import (
    ChatGeneration,
    ChoiceLikelihood,
    ContinuationLikelihood,
    EvaluationProtocol,
    MetricKind,
    TextGeneration,
    likelihood_metrics,
)
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ScriptGrader, TaskSpec, VerifyitGrader, grades_with_judge, verifyit_spec
from taskcompendium.pipeline.evaluation import EvaluationCase, evaluation_protocol
from taskcompendium.submission import conversation_messages
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.spec import JudgeSpec, MathJudgeSpec

from marin.evaluation.completion import completion_text
from marin.evaluation.likelihood import InvalidLikelihoodRequest, LikelihoodScorer, LikelihoodTransportError
from marin.evaluation.rollouts import rollout_records_from_sample
from marin.evaluation.selection import SELECTIONS_TABLE, EvaluationTokenizer, Selection, select_cases

ATTEMPTS_TABLE = "evaluation_attempts"
RUN_OBJECT = "evaluation/run.json"
DRIVER_REVISION = "2026.10.28.4"


class EvaluationInfrastructureError(RuntimeError):
    """All corpora were processed, but infrastructure failures remain eligible for resume."""


def _permanent_api_error(error: BaseException | None) -> bool:
    return isinstance(error, APIStatusError) and error.status_code < 500 and error.status_code not in (408, 409, 429)


class RunModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ModelIdentity(RunModel):
    name: str
    revision: str
    tokenizer: str
    tokenizer_revision: str
    context_window: int = Field(gt=0)
    serving: dict[str, str]


class JudgeIdentity(RunModel):
    deployment: str = Field(min_length=1)
    models: dict[str, str]
    default_model: str | None = None


class EvaluationRunConfig(RunModel):
    model: ModelIdentity
    max_tokens: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    limits: RolloutLimits
    source_limit: int | None = Field(default=None, gt=0)
    seed: int | None = None
    task_machine: MachineRuntimeSpec | None = None
    verifier_machine: MachineRuntimeSpec | None = None
    judge: JudgeIdentity | None = None


class AttemptPhase(StrEnum):
    STARTED = "started"
    FINISHED = "finished"


class Attempt(RunModel):
    dataset: str
    task_id: str
    source_row: str
    trial: int
    retry: int
    phase: AttemptPhase
    seed: int | None
    outcome: Outcome | None
    error: str | None
    result_blob: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


def generation_options(protocol: EvaluationProtocol, config: EvaluationRunConfig, trial: int) -> dict[str, Any]:
    """Resolve benchmark settings and explicit run budgets into transport options."""
    sampling = protocol.sampling
    assert sampling is not None
    extra: dict[str, Any] = {}
    if protocol.operation == "chat_generation":
        extra["chat_template_kwargs"] = sampling.chat_template_kwargs
    if sampling.top_k is not None:
        extra["top_k"] = sampling.top_k
    options: dict[str, Any] = {
        "temperature": sampling.temperature,
        "max_tokens": min(config.max_tokens, sampling.max_tokens or config.max_tokens),
        "extra_body": extra,
    }
    if sampling.top_p is not None:
        options["top_p"] = sampling.top_p
    if sampling.stop:
        options["stop"] = list(sampling.stop)
    seed = protocol.trials.seed_for(trial) if config.seed is None else config.seed + trial
    if seed is not None:
        options["seed"] = seed
    return options


def _sample(
    case: EvaluationCase, protocol: EvaluationProtocol, task: TaskSpec, rollout: Rollout, grade: GradeResult
) -> EvalSample:
    output = ""
    if rollout.conversations and rollout.conversations[-1].steps:
        turn = rollout.conversations[-1].steps[-1].turn
        output = completion_text(turn.metadata.get("evaluation_raw_message", turn.message), turn.stop_reason, "combine")
    detail = grade.detail or {}
    extracted = detail.get("extracted")
    metrics = detail.get("metrics", {})
    if not metrics and grade.reward is not None and grade.status in (Outcome.GRADED, Outcome.SUBMISSION_FAILURE):
        if len(protocol.metrics) == 1:
            metrics = {protocol.primary_metric: grade.reward}
        elif grade.status is Outcome.GRADED:
            raise ValueError(f"Multi-metric grader must return its named metrics: {protocol.benchmark}")
        # Submission extraction can fail before source component metrics exist.
        # Keep its terminal grade and evidence without inventing those scores.
    return EvalSample(
        task=protocol.benchmark,
        doc_id=task.id,
        kind=SampleKind.GENERATION,
        prompt_text=case.item.operation.context if isinstance(case.item.operation, TextGeneration) else None,
        prompt_messages=(
            None
            if isinstance(case.item.operation, TextGeneration)
            else [Message(**message) for message in conversation_messages(task.context.events)]
        ),
        output=output,
        extracted=(
            extracted if isinstance(extracted, str) or extracted is None else json.dumps(extracted, allow_nan=False)
        ),
        grading=Grading(
            method="taskcompendium:" + task.grader.kind,
            metric=protocol.primary_metric,
            score=grade.reward,
            passed=grade.passed,
            detail=json.dumps(dataclasses.asdict(grade), allow_nan=False),
        ),
        metrics=metrics,
        correct=(
            grade.reward == 1
            if grade.reward is not None
            and next(metric.kind for metric in protocol.metrics if metric.name == protocol.primary_metric)
            == MetricKind.BINARY
            else None
        ),
        doc=json.dumps({"source": case.item.source_row, "groups": case.item.groups}),
    )


def _attempt_key(dataset: str, source_row: str, trial: int) -> tuple[str, str, int]:
    return dataset, source_row, trial


async def generation_turn(
    request: ModelRequest, *, model: Model, operation: ChatGeneration | TextGeneration
) -> ModelTurn:
    """Keep raw channels and wire messages while selecting the source's grading text."""
    if isinstance(operation, ChatGeneration) and operation.historical_assistant_reasoning == "empty":
        messages = tuple(
            {**message, "reasoning_content": ""} if message["role"] == "assistant" else message
            for message in request.messages
        )
        request = dataclasses.replace(request, messages=messages)
    turn = await model(request)
    raw_message = dict(turn.message)
    if reasoning := turn.metadata.get("reasoning"):
        raw_message["reasoning"] = reasoning
    selection = operation.response_selection if isinstance(operation, ChatGeneration) else "combine"
    content = completion_text(raw_message, turn.stop_reason, selection)
    return dataclasses.replace(
        turn,
        message={**turn.message, "content": content},
        text=content,
        metadata={
            **turn.metadata,
            "evaluation_raw_message": raw_message,
            "evaluation_request": dataclasses.asdict(request),
        },
    )


def _save_attempt(
    store: DataStore,
    attempt: Attempt,
    sample: EvalSample,
    raw: dict,
    outcome: Outcome,
    error: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> None:
    trial_id = f"{attempt.dataset}/{attempt.trial}/{attempt.retry}"
    blob = f"evaluation/attempts/{canonical_sha256(attempt.model_dump(mode='json'))}.json"
    terminal = attempt.model_copy(
        update={
            "phase": AttemptPhase.FINISHED,
            "outcome": outcome,
            "error": error,
            "result_blob": blob,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    )
    with store.transaction() as transaction:
        transaction.table(ATTEMPTS_TABLE).add(terminal.model_dump(mode="json"))
        transaction.table(ARCHIVE_SAMPLES_TABLE).add(sample_to_archive_row(sample, trial_id=trial_id))
        for record in rollout_records_from_sample(sample, trial_id=trial_id):
            transaction.table(ARCHIVE_ROLLOUTS_TABLE).add(record.model_dump(mode="json"))
        transaction.write_object(blob, json.dumps(raw, allow_nan=False).encode())


async def _score_likelihood(
    case: EvaluationCase,
    protocol: EvaluationProtocol,
    scorer: LikelihoodScorer,
    store: DataStore,
    attempt: Attempt,
    timeout: float | None,
) -> Outcome:
    operation = case.item.operation
    assert isinstance(operation, ChoiceLikelihood | ContinuationLikelihood)
    requests = operation.requests if isinstance(operation, ChoiceLikelihood) else (operation.request,)
    results = []
    failure_evidence = None
    permanent_error = None
    outcome = Outcome.GRADED
    error = None
    try:
        async with asyncio.timeout(timeout):
            for request in requests:
                results.append(await scorer(request, seed=attempt.seed))
    except InvalidLikelihoodRequest as failure:
        outcome = Outcome.INVALID_TASK
        error = str(failure)
    except (APIError, LikelihoodTransportError, TimeoutError) as failure:
        # Preserve partial endpoint evidence and retry the same logical attempt on resume.
        outcome = Outcome.INFRA_ERROR
        error = f"{type(failure).__name__}: {failure}"
        if isinstance(failure, LikelihoodTransportError):
            failure_evidence = failure.evidence
        elif isinstance(failure, APIError):
            failure_evidence = {"provider_error": failure.body}
        if _permanent_api_error(failure):
            permanent_error = failure
    metrics = (
        likelihood_metrics(
            operation,
            tuple(result.loglikelihood for result in results),
            protocol,
            is_greedy=tuple(result.is_greedy for result in results),
        )
        if outcome is Outcome.GRADED
        else {}
    )
    primary = metrics.get(protocol.primary_metric)
    raw = {
        "task": case.task.model_dump(mode="json"),
        "item": case.item.model_dump(mode="json"),
        "scores": [dataclasses.asdict(result) for result in results],
        "outcome": outcome,
        "error": error,
        "failure_evidence": failure_evidence,
    }
    sample = EvalSample(
        task=protocol.benchmark,
        doc_id=case.task.id,
        kind=SampleKind.MULTIPLE_CHOICE if isinstance(operation, ChoiceLikelihood) else SampleKind.GENERATION,
        prompt_text=requests[0].context,
        choices=(
            [
                Choice(
                    label=str(index),
                    text=choice,
                    loglikelihood=results[index].loglikelihood if index < len(results) else None,
                    is_greedy=results[index].is_greedy if index < len(results) else None,
                )
                for index, choice in enumerate(operation.choices)
            ]
            if isinstance(operation, ChoiceLikelihood)
            else None
        ),
        model_choice=(
            max(range(len(results)), key=lambda index: results[index].loglikelihood)
            if outcome is Outcome.GRADED and results and isinstance(operation, ChoiceLikelihood)
            else None
        ),
        target_choice=operation.gold[0] if isinstance(operation, ChoiceLikelihood) else None,
        target_text=operation.request.continuation if isinstance(operation, ContinuationLikelihood) else None,
        grading=Grading(
            method="taskcompendium:likelihood",
            metric=protocol.primary_metric,
            score=primary,
            detail=json.dumps(raw["scores"], allow_nan=False),
        ),
        metrics=metrics,
        doc=json.dumps({"source": case.item.source_row, "groups": case.item.groups}),
    )
    usage = [result.usage for result in results]
    known_usage = outcome is Outcome.GRADED and all(value is not None for value in usage)
    _save_attempt(
        store,
        attempt,
        sample,
        raw,
        outcome,
        error,
        sum(value.prompt_tokens for value in usage if value is not None) if known_usage else None,
        sum(value.completion_tokens for value in usage if value is not None) if known_usage else None,
    )
    if permanent_error is not None:
        raise permanent_error
    return outcome


async def _score_generation(
    case: EvaluationCase,
    protocol: EvaluationProtocol,
    config: EvaluationRunConfig,
    model: Model,
    store: DataStore,
    attempt: Attempt,
    max_tokens: int,
    *,
    machine_factories: Mapping[str, MachineFactory],
    judge_connection: JudgeConnection | None,
) -> Outcome:
    task = case.task
    options = generation_options(protocol, config, attempt.trial)
    options["max_tokens"] = max_tokens
    engine = RolloutEngine(machine_factories, {EXACT_CHAT_HARNESS: ExactChatHarness(options)})
    spec = TaskRolloutSpec(
        task=task,
        harness=EXACT_CHAT_HARNESS,
        task_machine=config.task_machine,
        verifier_machine=(
            config.verifier_machine
            if isinstance(task.grader, ScriptGrader | VerifyitGrader)
            and task.grader.environment is not None
            and not grades_with_judge(task.grader)
            else None
        ),
        judge_connection=judge_connection,
        limits=config.limits,
        trace=TraceMode.MESSAGES,
    )
    operation = case.item.operation
    assert isinstance(operation, ChatGeneration | TextGeneration)

    policy_model = partial(generation_turn, model=model, operation=operation)

    failure_evidence = None
    permanent_error = None
    try:
        result = await engine.run_task(spec, policy_model)
        rollout, grade = result.rollout, result.grade
    except GradedRolloutInterrupted as error:
        rollout, grade = error.rollout, error.grade
        cause = error.__cause__
        if isinstance(cause, ModelResponseError):
            failure_evidence = {"response": cause.response, "response_json": cause.response_json}
        elif isinstance(cause, APIError):
            failure_evidence = {"provider_error": cause.body}
        if cause is not None and (
            _permanent_api_error(cause)
            or not isinstance(cause, (APIError, ModelResponseError, ConnectionError, TimeoutError, MachineTerminated))
        ):
            permanent_error = cause
    if rollout.failure is not None or grade.status in (Outcome.INFRA_ERROR, Outcome.UNAVAILABLE):
        outcome = Outcome.INFRA_ERROR
    else:
        outcome = grade.status
    sample = _sample(case, protocol, task, rollout, grade)
    turns = [step.turn for conversation in rollout.conversations for step in conversation.steps]
    raw = {
        "task": task.model_dump(mode="json"),
        "item": case.item.model_dump(mode="json"),
        "grade": dataclasses.asdict(grade),
        "conversations": [dataclasses.asdict(conversation) for conversation in rollout.conversations],
        "stop_reason": rollout.stop_reason,
        "failure": dataclasses.asdict(rollout.failure) if rollout.failure else None,
        "failure_evidence": failure_evidence,
        "options": options,
    }
    _save_attempt(
        store,
        attempt,
        sample,
        raw,
        outcome,
        grade.error,
        (
            sum(turn.usage.prompt_tokens if turn.usage else len(turn.prompt_token_ids) for turn in turns)
            if turns
            else None
        ),
        (
            sum(turn.usage.completion_tokens if turn.usage else len(turn.response_token_ids) for turn in turns)
            if turns
            else None
        ),
    )
    if permanent_error is not None:
        raise permanent_error
    return outcome


def _latest_attempts(reader: ReadView) -> dict[tuple[str, str, int], Attempt]:
    latest: dict[tuple[str, str, int], Attempt] = {}
    for row in reader.iter_rows(ATTEMPTS_TABLE):
        attempt = Attempt.model_validate({name: row[name] for name in Attempt.model_fields})
        key = _attempt_key(attempt.dataset, attempt.source_row, attempt.trial)
        old = latest.get(key)
        if old is None or (attempt.retry, attempt.phase == AttemptPhase.FINISHED) > (
            old.retry,
            old.phase == AttemptPhase.FINISHED,
        ):
            latest[key] = attempt
    return latest


def _validate_judge(task: TaskSpec, identity: JudgeIdentity | None) -> None:
    if not grades_with_judge(task.grader):
        return
    if identity is None:
        raise ValueError(f"Judged task requires an explicit judge identity and connection: {task.id}")
    assert isinstance(task.grader, VerifyitGrader)
    verifier = verifyit_spec(task.grader)
    judge = verifier.judge if isinstance(verifier, MathJudgeSpec) else verifier
    assert isinstance(judge, JudgeSpec)
    alias = judge.model.strip() or (identity.default_model or "").strip()
    if not alias or not identity.models.get(alias, "").strip():
        raise ValueError(f"Judge model has no recorded immutable revision: {alias!r} ({task.id})")


async def run_evaluation(
    datasets: tuple[str, ...],
    output: str,
    config: EvaluationRunConfig,
    model: Model,
    *,
    machine_factories: Mapping[str, MachineFactory],
    tokenizer: EvaluationTokenizer | None = None,
    judge_connection: JudgeConnection | None = None,
    completion_model: Model | None = None,
    likelihood: LikelihoodScorer | None = None,
) -> None:
    """Stream selected source rows through RolloutEngine; resume without resampling completed trials.

    Every physical attempt has an append-only start and finish. Its sample, terminal state and raw
    evidence commit together. Infrastructure failures remain retryable on a later invocation with the
    same run identity and seed. Summaries belong to the consumer of these raw results.
    """
    if (config.judge is None) != (judge_connection is None):
        raise ValueError("A judge requires both an explicit connection and a recorded judge identity")
    if config.judge is not None:
        assert judge_connection is not None
        judge_connection = dataclasses.replace(judge_connection, default_model=config.judge.default_model or "")
    corpora = [(root, *evaluation_protocol(root)) for root in datasets]
    if len({canonical_sha256(manifest.model_dump(mode="json")) for _, _, manifest in corpora}) != len(corpora):
        raise ValueError("A run cannot include the same dataset snapshot twice")
    for _, protocol, _ in corpora:
        if protocol.operation in ("choice_likelihood", "continuation_likelihood") and likelihood is None:
            raise ValueError(f"Likelihood scoring requires an explicit scorer: {protocol.benchmark}")
        if protocol.operation == "text_generation" and completion_model is None:
            raise ValueError(f"Text generation requires an explicit completion transport: {protocol.benchmark}")
    run = {
        "schema_version": 1,
        "driver_revision": DRIVER_REVISION,
        "config": config.model_dump(mode="json"),
        "datasets": [
            {
                "root": root,
                "dataset": canonical_sha256(manifest.model_dump(mode="json")),
                "protocol": protocol.model_dump(mode="json"),
                "manifest": manifest.model_dump(mode="json"),
            }
            for root, protocol, manifest in corpora
        ],
    }
    identity = canonical_sha256(run)
    previous = ReadView(output).read_blob(RUN_OBJECT)
    if previous is not None and canonical_sha256(json.loads(previous)) != identity:
        raise ValueError("Resume requires the identical model, corpora, protocol, selection and run settings")
    with DataStore.open(output) as store:
        EvaluationStore(store)
        store.table(
            ATTEMPTS_TABLE,
            schema=arrow_schema(Attempt),
            primary_key=("dataset", "source_row", "trial", "retry", "phase"),
            schema_version=1,
        )
        store.table(
            SELECTIONS_TABLE,
            schema=arrow_schema(Selection),
            primary_key=("dataset", "source_row"),
            schema_version=1,
        )
        latest = _latest_attempts(store.read_view())
        with store.transaction() as transaction:
            transaction.write_object(RUN_OBJECT, json.dumps(run, allow_nan=False).encode())

        queue: asyncio.Queue[tuple[EvaluationCase, EvaluationProtocol, Selection, int, int] | None] = asyncio.Queue(
            maxsize=config.concurrency * 2
        )
        failures = 0

        async def produce() -> None:
            for root, protocol, manifest in corpora:
                dataset_id = canonical_sha256(manifest.model_dump(mode="json"))
                records = []
                selected = 0
                for decision in select_cases(
                    root,
                    protocol,
                    dataset_id=dataset_id,
                    context_window=config.model.context_window,
                    max_tokens=config.max_tokens,
                    limit=config.source_limit,
                    tokenizer=tokenizer,
                ):
                    records.append(decision.record.model_dump(mode="json"))
                    if len(records) >= 500 or decision.case is not None:
                        with store.transaction() as transaction:
                            transaction.table(SELECTIONS_TABLE).extend(records)
                        records = []
                    case = decision.case
                    if case is None:
                        continue
                    _validate_judge(case.task, config.judge)
                    selected += 1
                    # Cohort decisions are durable before expanding trials, including on resumed runs.
                    for trial in range(protocol.trials.count):
                        old = latest.get(_attempt_key(dataset_id, case.item.source_row, trial))
                        if (
                            old is not None
                            and old.phase == AttemptPhase.FINISHED
                            and old.outcome in (Outcome.GRADED, Outcome.SUBMISSION_FAILURE, Outcome.INVALID_TASK)
                        ):
                            continue
                        await queue.put((case, protocol, decision.record, trial, 0 if old is None else old.retry + 1))
                if records:
                    with store.transaction() as transaction:
                        transaction.table(SELECTIONS_TABLE).extend(records)
                if not selected:
                    raise ValueError(f"No evaluation rows are eligible under this run: {protocol.benchmark}")
            for _ in range(config.concurrency):
                await queue.put(None)

        async def consume() -> None:
            nonlocal failures
            while (work := await queue.get()) is not None:
                case, protocol, selection, trial, retry = work
                task = case.task
                seed = protocol.trials.seed_for(trial) if config.seed is None else config.seed + trial
                attempt = Attempt(
                    dataset=selection.dataset,
                    task_id=task.id,
                    source_row=case.item.source_row,
                    trial=trial,
                    retry=retry,
                    phase=AttemptPhase.STARTED,
                    seed=seed,
                    outcome=None,
                    error=None,
                    result_blob=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                )
                with store.transaction() as transaction:
                    transaction.table(ATTEMPTS_TABLE).add(attempt.model_dump(mode="json"))
                if isinstance(case.item.operation, ChoiceLikelihood | ContinuationLikelihood):
                    assert likelihood is not None
                    outcome = await _score_likelihood(
                        case, protocol, likelihood, store, attempt, config.limits.attempt_timeout
                    )
                    failures += outcome is Outcome.INFRA_ERROR
                    continue
                transport = completion_model if isinstance(case.item.operation, TextGeneration) else model
                assert transport is not None and selection.max_tokens is not None
                outcome = await _score_generation(
                    case,
                    protocol,
                    config,
                    transport,
                    store,
                    attempt,
                    selection.max_tokens,
                    machine_factories=machine_factories,
                    judge_connection=judge_connection,
                )
                failures += outcome is Outcome.INFRA_ERROR

        async with asyncio.TaskGroup() as group:
            group.create_task(produce())
            for _ in range(config.concurrency):
                group.create_task(consume())
        if failures:
            raise EvaluationInfrastructureError(
                f"{failures} evaluation attempts failed; their raw evidence is saved for resume"
            )
        store.seal()
