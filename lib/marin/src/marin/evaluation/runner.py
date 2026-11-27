# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run endpoint-oriented evaluations as independent steps."""

import contextvars
import logging
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from functools import partial
from typing import NoReturn, Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from evalrecords.eval_stats import DEFAULT_MIN_COVERAGE
from evalrecords.model_identity import model_config_digest
from evalrecords.records import (
    EVALCHEMY_INFRASTRUCTURE_ERROR,
    EvalRef,
    EvalRunRecord,
    EvalTaskRef,
    HardwareRef,
    HostedJudgeRef,
    InferenceMetrics,
    ModelConfigRef,
    ModelRef,
    Provenance,
    RLAttemptRef,
    RunStatus,
    ServingParams,
    TaskCoverage,
    read_record,
    record_path,
    write_record,
)
from evalrecords.rollout_results import score_curated_record
from fray.client import JobHandle
from iris.client.client import Job, iris_ctx
from iris.rpc.proto_display import job_state_friendly
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join
from rigging.secrets import SecretSpec
from rigging.timing import retry_with_backoff

from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import AcceleratorChoice
from marin.evaluation.inference_metrics import InferenceMetricWindow
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.execution.step_runner import StepRunner, check_cache
from marin.execution.step_spec import StepSpec
from marin.inference.backend import OPENAI_API_SUFFIX
from marin.inference.config import RemoteInferenceConfig
from marin.inference.iris import (
    InferenceBackendState,
    InferenceStartupPhase,
    PreparedInference,
    RemoteInferenceSession,
    RemoteInferenceStartupError,
    prepare_inference,
    serve_inference,
)
from marin.rollouts.catalog import RolloutRunKind, record_rollout_run, rollout_run_record

logger = logging.getLogger(__name__)

_INFERENCE_ROLE = "inference"
_ORCHESTRATOR_ROLE = "orchestrator"
_JUDGE_ROLE = "judge"
_PREPARED_JUDGE_ROLE = "prepared-judge"
_UNCONSTRAINED = "unconstrained"
_REPORT_TAIL_LINES = 15


# Stop evaluator attempts before their HTTP retries exhaust during a model reload.
BACKEND_POLL_SECONDS = 5.0


class InferenceUnavailable(RuntimeError):
    """An evaluator attempt must stop while its inference backends recover."""


class InferenceBackendFinished(RuntimeError):
    """Inference ended permanently while an evaluator depended on it."""


def inference_backend_state(sessions: tuple[RemoteInferenceSession, ...]) -> InferenceBackendState:
    """Combine session states, prioritizing FINISHED over RECOVERING over READY."""
    states = tuple(session.backend_state() for session in sessions)
    if InferenceBackendState.FINISHED in states:
        return InferenceBackendState.FINISHED
    if InferenceBackendState.RECOVERING in states:
        return InferenceBackendState.RECOVERING
    return InferenceBackendState.READY


def check_inference_backend(state: InferenceBackendState) -> None:
    """Interrupt an attempt on a recoverable outage; fail on a terminal backend."""
    if state is InferenceBackendState.RECOVERING:
        raise InferenceUnavailable("inference backends are not ready")
    if state is InferenceBackendState.FINISHED:
        raise InferenceBackendFinished("inference backend finished while evaluation was running")


def inference_attempt_state(
    sessions: tuple[RemoteInferenceSession, ...], before: tuple[int, ...]
) -> InferenceBackendState:
    """Include preemptions that recovered between evaluator health polls."""
    state = inference_backend_state(sessions)
    if state is InferenceBackendState.READY and before != tuple(session.preemption_count() for session in sessions):
        return InferenceBackendState.RECOVERING
    return state


def run_with_inference_recovery[T](
    attempt: Callable[[int], T],
    sessions: tuple[RemoteInferenceSession, ...],
) -> T:
    """Wait for ready backends and retry attempts that raise InferenceUnavailable.

    Each attempt receives a zero-based attempt number. It must monitor
    dependencies and stop its work before raising InferenceUnavailable.
    Other exceptions propagate.
    """

    def check_dependencies() -> None:
        if inference_backend_state(sessions) is InferenceBackendState.FINISHED:
            check_inference_backend(InferenceBackendState.FINISHED)

    def wait_for_backends() -> None:
        for session in sessions:
            session.wait_until_ready(check_dependencies=check_dependencies)

    attempt_number = 0
    while True:
        state = inference_backend_state(sessions)
        if state is InferenceBackendState.RECOVERING:
            wait_for_backends()
        else:
            check_inference_backend(state)
        try:
            return attempt(attempt_number)
        except InferenceUnavailable:
            logger.warning("pausing evaluation while inference recovers")
            wait_for_backends()
            attempt_number += 1


@dataclass(frozen=True)
class EvaluationOutcome:
    metrics: dict[str, dict[str, float]]
    canonical_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    tasks: tuple[EvalTaskRef, ...] | None = None
    jobs: dict[str, str] = field(default_factory=dict)
    coverage: dict[str, TaskCoverage] = field(default_factory=dict)
    """Per-task item coverage for mechanisms that report an attempted-item count; empty otherwise."""


class EvaluationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: RunStatus,
        jobs: dict[str, str] | None = None,
        log_tails: dict[str, tuple[str, ...]] | None = None,
        coverage: dict[str, TaskCoverage] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.jobs = jobs or {}
        self.log_tails = log_tails or {}
        self.coverage = coverage or {}
        """Per-task coverage measured before the failure."""


class EvalExecutor(Protocol):
    """Execute one resumable attempt using the supplied endpoint and output_dir.

    Poll backend_state while child work runs. Stop all child work
    before raising InferenceUnavailable so the runner can safely reuse output_dir.
    Raise EvaluationError for evaluation failures; infrastructure failures can
    be retried when the attempt's inference dependencies were interrupted.
    """

    def __call__(
        self,
        session: RemoteInferenceSession,
        output_dir: str,
        env_vars: Mapping[str, str],
        *,
        backend_state: Callable[[], InferenceBackendState],
        judge: RemoteInferenceSession | None = None,
    ) -> EvaluationOutcome: ...


class EndpointRoute(StrEnum):
    """DIRECT resolves the registered serving address; CAPABILITY preserves the session endpoint."""

    DIRECT = "direct"
    CAPABILITY = "capability"


@dataclass(frozen=True)
class EvaluationIdentity:
    run_id: str
    created_at: str
    output_dir: str
    eval_ref: EvalRef
    eval_runtime: str | None = None


@dataclass(frozen=True)
class LaunchProvenance:
    git_sha: str
    launch_host: str


@dataclass(frozen=True)
class Evaluation:
    identity: EvaluationIdentity
    executor: EvalExecutor
    endpoint_route: EndpointRoute
    secret_env_keys: tuple[str, ...] = ()


type EvaluationStep = Callable[[Evaluation, str], dict[str, str]]


@dataclass(frozen=True)
class HostedJudge:
    """A second model served beside the evaluated model for verifier requests."""

    model: ModelConfig
    accelerator: AcceleratorChoice
    api_model: str | None


@dataclass(frozen=True)
class EvaluationBatch:
    group_id: str
    user: str
    version: str | None
    description: str | None
    records_prefix: str
    model: ModelConfig
    accelerator: AcceleratorChoice
    priority_band: int
    capability_origin: str
    api_model: str | None
    evaluations: tuple[Evaluation, ...]
    provenance: LaunchProvenance
    submission_cluster: str
    max_concurrent: int
    instances: int = 1
    judge: HostedJudge | None = None
    # Judges an evaluation serves for itself. The batch only prepares their weights, alongside the
    # candidate, so no evaluation downloads a judge while the candidate's accelerators are reserved.
    prepared_judges: tuple[HostedJudge, ...] = ()
    secret_env: Mapping[str, SecretSpec] = field(default_factory=dict)
    egress_allow: tuple[str, ...] = ()
    """Egress rules the orchestrator job and its children need beyond the cluster allowlist."""
    source_model_config: ModelConfigRef | None = None
    rl_attempt: RLAttemptRef | None = None
    excluded_nvlink_domains: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubmittedEvaluation:
    run_id: str
    eval_name: str


@dataclass(frozen=True)
class SubmittedEvaluationBatch:
    group_id: str
    job: Job
    records_prefix: str
    model_name: str
    evaluations: tuple[SubmittedEvaluation, ...]


def _record(
    batch: EvaluationBatch,
    identity: EvaluationIdentity,
    status: RunStatus,
    error: str | None,
    metrics: dict[str, dict[str, float]],
    jobs: dict[str, str],
    log_tails: dict[str, tuple[str, ...]],
    coverage: dict[str, TaskCoverage] | None = None,
    canonical_metrics: dict[str, dict[str, float]] | None = None,
    tasks: tuple[EvalTaskRef, ...] | None = None,
    serving: ServingParams | None = None,
    inference_metrics: InferenceMetrics | None = None,
) -> str:
    evaluation = identity.eval_ref
    if tasks is not None:
        evaluation = evaluation.model_copy(update={"tasks": tasks})
    if serving is None:
        serve = batch.model.serve
        geometry = batch.accelerator.geometry
        serving = ServingParams(
            tensor_parallel_size=serve.tensor_parallel_size,
            data_parallel_size=serve.data_parallel_size,
            pipeline_parallel_size=serve.pipeline_parallel_size,
            task_count=(geometry.task_count if geometry is not None else serve.pipeline_parallel_size) * batch.instances,
            max_model_len=serve.max_model_len,
        )
    evalchemy = identity.eval_ref.evalchemy
    serving = serving.model_copy(
        update={
            "max_gen_tokens": evalchemy.max_gen_toks if evalchemy is not None else None,
            "extra": dict(evalchemy.extra_gen_kwargs) if evalchemy is not None else {},
        }
    )
    model_config = ModelConfigRef.model_validate(asdict(batch.model))
    source_model_config = batch.source_model_config or model_config
    record = EvalRunRecord(
        run_id=identity.run_id,
        group_id=batch.group_id,
        created_at=identity.created_at,
        user=batch.user,
        version=batch.version,
        description=batch.description,
        rl_attempt=batch.rl_attempt,
        model=ModelRef(
            name=batch.model.name,
            location=batch.model.location,
            backend=batch.model.serve.backend.value,
            config=model_config,
            source_config=source_model_config if source_model_config != model_config else None,
            config_digest=model_config_digest(source_model_config),
        ),
        judge=(
            HostedJudgeRef(
                model=ModelRef(
                    name=batch.judge.model.name,
                    location=batch.judge.model.location,
                    backend=batch.judge.model.serve.backend.value,
                    config=ModelConfigRef.model_validate(asdict(batch.judge.model)),
                ),
                hardware=HardwareRef(
                    platform=batch.judge.accelerator.platform.value,
                    accelerator=batch.judge.accelerator.label,
                    region_or_cluster=(
                        batch.judge.accelerator.target_cluster or batch.judge.accelerator.region or _UNCONSTRAINED
                    ),
                ),
            )
            if batch.judge is not None
            else None
        ),
        eval=evaluation,
        hardware=HardwareRef(
            task_count=serving.task_count,
            instances=batch.instances,
            platform=batch.accelerator.platform.value,
            accelerator=batch.accelerator.label,
            region_or_cluster=(batch.accelerator.target_cluster or batch.accelerator.region or _UNCONSTRAINED),
        ),
        status=status,
        serving=serving,
        inference_metrics=inference_metrics,
        error=error,
        results_path=identity.output_dir,
        metrics=metrics,
        canonical_metrics=canonical_metrics or {},
        coverage=coverage or {},
        provenance=Provenance(
            git_sha=batch.provenance.git_sha,
            eval_runtime=identity.eval_runtime,
            launch_host=batch.provenance.launch_host,
        ),
        jobs=jobs,
        log_tails=log_tails,
    )
    path = write_record(record, batch.records_prefix)
    logger.info("wrote eval record %s (status=%s)", path, status.value)
    return path


def _job_role(role: str, index: int) -> str:
    return role if index == 0 else f"{role}-{index}"


def _session_job_ids(session: RemoteInferenceSession, role: str) -> dict[str, str]:
    return {_job_role(role, index): str(job.job_id) for index, job in enumerate(session.jobs)}


def _job_tail(handle: JobHandle) -> tuple[str, ...]:
    try:
        return handle.logs(max_lines=100)
    except Exception:
        logger.warning("could not fetch logs for job %s", handle.job_id, exc_info=True)
        return ()


def job_diagnostics(handles: tuple[JobHandle, ...], role: str) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Collect job identities and available log tails for an inference role."""
    jobs = {_job_role(role, index): str(handle.job_id) for index, handle in enumerate(handles)}
    tails = {_job_role(role, index): _job_tail(handle) for index, handle in enumerate(handles)}
    return jobs, tails


def _session_tails(session: RemoteInferenceSession, role: str) -> dict[str, tuple[str, ...]]:
    return {_job_role(role, index): _job_tail(handle) for index, handle in enumerate(session.jobs)}


def run_evaluation(
    executor: EvalExecutor,
    session: RemoteInferenceSession,
    output_dir: str,
    env_vars: Mapping[str, str],
    *,
    endpoint_route: EndpointRoute,
    judge: RemoteInferenceSession | None = None,
) -> EvaluationOutcome:
    """Retry evaluator attempts in the same output directory after inference outages."""
    sessions = (session,) if judge is None else (session, judge)

    def attempt(_attempt_number: int) -> EvaluationOutcome:
        before = tuple(inference.preemption_count() for inference in sessions)
        backend_state = partial(inference_attempt_state, sessions, before)
        execution_session = _local_endpoint_session(session) if endpoint_route is EndpointRoute.DIRECT else session
        # Adapters stop child work before returning or raising EvaluationError.
        try:
            outcome = executor(execution_session, output_dir, env_vars, backend_state=backend_state, judge=judge)
        except EvaluationError as error:
            if error.status is RunStatus.INFRA_FAILED:
                check_inference_backend(backend_state())
            raise
        transport_failures = any(
            entry.errors.get(EVALCHEMY_INFRASTRUCTURE_ERROR, 0) for entry in outcome.coverage.values()
        )
        interrupted = backend_state() is InferenceBackendState.RECOVERING
        if transport_failures and interrupted:
            raise InferenceUnavailable("inference interrupted before evaluator completion")
        return outcome

    return run_with_inference_recovery(attempt, sessions)


def _run_one_evaluation(
    batch: EvaluationBatch,
    evaluation: Evaluation,
    session: RemoteInferenceSession,
    orchestrator_job_id: str,
    env_vars: Mapping[str, str],
    judge: RemoteInferenceSession | None,
) -> str:
    jobs = {_ORCHESTRATOR_ROLE: orchestrator_job_id}
    jobs.update(_session_job_ids(session, _INFERENCE_ROLE))
    if judge is not None:
        jobs.update(_session_job_ids(judge, _JUDGE_ROLE))
    tails: dict[str, tuple[str, ...]] = {}
    metrics: dict[str, dict[str, float]] = {}
    coverage: dict[str, TaskCoverage] = {}
    canonical_metrics: dict[str, dict[str, float]] = {}
    tasks: tuple[EvalTaskRef, ...] | None = None
    status = RunStatus.SUCCEEDED
    error: str | None = None
    inference_metrics: InferenceMetrics | None = None
    metric_window: InferenceMetricWindow | None = None
    try:
        session.check_alive()
        if judge is not None:
            judge.check_alive()
        if session.metrics_url is not None and batch.max_concurrent == 1:
            try:
                metric_session = _local_endpoint_session(session)
                metric_window = InferenceMetricWindow.start(
                    metric_session,
                    speculative=batch.model.serve.speculative is not None,
                )
            except Exception:
                logger.warning(
                    "could not start inference metric capture for evaluation %s",
                    evaluation.identity.eval_ref.name,
                    exc_info=True,
                )
        allowed_env_keys = (*EVAL_RUNTIME_ENV_KEYS, *evaluation.secret_env_keys)
        evaluation_env = {key: env_vars[key] for key in allowed_env_keys if key in env_vars}
        outcome = run_evaluation(
            evaluation.executor,
            session,
            evaluation.identity.output_dir,
            evaluation_env,
            judge=judge,
            endpoint_route=evaluation.endpoint_route,
        )
        metrics = outcome.metrics
        coverage = outcome.coverage
        canonical_metrics = outcome.canonical_metrics
        tasks = outcome.tasks
        jobs |= outcome.jobs
        low_coverage = [
            f"{task}: {entry.n_scored}/{entry.n_attempted}"
            for task, entry in coverage.items()
            if entry.errors.get(EVALCHEMY_INFRASTRUCTURE_ERROR, 0)
            and entry.n_attempted is not None
            and entry.n_attempted > 0
            and entry.n_scored / entry.n_attempted < DEFAULT_MIN_COVERAGE
        ]
        if low_coverage:
            status = RunStatus.INFRA_FAILED
            error = f"infrastructure coverage below {DEFAULT_MIN_COVERAGE:.0%}: {', '.join(low_coverage)}"
    except Exception as exc:
        if isinstance(exc, EvaluationError):
            status = exc.status
            jobs |= exc.jobs
            tails = exc.log_tails
            coverage = exc.coverage
        else:
            logger.exception("unexpected failure in evaluation %s", evaluation.identity.eval_ref.name)
            status = RunStatus.FAILED
        error = f"{type(exc).__name__}: {exc}"
        try:
            session.check_alive()
        except Exception as serve_exc:
            status = RunStatus.INFRA_FAILED
            error = f"{error}; inference failed: {serve_exc}"
            tails |= _session_tails(session, _INFERENCE_ROLE)
        if judge is not None:
            try:
                judge.check_alive()
            except Exception as serve_exc:
                status = RunStatus.INFRA_FAILED
                error = f"{error}; judge inference failed: {serve_exc}"
                tails |= _session_tails(judge, _JUDGE_ROLE)

    if metric_window is not None:
        try:
            inference_metrics = metric_window.finish()
        except Exception:
            logger.warning(
                "could not preserve inference metrics for evaluation %s",
                evaluation.identity.eval_ref.name,
                exc_info=True,
            )

    effective = session.effective_serving
    serving = ServingParams(**asdict(effective), effective=True) if effective is not None else None
    path = _record(
        batch,
        evaluation.identity,
        status,
        error,
        metrics,
        jobs,
        tails,
        coverage,
        canonical_metrics,
        tasks,
        serving=serving,
        inference_metrics=inference_metrics,
    )
    record_rollout_run(
        rollout_run_record(
            run_id=evaluation.identity.run_id,
            run_kind=RolloutRunKind.EVALUATION,
            producer=evaluation.identity.eval_ref.mechanism,
            status=status.value,
            rollout_uri=evaluation.identity.output_dir,
            storage_format="finestore",
            artifact_uri=path,
            model=batch.model.name,
            job_id=orchestrator_job_id,
            attributes={
                "eval_name": evaluation.identity.eval_ref.name,
                **({"eval_runtime": evaluation.identity.eval_runtime} if evaluation.identity.eval_runtime else {}),
                "group_id": batch.group_id,
            },
        )
    )
    if error is not None:
        raise RuntimeError(f"{evaluation.identity.eval_ref.name} ({status.value}): {error}")
    return path


def _record_startup_failure(
    evaluation: Evaluation,
    _output_path: str,
    *,
    batch: EvaluationBatch,
    orchestrator_job_id: str,
    exc: RemoteInferenceStartupError,
    role: str,
    existing_jobs: Mapping[str, str] | None = None,
) -> NoReturn:
    failed_jobs, tails = job_diagnostics(exc.jobs, role)
    jobs = {_ORCHESTRATOR_ROLE: orchestrator_job_id, **(existing_jobs or {}), **failed_jobs}
    error = f"{type(exc).__name__}: {exc}"
    _record(batch, evaluation.identity, RunStatus.INFRA_FAILED, error, {}, jobs, tails)
    subject = "judge inference" if role.startswith((_JUDGE_ROLE, _PREPARED_JUDGE_ROLE)) else "inference"
    raise RuntimeError(f"{evaluation.identity.eval_ref.name} {subject} failed: {exc}") from exc


def run_evaluation_step(
    evaluation: Evaluation,
    _output_path: str,
    *,
    batch: EvaluationBatch,
    session: RemoteInferenceSession,
    judge: RemoteInferenceSession | None,
    orchestrator_job_id: str,
    evaluation_env: Mapping[str, str],
) -> dict[str, str]:
    """Run one evaluator against the supplied sessions and write its result record."""
    path = _run_one_evaluation(batch, evaluation, session, orchestrator_job_id, evaluation_env, judge)
    return {"record_path": path}


def run_evaluation_steps(
    batch: EvaluationBatch,
    build_steps: Callable[[EvaluationStep], tuple[StepSpec, ...]],
    _output_path: str,
) -> None:
    """Run pending evaluation steps against shared candidate and optional judge sessions."""
    configure_coreweave_s3()
    if not batch.evaluations:
        raise ValueError("an evaluation batch requires at least one evaluation")
    if all(
        check_cache(prefix_join(batch.records_prefix, evaluation.identity.run_id)) for evaluation in batch.evaluations
    ):
        return

    def run(execute: EvaluationStep) -> None:
        StepRunner().run(build_steps(execute), max_concurrent=batch.max_concurrent)

    orchestrator_job_id = str(iris_ctx().job_id)
    runtime_env = env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS)
    evaluation_env = {
        **runtime_env,
        **env_vars_from_keys(tuple(batch.secret_env)),
    }
    try:
        prepared = _prepare_all(_inference_configs(batch, runtime_env))
        with ExitStack() as stack:
            session, judge = _open_sessions(stack, prepared)
            run(
                partial(
                    run_evaluation_step,
                    batch=batch,
                    session=session,
                    judge=judge,
                    orchestrator_job_id=orchestrator_job_id,
                    evaluation_env=evaluation_env,
                )
            )
    except _RoleStartupError as failure:
        run(
            partial(
                _record_startup_failure,
                batch=batch,
                orchestrator_job_id=orchestrator_job_id,
                exc=failure.error,
                role=failure.role,
                existing_jobs=failure.existing_jobs,
            )
        )


def _inference_configs(batch: EvaluationBatch, runtime_env: Mapping[str, str]) -> dict[str, RemoteInferenceConfig]:
    """Inference configs keyed by job role: the candidate, the hosted judge, and judges prepared for evaluations."""
    configs = {
        _INFERENCE_ROLE: inference_config_for_model(
            batch.model,
            batch.accelerator,
            env_vars=runtime_env,
            capability_origin=batch.capability_origin,
            api_model=batch.api_model,
            priority=batch.priority_band,
            instances=batch.instances,
            excluded_nvlink_domains=batch.excluded_nvlink_domains,
            request_timeout_seconds=max(
                (
                    float(
                        evaluation.identity.eval_ref.evalchemy.extra_model_args.get(
                            "timeout", batch.model.evaluation_request_timeout
                        )
                    )
                    for evaluation in batch.evaluations
                    if evaluation.identity.eval_ref.evalchemy is not None
                ),
                default=batch.model.evaluation_request_timeout,
            ),
        )
    }
    if batch.judge is not None:
        configs[_JUDGE_ROLE] = inference_config_for_model(
            batch.judge.model,
            batch.judge.accelerator,
            env_vars=runtime_env,
            capability_origin=batch.capability_origin,
            api_model=batch.judge.api_model,
            priority=batch.priority_band,
        )
    for index, judge in enumerate(batch.prepared_judges):
        configs[f"{_PREPARED_JUDGE_ROLE}-{index}"] = inference_config_for_model(
            judge.model,
            judge.accelerator,
            env_vars=runtime_env,
            api_model=judge.api_model,
            priority=batch.priority_band,
        )
    return configs


class _RoleStartupError(Exception):
    """One role's inference failed to prepare or start; ``existing_jobs`` are the jobs of roles already serving."""

    def __init__(self, role: str, error: RemoteInferenceStartupError, existing_jobs: Mapping[str, str] | None = None):
        super().__init__(role, error)
        self.role = role
        self.error = error
        self.existing_jobs = dict(existing_jobs or {})


def _prepare_all(configs: Mapping[str, RemoteInferenceConfig]) -> dict[str, PreparedInference]:
    """Prepare every role's weights in parallel and wait for all of them before any serving job is submitted.

    Raises:
        _RoleStartupError: for the first role, in ``configs`` order, whose preparation failed.
    """
    with ThreadPoolExecutor(max_workers=len(configs)) as pool:
        futures = {
            role: pool.submit(contextvars.copy_context().run, prepare_inference, config)
            for role, config in configs.items()
        }
    prepared: dict[str, PreparedInference] = {}
    for role, future in futures.items():
        try:
            prepared[role] = future.result()
        except RemoteInferenceStartupError as exc:
            raise _RoleStartupError(role, exc) from exc
    return prepared


def _open_sessions(
    stack: ExitStack, prepared: Mapping[str, PreparedInference]
) -> tuple[RemoteInferenceSession, RemoteInferenceSession | None]:
    """Serve the prepared candidate, then the prepared judge if there is one.

    Raises:
        _RoleStartupError: for the role whose serving failed to start.
    """
    try:
        session = _inference_session(stack, prepared[_INFERENCE_ROLE])
    except RemoteInferenceStartupError as exc:
        raise _RoleStartupError(_INFERENCE_ROLE, exc) from exc
    if _JUDGE_ROLE not in prepared:
        return session, None
    try:
        return session, _inference_session(stack, prepared[_JUDGE_ROLE])
    except RemoteInferenceStartupError as exc:
        raise _RoleStartupError(_JUDGE_ROLE, exc, _session_job_ids(session, _INFERENCE_ROLE)) from exc


def _inference_session(stack: ExitStack, prepared: PreparedInference) -> RemoteInferenceSession:
    """Start an eval server, retrying readiness timeouts within its failure budget."""
    return retry_with_backoff(
        lambda: stack.enter_context(serve_inference(prepared)),
        retryable=lambda error: isinstance(error, RemoteInferenceStartupError)
        and error.phase is InferenceStartupPhase.SERVING
        and isinstance(error.__cause__, TimeoutError),
        max_attempts=prepared.config.iris.max_retries_failure + 1,
        operation="evaluation inference startup",
    )


def _local_endpoint_session(session: RemoteInferenceSession) -> RemoteInferenceSession:
    """Return an eval session that reaches the serving endpoint directly."""
    try:
        address = iris_ctx().client.resolve_endpoint(session.endpoint_name).rstrip("/")
    except ConnectError as exc:
        if exc.code != Code.NOT_FOUND:
            raise
        raise InferenceUnavailable("inference endpoint registration is unavailable") from exc
    endpoint = replace(session.model.endpoint, base_url=f"{address}{OPENAI_API_SUFFIX}")
    return replace(
        session,
        model=replace(session.model, endpoint=endpoint),
        metrics_url=f"{address}/metrics" if session.metrics_url is not None else None,
    )


def print_record(record: EvalRunRecord) -> EvalRunRecord:
    """Print a run's status, failure context, coverage and metrics.

    Curated records carry no metrics; their scores are derived from the run's raw archive.
    """
    record = score_curated_record(record)
    print(f"{record.run_id}  [{record.status.value}]  {record.model.name} / {record.evaluation.name}")
    if record.error:
        print(f"  error: {record.error}")
        for role, job_path in sorted(record.jobs.items()):
            print(f"  {role} job: {job_path}")
        for role, lines in sorted(record.log_tails.items()):
            if not lines:
                continue
            print(f"  last {min(len(lines), _REPORT_TAIL_LINES)} log lines of the {role} child:")
            for line in lines[-_REPORT_TAIL_LINES:]:
                print(f"    {line}")
    print(f"  results: {record.results_path}")
    for task, coverage in sorted(record.coverage.items()):
        errors = ", ".join(f"{name}={count}" for name, count in sorted(coverage.errors.items()))
        print(
            f"  {task:<40} scored {coverage.n_scored}/{coverage.n_attempted} selected "
            f"of {coverage.n_benchmark}" + (f"  errors: {errors}" if errors else "")
        )
    if not record.metrics:
        print("  (no metrics)")
    for task in sorted(record.metrics):
        for metric in sorted(record.metrics[task]):
            print(f"  {task:<40} {metric:<24} {record.metrics[task][metric]:.4f}")
    return record


def wait_and_report(
    batches: list[SubmittedEvaluationBatch],
    *,
    report_group: Callable[[list[EvalRunRecord]], None] | None = None,
) -> bool:
    """Wait for submitted batches and print their durable records.

    Returns whether every evaluation wrote a record with succeeded status. For one that wrote no
    record, prints the orchestrator job's terminal state and error, which explain rejections before
    any record exists.
    """
    configure_coreweave_s3()
    complete = True
    for batch in batches:
        records = []
        status = batch.job.wait(timeout=float("inf"), raise_on_failure=False)
        for evaluation in batch.evaluations:
            path = record_path(batch.records_prefix, evaluation.run_id)
            try:
                record = read_record(path)
            except Exception as error:
                complete = False
                print(f"{evaluation.run_id}  [no record]  {batch.model_name} / {evaluation.eval_name}")
                print(f"  job {batch.job.job_id}: {job_state_friendly(status.state)}")
                if status.error:
                    print(f"  error: {status.error}")
                print(f"  record {path} is unreadable: {type(error).__name__}: {error}")
                continue
            records.append(print_record(record))
            complete = complete and record.status is RunStatus.SUCCEEDED
        if report_group is not None:
            report_group(records)
    return complete
