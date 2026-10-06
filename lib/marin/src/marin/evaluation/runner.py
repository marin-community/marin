# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run endpoint-oriented evaluations as independent steps."""

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import NoReturn, Protocol

from fray.client import JobHandle
from iris.client.client import Job, iris_ctx
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import SecretSpec

from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.eval_stats import DEFAULT_MIN_COVERAGE
from marin.evaluation.hardware import AcceleratorChoice
from marin.evaluation.inference_metrics import InferenceMetricWindow
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.model_identity import model_config_digest
from marin.evaluation.records import (
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
    RunStatus,
    ServingParams,
    TaskCoverage,
    read_record,
    record_path,
    write_record,
)
from marin.evaluation.serving_config import inference_config_for_model
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.inference.backend import OPENAI_API_SUFFIX
from marin.inference.iris import RemoteInferenceSession, RemoteInferenceStartupError, remote_inference
from marin.rollouts.catalog import RolloutRunKind, record_rollout_run, rollout_run_record

logger = logging.getLogger(__name__)

_INFERENCE_ROLE = "inference"
_ORCHESTRATOR_ROLE = "orchestrator"
_JUDGE_ROLE = "judge"
_UNCONSTRAINED = "unconstrained"
_REPORT_TAIL_LINES = 15


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
        """Coverage measured before the failure, so a rejected run records why it was rejected as
        structured counts rather than only as an error string."""


class EvalExecutor(Protocol):
    """Execute one evaluation mechanism against an inference session."""

    def __call__(
        self,
        session: RemoteInferenceSession,
        output_dir: str,
        env_vars: Mapping[str, str],
        *,
        judge: RemoteInferenceSession | None = None,
    ) -> EvaluationOutcome: ...


class EndpointRoute(StrEnum):
    DIRECT = "direct"
    CAPABILITY = "capability"


@dataclass(frozen=True)
class EvaluationIdentity:
    run_id: str
    created_at: str
    output_dir: str
    eval_ref: EvalRef
    eval_runtime: str


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
    judge: HostedJudge | None = None
    secret_env: Mapping[str, SecretSpec] = field(default_factory=dict)
    source_model_config: ModelConfigRef | None = None


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
        serving = ServingParams(
            tensor_parallel_size=serve.tensor_parallel_size,
            data_parallel_size=serve.data_parallel_size,
            pipeline_parallel_size=serve.pipeline_parallel_size,
            task_count=serve.pipeline_parallel_size,
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


def _job_diagnostics(handles: tuple[JobHandle, ...], role: str) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    jobs = {_job_role(role, index): str(handle.job_id) for index, handle in enumerate(handles)}
    tails = {_job_role(role, index): _job_tail(handle) for index, handle in enumerate(handles)}
    return jobs, tails


def _session_tails(session: RemoteInferenceSession, role: str) -> dict[str, tuple[str, ...]]:
    return {_job_role(role, index): _job_tail(handle) for index, handle in enumerate(session.jobs)}


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
        if evaluation.endpoint_route is EndpointRoute.DIRECT:
            execution_session = _local_endpoint_session(session)
        else:
            execution_session = session
        if session.metrics_url is not None:
            try:
                metric_session = execution_session
                if evaluation.endpoint_route is EndpointRoute.CAPABILITY:
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
        outcome = evaluation.executor(
            execution_session,
            evaluation.identity.output_dir,
            evaluation_env,
            judge=judge,
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
                "eval_runtime": evaluation.identity.eval_runtime,
                "group_id": batch.group_id,
            },
        )
    )
    if error is not None:
        raise RuntimeError(f"{evaluation.identity.eval_ref.name} ({status.value}): {error}")
    return path


def _record_startup_failure(
    batch: EvaluationBatch,
    evaluation: Evaluation,
    orchestrator_job_id: str,
    exc: RemoteInferenceStartupError,
    role: str,
    existing_jobs: Mapping[str, str] | None = None,
) -> NoReturn:
    failed_jobs, tails = _job_diagnostics(exc.jobs, role)
    jobs = {_ORCHESTRATOR_ROLE: orchestrator_job_id, **(existing_jobs or {}), **failed_jobs}
    error = f"{type(exc).__name__}: {exc}"
    _record(batch, evaluation.identity, RunStatus.INFRA_FAILED, error, {}, jobs, tails)
    subject = "judge inference" if role == _JUDGE_ROLE else "inference"
    raise RuntimeError(f"{evaluation.identity.eval_ref.name} {subject} failed: {exc}") from exc


def _evaluate_with_hosted_judge(
    batch: EvaluationBatch,
    evaluation: Evaluation,
    session: RemoteInferenceSession,
    orchestrator_job_id: str,
    runtime_env: Mapping[str, str],
    evaluation_env: Mapping[str, str],
) -> str:
    if batch.judge is None:
        return _run_one_evaluation(batch, evaluation, session, orchestrator_job_id, evaluation_env, None)
    judge_inference = inference_config_for_model(
        batch.judge.model,
        batch.judge.accelerator,
        env_vars=runtime_env,
        capability_origin=batch.capability_origin,
        api_model=batch.judge.api_model,
        priority=batch.priority_band,
    )
    try:
        with remote_inference(judge_inference) as judge:
            return _run_one_evaluation(batch, evaluation, session, orchestrator_job_id, evaluation_env, judge)
    except RemoteInferenceStartupError as exc:
        _record_startup_failure(
            batch,
            evaluation,
            orchestrator_job_id,
            exc,
            _JUDGE_ROLE,
            _session_job_ids(session, _INFERENCE_ROLE),
        )


def run_evaluation_step(batch: EvaluationBatch, evaluation: Evaluation, _output_path: str) -> dict[str, str]:
    """Serve one evaluation inside its StepSpec and write its result record."""
    configure_coreweave_s3()
    orchestrator_job_id = str(iris_ctx().job_id)
    runtime_env = env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS)
    evaluation_env = {
        **runtime_env,
        **env_vars_from_keys(tuple(batch.secret_env)),
    }
    inference = inference_config_for_model(
        batch.model,
        batch.accelerator,
        env_vars=runtime_env,
        capability_origin=batch.capability_origin,
        api_model=batch.api_model,
        priority=batch.priority_band,
    )
    try:
        with remote_inference(inference) as session:
            path = _evaluate_with_hosted_judge(
                batch,
                evaluation,
                session,
                orchestrator_job_id,
                runtime_env,
                evaluation_env,
            )
    except RemoteInferenceStartupError as exc:
        _record_startup_failure(batch, evaluation, orchestrator_job_id, exc, _INFERENCE_ROLE)
    return {"record_path": path}


def run_evaluation_steps(steps: tuple[StepSpec, ...], max_concurrent: int) -> None:
    """Run each independent evaluation step, skipping successful steps on restart."""
    if not steps:
        raise ValueError("an evaluation batch requires at least one evaluation")
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be at least 1")
    StepRunner().run(steps, max_concurrent=max_concurrent)


def _local_endpoint_session(session: RemoteInferenceSession) -> RemoteInferenceSession:
    """Return an eval session that reaches the serving endpoint directly."""
    address = iris_ctx().client.resolve_endpoint(session.endpoint_name).rstrip("/")
    endpoint = replace(session.model.endpoint, base_url=f"{address}{OPENAI_API_SUFFIX}")
    return replace(
        session,
        model=replace(session.model, endpoint=endpoint),
        metrics_url=f"{address}/metrics" if session.metrics_url is not None else None,
    )


def _print_record(record: EvalRunRecord) -> None:
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
    if not record.metrics:
        print("  (no metrics)")
        return
    for task in sorted(record.metrics):
        for metric in sorted(record.metrics[task]):
            print(f"  {task:<40} {metric:<24} {record.metrics[task][metric]:.4f}")


def wait_and_report(batches: list[SubmittedEvaluationBatch]) -> None:
    """Wait for submitted batches and print their durable records."""
    configure_coreweave_s3()
    for batch in batches:
        batch.job.wait(timeout=float("inf"), raise_on_failure=False)
        for evaluation in batch.evaluations:
            path = record_path(batch.records_prefix, evaluation.run_id)
            try:
                record = read_record(path)
            except Exception:
                logger.warning(
                    "no readable record.json for run %s at %s",
                    evaluation.run_id,
                    path,
                    exc_info=True,
                )
                print(f"{evaluation.run_id}  [no record]  {batch.model_name} / {evaluation.eval_name}")
                continue
            _print_record(record)
