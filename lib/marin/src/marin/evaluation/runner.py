# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve one model and run a batch of endpoint-oriented evaluations."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from fray.client import JobHandle
from iris.client import IrisClient, Job, iris_ctx
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp, region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from rigging.filesystem.s3_compat import configure_coreweave_s3

from marin.evaluation.eval_env import EVAL_ENV_KEYS, EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import AcceleratorChoice
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import (
    EvalRef,
    EvalRunRecord,
    HardwareRef,
    ModelRef,
    Provenance,
    RunStatus,
    read_record,
    record_path,
    write_record,
)
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import RemoteInferenceSession, RemoteInferenceStartupError, remote_inference
from marin.inference.types import RunningModel

logger = logging.getLogger(__name__)

_INFERENCE_ROLE = "inference"
_ORCHESTRATOR_ROLE = "orchestrator"
_ORCHESTRATOR_CPU = 4.0
_ORCHESTRATOR_MEMORY = "16g"
_ORCHESTRATOR_DISK = "16g"
_REPORT_TAIL_LINES = 15


@dataclass(frozen=True)
class EvaluationOutcome:
    metrics: dict[str, dict[str, float]]
    jobs: dict[str, str] = field(default_factory=dict)


class EvaluationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: RunStatus,
        jobs: dict[str, str] | None = None,
        log_tails: dict[str, tuple[str, ...]] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.jobs = jobs or {}
        self.log_tails = log_tails or {}


class EvalExecutor(Protocol):
    """Execute one evaluation mechanism against an already-running OpenAI endpoint."""

    def __call__(
        self,
        model: RunningModel,
        output_dir: str,
        env_vars: Mapping[str, str],
    ) -> EvaluationOutcome: ...


@dataclass(frozen=True)
class EvaluationIdentity:
    run_id: str
    created_at: str
    output_dir: str
    eval_ref: EvalRef


@dataclass(frozen=True)
class Evaluation:
    identity: EvaluationIdentity
    executor: EvalExecutor


@dataclass(frozen=True)
class EvaluationBatch:
    group_id: str
    user: str
    version: str | None
    description: str | None
    records_prefix: str
    model: ModelConfig
    accelerator: AcceleratorChoice
    capability_origin: str
    api_model: str | None
    evaluations: tuple[Evaluation, ...]
    provenance: Provenance


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
) -> str:
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
        ),
        eval=identity.eval_ref,
        hardware=HardwareRef(
            platform=batch.accelerator.platform.value,
            accelerator=batch.accelerator.label,
            region_or_cluster=(batch.accelerator.target_cluster or batch.accelerator.region or "unconstrained"),
        ),
        status=status,
        error=error,
        results_path=identity.output_dir,
        metrics=metrics,
        provenance=batch.provenance,
        jobs=jobs,
        log_tails=log_tails,
    )
    path = write_record(record, batch.records_prefix)
    logger.info("wrote eval record %s (status=%s)", path, status.value)
    return path


def _inference_role(index: int) -> str:
    return _INFERENCE_ROLE if index == 0 else f"{_INFERENCE_ROLE}-{index}"


def _inference_job_ids(session: RemoteInferenceSession) -> dict[str, str]:
    return {_inference_role(index): str(job.job_id) for index, job in enumerate(session.jobs)}


def _job_tail(handle: JobHandle) -> tuple[str, ...]:
    try:
        return handle.logs(max_lines=100)
    except Exception:
        logger.warning("could not fetch logs for job %s", handle.job_id, exc_info=True)
        return ()


def _startup_diagnostics(exc: RemoteInferenceStartupError) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    jobs: dict[str, str] = {}
    tails: dict[str, tuple[str, ...]] = {}
    for index, handle in enumerate(exc.jobs):
        role = _inference_role(index)
        jobs[role] = str(handle.job_id)
        tails[role] = _job_tail(handle)
    return jobs, tails


def _session_tail(session: RemoteInferenceSession) -> dict[str, tuple[str, ...]]:
    return {_inference_role(index): _job_tail(handle) for index, handle in enumerate(session.jobs)}


def _record_unstarted(
    batch: EvaluationBatch,
    evaluations: tuple[Evaluation, ...],
    error: Exception,
    jobs: dict[str, str],
    tails: dict[str, tuple[str, ...]],
    paths: list[str],
) -> None:
    message = f"{type(error).__name__}: {error}"
    for evaluation in evaluations:
        paths.append(
            _record(
                batch,
                evaluation.identity,
                RunStatus.INFRA_FAILED,
                message,
                {},
                jobs,
                tails,
            )
        )


@dataclass(frozen=True)
class _EvaluationExecution:
    record_path: str
    failure: str | None
    inference_failure: Exception | None
    jobs: dict[str, str]
    log_tails: dict[str, tuple[str, ...]]


def _run_one_evaluation(
    batch: EvaluationBatch,
    evaluation: Evaluation,
    session: RemoteInferenceSession,
    orchestrator_job_id: str,
    env_vars: Mapping[str, str],
) -> _EvaluationExecution:
    jobs = {_ORCHESTRATOR_ROLE: orchestrator_job_id}
    jobs.update(_inference_job_ids(session))
    tails: dict[str, tuple[str, ...]] = {}
    metrics: dict[str, dict[str, float]] = {}
    status = RunStatus.SUCCEEDED
    error: str | None = None
    inference_failure: Exception | None = None
    try:
        session.check_alive()
        outcome = evaluation.executor(session.model, evaluation.identity.output_dir, env_vars)
        metrics = outcome.metrics
        jobs |= outcome.jobs
    except Exception as exc:
        if isinstance(exc, EvaluationError):
            status = exc.status
            jobs |= exc.jobs
            tails = exc.log_tails
        else:
            logger.exception("unexpected failure in evaluation %s", evaluation.identity.eval_ref.name)
            status = RunStatus.FAILED
        error = f"{type(exc).__name__}: {exc}"
        try:
            session.check_alive()
        except Exception as serve_exc:
            status = RunStatus.INFRA_FAILED
            error = f"{error}; inference failed: {serve_exc}"
            tails |= _session_tail(session)
            inference_failure = serve_exc

    path = _record(batch, evaluation.identity, status, error, metrics, jobs, tails)
    failure = f"{evaluation.identity.eval_ref.name} ({status.value})" if error is not None else None
    return _EvaluationExecution(
        record_path=path,
        failure=failure,
        inference_failure=inference_failure,
        jobs=jobs,
        log_tails=tails,
    )


def evaluate_batch(
    batch: EvaluationBatch,
    session: RemoteInferenceSession,
    *,
    orchestrator_job_id: str,
    env_vars: Mapping[str, str],
) -> list[str]:
    """Run a batch against one inference context and persist a record per evaluation."""
    paths: list[str] = []
    failed: list[str] = []

    for index, evaluation in enumerate(batch.evaluations):
        execution = _run_one_evaluation(batch, evaluation, session, orchestrator_job_id, env_vars)
        paths.append(execution.record_path)
        if execution.failure is not None:
            failed.append(execution.failure)
        if execution.inference_failure is None:
            continue
        remaining = batch.evaluations[index + 1 :]
        _record_unstarted(
            batch,
            remaining,
            execution.inference_failure,
            execution.jobs,
            execution.log_tails,
            paths,
        )
        failed.extend(f"{rest.identity.eval_ref.name} ({RunStatus.INFRA_FAILED.value})" for rest in remaining)
        break

    if failed:
        raise RuntimeError(f"{len(failed)} of {len(batch.evaluations)} evals failed: {', '.join(failed)}")
    return paths


def run_evaluation_batch(batch: EvaluationBatch) -> list[str]:
    """Serve once, run every evaluation, and write each record as it finishes."""
    configure_coreweave_s3()
    if not batch.evaluations:
        raise ValueError("an evaluation batch requires at least one evaluation")
    orchestrator_job_id = str(iris_ctx().job_id)
    runtime_env = env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS)
    inference = inference_config_for_model(
        batch.model,
        batch.accelerator,
        env_vars=runtime_env,
        capability_origin=batch.capability_origin,
        api_model=batch.api_model,
    )
    try:
        with remote_inference(inference) as session:
            return evaluate_batch(
                batch,
                session,
                orchestrator_job_id=orchestrator_job_id,
                env_vars=runtime_env,
            )
    except RemoteInferenceStartupError as exc:
        inference_jobs, tails = _startup_diagnostics(exc)
        jobs = {_ORCHESTRATOR_ROLE: orchestrator_job_id}
        jobs.update(inference_jobs)
        paths: list[str] = []
        _record_unstarted(batch, batch.evaluations, exc, jobs, tails, paths)
        raise RuntimeError(f"evaluation batch inference failed: {exc}") from exc


def submit_evaluation_batch(batch: EvaluationBatch, client: IrisClient) -> SubmittedEvaluationBatch:
    """Submit a resolved batch to one CPU orchestrator."""
    constraints = None
    if batch.accelerator.target_cluster:
        constraints = [
            Constraint.create(
                key=CLUSTER_CONSTRAINT_KEY,
                op=ConstraintOp.EQ,
                value=batch.accelerator.target_cluster,
            )
        ]
    elif batch.accelerator.region:
        constraints = [region_constraint([batch.accelerator.region])]
    job = client.submit(
        entrypoint=Entrypoint.from_callable(run_evaluation_batch, batch),
        name=f"eval-{batch.group_id}",
        resources=ResourceSpec(
            cpu=_ORCHESTRATOR_CPU,
            memory=_ORCHESTRATOR_MEMORY,
            disk=_ORCHESTRATOR_DISK,
        ),
        environment=EnvironmentSpec(env_vars=env_vars_from_keys(EVAL_ENV_KEYS)),
        constraints=constraints,
        max_retries_failure=0,
    )
    logger.info("submitted eval batch %s (%d evals) as job %s", batch.group_id, len(batch.evaluations), job)
    return SubmittedEvaluationBatch(
        group_id=batch.group_id,
        job=job,
        records_prefix=batch.records_prefix,
        model_name=batch.model.name,
        evaluations=tuple(
            SubmittedEvaluation(
                run_id=evaluation.identity.run_id,
                eval_name=evaluation.identity.eval_ref.name,
            )
            for evaluation in batch.evaluations
        ),
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
