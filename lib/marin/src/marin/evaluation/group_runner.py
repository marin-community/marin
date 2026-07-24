# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a resolved evaluation group against one shared inference session."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from fray.iris_backend import IrisJobHandle
from iris.client import iris_ctx
from iris.cluster.types import EndpointAccess
from rigging.filesystem.s3_compat import configure_coreweave_s3

from marin.evaluation.definitions import (
    EvalRunIdentity,
    ResolvedEvalchemyRun,
    ResolvedEvalRun,
    ResolvedHarborRun,
)
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.evalchemy import (
    EndpointMovedError,
    EvalPipelineError,
    PipelineStage,
    job_log_tail,
    run_evalchemy,
)
from marin.evaluation.harbor_runner import run_harbor
from marin.evaluation.records import (
    EvalRunRecord,
    HardwareRef,
    ModelRef,
    Provenance,
    RunStatus,
    write_record,
)
from marin.evaluation.serving_config import EvaluationServingConfig, resolve_inference_launch
from marin.inference.iris import RemoteInferenceSession, RemoteInferenceStartupError, remote_inference
from marin.inference.types import EndpointRoute, RunningModel

logger = logging.getLogger(__name__)

_ENDPOINT_RESTART_TIMEOUT = 2400.0
_SERVE_ROLE = "serve"


@dataclass(frozen=True)
class EvalGroupParams:
    """Complete cloudpickle-safe payload for one evaluation worker."""

    group_id: str
    user: str
    version: str | None
    description: str | None
    records_prefix: str
    serving: EvaluationServingConfig
    runs: tuple[ResolvedEvalRun, ...]
    model_ref: ModelRef
    hardware_ref: HardwareRef
    provenance: Provenance
    capability_origin: str | None = None

    def __post_init__(self) -> None:
        has_harbor = any(isinstance(run, ResolvedHarborRun) for run in self.runs)
        if has_harbor and self.capability_origin is None:
            raise ValueError("Harbor groups require a capability origin")
        if has_harbor and self.serving.spec.broker is not None:
            raise ValueError("Harbor groups require direct inference")
        if has_harbor and self.serving.spec.endpoint_access != EndpointAccess.ENDPOINT_ACCESS_LINK:
            raise ValueError("Harbor groups require a link-accessible endpoint")


def _record(
    group: EvalGroupParams,
    identity: EvalRunIdentity,
    status: RunStatus,
    error: str | None,
    metrics: dict[str, dict[str, float]],
    jobs: dict[str, str],
    log_tails: dict[str, tuple[str, ...]],
) -> str:
    record = EvalRunRecord(
        run_id=identity.run_id,
        group_id=group.group_id,
        created_at=identity.created_at,
        user=group.user,
        version=group.version,
        description=group.description,
        model=group.model_ref,
        eval=identity.eval_ref,
        hardware=group.hardware_ref,
        status=status,
        error=error,
        results_path=identity.output_dir,
        metrics=metrics,
        provenance=group.provenance,
        jobs=jobs,
        log_tails=log_tails,
    )
    path = write_record(record, group.records_prefix)
    logger.info("wrote eval record %s (status=%s)", path, status.value)
    return path


def _serve_jobs(session: RemoteInferenceSession) -> dict[str, str]:
    jobs: dict[str, str] = {}
    for index, job in enumerate(session.jobs):
        role = _serve_role(index)
        jobs[role] = str(job.job_id)
    return jobs


def _serve_role(index: int) -> str:
    return _SERVE_ROLE if index == 0 else f"{_SERVE_ROLE}-{index}"


def _startup_diagnostics(exc: RemoteInferenceStartupError) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    jobs: dict[str, str] = {}
    tails: dict[str, tuple[str, ...]] = {}
    for index, handle in enumerate(exc.jobs):
        role = _serve_role(index)
        jobs[role] = str(handle.job_id)
        if isinstance(handle, IrisJobHandle):
            tails[role] = job_log_tail(handle.iris_job)
    return jobs, tails


def _session_tail(session: RemoteInferenceSession) -> dict[str, tuple[str, ...]]:
    tails: dict[str, tuple[str, ...]] = {}
    for index, handle in enumerate(session.jobs):
        if not isinstance(handle, IrisJobHandle):
            continue
        role = _serve_role(index)
        tails[role] = job_log_tail(handle.iris_job)
    return tails


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _pipeline_status(exc: Exception) -> RunStatus:
    if isinstance(exc, EvalPipelineError) and exc.stage is PipelineStage.EVAL:
        return RunStatus.FAILED
    return RunStatus.INFRA_FAILED


def _run_evalchemy_once(
    session: RemoteInferenceSession,
    model: RunningModel,
    run: ResolvedEvalchemyRun,
    env_vars: Mapping[str, str],
):
    return run_evalchemy(
        model,
        run.config,
        run.identity.output_dir,
        observer=session,
        env_vars=env_vars,
    )


def _run_evalchemy_with_restart(
    session: RemoteInferenceSession,
    run: ResolvedEvalchemyRun,
    env_vars: Mapping[str, str],
):
    model = session.resolve_model()
    try:
        return _run_evalchemy_once(session, model, run, env_vars)
    except EndpointMovedError:
        logger.info("eval %s: inference moved mid-run; retrying once", run.identity.eval_ref.name)
        model = session.wait_model(_ENDPOINT_RESTART_TIMEOUT)
        return _run_evalchemy_once(session, model, run, env_vars)


def _record_unstarted(
    group: EvalGroupParams,
    runs: tuple[ResolvedEvalRun, ...],
    error: Exception,
    jobs: dict[str, str],
    tails: dict[str, tuple[str, ...]],
    paths: list[str],
) -> None:
    for run in runs:
        paths.append(
            _record(
                group,
                run.identity,
                RunStatus.INFRA_FAILED,
                _error_text(error),
                {},
                jobs,
                tails,
            )
        )


def _capability_model(
    group: EvalGroupParams,
    session: RemoteInferenceSession,
) -> RunningModel | None:
    if not any(isinstance(run, ResolvedHarborRun) for run in group.runs):
        return None
    assert group.capability_origin is not None
    return session.resolve_model(
        EndpointRoute.CAPABILITY,
        public_origin=group.capability_origin,
    )


def _run_mechanism(
    session: RemoteInferenceSession,
    run: ResolvedEvalRun,
    capability_model: RunningModel | None,
    env_vars: Mapping[str, str],
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    if isinstance(run, ResolvedEvalchemyRun):
        outcome = _run_evalchemy_with_restart(session, run, env_vars)
        metrics = outcome.result.task_metrics()
        if not metrics:
            raise RuntimeError(f"eval finished but no task metrics were readable under {run.identity.output_dir!r}")
        return metrics, outcome.jobs

    assert capability_model is not None
    result = run_harbor(
        capability_model,
        run.config,
        run.identity.output_dir,
        hf_token=env_vars.get("HF_TOKEN"),
    )
    if not result.total_trials:
        raise RuntimeError(f"Harbor eval finished with no trials under {run.identity.output_dir!r}")
    return result.task_metrics(), {}


def _run_ready_group(
    group: EvalGroupParams,
    session: RemoteInferenceSession,
    base_jobs: dict[str, str],
    env_vars: Mapping[str, str],
) -> list[str]:
    paths: list[str] = []
    failed: list[str] = []
    serve_jobs = _serve_jobs(session)
    try:
        capability_model = _capability_model(group, session)
    except Exception as exc:
        _record_unstarted(
            group,
            group.runs,
            exc,
            base_jobs | serve_jobs,
            _session_tail(session),
            paths,
        )
        raise RuntimeError("failed to create the Harbor capability route") from exc

    for index, run in enumerate(group.runs):
        jobs = base_jobs | serve_jobs
        tails: dict[str, tuple[str, ...]] = {}
        metrics: dict[str, dict[str, float]] = {}
        status = RunStatus.SUCCEEDED
        error: str | None = None
        try:
            session.check_alive()
            metrics, mechanism_jobs = _run_mechanism(session, run, capability_model, env_vars)
            jobs |= mechanism_jobs
        except Exception as exc:
            status = _pipeline_status(exc) if isinstance(run, ResolvedEvalchemyRun) else RunStatus.FAILED
            error = _error_text(exc)
            if isinstance(exc, EvalPipelineError):
                jobs |= exc.jobs
                tails = exc.log_tails
            try:
                session.check_alive()
            except Exception as serve_exc:
                status = RunStatus.INFRA_FAILED
                error = f"{error}; inference failed: {serve_exc}"
                tails |= _session_tail(session)
                paths.append(_record(group, run.identity, status, error, {}, jobs, tails))
                failed.append(f"{run.identity.eval_ref.name} ({status.value})")
                _record_unstarted(
                    group,
                    group.runs[index + 1 :],
                    serve_exc,
                    jobs,
                    tails,
                    paths,
                )
                failed.extend(
                    f"{rest.identity.eval_ref.name} ({RunStatus.INFRA_FAILED.value})" for rest in group.runs[index + 1 :]
                )
                break

        paths.append(_record(group, run.identity, status, error, metrics, jobs, tails))
        if error is not None:
            failed.append(f"{run.identity.eval_ref.name} ({status.value})")

    if failed:
        raise RuntimeError(f"{len(failed)} of {len(group.runs)} evals failed: {', '.join(failed)}")
    return paths


def run_eval_group(group: EvalGroupParams) -> list[str]:
    """Serve once, run every resolved evaluation, and record each result progressively."""
    configure_coreweave_s3()
    if not group.runs:
        raise ValueError("an evaluation group requires at least one run")
    base_jobs = {"orchestrator": str(iris_ctx().job_id)}
    runtime_env = env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS)
    inference = resolve_inference_launch(group.serving, runtime_env)
    try:
        with remote_inference(
            inference.model,
            inference.engine,
            inference.iris,
            instances=inference.instances,
            broker=inference.broker,
            endpoint_access=inference.endpoint_access,
        ) as session:
            return _run_ready_group(group, session, base_jobs, runtime_env)
    except RemoteInferenceStartupError as exc:
        jobs, tails = _startup_diagnostics(exc)
        paths: list[str] = []
        _record_unstarted(group, group.runs, exc, base_jobs | jobs, tails, paths)
        raise RuntimeError(f"evaluation group inference failed: {exc}") from exc
