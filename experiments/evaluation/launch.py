# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit eval groups and record their outcomes.

:func:`plan_runs` resolves a :class:`LaunchSpec` into one :class:`RunPlan` per eval;
:func:`launch_group` submits ONE CPU orchestrator job for the whole launch, which runs
:func:`run_eval_group`: serve the model once, evaluate every eval against the endpoint in order, and
write one ``record.json`` per eval as it finishes (records share a ``group_id``).
:func:`wait_and_report` waits on the group job, reads the records back, and prints per-run metrics.

The GPU path routes the whole orchestrator to a CoreWeave peer via a ``cluster`` constraint, so its
serve + eval child jobs land on that cluster too (they inherit the orchestrator's in-cluster client).
"""

from __future__ import annotations

import getpass
import logging
import os
import socket
import subprocess
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from iris.client import IrisClient, Job, iris_ctx
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from marin.evaluation.eval_result import EvalchemyResult
from marin.evaluation.evaluators.harbor_evaluator import HARBOR_EVAL_ENV_KEYS, env_vars_from_keys
from marin.evaluation.records import (
    CW_RECORDS_PREFIX,
    DEFAULT_RECORDS_PREFIX,
    EvalRef,
    EvalRunRecord,
    EvalTaskRef,
    HardwareRef,
    ModelRef,
    Provenance,
    RunStatus,
    read_record,
    record_path,
    write_record,
)
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.evals.evalchemy.image import EVALCHEMY_IMAGE
from experiments.evals.evalchemy.serve_and_eval import (
    EvalPipelineError,
    EvalSession,
    EvalUnit,
    PipelineStage,
    ServeSpec,
    run_eval_units,
)
from experiments.evaluation.evals import EVALS, EvalMechanism, EvalSuiteConfig
from experiments.evaluation.hardware import AcceleratorChoice, Platform, select_accelerator
from experiments.evaluation.models import MODELS, EvalModelConfig

logger = logging.getLogger(__name__)

# The orchestrator is a lightweight CPU job: it submits the serve + eval children and waits. The
# serving slice rides on the run's ServeSpec, not on this resource.
_ORCHESTRATOR_CPU = 1.0
_ORCHESTRATOR_MEMORY = "4g"
_ORCHESTRATOR_DISK = "16g"


# --------------------------------------------------------------------------------------------------
# Worker entrypoint: serve once, run every eval, write a record per eval (the CPU orchestrator job).
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRunParams:
    """One recorded eval within a group: its identity, its durable results path, and its eval unit."""

    run_id: str
    created_at: str
    out_path: str
    eval_ref: EvalRef
    unit: EvalUnit


@dataclass(frozen=True)
class EvalGroupParams:
    """Everything one orchestrator job needs, cloudpickled into it.

    The record fields are prebuilt at launch (git SHA, host, user) because the worker cannot recover
    them. One group serves the model once and writes one record per entry in ``runs``.
    """

    group_id: str
    user: str
    records_prefix: str
    session: EvalSession
    runs: tuple[EvalRunParams, ...]
    model_ref: ModelRef
    hardware_ref: HardwareRef
    provenance: Provenance


def _classify_failure(exc: Exception) -> RunStatus:
    """An eval-stage failure is a result about the model (``FAILED``); anything else is infra."""
    if isinstance(exc, EvalPipelineError) and exc.stage is PipelineStage.EVAL:
        return RunStatus.FAILED
    return RunStatus.INFRA_FAILED


def _build_record(
    group: EvalGroupParams,
    run: EvalRunParams,
    status: RunStatus,
    error: str | None,
    metrics: dict[str, dict[str, float]],
    jobs: dict[str, str],
    log_tails: dict[str, tuple[str, ...]],
) -> EvalRunRecord:
    return EvalRunRecord(
        run_id=run.run_id,
        group_id=group.group_id,
        created_at=run.created_at,
        user=group.user,
        model=group.model_ref,
        evaluation=run.eval_ref,
        hardware=group.hardware_ref,
        status=status,
        error=error,
        results_path=run.out_path,
        metrics=metrics,
        provenance=group.provenance,
        jobs=jobs,
        log_tails=log_tails,
    )


def run_eval_group(params: EvalGroupParams) -> list[str]:
    """Serve once, run every eval in the group, and write one ``record.json`` per eval as it finishes.

    Each record is written before the next eval starts, so results land progressively across an
    hours-long suite. A failed eval is recorded (``FAILED`` for an eval-stage failure,
    ``INFRA_FAILED`` otherwise, with the failed child's log tail) and the group continues -- unless
    the served endpoint itself died, which fails the remaining evals as serve failures. The
    orchestrator job itself fails at the end if any eval failed.
    """
    configure_coreweave_s3()
    base_jobs = {"orchestrator": str(iris_ctx().job_id)}
    runs_by_unit = {run.unit.name: run for run in params.runs}
    paths: list[str] = []
    failed: list[str] = []
    for outcome in run_eval_units(params.session, tuple(run.unit for run in params.runs)):
        run = runs_by_unit[outcome.unit.name]
        status = RunStatus.SUCCEEDED
        error: str | None = None
        metrics: dict[str, dict[str, float]] = {}
        jobs = base_jobs | outcome.jobs
        log_tails: dict[str, tuple[str, ...]] = {}
        if outcome.error is not None:
            status = _classify_failure(outcome.error)
            error = f"{type(outcome.error).__name__}: {outcome.error}"
            jobs |= outcome.error.jobs
            log_tails = outcome.error.log_tails
        else:
            try:
                metrics = EvalchemyResult(path=run.out_path).task_metrics()
                if not metrics:
                    raise RuntimeError(f"eval finished but no task metrics were readable under {run.out_path!r}")
            except Exception as exc:
                status = RunStatus.INFRA_FAILED
                error = f"{type(exc).__name__}: {exc}"
                logger.exception("eval run %s: metrics unreadable", run.run_id)
        if error is not None:
            failed.append(f"{run.unit.name} ({status.value})")
            logger.error("eval run %s failed (%s): %s", run.run_id, status.value, error)
        path = write_record(_build_record(params, run, status, error, metrics, jobs, log_tails), params.records_prefix)
        logger.info("wrote eval record %s (status=%s)", path, status.value)
        paths.append(path)
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(params.runs)} evals failed: {', '.join(failed)}")
    return paths


# --------------------------------------------------------------------------------------------------
# Launcher side: plan, submit, and report (runs on the user's machine / CI).
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchSpec:
    """A launch request: one model, one or more eval suites, and where to run and record them.

    ``records_prefix=None`` means per-run resolution: CoreWeave-routed runs record to the CW-local
    object store (their workers have no GCP credentials), everything else to the GCS default.
    """

    model: str
    evals: tuple[str, ...]
    platform: Platform
    accelerator: str | None
    limit: int | None
    records_prefix: str | None
    cluster: str


@dataclass(frozen=True)
class RunPlan:
    """One resolved run: the model + suite + slice + serve spec that will be submitted."""

    model_key: str
    eval_key: str
    model: EvalModelConfig
    suite: EvalSuiteConfig
    accel: AcceleratorChoice
    serve: ServeSpec
    limit: int | None


@dataclass(frozen=True)
class GroupRunRef:
    """One eval's identity within a submitted group, enough to read its record back."""

    run_id: str
    eval_key: str


@dataclass(frozen=True)
class SubmittedGroup:
    """A submitted group orchestrator job and the run identities needed to read its records back."""

    group_id: str
    job: Job
    records_prefix: str
    model_key: str
    runs: tuple[GroupRunRef, ...]


def _serve_spec(model: EvalModelConfig, accel: AcceleratorChoice) -> ServeSpec:
    if accel.platform == Platform.GPU:
        spec = ServeSpec(
            backend=model.backend,
            tpu_type=None,
            gpu_type=accel.gpu_type,
            gpu_count=accel.gpu_count,
            max_model_len=model.max_model_len,
            tensor_parallel_size=model.tensor_parallel_size,
            region=accel.region,
            vllm_extra_args=model.vllm_extra_args,
            chat_template_content=model.chat_template,
        )
    else:
        spec = ServeSpec(
            backend=model.backend,
            tpu_type=accel.tpu_type,
            gpu_type=None,
            gpu_count=None,
            max_model_len=model.max_model_len,
            tensor_parallel_size=model.tensor_parallel_size,
            region=accel.region,
            vllm_extra_args=model.vllm_extra_args,
            chat_template_content=model.chat_template,
        )
    if model.serve_memory is not None:
        spec = replace(spec, serve_memory=model.serve_memory)
    return spec


def plan_runs(spec: LaunchSpec) -> list[RunPlan]:
    """Resolve a launch request into one :class:`RunPlan` per eval, sizing each serving slice.

    Raises ``NotImplementedError`` for a suite whose mechanism the launcher does not run yet (Harbor).
    """
    model = MODELS[spec.model]
    plans: list[RunPlan] = []
    for eval_key in spec.evals:
        suite = EVALS[eval_key]
        if suite.mechanism is not EvalMechanism.EVALCHEMY:
            raise NotImplementedError(
                f"eval {eval_key!r} uses the {suite.mechanism.value!r} mechanism, which this launcher does not "
                "run yet; only the evalchemy mechanism is wired."
            )
        accel = select_accelerator(model, spec.platform, spec.accelerator)
        limit = spec.limit if spec.limit is not None else suite.max_eval_instances
        plans.append(
            RunPlan(
                model_key=spec.model,
                eval_key=eval_key,
                model=model,
                suite=suite,
                accel=accel,
                serve=_serve_spec(model, accel),
                limit=limit,
            )
        )
    return plans


def _git_sha() -> str:
    for key in ("MARIN_GIT_SHA", "GIT_COMMIT"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _launch_user() -> str:
    return os.environ.get("MARIN_EVAL_USER") or getpass.getuser()


def _run_id(model_key: str, eval_key: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{model_key}-{eval_key}-{uuid.uuid4().hex[:4]}"


def _group_id(model_key: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{model_key}-{uuid.uuid4().hex[:4]}"


def records_prefix_for(accel: AcceleratorChoice, spec: LaunchSpec) -> str:
    """The records prefix a run writes to: the caller's override, else CW S3 for CoreWeave-routed
    runs (whose workers hold only CW object-store credentials), else the GCS default."""
    if spec.records_prefix:
        return spec.records_prefix
    if accel.target_cluster:
        return CW_RECORDS_PREFIX
    return DEFAULT_RECORDS_PREFIX


def _group_params(plans: list[RunPlan], spec: LaunchSpec, provenance: Provenance, user: str) -> EvalGroupParams:
    """Build the group orchestrator's params: one shared serve session plus one run per plan.

    Every plan in a launch shares the model and accelerator (they differ only in eval), so the first
    plan's serve spec is the session's.
    """
    first = plans[0]
    records_prefix = records_prefix_for(first.accel, spec)
    created_at = datetime.now(UTC).isoformat()
    runs: list[EvalRunParams] = []
    for plan in plans:
        run_id = _run_id(plan.model_key, plan.eval_key)
        out_path = f"{records_prefix.rstrip('/')}/{run_id}/results"
        runs.append(
            EvalRunParams(
                run_id=run_id,
                created_at=created_at,
                out_path=out_path,
                eval_ref=EvalRef(
                    name=plan.suite.name,
                    mechanism=plan.suite.mechanism.value,
                    tasks=tuple(EvalTaskRef(name=t.name, num_fewshot=t.num_fewshot) for t in plan.suite.tasks),
                ),
                unit=EvalUnit(
                    name=plan.eval_key,
                    tasks=plan.suite.tasks,
                    out_path=out_path,
                    max_gen_toks=plan.suite.max_gen_toks,
                    max_eval_instances=plan.limit,
                ),
            )
        )
    if len({run.unit.name for run in runs}) != len(runs):
        raise ValueError(f"duplicate eval keys in one launch: {[run.unit.name for run in runs]}")
    return EvalGroupParams(
        group_id=_group_id(first.model_key),
        user=user,
        records_prefix=records_prefix,
        session=EvalSession(
            model=first.model.location,
            # The serve child's self-stop backstop must outlive the whole suite running sequentially
            # against it: budget two hours per eval on top of boot time.
            serve=replace(first.serve, timeout_hours=2.0 + 2.0 * len(plans)),
            tokenizer=first.model.tokenizer,
            apply_chat_template=first.model.apply_chat_template,
        ),
        runs=tuple(runs),
        model_ref=ModelRef(name=first.model.name, location=first.model.location, backend=first.model.backend.value),
        hardware_ref=HardwareRef(
            platform=first.accel.platform.value,
            accelerator=first.accel.label,
            region_or_cluster=first.accel.target_cluster or first.accel.region or spec.cluster,
        ),
        provenance=provenance,
    )


def run_inline(spec: LaunchSpec) -> list[str]:
    """Run ``spec``'s whole group in this process and return the written record paths.

    The pipeline twin of ``launch_group`` + ``wait_and_report``: instead of submitting a CPU
    orchestrator job, the calling process -- which must itself be an Iris job, e.g. a pipeline
    step -- acts as the orchestrator and spawns the serve/eval children directly.
    """
    provenance = Provenance(git_sha=_git_sha(), evalchemy_image=EVALCHEMY_IMAGE, launch_host=socket.gethostname())
    user = _launch_user()
    return run_eval_group(_group_params(plan_runs(spec), spec, provenance, user))


def launch_group(spec: LaunchSpec, client: IrisClient) -> SubmittedGroup:
    """Submit one CPU orchestrator job for the whole launch and return a handle to it."""
    provenance = Provenance(git_sha=_git_sha(), evalchemy_image=EVALCHEMY_IMAGE, launch_host=socket.gethostname())
    user = _launch_user()
    plans = plan_runs(spec)
    params = _group_params(plans, spec, provenance, user)
    constraints = None
    if plans[0].accel.target_cluster:
        constraints = [
            Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=plans[0].accel.target_cluster)
        ]
    job = client.submit(
        entrypoint=Entrypoint.from_callable(run_eval_group, params),
        name=f"eval-{params.group_id}",
        resources=ResourceSpec(cpu=_ORCHESTRATOR_CPU, memory=_ORCHESTRATOR_MEMORY, disk=_ORCHESTRATOR_DISK),
        environment=EnvironmentSpec(env_vars=env_vars_from_keys(HARBOR_EVAL_ENV_KEYS)),
        constraints=constraints,
        max_retries_failure=0,
    )
    logger.info("submitted eval group %s (%d evals) as job %s", params.group_id, len(params.runs), job)
    return SubmittedGroup(
        group_id=params.group_id,
        job=job,
        records_prefix=params.records_prefix,
        model_key=plans[0].model_key,
        runs=tuple(GroupRunRef(run_id=run.run_id, eval_key=run.unit.name) for run in params.runs),
    )


# How much of a failed child's recorded log tail the CLI report prints; the full tail is in record.json.
_REPORT_TAIL_LINES = 15


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


def wait_and_report(groups: list[SubmittedGroup]) -> None:
    """Wait on each group, read its run records back, and print them."""
    configure_coreweave_s3()
    for group in groups:
        group.job.wait(timeout=float("inf"), raise_on_failure=False)
        for ref in group.runs:
            path = record_path(group.records_prefix, ref.run_id)
            try:
                record = read_record(path)
            except Exception:
                logger.warning("no readable record.json for run %s at %s", ref.run_id, path, exc_info=True)
                print(f"{ref.run_id}  [no record]  {group.model_key} / {ref.eval_key}")
                continue
            _print_record(record)
