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

from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.client import IrisClient, Job, iris_ctx
from iris.cluster.config import load_config
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from marin.evaluation.eval_env import EVAL_ENV_KEYS, daytona_sdk_env, env_vars_from_keys
from marin.evaluation.eval_result import EvalchemyResult
from marin.evaluation.harbor_runner import HarborRunConfig, canonical_served_name, run_harbor_eval
from marin.evaluation.records import (
    CW_RECORDS_PREFIX,
    DEFAULT_RECORDS_PREFIX,
    EvalRef,
    EvalRunRecord,
    EvalTaskRef,
    HarborRef,
    HardwareRef,
    ModelRef,
    Provenance,
    RunStatus,
    read_record,
    record_path,
    write_record,
)
from rigging.config_discovery import resolve_cluster_config
from rigging.connect import capability_path
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration

from experiments.evals.evalchemy.image import EVALCHEMY_IMAGE
from experiments.evals.evalchemy.serve_and_eval import (
    EvalPipelineError,
    EvalSession,
    EvalUnit,
    PipelineStage,
    ServeSpec,
    run_eval_units,
    serve_model,
)
from experiments.evaluation.evals import EVALS, EvalMechanism, EvalSuiteConfig, HarborSpec
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
    """One recorded eval within a group: its identity, its durable results path, and its mechanism.

    Exactly one of ``unit`` (evalchemy) or ``harbor`` (Harbor) is set; ``limit`` caps the run's
    instances for either mechanism.
    """

    run_id: str
    created_at: str
    out_path: str
    eval_ref: EvalRef
    unit: EvalUnit | None = None
    harbor: HarborSpec | None = None
    limit: int | None = None


@dataclass(frozen=True)
class EvalGroupParams:
    """Everything one orchestrator job needs, cloudpickled into it.

    The record fields are prebuilt at launch (git SHA, host, user) because the worker cannot recover
    them. One group serves the model once and writes one record per entry in ``runs``.
    """

    group_id: str
    user: str
    version: str | None
    description: str | None
    records_prefix: str
    session: EvalSession
    runs: tuple[EvalRunParams, ...]
    model_ref: ModelRef
    hardware_ref: HardwareRef
    provenance: Provenance
    mint_origin: str | None = None
    """Public origin (e.g. ``https://iris.oa.dev``) for minting a capability URL to the served
    endpoint, so an off-cluster Harbor sandbox can reach it. ``None`` for evalchemy groups (the
    in-cluster ``base_url`` suffices) or when the cluster config carries no dashboard URL."""


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
        version=group.version,
        description=group.description,
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
    if any(run.harbor is not None for run in params.runs):
        return _run_harbor_group(params)
    base_jobs = {"orchestrator": str(iris_ctx().job_id)}
    units = [run.unit for run in params.runs if run.unit is not None]
    runs_by_unit = {run.unit.name: run for run in params.runs if run.unit is not None}
    paths: list[str] = []
    failed: list[str] = []
    for outcome in run_eval_units(params.session, tuple(units)):
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
            failed.append(f"{run.eval_ref.name} ({status.value})")
            logger.error("eval run %s failed (%s): %s", run.run_id, status.value, error)
        path = write_record(_build_record(params, run, status, error, metrics, jobs, log_tails), params.records_prefix)
        logger.info("wrote eval record %s (status=%s)", path, status.value)
        paths.append(path)
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(params.runs)} evals failed: {', '.join(failed)}")
    return paths


# A minted endpoint token must outlive the longest Harbor suite it authorizes (an agentic run can take
# many hours); 96h matches the scoped-token ceiling in iris (#7551).
_MINT_TTL = Duration.from_hours(96)


def _mint_capability_url(endpoint_name: str, origin: str) -> str:
    """Mint a capability URL for ``endpoint_name`` and return its OpenAI base (``.../v1``).

    The scoped token rides in the path, so possession of the URL is the credential -- an off-cluster
    Harbor sandbox can call the served model with no auth header.
    """
    resp = iris_ctx().client._cluster_client.mint_endpoint_token(endpoint_name, ttl=_MINT_TTL)
    return f"{origin.rstrip('/')}{capability_path(endpoint_name, resp.token)}/v1"


def _harbor_api_base(endpoint, mint_origin: str | None) -> str:
    """The URL a Harbor agent calls: a minted capability URL when an origin is known (reachable from an
    off-cluster sandbox), else the in-cluster ``base_url`` (fine for a client-side agent)."""
    if endpoint.name is not None and mint_origin:
        try:
            url = _mint_capability_url(endpoint.name, mint_origin)
            logger.info("minted capability URL for Harbor agent -> endpoint %s", endpoint.name)
            return url
        except Exception:
            logger.warning("could not mint a capability URL for %s; using in-VPC base_url", endpoint.name, exc_info=True)
    return endpoint.base_url


def _run_harbor_group(params: EvalGroupParams) -> list[str]:
    """Serve the model once, run each Harbor dataset against it, and write one record per run."""
    base_jobs = {"orchestrator": str(iris_ctx().job_id)}
    session = params.session
    served_name = canonical_served_name(params.model_ref.name)
    paths: list[str] = []
    failed: list[str] = []
    try:
        with serve_model(session.model, session.tokenizer or session.model, session.serve) as endpoint:
            api_base = _harbor_api_base(endpoint, params.mint_origin)
            serve_jobs = {"serve": endpoint.job}
            for run in params.runs:
                assert run.harbor is not None
                status = RunStatus.SUCCEEDED
                error: str | None = None
                metrics: dict[str, dict[str, float]] = {}
                try:
                    result = run_harbor_eval(
                        HarborRunConfig(
                            dataset=run.harbor.dataset,
                            version=run.harbor.version,
                            agent=run.harbor.agent,
                            served_model_name=served_name,
                            api_base=api_base,
                            env=run.harbor.env,
                            n_concurrent=run.harbor.n_concurrent,
                            max_output_tokens=run.harbor.max_output_tokens,
                            task_limit=run.limit,
                            agent_kwargs=dict(run.harbor.agent_kwargs),
                        ),
                        run.out_path,
                    )
                    metrics = result.task_metrics()
                    if not result.total_trials:
                        raise RuntimeError(f"Harbor eval finished with no trials under {run.out_path!r}")
                except Exception as exc:
                    status = RunStatus.FAILED
                    error = f"{type(exc).__name__}: {exc}"
                    logger.exception("harbor run %s failed", run.run_id)
                if error is not None:
                    failed.append(f"{run.eval_ref.name} ({status.value})")
                path = write_record(
                    _build_record(params, run, status, error, metrics, base_jobs | serve_jobs, {}),
                    params.records_prefix,
                )
                logger.info("wrote harbor record %s (status=%s)", path, status.value)
                paths.append(path)
    except EvalPipelineError as exc:
        # The serve never came up: record every run as an infra failure with the serve log tail.
        for run in params.runs:
            path = write_record(
                _build_record(params, run, RunStatus.INFRA_FAILED, str(exc), {}, base_jobs | exc.jobs, exc.log_tails),
                params.records_prefix,
            )
            paths.append(path)
        raise RuntimeError(f"harbor group serve failed: {exc}") from exc
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(params.runs)} harbor evals failed: {', '.join(failed)}")
    return paths


def _mint_origin_for(cluster: str) -> str | None:
    """The cluster's public dashboard origin for minting capability URLs, or None if unresolved."""
    try:
        config = load_config(resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS))
        return config.dashboard_url or None
    except Exception:
        logger.warning("could not resolve a dashboard origin for cluster %r; Harbor will use base_url", cluster)
        return None


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
    version: str | None = None
    description: str | None = None


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

    Every mechanism serves the model on the sized slice; they differ in what runs against the endpoint
    (an evalchemy client vs. Harbor trials). A group mixes only one mechanism.
    """
    model = MODELS[spec.model]
    mechanisms = {EVALS[eval_key].mechanism for eval_key in spec.evals}
    if len(mechanisms) > 1:
        raise ValueError(f"a launch group must be one mechanism, got {sorted(m.value for m in mechanisms)}")
    plans: list[RunPlan] = []
    for eval_key in spec.evals:
        suite = EVALS[eval_key]
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
    is_harbor = first.suite.mechanism is EvalMechanism.HARBOR
    records_prefix = records_prefix_for(first.accel, spec)
    created_at = datetime.now(UTC).isoformat()
    runs: list[EvalRunParams] = []
    for plan in plans:
        run_id = _run_id(plan.model_key, plan.eval_key)
        out_path = f"{records_prefix.rstrip('/')}/{run_id}/results"
        harbor = plan.suite.harbor
        eval_ref = EvalRef(
            name=plan.suite.name,
            mechanism=plan.suite.mechanism.value,
            tasks=tuple(EvalTaskRef(name=t.name, num_fewshot=t.num_fewshot) for t in plan.suite.tasks),
            harbor=(
                HarborRef(dataset=harbor.dataset, version=harbor.version, agent=harbor.agent, env=harbor.env)
                if harbor is not None
                else None
            ),
        )
        unit = (
            None
            if harbor is not None
            else EvalUnit(
                name=plan.eval_key,
                tasks=plan.suite.tasks,
                out_path=out_path,
                max_gen_toks=plan.model.max_gen_toks or plan.suite.max_gen_toks,
                max_eval_instances=plan.limit,
            )
        )
        runs.append(
            EvalRunParams(
                run_id=run_id,
                created_at=created_at,
                out_path=out_path,
                eval_ref=eval_ref,
                unit=unit,
                harbor=harbor,
                limit=plan.limit,
            )
        )
    if len({run.eval_ref.name for run in runs}) != len(runs):
        raise ValueError(f"duplicate eval keys in one launch: {[run.eval_ref.name for run in runs]}")
    # Harbor's agent addresses the model as hosted_vllm/<served-name>, which must be slash-free; serve
    # it under that canonical name so the agent's request model matches what vLLM answers to.
    serve = first.serve
    if is_harbor:
        served_name = canonical_served_name(first.model.name)
        serve = replace(serve, vllm_extra_args=(*serve.vllm_extra_args, "--served-model-name", served_name))
    return EvalGroupParams(
        group_id=_group_id(first.model_key),
        user=user,
        version=spec.version,
        description=spec.description,
        records_prefix=records_prefix,
        mint_origin=_mint_origin_for(spec.cluster) if is_harbor else None,
        session=EvalSession(
            model=first.model.location,
            serve=serve,
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
    provenance = Provenance(git_sha=_git_sha(), eval_image=EVALCHEMY_IMAGE, launch_host=socket.gethostname())
    user = _launch_user()
    return run_eval_group(_group_params(plan_runs(spec), spec, provenance, user))


def launch_group(spec: LaunchSpec, client: IrisClient) -> SubmittedGroup:
    """Submit one CPU orchestrator job for the whole launch and return a handle to it."""
    provenance = Provenance(git_sha=_git_sha(), eval_image=EVALCHEMY_IMAGE, launch_host=socket.gethostname())
    user = _launch_user()
    plans = plan_runs(spec)
    params = _group_params(plans, spec, provenance, user)
    is_harbor = plans[0].suite.mechanism is EvalMechanism.HARBOR
    constraints = None
    if plans[0].accel.target_cluster:
        constraints = [
            Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=plans[0].accel.target_cluster)
        ]
    # A Harbor group runs Harbor as an isolated uv subprocess from the orchestrator (see
    # marin.evaluation.harbor_runner), so the orchestrator needs no harbor extra -- just more CPU and
    # memory than the evalchemy orchestrator, and the Daytona credential in its env.
    job = client.submit(
        entrypoint=Entrypoint.from_callable(run_eval_group, params),
        name=f"eval-{params.group_id}",
        resources=ResourceSpec(
            cpu=4.0 if is_harbor else _ORCHESTRATOR_CPU,
            memory="16g" if is_harbor else _ORCHESTRATOR_MEMORY,
            disk=_ORCHESTRATOR_DISK,
        ),
        environment=EnvironmentSpec(env_vars=env_vars_from_keys(EVAL_ENV_KEYS) | daytona_sdk_env()),
        constraints=constraints,
        max_retries_failure=0,
    )
    logger.info("submitted eval group %s (%d evals) as job %s", params.group_id, len(params.runs), job)
    return SubmittedGroup(
        group_id=params.group_id,
        job=job,
        records_prefix=params.records_prefix,
        model_key=plans[0].model_key,
        runs=tuple(GroupRunRef(run_id=run.run_id, eval_key=run.eval_ref.name) for run in params.runs),
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
