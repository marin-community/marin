# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve experiment catalogs, submit generic eval groups, and report their records."""

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
from iris.client import IrisClient, Job
from iris.cluster.config import load_config
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.types import EndpointAccess, Entrypoint, EnvironmentSpec, ResourceSpec
from marin.evaluation.definitions import (
    EvalchemyDefinition,
    EvalDefinition,
    EvalRunIdentity,
    HarborDefinition,
    ResolvedEvalchemyRun,
    ResolvedEvalRun,
    ResolvedHarborRun,
)
from marin.evaluation.eval_env import EVAL_ENV_KEYS, daytona_sdk_env, env_vars_from_keys
from marin.evaluation.evalchemy import EvalchemyRunConfig, EvalchemyRuntimeConfig
from marin.evaluation.evalchemy_runtime import EVALCHEMY_IMAGE
from marin.evaluation.group_runner import EvalGroupParams, run_eval_group
from marin.evaluation.harbor_runner import canonical_served_name
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.planning import (
    EvalLaunchRequest,
    RunPlan,
)
from marin.evaluation.planning import (
    plan_runs as plan_selected_runs,
)
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
    read_record,
    record_path,
)
from marin.evaluation.serving_config import EvaluationServingConfig
from rigging.config_discovery import resolve_cluster_config
from rigging.filesystem import prefix_join
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.evaluation.evals import EVALS
from experiments.evaluation.hardware import MARIN_EVAL_HARDWARE
from experiments.evaluation.models import MODELS

logger = logging.getLogger(__name__)

_ORCHESTRATOR_CPU = 1.0
_ORCHESTRATOR_MEMORY = "4g"
_ORCHESTRATOR_DISK = "16g"
_REPORT_TAIL_LINES = 15


@dataclass(frozen=True)
class LaunchSpec:
    """One model, selected evaluations, execution target, and record destination."""

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
class GroupRunRef:
    """One submitted run's record identity."""

    run_id: str
    eval_key: str


@dataclass(frozen=True)
class SubmittedGroup:
    """A submitted orchestrator and the records it will write."""

    group_id: str
    job: Job
    records_prefix: str
    model_key: str
    runs: tuple[GroupRunRef, ...]


def plan_runs(spec: LaunchSpec) -> list[RunPlan]:
    """Resolve selected experiment definitions without passing global catalogs to workers."""
    model = MODELS[spec.model]
    return plan_selected_runs(
        EvalLaunchRequest(
            model_key=spec.model,
            model=model,
            evaluations=tuple((key, EVALS[key]) for key in spec.evals),
            platform=spec.platform,
            accelerator=spec.accelerator,
            limit=spec.limit,
        ),
        MARIN_EVAL_HARDWARE,
    )


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


def _mint_origin_for(cluster: str) -> str:
    config = load_config(resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS))
    if not config.dashboard_url:
        raise ValueError(f"cluster {cluster!r} has no public dashboard URL for Harbor capability routing")
    return config.dashboard_url


def records_prefix_for(accel: AcceleratorChoice, spec: LaunchSpec) -> str:
    """Resolve a region-local records prefix unless the caller supplied one."""
    if spec.records_prefix:
        return spec.records_prefix
    if accel.target_cluster:
        return CW_RECORDS_PREFIX
    return DEFAULT_RECORDS_PREFIX


def _eval_ref(definition: EvalDefinition) -> EvalRef:
    if isinstance(definition, EvalchemyDefinition):
        return EvalRef(
            name=definition.name,
            mechanism="evalchemy",
            tasks=tuple(EvalTaskRef(name=task.name, num_fewshot=task.num_fewshot) for task in definition.tasks),
        )
    config = definition.config
    return EvalRef(
        name=definition.name,
        mechanism="harbor",
        harbor=HarborRef(
            dataset=config.dataset,
            version=config.version,
            agent=config.agent,
            env=config.env,
        ),
    )


def _resolved_run(
    plan: RunPlan,
    identity: EvalRunIdentity,
) -> ResolvedEvalRun:
    definition = plan.suite
    if isinstance(definition, EvalchemyDefinition):
        return ResolvedEvalchemyRun(
            identity=identity,
            config=EvalchemyRunConfig(
                name=plan.eval_key,
                tasks=definition.tasks,
                apply_chat_template=plan.model.apply_chat_template,
                max_gen_toks=plan.model.generation.max_gen_toks or definition.max_gen_toks,
                max_eval_instances=plan.limit,
                extra_gen_kwargs=dict(plan.model.generation.extra_gen_kwargs),
                runtime=EvalchemyRuntimeConfig(region=plan.serve.region),
            ),
        )
    config = definition.config
    if plan.model.agent.agent_kwargs:
        config = replace(
            config,
            agent_kwargs={**plan.model.agent.agent_kwargs, **config.agent_kwargs},
        )
    return ResolvedHarborRun(
        identity=identity,
        config=replace(config, task_limit=plan.limit),
    )


def _group_params(
    plans: list[RunPlan],
    spec: LaunchSpec,
    provenance: Provenance,
    user: str,
) -> EvalGroupParams:
    if not plans:
        raise ValueError("cannot build an empty evaluation group")
    first = plans[0]
    is_harbor = isinstance(first.suite, HarborDefinition)
    records_prefix = records_prefix_for(first.accel, spec)
    created_at = datetime.now(UTC).isoformat()
    runs: list[ResolvedEvalRun] = []
    for plan in plans:
        run_id = _run_id(plan.model_key, plan.eval_key)
        output_dir = prefix_join(records_prefix, f"{run_id}/results")
        identity = EvalRunIdentity(
            run_id=run_id,
            created_at=created_at,
            output_dir=output_dir,
            eval_ref=_eval_ref(plan.suite),
        )
        runs.append(_resolved_run(plan, identity))
    if len({run.identity.eval_ref.name for run in runs}) != len(runs):
        raise ValueError(f"duplicate eval keys in one launch: {[run.identity.eval_ref.name for run in runs]}")

    serve = first.serve
    served_model_name = None
    if is_harbor:
        served_model_name = canonical_served_name(first.model.name)
        serve = replace(
            serve,
            endpoint_access=EndpointAccess.ENDPOINT_ACCESS_LINK,
        )
    tokenizer = first.model.tokenizer or first.model.location
    return EvalGroupParams(
        group_id=_group_id(first.model_key),
        user=user,
        version=spec.version,
        description=spec.description,
        records_prefix=records_prefix,
        capability_origin=_mint_origin_for(spec.cluster) if is_harbor else None,
        serving=EvaluationServingConfig(
            model=first.model.location,
            tokenizer=tokenizer,
            spec=serve,
            revision=first.model.revision,
            served_model_name=served_model_name,
        ),
        runs=tuple(runs),
        model_ref=ModelRef(
            name=first.model.name,
            location=first.model.location,
            backend=first.model.serve.backend.value,
        ),
        hardware_ref=HardwareRef(
            platform=first.accel.platform.value,
            accelerator=first.accel.label,
            region_or_cluster=first.accel.target_cluster or first.accel.region or spec.cluster,
        ),
        provenance=provenance,
    )


def run_inline(spec: LaunchSpec) -> list[str]:
    """Run the resolved group in the current Iris job."""
    provenance = Provenance(
        git_sha=_git_sha(),
        eval_image=EVALCHEMY_IMAGE,
        launch_host=socket.gethostname(),
    )
    return run_eval_group(_group_params(plan_runs(spec), spec, provenance, _launch_user()))


def launch_group(spec: LaunchSpec, client: IrisClient) -> SubmittedGroup:
    """Submit one CPU orchestrator for a resolved group."""
    provenance = Provenance(
        git_sha=_git_sha(),
        eval_image=EVALCHEMY_IMAGE,
        launch_host=socket.gethostname(),
    )
    plans = plan_runs(spec)
    params = _group_params(plans, spec, provenance, _launch_user())
    is_harbor = isinstance(plans[0].suite, HarborDefinition)
    constraints = None
    if plans[0].accel.target_cluster:
        constraints = [
            Constraint.create(
                key=CLUSTER_CONSTRAINT_KEY,
                op=ConstraintOp.EQ,
                value=plans[0].accel.target_cluster,
            )
        ]
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
        runs=tuple(
            GroupRunRef(
                run_id=run.identity.run_id,
                eval_key=run.identity.eval_ref.name,
            )
            for run in params.runs
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


def wait_and_report(groups: list[SubmittedGroup]) -> None:
    """Wait for groups and print their durable records."""
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
