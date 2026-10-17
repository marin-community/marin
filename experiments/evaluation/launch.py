# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve evaluation launches and bind their executable StepSpecs."""

from __future__ import annotations

import getpass
import json
import logging
import os
import re
import socket
import subprocess
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from functools import partial

from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.client.client import IrisClient
from iris.cluster.config import load_config
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp, region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from marin.evaluation.eval_env import EVAL_ENV_KEYS, env_vars_from_keys
from marin.evaluation.eval_policy import RUNTIME_COMMITS, policy_violations, runtime_violations
from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.evalchemy.runner import EvalchemyExecutor
from marin.evaluation.harbor.dataset import validate_harbor_dataset_source
from marin.evaluation.harbor.driver_config import (
    HARBOR_RUNTIME_PROJECT,
    ValidatedHarborConfig,
    harbor_runtime_descriptor,
    preflight_harbor_configs,
)
from marin.evaluation.harbor.runner import canonical_served_name
from marin.evaluation.hardware import AcceleratorChoice, Platform, default_platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import (
    CW_RECORDS_PREFIX,
    DEFAULT_RECORDS_PREFIX,
    EvalRef,
    ModelConfigRef,
    ModelRef,
    RunStatus,
    list_records,
)
from marin.evaluation.runner import (
    EndpointRoute,
    EvalExecutor,
    Evaluation,
    EvaluationBatch,
    EvaluationIdentity,
    EvaluationStep,
    HostedJudge,
    LaunchProvenance,
    SubmittedEvaluation,
    SubmittedEvaluationBatch,
    run_evaluation_steps,
)
from marin.evaluation.serving_config import resolved_serve_config
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.external_dependencies import EVALCHEMY
from rigging.config_discovery import resolve_cluster_config
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join
from rigging.secrets import SecretSpec, resolve_secret_spec

from experiments.evaluation.evals import (
    EVALS,
    EvalchemyDefinition,
    EvaluationDefinition,
    HarborDefinition,
    harbor_model_agent_kwargs,
)
from experiments.evaluation.fleet import MARIN_EVAL_HARDWARE

EVALUATION_CONTROLLER_CLUSTER = "marin"
DEFAULT_EVAL_CONCURRENCY = 1
_ORCHESTRATOR_CPU = 4.0
_ORCHESTRATOR_MEMORY = "16g"
_ORCHESTRATOR_DISK = "16g"
_SERVE_THROUGHPUT_FIELDS = {"max_num_batched_tokens", "max_num_seqs", "gpu_memory_utilization"}
_EVALCHEMY_THROUGHPUT_FIELDS = {"num_concurrent"}

logger = logging.getLogger(__name__)


def _model_identity(model: ModelRef) -> dict:
    return model.model_copy(update={"source_config": model.source_config or model.config}).model_dump(
        mode="json",
        exclude={
            "config_digest": True,
            "config": {"serve": _SERVE_THROUGHPUT_FIELDS},
            "source_config": {"serve": _SERVE_THROUGHPUT_FIELDS},
        },
    )


def _evaluation_identity(evaluation: EvalRef) -> dict:
    excluded = {"tasks": {"__all__": {"benchmark"}}}
    if evaluation.evalchemy is not None:
        excluded.update({"source_digest": True, "evalchemy": _EVALCHEMY_THROUGHPUT_FIELDS})
    return evaluation.model_dump(mode="json", exclude=excluded)


def _same_evaluation(prior: EvalRef, current: EvalRef) -> bool:
    before, after = _evaluation_identity(prior), _evaluation_identity(current)
    if (
        current.evalchemy is not None
        and current.source_digest is not None
        and current.source_digest == prior.source_digest
    ):
        # Completed group tasks expand into benchmark leaves in the record. An unchanged launch
        # configuration establishes the original task selection; compare the remaining controls.
        before.pop("tasks")
        after.pop("tasks")
    return before == after


@dataclass(frozen=True)
class LaunchSpec:
    """One model, evaluation selection, execution target, and record destination."""

    model: ModelConfig
    evals: tuple[str, ...]
    evalchemy_definitions: tuple[EvalchemyDefinition, ...]
    harbor_definitions: tuple[HarborDefinition, ...]
    platform: Platform
    accelerator: str | None
    limit: int | None
    records_prefix: str | None
    submission_cluster: str
    federated_cluster: str | None
    priority_band: int
    max_concurrent: int = DEFAULT_EVAL_CONCURRENCY
    instances: int = 1
    judge_model: ModelConfig | None = None
    judge_accelerator: str | None = None
    seed: int | None = None
    version: str | None = None
    description: str | None = None
    cache_key: str | None = None
    """Reuse evaluation records for this key and the same resolved configuration."""
    run_id: str | None = None


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


def _run_id(model_name: str, eval_key: str, cache_key: str, group_id: str | None) -> str:
    prefix = group_id if group_id is not None else f"pipeline-{model_name}"
    return f"{prefix}-{eval_key}-{uuid.uuid5(uuid.NAMESPACE_URL, cache_key).hex}"


def _group_id(model_name: str, cache_key: str | None) -> str:
    if cache_key is not None:
        return f"pipeline-{model_name}-{uuid.uuid5(uuid.NAMESPACE_URL, cache_key).hex}"
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{model_name}-{uuid.uuid4().hex}"


def _capability_origin(cluster: str) -> str:
    config = load_config(resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS))
    if not config.dashboard_url:
        raise ValueError(f"cluster {cluster!r} has no public dashboard URL for inference endpoint routing")
    return config.dashboard_url


def records_prefix_for(accel: AcceleratorChoice, spec: LaunchSpec) -> str:
    """Resolve the configured TPU or CoreWeave records store."""
    if spec.records_prefix:
        return spec.records_prefix if "://" in spec.records_prefix else os.path.abspath(spec.records_prefix)
    if accel.target_cluster:
        return CW_RECORDS_PREFIX
    return DEFAULT_RECORDS_PREFIX


def _evaluation_definitions(spec: LaunchSpec) -> tuple[tuple[str, EvaluationDefinition], ...]:
    registry_definitions: tuple[tuple[str, EvaluationDefinition], ...] = tuple(
        (eval_key, EVALS[eval_key]) for eval_key in spec.evals
    )
    evalchemy_definitions: tuple[tuple[str, EvaluationDefinition], ...] = tuple(
        (definition.name, definition) for definition in spec.evalchemy_definitions
    )
    harbor_definitions: tuple[tuple[str, EvaluationDefinition], ...] = tuple(
        (definition.name, definition) for definition in spec.harbor_definitions
    )
    definitions = registry_definitions + evalchemy_definitions + harbor_definitions
    if not definitions:
        raise ValueError("at least one evaluation is required")
    names = [name for name, _ in definitions]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate eval names in one launch: {names}")
    return definitions


@dataclass(frozen=True)
class _ResolvedDefinition:
    record_ref: EvalRef
    runtime_descriptor: str
    executor: EvalExecutor
    endpoint_route: EndpointRoute
    secret_env: dict[str, SecretSpec]


def _resolve_definitions(
    definitions: tuple[tuple[str, EvaluationDefinition], ...],
    model: ModelConfig,
    limit: int | None,
    seed: int | None,
    version: str | None,
) -> tuple[tuple[str, _ResolvedDefinition], ...]:
    evalchemy_definitions = [definition for _, definition in definitions if isinstance(definition, EvalchemyDefinition)]
    evalchemy_sources = iter(load_evalchemy_config(definition.config_path) for definition in evalchemy_definitions)
    harbor_definitions = [definition for _, definition in definitions if isinstance(definition, HarborDefinition)]
    model_agent_kwargs = harbor_model_agent_kwargs(model)
    requests = [(definition.config_path, model_agent_kwargs) for definition in harbor_definitions]
    runtime_commits = RUNTIME_COMMITS.get(version or "", {})
    harbor_commit = runtime_commits.get("harbor")
    harbor_project = (
        f"{HARBOR_RUNTIME_PROJECT}/pins/{harbor_commit}" if harbor_commit is not None else HARBOR_RUNTIME_PROJECT
    )
    validated_configs = iter(preflight_harbor_configs(requests, runtime_project=harbor_project))
    evalchemy_commit = runtime_commits.get("evalchemy", EVALCHEMY.commit)
    evalchemy_dependency = replace(EVALCHEMY, commit=evalchemy_commit)

    resolved: list[tuple[str, _ResolvedDefinition]] = []
    for name, definition in definitions:
        if isinstance(definition, EvalchemyDefinition):
            source = next(evalchemy_sources)
            config = definition.config_for(source, model, limit, evalchemy_dependency)
            if seed is not None:
                config = replace(config, seed=seed)
            secret_env = definition.secret_env_for(config)
            resolved.append(
                (
                    name,
                    _ResolvedDefinition(
                        record_ref=definition.record_ref_for(config),
                        runtime_descriptor=config.runtime.requirement,
                        executor=EvalchemyExecutor(config),
                        endpoint_route=EndpointRoute.DIRECT,
                        secret_env=dict(secret_env),
                    ),
                )
            )
            continue

        config: ValidatedHarborConfig = next(validated_configs)
        validate_harbor_dataset_source(config)
        runtime_task_limit = definition.max_eval_instances if limit is None else limit
        resolved.append(
            (
                name,
                _ResolvedDefinition(
                    record_ref=definition.record_ref_for(config, runtime_task_limit),
                    runtime_descriptor=harbor_runtime_descriptor(config.error_taxonomy.commit, config.runtime_project),
                    executor=definition.executor_for(config, model, runtime_task_limit),
                    endpoint_route=EndpointRoute.CAPABILITY,
                    secret_env=dict(definition.secret_env_for(config)),
                ),
            )
        )
    return tuple(resolved)


def _resolve_hosted_judge(spec: LaunchSpec, candidate_accelerator: AcceleratorChoice) -> HostedJudge | None:
    if spec.judge_model is None and spec.judge_accelerator is not None:
        raise ValueError("--judge-accelerator requires --judge-model or --judge-model-config")
    if spec.judge_model is None:
        return None

    judge_accelerator = MARIN_EVAL_HARDWARE.select(
        spec.judge_model,
        default_platform(spec.judge_model),
        spec.judge_accelerator,
    )
    if spec.federated_cluster is not None:
        if judge_accelerator.platform is not Platform.GPU:
            raise ValueError("a federated hosted judge requires a GPU accelerator")
        judge_accelerator = replace(judge_accelerator, target_cluster=spec.federated_cluster)
    candidate_location = candidate_accelerator.target_cluster or candidate_accelerator.region
    judge_location = judge_accelerator.target_cluster or judge_accelerator.region
    if candidate_location != judge_location:
        raise ValueError(
            "the evaluated model and hosted judge must run in the same cluster or region; "
            f"got {candidate_location!r} and {judge_location!r}"
        )
    return HostedJudge(
        model=spec.judge_model,
        accelerator=judge_accelerator,
        api_model=canonical_served_name(spec.judge_model.name),
    )


def build_evaluation_batch(
    spec: LaunchSpec,
    provenance: LaunchProvenance,
    user: str,
) -> EvaluationBatch:
    """Resolve experiment names into one configurable evaluation batch."""
    if spec.run_id is not None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", spec.run_id) is None:
            raise ValueError("--run-id must contain only letters, digits, dots, underscores, and hyphens")
        if spec.cache_key is not None:
            raise ValueError("--run-id and pipeline cache_key cannot be combined")
    if spec.max_concurrent < 1:
        raise ValueError("max_concurrent must be at least 1")
    if spec.instances < 1:
        raise ValueError("instances must be at least 1")
    if spec.instances > 1 and spec.model.serve.pipeline_parallel_size > 1:
        raise ValueError("pipeline parallelism requires one serving instance")
    model = spec.model
    source_model_config = ModelConfigRef.model_validate(asdict(model))
    accelerator = MARIN_EVAL_HARDWARE.select(model, spec.platform, spec.accelerator)
    if spec.federated_cluster is not None:
        if accelerator.platform is not Platform.GPU:
            raise ValueError("--federated_cluster requires a GPU accelerator")
        accelerator = replace(accelerator, target_cluster=spec.federated_cluster)
    judge = _resolve_hosted_judge(spec, accelerator)
    requested_definitions = _evaluation_definitions(spec)
    if model.serve.max_model_len is not None and any(
        isinstance(definition, HarborDefinition) for _, definition in requested_definitions
    ):
        model = replace(model, serve=resolved_serve_config(model))
    definitions = _resolve_definitions(requested_definitions, model, spec.limit, spec.seed, spec.version)
    model_ref = ModelRef(
        name=model.name,
        location=model.location,
        backend=model.serve.backend.value,
        config=ModelConfigRef.model_validate(asdict(model)),
        source_config=source_model_config,
    )
    for name, definition in definitions:
        violations = (
            *policy_violations(spec.version, model_ref, definition.record_ref),
            *runtime_violations(spec.version, definition.record_ref, definition.runtime_descriptor),
        )
        if violations:
            raise ValueError(f"{spec.version} pre-submit check failed for {name}: {'; '.join(violations)}")
    if judge is not None and any(isinstance(definition.executor, EvalchemyExecutor) for _, definition in definitions):
        raise ValueError("--judge-model serves Harbor verifiers only; remove it or drop the Evalchemy evaluations")
    records_prefix = records_prefix_for(accelerator, spec)
    group_id = spec.run_id or _group_id(model.name, spec.cache_key)
    created_at = datetime.now(UTC).isoformat()
    secret_env: dict[str, SecretSpec] = {}
    for _, definition in definitions:
        for name, spec_value in definition.secret_env.items():
            if name in secret_env and secret_env[name] != spec_value:
                raise ValueError(f"evaluations declare conflicting secret specifications for {name}")
            secret_env[name] = spec_value

    evaluations: list[Evaluation] = []
    for eval_key, definition in definitions:
        cache_key = json.dumps(
            {
                "pipeline": spec.cache_key or group_id,
                "model": _model_identity(model_ref),
                "evaluation": _evaluation_identity(definition.record_ref),
                "runtime": definition.runtime_descriptor if spec.cache_key is not None else None,
                "judge": (
                    ModelConfigRef.model_validate(asdict(judge.model)).model_dump(mode="json")
                    if judge is not None
                    else None
                ),
                "policy": spec.version,
            },
            sort_keys=True,
        )
        run_id = _run_id(model.name, eval_key, cache_key, group_id if spec.cache_key is None else None)
        evaluations.append(
            Evaluation(
                identity=EvaluationIdentity(
                    run_id=run_id,
                    created_at=created_at,
                    output_dir=prefix_join(records_prefix, f"{run_id}/results"),
                    eval_ref=definition.record_ref,
                    eval_runtime=definition.runtime_descriptor,
                ),
                executor=definition.executor,
                endpoint_route=definition.endpoint_route,
                secret_env_keys=tuple(definition.secret_env),
            )
        )

    endpoint_cluster = accelerator.target_cluster or spec.submission_cluster
    return EvaluationBatch(
        group_id=group_id,
        user=user,
        version=spec.version,
        description=spec.description,
        records_prefix=records_prefix,
        model=model,
        accelerator=accelerator,
        priority_band=spec.priority_band,
        capability_origin=_capability_origin(endpoint_cluster),
        api_model=canonical_served_name(model.name),
        evaluations=tuple(evaluations),
        provenance=provenance,
        submission_cluster=spec.submission_cluster,
        max_concurrent=spec.max_concurrent,
        instances=spec.instances,
        judge=judge,
        secret_env=secret_env,
        source_model_config=source_model_config,
    )


def build_evaluation_steps(batch: EvaluationBatch, execute: EvaluationStep) -> tuple[StepSpec, ...]:
    """Build per-evaluation steps with the supplied execution function."""
    return tuple(
        StepSpec(
            name=f"eval/{batch.model.name}/{evaluation.identity.eval_ref.name}",
            override_output_path=prefix_join(batch.records_prefix, evaluation.identity.run_id),
            fn=lambda output_path, evaluation=evaluation: execute(evaluation, output_path),
            hash_attrs={"run_id": evaluation.identity.run_id},
        )
        for evaluation in batch.evaluations
    )


def prepare_evaluation_batch(spec: LaunchSpec) -> EvaluationBatch:
    """Resolve one launch before an Iris client is opened."""
    provenance = LaunchProvenance(
        git_sha=_git_sha(),
        launch_host=socket.gethostname(),
    )
    batch = build_evaluation_batch(spec, provenance, _launch_user())
    if spec.run_id is None:
        return batch
    if batch.records_prefix.startswith("s3://"):
        configure_coreweave_s3()
    model = _model_identity(
        ModelRef(
            name=batch.model.name,
            location=batch.model.location,
            backend=batch.model.serve.backend.value,
            config=ModelConfigRef.model_validate(asdict(batch.model)),
            source_config=batch.source_model_config,
        )
    )
    judge = None if batch.judge is None else ModelConfigRef.model_validate(asdict(batch.judge.model))
    records = sorted(
        (
            record
            for record in list_records(batch.records_prefix)
            if record.group_id == batch.group_id
            and record.version == batch.version
            and _model_identity(record.model) == model
            and (record.judge.model.source_config or record.judge.model.config if record.judge is not None else None)
            == judge
        ),
        key=lambda record: (record.status is RunStatus.SUCCEEDED, record.created_at),
        reverse=True,
    )
    evaluations = []
    for evaluation in batch.evaluations:
        prior = next(
            (record for record in records if _same_evaluation(record.evaluation, evaluation.identity.eval_ref)),
            None,
        )
        if prior is not None:
            evaluation = replace(
                evaluation,
                identity=replace(evaluation.identity, run_id=prior.run_id, output_dir=prior.results_path),
            )
        evaluations.append(evaluation)
    return replace(batch, evaluations=tuple(evaluations))


def run_evaluation_group(batch: EvaluationBatch) -> None:
    """Run the group's evaluation steps under one shared-inference ownership lock."""
    configure_coreweave_s3()
    step = StepSpec(
        name=f"eval-group/{batch.model.name}",
        output_path_prefix=prefix_join(batch.records_prefix, f"groups/{batch.group_id}"),
        fn=partial(run_evaluation_steps, batch, partial(build_evaluation_steps, batch)),
        hash_attrs={"run_ids": tuple(evaluation.identity.run_id for evaluation in batch.evaluations)},
    )
    StepRunner().run((step,), max_concurrent=1)


def launch_group(batch: EvaluationBatch, client: IrisClient) -> SubmittedEvaluationBatch:
    """Bind final evaluation steps and submit one CPU orchestrator."""
    constraints = None
    if batch.accelerator.target_cluster and batch.accelerator.target_cluster != batch.submission_cluster:
        constraints = [
            Constraint.create(
                key=CLUSTER_CONSTRAINT_KEY,
                op=ConstraintOp.EQ,
                value=batch.accelerator.target_cluster,
            )
        ]
    elif batch.accelerator.region:
        constraints = [region_constraint([batch.accelerator.region])]
    launch_env = env_vars_from_keys(EVAL_ENV_KEYS)
    for name, spec in sorted(batch.secret_env.items()):
        launch_env[name] = resolve_secret_spec(spec).value
    job = client.submit(
        entrypoint=Entrypoint.from_callable(run_evaluation_group, batch),
        name=f"eval-{batch.group_id}-{uuid.uuid4().hex[:8]}",
        resources=ResourceSpec(cpu=_ORCHESTRATOR_CPU, memory=_ORCHESTRATOR_MEMORY, disk=_ORCHESTRATOR_DISK),
        environment=EnvironmentSpec(env_vars=launch_env),
        constraints=constraints,
        max_retries_failure=0,
        priority_band=batch.priority_band,
    )
    logger.info("submitted eval batch %s (%d evals) as job %s", batch.group_id, len(batch.evaluations), job)
    return SubmittedEvaluationBatch(
        group_id=batch.group_id,
        job=job,
        records_prefix=batch.records_prefix,
        model_name=batch.model.name,
        evaluations=tuple(
            SubmittedEvaluation(run_id=evaluation.identity.run_id, eval_name=evaluation.identity.eval_ref.name)
            for evaluation in batch.evaluations
        ),
    )
