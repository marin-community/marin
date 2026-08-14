# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve experiment catalogs into shared evaluation batches."""

from __future__ import annotations

import getpass
import os
import socket
import subprocess
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.client import IrisClient
from iris.cluster.config import load_config
from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.evalchemy.runner import EvalchemyExecutor
from marin.evaluation.harbor.dataset import validate_harbor_dataset_source
from marin.evaluation.harbor.driver_config import (
    ValidatedHarborConfig,
    preflight_harbor_configs,
)
from marin.evaluation.harbor.runner import HarborExecutor, canonical_served_name, validate_harbor_resume_root
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import (
    CW_RECORDS_PREFIX,
    DEFAULT_RECORDS_PREFIX,
    EvalRef,
)
from marin.evaluation.runner import (
    EvalExecutor,
    Evaluation,
    EvaluationBatch,
    EvaluationIdentity,
    LaunchProvenance,
    SubmittedEvaluationBatch,
    submit_evaluation_batch,
)
from rigging.config_discovery import resolve_cluster_config
from rigging.filesystem import prefix_join
from rigging.secrets import SecretSpec

from experiments.evaluation.evals import (
    EVALS,
    EvalchemyDefinition,
    EvaluationDefinition,
    HarborDefinition,
)
from experiments.evaluation.fleet import MARIN_EVAL_HARDWARE

EVALUATION_CONTROLLER_CLUSTER = "marin"


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
    version: str | None = None
    description: str | None = None
    resume_results_path: str | None = None


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


def _run_id(model_name: str, eval_key: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{model_name}-{eval_key}-{uuid.uuid4().hex[:4]}"


def _group_id(model_name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{model_name}-{uuid.uuid4().hex[:4]}"


def _capability_origin(cluster: str) -> str:
    config = load_config(resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS))
    if not config.dashboard_url:
        raise ValueError(f"cluster {cluster!r} has no public dashboard URL for inference endpoint routing")
    return config.dashboard_url


def records_prefix_for(accel: AcceleratorChoice, spec: LaunchSpec) -> str:
    """Resolve the configured TPU or CoreWeave records store."""
    if spec.records_prefix:
        return spec.records_prefix
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
    secret_env: dict[str, SecretSpec]


def _resolve_definitions(
    definitions: tuple[tuple[str, EvaluationDefinition], ...],
    model: ModelConfig,
    limit: int | None,
) -> tuple[tuple[str, _ResolvedDefinition], ...]:
    evalchemy_definitions = [definition for _, definition in definitions if isinstance(definition, EvalchemyDefinition)]
    evalchemy_sources = iter(load_evalchemy_config(definition.config_path) for definition in evalchemy_definitions)
    harbor_definitions = [definition for _, definition in definitions if isinstance(definition, HarborDefinition)]
    requests = [(definition.config_path, dict(model.agent.agent_kwargs)) for definition in harbor_definitions]
    validated_configs = iter(preflight_harbor_configs(requests))

    resolved: list[tuple[str, _ResolvedDefinition]] = []
    for name, definition in definitions:
        if isinstance(definition, EvalchemyDefinition):
            source = next(evalchemy_sources)
            config = definition.config_for(source, model, limit)
            resolved.append(
                (
                    name,
                    _ResolvedDefinition(
                        record_ref=definition.record_ref_for(config),
                        runtime_descriptor=config.runtime.requirement,
                        executor=EvalchemyExecutor(config),
                        secret_env=dict(definition.secret_env),
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
                    runtime_descriptor=definition.runtime_descriptor,
                    executor=definition.executor_for(config, model, runtime_task_limit),
                    secret_env=dict(definition.secret_env_for(config)),
                ),
            )
        )
    return tuple(resolved)


def build_evaluation_batch(
    spec: LaunchSpec,
    provenance: LaunchProvenance,
    user: str,
) -> EvaluationBatch:
    """Resolve experiment names into one model-serving evaluation batch."""
    model = spec.model
    accelerator = MARIN_EVAL_HARDWARE.select(model, spec.platform, spec.accelerator)
    if spec.federated_cluster is not None:
        if accelerator.platform is not Platform.GPU:
            raise ValueError("--federated_cluster requires a GPU accelerator")
        accelerator = replace(accelerator, target_cluster=spec.federated_cluster)
    definitions = _resolve_definitions(_evaluation_definitions(spec), model, spec.limit)
    if spec.resume_results_path is not None:
        if len(definitions) != 1 or not isinstance(definitions[0][1].executor, HarborExecutor):
            raise ValueError("--resume-results-path requires exactly one Harbor evaluation")
        if "://" not in spec.resume_results_path:
            raise ValueError("--resume-results-path must be an object-store path")
        validate_harbor_resume_root(spec.resume_results_path, definitions[0][1].executor.config)
    records_prefix = records_prefix_for(accelerator, spec)
    created_at = datetime.now(UTC).isoformat()
    evaluations: list[Evaluation] = []
    secret_env: dict[str, SecretSpec] = {}
    for eval_key, definition in definitions:
        for name, spec_value in definition.secret_env.items():
            if name in secret_env and secret_env[name] != spec_value:
                raise ValueError(f"evaluations declare conflicting secret specifications for {name}")
            secret_env[name] = spec_value
        run_id = _run_id(model.name, eval_key)
        output_dir = spec.resume_results_path or prefix_join(records_prefix, f"{run_id}/results")
        evaluations.append(
            Evaluation(
                identity=EvaluationIdentity(
                    run_id=run_id,
                    created_at=created_at,
                    output_dir=output_dir,
                    eval_ref=definition.record_ref,
                    eval_runtime=definition.runtime_descriptor,
                ),
                executor=definition.executor,
                secret_env_keys=tuple(definition.secret_env),
            )
        )

    endpoint_cluster = accelerator.target_cluster or spec.submission_cluster
    return EvaluationBatch(
        group_id=_group_id(model.name),
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
        secret_env=secret_env,
    )


def prepare_evaluation_batch(spec: LaunchSpec) -> EvaluationBatch:
    """Resolve one launch before an Iris client is opened."""
    provenance = LaunchProvenance(
        git_sha=_git_sha(),
        launch_host=socket.gethostname(),
    )
    return build_evaluation_batch(spec, provenance, _launch_user())


def launch_group(batch: EvaluationBatch, client: IrisClient) -> SubmittedEvaluationBatch:
    return submit_evaluation_batch(batch, client)
