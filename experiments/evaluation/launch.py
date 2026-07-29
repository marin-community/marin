# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve experiment catalogs into shared evaluation batches."""

from __future__ import annotations

import getpass
import json
import os
import socket
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.client import IrisClient
from iris.cluster.config import load_config
from marin.evaluation.harbor.dataset import validate_harbor_dataset_source
from marin.evaluation.harbor.driver_config import (
    ValidatedHarborConfig,
    legacy_harbor_policy_document,
    preflight_harbor_configs,
)
from marin.evaluation.harbor.runner import canonical_served_name
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
    harbor_definition,
)
from experiments.evaluation.fleet import MARIN_EVAL_HARDWARE
from experiments.evaluation.models import models


@dataclass(frozen=True)
class HarborConfigSelection:
    """One Harbor config file supplied at launch."""

    name: str
    path: Path


@dataclass(frozen=True)
class LaunchSpec:
    """One model, evaluation selection, execution target, and record destination."""

    model: str
    evals: tuple[str, ...]
    harbor_configs: tuple[HarborConfigSelection, ...]
    platform: Platform
    accelerator: str | None
    limit: int | None
    records_prefix: str | None
    cluster: str
    version: str | None = None
    description: str | None = None


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
    config_definitions: tuple[tuple[str, EvaluationDefinition], ...] = tuple(
        (selection.name, harbor_definition(selection.name, selection.path)) for selection in spec.harbor_configs
    )
    definitions = registry_definitions + config_definitions
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


def _preflight_definitions(
    definitions: tuple[tuple[str, EvaluationDefinition], ...],
    model: ModelConfig,
    limit: int | None,
) -> tuple[tuple[str, _ResolvedDefinition], ...]:
    harbor_definitions = [definition for _, definition in definitions if isinstance(definition, HarborDefinition)]
    with tempfile.TemporaryDirectory(prefix="marin-harbor-legacy-policies-") as temp_dir:
        requests: list[tuple[Path, dict[str, object]]] = []
        for index, definition in enumerate(harbor_definitions):
            if definition.config_path is not None:
                path = definition.config_path
            else:
                assert definition.legacy_config is not None
                path = Path(temp_dir) / f"{index:04d}-{definition.name}.json"
                path.write_text(json.dumps(legacy_harbor_policy_document(definition.legacy_config)))
                path.chmod(0o600)
            requests.append((path, dict(model.agent.agent_kwargs)))
        validated_configs = iter(preflight_harbor_configs(requests))

    resolved: list[tuple[str, _ResolvedDefinition]] = []
    for name, definition in definitions:
        if isinstance(definition, EvalchemyDefinition):
            resolved.append(
                (
                    name,
                    _ResolvedDefinition(
                        record_ref=definition.record_ref,
                        runtime_descriptor=definition.runtime_descriptor,
                        executor=definition.executor_for(model, limit),
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
    model = models()[spec.model]
    accelerator = MARIN_EVAL_HARDWARE.select(model, spec.platform, spec.accelerator)
    definitions = _preflight_definitions(_evaluation_definitions(spec), model, spec.limit)
    records_prefix = records_prefix_for(accelerator, spec)
    created_at = datetime.now(UTC).isoformat()
    evaluations: list[Evaluation] = []
    secret_env: dict[str, SecretSpec] = {}
    for eval_key, definition in definitions:
        for name, spec_value in definition.secret_env.items():
            if name in secret_env and secret_env[name] != spec_value:
                raise ValueError(f"evaluations declare conflicting secret specifications for {name}")
            secret_env[name] = spec_value
        run_id = _run_id(spec.model, eval_key)
        output_dir = prefix_join(records_prefix, f"{run_id}/results")
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

    endpoint_cluster = accelerator.target_cluster or spec.cluster
    return EvaluationBatch(
        group_id=_group_id(spec.model),
        user=user,
        version=spec.version,
        description=spec.description,
        records_prefix=records_prefix,
        model=model,
        accelerator=accelerator,
        capability_origin=_capability_origin(endpoint_cluster),
        api_model=canonical_served_name(model.name),
        evaluations=tuple(evaluations),
        provenance=provenance,
        secret_env=secret_env,
    )


def prepare_evaluation_batch(spec: LaunchSpec) -> EvaluationBatch:
    """Resolve and preflight one launch before an Iris client is opened."""
    provenance = LaunchProvenance(
        git_sha=_git_sha(),
        launch_host=socket.gethostname(),
    )
    return build_evaluation_batch(spec, provenance, _launch_user())


def launch_group(batch: EvaluationBatch, client: IrisClient) -> SubmittedEvaluationBatch:
    """Submit one preflighted CPU orchestrator batch."""
    return submit_evaluation_batch(batch, client)
