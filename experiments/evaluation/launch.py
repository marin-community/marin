# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve experiment catalogs into shared evaluation batches."""

from __future__ import annotations

import getpass
import os
import socket
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.client import IrisClient
from iris.cluster.config import load_config
from marin.evaluation.evalchemy.runtime import EVALCHEMY_IMAGE
from marin.evaluation.harbor.runner import canonical_served_name
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.records import (
    CW_RECORDS_PREFIX,
    DEFAULT_RECORDS_PREFIX,
    Provenance,
)
from marin.evaluation.runner import (
    Evaluation,
    EvaluationBatch,
    EvaluationIdentity,
    SubmittedEvaluationBatch,
    submit_evaluation_batch,
)
from rigging.config_discovery import resolve_cluster_config
from rigging.filesystem import prefix_join

from experiments.evaluation.evals import EVALS
from experiments.evaluation.fleet import MARIN_EVAL_HARDWARE
from experiments.evaluation.models import models


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


def build_evaluation_batch(
    spec: LaunchSpec,
    provenance: Provenance,
    user: str,
) -> EvaluationBatch:
    """Resolve experiment names into one model-serving evaluation batch."""
    if not spec.evals:
        raise ValueError("at least one evaluation is required")
    if len(set(spec.evals)) != len(spec.evals):
        raise ValueError(f"duplicate eval keys in one launch: {list(spec.evals)}")

    model = models()[spec.model]
    accelerator = MARIN_EVAL_HARDWARE.select(model, spec.platform, spec.accelerator)
    records_prefix = records_prefix_for(accelerator, spec)
    created_at = datetime.now(UTC).isoformat()
    evaluations: list[Evaluation] = []
    for eval_key in spec.evals:
        definition = EVALS[eval_key]
        run_id = _run_id(spec.model, eval_key)
        output_dir = prefix_join(records_prefix, f"{run_id}/results")
        evaluations.append(
            Evaluation(
                identity=EvaluationIdentity(
                    run_id=run_id,
                    created_at=created_at,
                    output_dir=output_dir,
                    eval_ref=definition.record_ref,
                ),
                executor=definition.executor_for(model, spec.limit),
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
    )


def launch_group(spec: LaunchSpec, client: IrisClient) -> SubmittedEvaluationBatch:
    """Submit one CPU orchestrator for a resolved batch."""
    provenance = Provenance(
        git_sha=_git_sha(),
        eval_image=EVALCHEMY_IMAGE,
        launch_host=socket.gethostname(),
    )
    batch = build_evaluation_batch(spec, provenance, _launch_user())
    return submit_evaluation_batch(batch, client)
