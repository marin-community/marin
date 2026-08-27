# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and assemble a local transfer-optimization campaign.

A campaign assigns target, source, objective-reference, and noise-reference
roles to registered swarms. Remote campaigns pin a Hugging
Face commit; every referenced local artifact is checked against its SHA-256
before it becomes a typed ``Campaign``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypedDict, cast

import numpy as np

from experiments.datakit.mixprior.data import (
    ArtifactReference,
    ContentRecord,
    ObservationArtifactReference,
    Swarm,
    SwarmProvenance,
    load_observations,
    read_record,
    sha256,
)
from experiments.datakit.mixprior.objective import (
    ScalarObjective,
    fit_harrier_hinge_objective,
    objective_metadata,
)

CAMPAIGN_MANIFEST = "transfer_campaign.parquet"


class SwarmReference(ArtifactReference):
    swarm_id: str


class ContentBasisReference(ArtifactReference):
    basis_id: str


class SwarmRegistry(TypedDict):
    schema_version: int
    content_bases: list[ContentBasisReference]
    swarms: list[SwarmReference]


class CampaignManifest(TypedDict):
    schema_version: int
    registry: ArtifactReference
    target_swarm: str
    source_swarms: list[str]
    objective_reference_swarm: str
    noise_reference_swarm: str
    response_tasks: list[str]
    objective_epsilon: float


class ContentBasisManifest(TypedDict):
    schema_version: int
    basis_id: str
    lookup: ArtifactReference


class SwarmManifest(TypedDict):
    schema_version: int
    swarm_id: str
    observations: ObservationArtifactReference
    buckets: ArtifactReference
    content: ArtifactReference
    phase_budgets: list[float]
    store_uri: str
    store_artifact_uri: str
    training_recipe: str
    tokenizer: str
    evaluation_suite: str
    model_active_parameters: int
    model_total_parameters: int
    physical_training_tokens: int
    simulated_training_tokens: int


@dataclass(frozen=True)
class Campaign:
    target: Swarm
    sources: tuple[Swarm, ...]
    objective: ScalarObjective
    objective_metadata: dict[str, object]


class CampaignInputs(NamedTuple):
    target: Swarm
    sources: tuple[Swarm, ...]
    objective_reference: Swarm
    noise_reference: Swarm
    objective_metrics: tuple[str, ...]
    hinge_tolerance: float


def load_campaign_inputs(manifest_path: Path) -> CampaignInputs:
    manifest = cast(CampaignManifest, read_record(manifest_path))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported transfer campaign schema")
    root = manifest_path.parent
    registry_path = _checked_path(root, manifest["registry"])
    registry = cast(SwarmRegistry, read_record(registry_path))
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported swarm registry schema")
    registry_references = {reference["swarm_id"]: reference for reference in registry["swarms"]}
    if len(registry_references) != len(registry["swarms"]):
        raise ValueError("Swarm registry IDs must be unique")
    selected_ids = [manifest["target_swarm"], *manifest["source_swarms"]]
    missing_swarms = sorted(set(selected_ids) - set(registry_references))
    if missing_swarms:
        raise ValueError(f"Campaign swarms are missing from the registry: {missing_swarms}")
    references = [registry_references[swarm_id] for swarm_id in selected_ids]
    reference_ids = [reference["swarm_id"] for reference in references]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Campaign swarm IDs must be unique")

    loaded: dict[str, Swarm] = {}
    for reference in references:
        swarm_path = _checked_path(root, reference)
        swarm = load_swarm(swarm_path)
        if swarm.swarm_id != reference["swarm_id"]:
            raise ValueError(f"Campaign expected swarm {reference['swarm_id']}, got {swarm.swarm_id}")
        loaded[swarm.swarm_id] = swarm

    role_fields = (
        "objective_reference_swarm",
        "noise_reference_swarm",
    )
    for field in role_fields:
        if manifest[field] not in loaded:
            raise ValueError(f"{field} is missing from the campaign")
    objective_metrics = tuple(manifest["response_tasks"])
    if not objective_metrics or len(objective_metrics) != len(set(objective_metrics)):
        raise ValueError("Objective metrics must be a non-empty unique list")

    basis_ids = {swarm.content_basis_id for swarm in loaded.values()}
    dimensions = {swarm.content_matrix.shape[1] for swarm in loaded.values()}
    if len(basis_ids) != 1 or len(dimensions) != 1:
        raise ValueError("Campaign swarms must share one content basis and dimension")
    basis_references = {reference["basis_id"]: reference for reference in registry["content_bases"]}
    if len(basis_references) != len(registry["content_bases"]):
        raise ValueError("Content basis IDs must be unique")
    basis_id = next(iter(basis_ids))
    if basis_id not in basis_references:
        raise ValueError(f"Content basis is missing from the registry: {basis_id}")
    basis_path = _checked_path(root, basis_references[basis_id])
    basis = cast(ContentBasisManifest, read_record(basis_path))
    if basis.get("schema_version") != 1 or basis.get("basis_id") != basis_id:
        raise ValueError("Content basis manifest does not match the registered basis")
    _checked_path(basis_path.parent, basis["lookup"])

    target_id = manifest["target_swarm"]
    return CampaignInputs(
        target=loaded[target_id],
        sources=tuple(loaded[swarm_id] for swarm_id in manifest["source_swarms"]),
        objective_reference=loaded[manifest["objective_reference_swarm"]],
        noise_reference=loaded[manifest["noise_reference_swarm"]],
        objective_metrics=objective_metrics,
        hinge_tolerance=float(manifest["objective_epsilon"]),
    )


def build_variance_normalized_campaign(inputs: CampaignInputs) -> Campaign:
    objective = fit_harrier_hinge_objective(
        inputs.objective_reference.data,
        inputs.noise_reference.data,
        metrics=inputs.objective_metrics,
        epsilon=inputs.hinge_tolerance,
    )
    return build_campaign(inputs, objective, objective_metadata(objective))


def build_campaign(
    inputs: CampaignInputs,
    objective: ScalarObjective,
    objective_metadata: dict[str, object],
) -> Campaign:
    return Campaign(
        target=inputs.target,
        sources=inputs.sources,
        objective=objective,
        objective_metadata=objective_metadata,
    )


def load_campaign(manifest_path: Path) -> Campaign:
    return build_variance_normalized_campaign(load_campaign_inputs(manifest_path))


def load_swarm(manifest_path: Path) -> Swarm:
    spec = cast(SwarmManifest, read_record(manifest_path))
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported swarm schema")
    root = manifest_path.parent
    if spec["observations"].get("schema") != "mixture-observations-v1":
        raise ValueError("Swarm observations must use mixture-observations-v1")
    observations = _checked_path(root, spec["observations"])
    components = _checked_path(root, spec["buckets"])
    data = load_observations(observations, components, spec["swarm_id"])
    content_path = _checked_path(root, spec["content"])
    content = cast(ContentRecord, read_record(content_path))
    if content["cells"] != data.mixture_components:
        raise ValueError("Content rows must use the mixture-component order")
    return Swarm(
        swarm_id=spec["swarm_id"],
        data=data,
        phase_budgets=np.asarray(spec["phase_budgets"], dtype=np.float64),
        content_basis_id=content["basis_id"],
        content_matrix=np.asarray(content["matrix"], dtype=np.float64),
        provenance=SwarmProvenance(
            store_uri=spec["store_uri"],
            store_artifact_uri=spec["store_artifact_uri"],
            training_recipe=spec["training_recipe"],
            tokenizer=spec["tokenizer"],
            evaluation_suite=spec["evaluation_suite"],
            model_active_parameters=int(spec["model_active_parameters"]),
            model_total_parameters=int(spec["model_total_parameters"]),
            physical_training_tokens=int(spec["physical_training_tokens"]),
            simulated_training_tokens=int(spec["simulated_training_tokens"]),
            content_provenance=dict(content["provenance"]),
        ),
    )


def _checked_path(root: Path, reference: ArtifactReference) -> Path:
    if not reference.get("path") or not reference.get("sha256"):
        raise ValueError("Artifact references require a path and SHA-256")
    relative = Path(reference["path"])
    if relative.is_absolute():
        raise ValueError("Artifact paths must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its manifest root")
    if sha256(path) != reference["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return path
