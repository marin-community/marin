# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a Bayesian-optimization campaign from pinned swarm manifests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NamedTuple, TypedDict, cast

import numpy as np
from huggingface_hub import HfFileSystem

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
    HingeObjective,
    VarianceNormalizedObjective,
    pooled_replicate_sd,
)

PROPORTIONAL_REFERENCE_GROUP = "marin_proportional"
CAMPAIGN_MANIFEST = "transfer_campaign.parquet"
HF_COMMIT_URI = re.compile(r"^hf://datasets/[^/@]+/[^/@]+@[0-9a-f]{40}/.+$")


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
    kernel_reference_swarm: str
    response_tasks: list[str]
    objective_epsilon: float
    max_cumulative_epochs: float


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
    objective: HingeObjective
    observation_sd: dict[str, float]
    objective_metrics: tuple[str, ...]
    kernel_reference_swarm: str
    max_cumulative_epochs: float

    def __post_init__(self) -> None:
        missing_metrics = sorted(set(self.objective_metrics) - set(self.objective.labels))
        if missing_metrics:
            raise ValueError(f"Objective is missing metrics: {missing_metrics}")
        missing_noise = sorted(set(self.objective_metrics) - set(self.observation_sd))
        if missing_noise:
            raise ValueError(f"Observation-noise estimates are missing metrics: {missing_noise}")
        response_sd = np.asarray([self.observation_sd[label] for label in self.objective_metrics])
        if np.any(response_sd <= 0) or not np.isfinite(response_sd).all():
            raise ValueError("Observation standard deviations must be finite and positive")


class CampaignInputs(NamedTuple):
    target: Swarm
    sources: tuple[Swarm, ...]
    objective_reference: Swarm
    noise_reference: Swarm
    objective_metrics: tuple[str, ...]
    hinge_tolerance: float
    kernel_reference_swarm: str
    max_cumulative_epochs: float


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
        "kernel_reference_swarm",
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
    max_cumulative_epochs = float(manifest["max_cumulative_epochs"])
    if not np.isfinite(max_cumulative_epochs) or max_cumulative_epochs <= 0:
        raise ValueError("Maximum cumulative epochs must be finite and positive")
    return CampaignInputs(
        target=loaded[target_id],
        sources=tuple(loaded[swarm_id] for swarm_id in manifest["source_swarms"]),
        objective_reference=loaded[manifest["objective_reference_swarm"]],
        noise_reference=loaded[manifest["noise_reference_swarm"]],
        objective_metrics=objective_metrics,
        hinge_tolerance=float(manifest["objective_epsilon"]),
        kernel_reference_swarm=manifest["kernel_reference_swarm"],
        max_cumulative_epochs=max_cumulative_epochs,
    )


def build_variance_normalized_campaign(inputs: CampaignInputs) -> Campaign:
    """Build the hinge objective and observation noise from reference data."""
    objective_data = inputs.objective_reference.data
    objective = VarianceNormalizedObjective.fit(
        objective_data.labels,
        objective_data.outcomes,
        objective_data.frame.group.eq(PROPORTIONAL_REFERENCE_GROUP).to_numpy(),
        epsilon=inputs.hinge_tolerance,
    )
    noise_data = inputs.noise_reference.data
    reference_sd = pooled_replicate_sd(noise_data)
    sd_by_label = dict(zip(noise_data.labels, reference_sd, strict=True))
    return build_campaign(inputs, objective, sd_by_label)


def build_campaign(
    inputs: CampaignInputs,
    objective: HingeObjective,
    observation_sd: dict[str, float],
) -> Campaign:
    return Campaign(
        target=inputs.target,
        sources=inputs.sources,
        objective=objective,
        observation_sd=observation_sd,
        objective_metrics=inputs.objective_metrics,
        kernel_reference_swarm=inputs.kernel_reference_swarm,
        max_cumulative_epochs=inputs.max_cumulative_epochs,
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


def download_campaign(campaign_uri: str, campaign_sha256: str, destination: Path) -> Path:
    """Materialize one commit-pinned campaign from Hugging Face."""
    filesystem = HfFileSystem(token=False)
    if not HF_COMMIT_URI.fullmatch(campaign_uri):
        raise ValueError("Campaign URI must pin a 40-character Hugging Face commit")
    campaign_root_uri = campaign_uri.rsplit("/", 1)[0]
    destination.mkdir(parents=True, exist_ok=False)
    manifest_path = destination / CAMPAIGN_MANIFEST
    _download(filesystem, campaign_uri, manifest_path)
    if sha256(manifest_path) != campaign_sha256:
        raise ValueError("Campaign manifest hash mismatch")

    manifest = cast(CampaignManifest, read_record(manifest_path))
    registry_reference = manifest["registry"]
    registry_path = destination / _safe_relative_path(registry_reference["path"])
    _download_reference(filesystem, campaign_root_uri, destination, registry_reference, registry_path)
    registry = cast(SwarmRegistry, read_record(registry_path))
    swarms_by_id = {reference["swarm_id"]: reference for reference in registry["swarms"]}

    basis_ids = set()
    selected_ids = [manifest["target_swarm"], *manifest["source_swarms"]]
    for swarm_id in selected_ids:
        swarm_reference = swarms_by_id[swarm_id]
        swarm_path = destination / _safe_relative_path(swarm_reference["path"])
        _download_reference(filesystem, campaign_root_uri, destination, swarm_reference, swarm_path)
        swarm = cast(SwarmManifest, read_record(swarm_path))
        for reference in (swarm["observations"], swarm["buckets"], swarm["content"]):
            artifact_path = swarm_path.parent / _safe_relative_path(reference["path"])
            _download_reference(filesystem, campaign_root_uri, destination, reference, artifact_path)
        content_path = swarm_path.parent / _safe_relative_path(swarm["content"]["path"])
        basis_ids.add(read_record(content_path)["basis_id"])

    bases_by_id = {reference["basis_id"]: reference for reference in registry["content_bases"]}
    for basis_id in basis_ids:
        basis_reference = bases_by_id[basis_id]
        basis_path = destination / _safe_relative_path(basis_reference["path"])
        _download_reference(filesystem, campaign_root_uri, destination, basis_reference, basis_path)
        basis = cast(ContentBasisManifest, read_record(basis_path))
        lookup_reference = basis["lookup"]
        lookup_path = basis_path.parent / _safe_relative_path(lookup_reference["path"])
        _download_reference(filesystem, campaign_root_uri, destination, lookup_reference, lookup_path)
    return manifest_path


def _download_reference(
    filesystem: HfFileSystem,
    campaign_root_uri: str,
    campaign_directory: Path,
    reference: ArtifactReference,
    destination: Path,
) -> None:
    relative = destination.relative_to(campaign_directory).as_posix()
    _download(filesystem, f"{campaign_root_uri}/{relative}", destination)
    if sha256(destination) != reference["sha256"]:
        raise ValueError(f"Downloaded artifact hash mismatch: {destination}")


def _download(filesystem: HfFileSystem, source_uri: str, destination: Path) -> None:
    if not HF_COMMIT_URI.fullmatch(source_uri):
        raise ValueError("Campaign artifacts must use commit-pinned Hugging Face URIs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    filesystem.get_file(source_uri, destination)


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Campaign artifact path escapes its root: {value}")
    return path
