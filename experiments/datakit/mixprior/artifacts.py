# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist one candidate decision and its reproducibility artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple, TypedDict, cast

import numpy as np

from experiments.datakit.mixprior.campaign import Campaign
from experiments.datakit.mixprior.data import canonical_mixture_rows, read_record, record_sha256, sha256, write_record
from experiments.datakit.mixprior.diagnostics import CandidateDiagnostics
from experiments.datakit.mixprior.search import (
    MIXTURE_DENOMINATOR,
    CandidateSelection,
    PoolProvenance,
    exclude_observed,
    validate_candidate_pool,
)
from experiments.datakit.mixprior.surrogate import ModelMetadata

CANDIDATE_ARTIFACT = "candidate.parquet"
POOL_ARTIFACT = "candidate_pool.npz"
ACQUISITION_ARTIFACT = "acquisition_values.npy"
BUNDLE_MANIFEST_ARTIFACT = "bundle_manifest.parquet"


class CandidateDecision(NamedTuple):
    model_metadata: ModelMetadata
    pool: np.ndarray
    selection: CandidateSelection
    diagnostics: CandidateDiagnostics
    proposal: PoolProvenance
    pool_seeds: tuple[int, ...]


class CandidateModelArtifact(TypedDict):
    kind: str
    device: str
    details: dict[str, object]
    objective_sha256: str
    phase_token_fractions: list[float]


class RankedPoolRow(TypedDict):
    pool_index: int
    acquisition_value: float


class CandidateAcquisitionArtifact(TypedDict):
    function: str
    selection_rule: str
    pool_seeds: list[int]
    acquisition_seed: int | None
    pool_size: int
    pool_weights_sha256: str
    pool_artifact: str
    pool_artifact_sha256: str
    values_artifact: str
    values_artifact_sha256: str
    selected_pool_index: int
    selected_acquisition_value: float
    top_pool_rows: list[RankedPoolRow]
    proposal: PoolProvenance
    observed_target_mixtures_excluded: bool


class CandidateConstraintsArtifact(TypedDict):
    simplex_count: int
    mixture_denominator: int


class CandidateArtifact(TypedDict):
    schema_version: int
    candidate_id: str
    campaign_manifest_sha256: str
    dependency_lock_sha256: str
    target_swarm: str
    source_swarms: list[str]
    candidate_store_uri: str
    model: CandidateModelArtifact
    acquisition: CandidateAcquisitionArtifact
    constraints: CandidateConstraintsArtifact
    mixture_components: list[str]
    phase_weights: list[dict[str, float]]
    diagnostics: CandidateDiagnostics


class BundleGenerationArtifact(TypedDict):
    acquisition_function: str
    selection_rule: str
    pool_size: int
    pool_seeds: list[int]
    acquisition_seed: int | None


class BundleManifest(TypedDict):
    schema_version: int
    campaign_uri: str
    campaign_manifest_sha256: str
    candidate_id: str
    target_swarm: str
    source_swarms: list[str]
    generation: BundleGenerationArtifact
    artifact_sha256: dict[str, str]


def candidate_id(weights: np.ndarray) -> str:
    return hashlib.sha256(canonical_mixture_rows(weights[None]).tobytes()).hexdigest()[:16]


def pool_sha256(weights: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest()


def write_candidate_bundle(
    *,
    campaign_manifest: Path,
    campaign: Campaign,
    decision: CandidateDecision,
    output_dir: Path,
    dependency_lock: Path,
) -> CandidateArtifact:
    """Write the selected candidate and the exact artifacts supporting it."""
    model_metadata = decision.model_metadata
    pool = decision.pool
    acquired = decision.selection.acquired
    selected_weights = decision.selection.weights
    pool = validate_candidate_pool(pool, campaign.target.data.weights.shape[1:])
    if acquired.acquisition_values.shape != (len(pool),):
        raise ValueError("Acquisition values must align with the candidate pool")
    if acquired.pool_index < 0 or acquired.pool_index >= len(pool):
        raise ValueError("Selected pool index is outside the candidate pool")
    if acquired.acquisition_value != acquired.acquisition_values[acquired.pool_index]:
        raise ValueError("Selected acquisition value does not match its pool row")
    if not np.array_equal(selected_weights, pool[acquired.pool_index]):
        raise ValueError("Selected weights do not match the selected pool row")
    output_dir.mkdir(parents=True, exist_ok=False)
    pool_path = output_dir / POOL_ARTIFACT
    acquisition_path = output_dir / ACQUISITION_ARTIFACT
    candidate_path = output_dir / CANDIDATE_ARTIFACT

    np.savez_compressed(pool_path, weights=pool)
    np.save(acquisition_path, acquired.acquisition_values)

    order = np.argsort(acquired.acquisition_values)[::-1][:10]
    payload: CandidateArtifact = {
        "schema_version": 5,
        "candidate_id": candidate_id(selected_weights),
        "campaign_manifest_sha256": sha256(campaign_manifest),
        "dependency_lock_sha256": sha256(dependency_lock),
        "target_swarm": campaign.target.swarm_id,
        "source_swarms": [source.swarm_id for source in campaign.sources],
        "candidate_store_uri": campaign.target.provenance.store_uri,
        "model": {
            "kind": model_metadata["kind"],
            "device": model_metadata["device"],
            "details": model_metadata["details"],
            "objective_sha256": record_sha256(campaign.objective_metadata),
            "phase_token_fractions": (campaign.target.phase_budgets / campaign.target.phase_budgets.sum()).tolist(),
        },
        "acquisition": {
            "function": decision.selection.acquisition_function,
            "selection_rule": decision.selection.selection_rule,
            "pool_seeds": list(decision.pool_seeds),
            "acquisition_seed": decision.selection.acquisition_seed,
            "pool_size": len(pool),
            "pool_weights_sha256": pool_sha256(pool),
            "pool_artifact": POOL_ARTIFACT,
            "pool_artifact_sha256": sha256(pool_path),
            "values_artifact": ACQUISITION_ARTIFACT,
            "values_artifact_sha256": sha256(acquisition_path),
            "selected_pool_index": acquired.pool_index,
            "selected_acquisition_value": acquired.acquisition_value,
            "top_pool_rows": [
                {
                    "pool_index": int(index),
                    "acquisition_value": float(acquired.acquisition_values[index]),
                }
                for index in order
            ],
            "proposal": decision.proposal,
            "observed_target_mixtures_excluded": len(exclude_observed(pool, campaign.target.data.weights)) == len(pool),
        },
        "constraints": {
            "simplex_count": len(campaign.target.phase_budgets),
            "mixture_denominator": MIXTURE_DENOMINATOR,
        },
        "mixture_components": campaign.target.data.mixture_components,
        "phase_weights": [
            dict(
                zip(
                    campaign.target.data.mixture_components,
                    phase.tolist(),
                    strict=True,
                )
            )
            for phase in selected_weights
        ],
        "diagnostics": decision.diagnostics,
    }
    write_record(candidate_path, payload)
    return payload


def write_bundle_manifest(output_dir: Path, campaign_uri: str) -> Path:
    """Write the remote input and hashes for the complete candidate bundle."""
    candidate = cast(CandidateArtifact, read_record(output_dir / CANDIDATE_ARTIFACT))
    artifact_hashes = {
        name: sha256(output_dir / name)
        for name in (
            CANDIDATE_ARTIFACT,
            POOL_ARTIFACT,
            ACQUISITION_ARTIFACT,
        )
    }
    acquisition = candidate["acquisition"]
    bundle_manifest: BundleManifest = {
        "schema_version": 4,
        "campaign_uri": campaign_uri,
        "campaign_manifest_sha256": candidate["campaign_manifest_sha256"],
        "candidate_id": candidate["candidate_id"],
        "target_swarm": candidate["target_swarm"],
        "source_swarms": candidate["source_swarms"],
        "generation": {
            "acquisition_function": acquisition["function"],
            "selection_rule": acquisition["selection_rule"],
            "pool_size": acquisition["pool_size"],
            "pool_seeds": acquisition["pool_seeds"],
            "acquisition_seed": acquisition["acquisition_seed"],
        },
        "artifact_sha256": artifact_hashes,
    }
    manifest_path = output_dir / BUNDLE_MANIFEST_ARTIFACT
    write_record(manifest_path, bundle_manifest)
    return manifest_path
