# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist one candidate decision and its reproducibility artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch

from experiments.datakit.mixprior.campaign import Campaign
from experiments.datakit.mixprior.data import (
    AcquiredCandidate,
    PoolProvenance,
    record_sha256,
    sha256,
    write_record,
)

CANDIDATE_ARTIFACT = "candidate.parquet"
POOL_ARTIFACT = "candidate_pool.npz"
MODEL_ARTIFACT = "fitted_model.pt"
ACQUISITION_ARTIFACT = "acquisition_values.npy"
CYCLE_ARTIFACT = "cycle.parquet"


class ModelMetadata(TypedDict):
    kind: str
    device: str
    details: dict[str, Any]


def candidate_id(weights: np.ndarray) -> str:
    rounded = np.round(weights, decimals=12)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def pool_sha256(weights: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest()


def write_candidate_bundle(
    *,
    campaign_manifest: Path,
    campaign: Campaign,
    model_payload: dict,
    model_metadata: ModelMetadata,
    pool: np.ndarray,
    acquired: AcquiredCandidate,
    selected_weights: np.ndarray,
    diagnostics: dict,
    phase_token_fractions: np.ndarray,
    output_dir: Path,
    seed: int,
    proposal: PoolProvenance,
    acquisition_function: str,
    selection_rule: str,
    dependency_lock: Path,
) -> dict:
    """Write the selected candidate and the exact artifacts supporting it."""
    if acquired.acquisition_values.shape != (len(pool),):
        raise ValueError("Acquisition values must align with the candidate pool")
    if acquired.pool_index < 0 or acquired.pool_index >= len(pool):
        raise ValueError("Selected pool index is outside the candidate pool")
    if acquired.acquisition_value != acquired.acquisition_values[acquired.pool_index]:
        raise ValueError("Selected acquisition value does not match its pool row")
    if not np.array_equal(selected_weights, pool[acquired.pool_index]):
        raise ValueError("Selected weights do not match the selected pool row")
    if not proposal["parameters"]:
        raise ValueError("Pool provenance requires at least one parameter")
    if not model_metadata["details"]:
        raise ValueError("Model metadata requires at least one detail")
    output_dir.mkdir(parents=True, exist_ok=False)
    pool_path = output_dir / POOL_ARTIFACT
    model_path = output_dir / MODEL_ARTIFACT
    acquisition_path = output_dir / ACQUISITION_ARTIFACT
    candidate_path = output_dir / CANDIDATE_ARTIFACT

    np.savez_compressed(pool_path, weights=pool)
    np.save(acquisition_path, acquired.acquisition_values)
    torch.save(model_payload, model_path)

    objective_payload = {
        "reference": campaign.objective.payload(),
        "objective_metrics": list(campaign.objective_metrics),
    }
    order = np.argsort(acquired.acquisition_values)[::-1][:10]
    payload = {
        "schema_version": 3,
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
            "objective_metrics": list(campaign.objective_metrics),
            "objective_sha256": record_sha256(objective_payload),
            "phase_token_fractions": phase_token_fractions.tolist(),
            "artifact": MODEL_ARTIFACT,
            "artifact_sha256": sha256(model_path),
        },
        "acquisition": {
            "function": acquisition_function,
            "selection_rule": selection_rule,
            "seed": seed,
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
            "proposal": proposal,
        },
        "constraints": {
            "simplex_count": len(campaign.target.phase_budgets),
            "max_cumulative_epochs": campaign.max_cumulative_epochs,
            "observed_target_mixtures_excluded": _excludes_observed(pool, campaign.target.data.weights),
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
        "diagnostics": diagnostics,
    }
    write_record(candidate_path, payload)
    return payload


def write_cycle_record(output_dir: Path, campaign_uri: str, candidate: dict) -> Path:
    """Write the remote-input and generated-artifact record for one cycle."""
    artifact_hashes = {
        name: sha256(output_dir / name)
        for name in (
            CANDIDATE_ARTIFACT,
            POOL_ARTIFACT,
            MODEL_ARTIFACT,
            ACQUISITION_ARTIFACT,
        )
    }
    acquisition = candidate["acquisition"]
    cycle = {
        "schema_version": 2,
        "campaign_uri": campaign_uri,
        "campaign_manifest_sha256": candidate["campaign_manifest_sha256"],
        "candidate_id": candidate["candidate_id"],
        "target_swarm": candidate["target_swarm"],
        "source_swarms": candidate["source_swarms"],
        "generation": {
            "acquisition_function": acquisition["function"],
            "selection_rule": acquisition["selection_rule"],
            "pool_size": acquisition["pool_size"],
            "seed": acquisition["seed"],
        },
        "artifact_sha256": artifact_hashes,
    }
    cycle_path = output_dir / CYCLE_ARTIFACT
    write_record(cycle_path, cycle)
    return cycle_path


def _excludes_observed(pool: np.ndarray, observed: np.ndarray) -> bool:
    observed_rows = {row.tobytes() for row in np.round(observed.reshape(len(observed), -1), decimals=12)}
    return all(row.tobytes() not in observed_rows for row in np.round(pool.reshape(len(pool), -1), decimals=12))
