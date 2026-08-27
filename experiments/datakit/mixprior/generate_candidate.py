# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit the quadratic exposure GP and persist one candidate decision."""

from __future__ import annotations

from pathlib import Path

import jax

from experiments.datakit.mixprior.artifacts import CandidateArtifact, CandidateDecision, write_candidate_bundle
from experiments.datakit.mixprior.campaign import Campaign, load_campaign
from experiments.datakit.mixprior.diagnostics import candidate_diagnostics
from experiments.datakit.mixprior.quadratic_exposure import fit_quadratic_exposure_model
from experiments.datakit.mixprior.search import (
    Acquisition,
    campaign_pool_inputs,
    sample_standard_candidate_pool,
    select_candidate,
)


def search_candidate(
    campaign: Campaign,
    *,
    acquisition: Acquisition,
    pool_size_per_seed: int,
    pool_seeds: tuple[int, ...],
    device: jax.Device,
) -> CandidateDecision:
    """Fit the quadratic exposure GP and acquire from the standard pool."""
    model = fit_quadratic_exposure_model(campaign, device)
    pool = sample_standard_candidate_pool(campaign_pool_inputs(campaign), pool_size_per_seed, pool_seeds)
    selection = select_candidate(campaign, model, pool.weights, acquisition)
    diagnostics = candidate_diagnostics(campaign.target, selection.weights, selection.posterior)
    return CandidateDecision(
        model.model_metadata,
        pool.weights,
        selection,
        diagnostics,
        pool.provenance,
        pool.seeds,
    )


def generate_candidate(
    *,
    campaign_manifest: Path,
    output_dir: Path,
    dependency_lock: Path,
    acquisition: Acquisition,
    pool_size_per_seed: int,
    pool_seeds: tuple[int, ...],
    device: jax.Device,
) -> CandidateArtifact:
    """Load a campaign, fit the quadratic GP, and persist its decision."""
    campaign = load_campaign(campaign_manifest)
    decision = search_candidate(
        campaign,
        acquisition=acquisition,
        pool_size_per_seed=pool_size_per_seed,
        pool_seeds=pool_seeds,
        device=device,
    )
    return write_candidate_bundle(
        campaign_manifest=campaign_manifest,
        campaign=campaign,
        decision=decision,
        output_dir=output_dir,
        dependency_lock=dependency_lock,
    )
