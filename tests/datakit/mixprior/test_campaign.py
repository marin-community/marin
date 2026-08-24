# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from experiments.datakit.mixprior.campaign import Campaign, download_campaign, load_campaign
from experiments.datakit.mixprior.data import load_observations, write_record


def test_canonical_observations_reject_missing_phase(tmp_path: Path) -> None:
    swarm_id = "strict-swarm"
    observations = tmp_path / "observations.parquet"
    buckets = tmp_path / "buckets.parquet"
    weights = {"c00q0": 0.25, "c00q1": 0.75}
    pd.DataFrame(
        [
            {
                "observation_id": f"{swarm_id}:run",
                "swarm_id": swarm_id,
                "run_name": "run",
                "group": "test",
                "phase0_weights": weights,
                "phase1_weights": None,
                "grouped_bpb": {"boolq_0shot": 1.0},
            }
        ]
    ).to_parquet(observations, index=False)
    write_record(
        buckets,
        {
            "cells": [
                {
                    "cell": "c00q0",
                    "domain": "domain-0",
                    "quality": 0,
                    "available_tokens": 1,
                },
                {
                    "cell": "c00q1",
                    "domain": "domain-0",
                    "quality": 1,
                    "available_tokens": 1,
                },
            ]
        },
    )

    with pytest.raises(TypeError):
        load_observations(observations, buckets, swarm_id)


def test_campaign_rejects_corrupt_pinned_artifact(
    campaign_bundle: tuple[Path, Path],
) -> None:
    manifest_path, observations = campaign_bundle
    observations.write_bytes(observations.read_bytes() + b"corrupt")

    with pytest.raises(ValueError):
        load_campaign(manifest_path)


def test_campaign_download_requires_hugging_face_commit(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        download_campaign(
            "hf://datasets/marin-community/grug-moe-mix-swarm/registry/v1/transfer_campaign.parquet",
            "unused",
            tmp_path / "campaign",
        )


def test_swarm_provenance_rejects_more_active_than_total_parameters(
    tiny_campaign: Campaign,
) -> None:
    with pytest.raises(ValueError):
        replace(
            tiny_campaign.target.provenance,
            model_active_parameters=2,
            model_total_parameters=1,
        )


def test_swarm_phase_budgets_match_provenance_token_horizon(
    tiny_campaign: Campaign,
) -> None:
    provenance = replace(tiny_campaign.target.provenance, simulated_training_tokens=3)

    with pytest.raises(ValueError):
        replace(tiny_campaign.target, provenance=provenance)
