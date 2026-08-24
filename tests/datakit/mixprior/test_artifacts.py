# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.datakit.mixprior.artifacts import (
    ACQUISITION_ARTIFACT,
    CANDIDATE_ARTIFACT,
    MODEL_ARTIFACT,
    POOL_ARTIFACT,
    candidate_id,
    write_candidate_bundle,
    write_cycle_record,
)
from experiments.datakit.mixprior.campaign import build_campaign, load_campaign_inputs
from experiments.datakit.mixprior.data import AcquiredCandidate, read_record, record_sha256, sha256
from experiments.datakit.mixprior.model import fit_additive_hellinger_model, prepare_hellinger_transfer_data
from experiments.datakit.mixprior.objective import VarianceNormalizedObjective
from experiments.datakit.mixprior.search import (
    build_candidate_selection,
    candidate_diagnostics,
    prepare_candidate_features,
)


def test_custom_stages_persist_selected_candidate_and_cycle_provenance(
    tmp_path: Path,
    campaign_bundle: tuple[Path, Path],
    objective: VarianceNormalizedObjective,
) -> None:
    manifest_path, _ = campaign_bundle
    output_dir = tmp_path / "generated"
    dependency_lock = tmp_path / "uv.lock"
    dependency_lock.write_text("test dependency lock")
    inputs = load_campaign_inputs(manifest_path)
    custom_objective = replace(objective, epsilon=0.75)
    campaign = build_campaign(
        inputs,
        custom_objective,
        {label: 0.1 for label in custom_objective.labels},
    )
    model = fit_additive_hellinger_model(prepare_hellinger_transfer_data(campaign), torch.device("cpu"))
    custom_pool = np.asarray(
        [
            [[0.7, 0.3], [0.6, 0.4]],
            [[0.3, 0.7], [0.4, 0.6]],
        ]
    )
    candidates = prepare_candidate_features(campaign.target, custom_pool)
    acquisition_values = np.asarray([0.0, 1.0])
    acquired = AcquiredCandidate(1, 1.0, acquisition_values)
    selection = build_candidate_selection(
        campaign,
        model,
        candidates,
        acquired,
        acquisition_function="test acquisition",
        selection_rule="choose the second feasible row",
    )
    diagnostics = candidate_diagnostics(
        campaign.target,
        selection.weights,
        selection.posterior,
        objective_name="negative_hinge_loss",
        hinge_tolerance=campaign.objective.epsilon,
        acquisition_function=selection.acquisition_function,
        selection_rule=selection.selection_rule,
    )
    payload = write_candidate_bundle(
        campaign_manifest=manifest_path,
        campaign=campaign,
        model_payload=model.model_state(),
        model_metadata={
            "kind": "test model",
            "device": "cpu",
            "details": {"fixture": "custom stages"},
        },
        pool=custom_pool,
        acquired=selection.acquired,
        selected_weights=selection.weights,
        diagnostics=diagnostics,
        phase_token_fractions=candidates.phase_token_fractions,
        output_dir=output_dir,
        seed=17,
        proposal={"kind": "hand-authored", "parameters": {"fixture": "second-row"}},
        acquisition_function=selection.acquisition_function,
        selection_rule=selection.selection_rule,
        dependency_lock=dependency_lock,
    )

    persisted_pool = np.load(output_dir / POOL_ARTIFACT)["weights"]
    persisted_weights = np.asarray(
        [[phase[component] for component in payload["mixture_components"]] for phase in payload["phase_weights"]]
    )
    expected_objective_hash = record_sha256(
        {
            "reference": custom_objective.payload(),
            "objective_metrics": list(campaign.objective_metrics),
        }
    )

    assert payload["schema_version"] == 3
    assert payload["acquisition"]["selected_pool_index"] == 1
    assert payload["acquisition"]["function"] == "test acquisition"
    assert payload["acquisition"]["selection_rule"] == ("choose the second feasible row")
    assert payload["model"]["objective_metrics"] == list(campaign.objective_metrics)
    assert set(payload["diagnostics"]["posterior"]) == {
        "objective_mean",
        "objective_sd",
        "incumbent_objective_value",
        "probability_of_improvement",
    }
    assert np.array_equal(persisted_pool, custom_pool)
    assert np.array_equal(persisted_weights, custom_pool[1])
    assert payload["candidate_id"] == candidate_id(custom_pool[1])
    assert payload["model"]["objective_sha256"] == expected_objective_hash
    assert payload["diagnostics"]["summary"]["hinge_tolerance"] == 0.75
    assert sha256(output_dir / POOL_ARTIFACT) == payload["acquisition"]["pool_artifact_sha256"]
    assert sha256(output_dir / MODEL_ARTIFACT) == payload["model"]["artifact_sha256"]
    assert sha256(output_dir / ACQUISITION_ARTIFACT) == payload["acquisition"]["values_artifact_sha256"]
    assert read_record(output_dir / CANDIDATE_ARTIFACT) == payload

    campaign_uri = "hf://datasets/test/campaign@0000000000000000000000000000000000000000/transfer_campaign.parquet"
    cycle_path = write_cycle_record(output_dir, campaign_uri, payload)
    cycle = read_record(cycle_path)
    assert cycle["schema_version"] == 2
    assert cycle["candidate_id"] == payload["candidate_id"]
    assert cycle["generation"] == {
        "acquisition_function": "test acquisition",
        "selection_rule": "choose the second feasible row",
        "pool_size": 2,
        "seed": 17,
    }
    assert cycle["artifact_sha256"] == {
        CANDIDATE_ARTIFACT: sha256(output_dir / CANDIDATE_ARTIFACT),
        POOL_ARTIFACT: sha256(output_dir / POOL_ARTIFACT),
        MODEL_ARTIFACT: sha256(output_dir / MODEL_ARTIFACT),
        ACQUISITION_ARTIFACT: sha256(output_dir / ACQUISITION_ARTIFACT),
    }


def test_candidate_bundle_rejects_weights_from_another_pool_row(
    tmp_path: Path,
    campaign_bundle: tuple[Path, Path],
    objective: VarianceNormalizedObjective,
) -> None:
    manifest_path, _ = campaign_bundle
    inputs = load_campaign_inputs(manifest_path)
    campaign = build_campaign(inputs, objective, {label: 0.1 for label in objective.labels})
    model = fit_additive_hellinger_model(prepare_hellinger_transfer_data(campaign), torch.device("cpu"))
    pool = np.asarray(
        [
            [[0.7, 0.3], [0.6, 0.4]],
            [[0.3, 0.7], [0.4, 0.6]],
        ]
    )
    acquired = AcquiredCandidate(1, 1.0, np.asarray([0.0, 1.0]))

    with pytest.raises(ValueError, match="selected pool row"):
        write_candidate_bundle(
            campaign_manifest=manifest_path,
            campaign=campaign,
            model_payload=model.model_state(),
            model_metadata={
                "kind": "test",
                "device": "cpu",
                "details": {"fixture": "mismatch"},
            },
            pool=pool,
            acquired=acquired,
            selected_weights=pool[0],
            diagnostics={"fixture": "mismatch"},
            phase_token_fractions=np.asarray([0.75, 0.25]),
            output_dir=tmp_path / "invalid",
            seed=17,
            proposal={"kind": "hand-authored", "parameters": {"fixture": "mismatch"}},
            acquisition_function="test acquisition",
            selection_rule="choose second row",
            dependency_lock=manifest_path,
        )
