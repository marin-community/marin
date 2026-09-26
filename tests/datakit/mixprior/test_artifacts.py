# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from experiments.datakit.mixprior.artifacts import (
    ACQUISITION_ARTIFACT,
    BUNDLE_MANIFEST_ARTIFACT,
    CANDIDATE_ARTIFACT,
    POOL_ARTIFACT,
    CandidateDecision,
    candidate_id,
    write_bundle_manifest,
    write_candidate_bundle,
)
from experiments.datakit.mixprior.campaign import build_campaign, load_campaign_inputs
from experiments.datakit.mixprior.data import Swarm, read_record, record_sha256, sha256
from experiments.datakit.mixprior.diagnostics import candidate_diagnostics
from experiments.datakit.mixprior.objective import ObjectiveObservations, VarianceNormalizedObjective
from experiments.datakit.mixprior.search import (
    POSTERIOR_MEAN,
    AcquiredCandidate,
    build_candidate_selection,
)
from experiments.datakit.mixprior.surrogate import PredictiveMoments


class ConstantPredictor:
    def predict(self, _swarm: Swarm, weights: np.ndarray) -> PredictiveMoments:
        return PredictiveMoments(mean=np.zeros(len(weights)), latent_variance=np.full(len(weights), 0.1))


class ConstantObjective:
    def observations(self, _swarm: Swarm) -> ObjectiveObservations:
        return ObjectiveObservations(
            np.zeros(len(_swarm.data.weights)),
            np.full(len(_swarm.data.weights), 0.01),
        )


def test_custom_stages_persist_selected_candidate_and_bundle_manifest(
    tmp_path: Path,
    campaign_bundle: tuple[Path, Path],
) -> None:
    manifest_path, _ = campaign_bundle
    output_dir = tmp_path / "generated"
    dependency_lock = tmp_path / "uv.lock"
    dependency_lock.write_text("test dependency lock")
    inputs = load_campaign_inputs(manifest_path)
    custom_objective = ConstantObjective()
    objective_metadata = {"kind": "constant-test-objective"}
    campaign = build_campaign(inputs, custom_objective, objective_metadata)
    model = ConstantPredictor()
    custom_pool = np.asarray(
        [
            [[0.75, 0.25], [0.625, 0.375]],
            [[0.25, 0.75], [0.375, 0.625]],
        ]
    )
    acquisition_values = np.asarray([0.0, 1.0])
    acquired = AcquiredCandidate(1, 1.0, acquisition_values)
    selection = build_candidate_selection(
        campaign,
        model,
        custom_pool,
        acquired,
        acquisition=POSTERIOR_MEAN._replace(
            name="test acquisition",
            selection_rule="choose the second candidate row",
            seed=23,
        ),
    )
    diagnostics = candidate_diagnostics(
        campaign.target,
        selection.weights,
        selection.posterior,
    )
    payload = write_candidate_bundle(
        campaign_manifest=manifest_path,
        campaign=campaign,
        decision=CandidateDecision(
            model_metadata={
                "kind": "test model",
                "device": "cpu",
                "details": {"fixture": "custom stages"},
            },
            pool=custom_pool,
            selection=selection,
            diagnostics=diagnostics,
            pool_seeds=(17, 19),
            proposal={"kind": "hand-authored", "parameters": {"fixture": "second-row"}},
        ),
        output_dir=output_dir,
        dependency_lock=dependency_lock,
    )

    persisted_pool = np.load(output_dir / POOL_ARTIFACT)["weights"]
    persisted_weights = np.asarray(
        [[phase[component] for component in payload["mixture_components"]] for phase in payload["phase_weights"]]
    )
    expected_objective_hash = record_sha256(objective_metadata)

    assert payload["schema_version"] == 5
    assert payload["acquisition"]["selected_pool_index"] == 1
    assert payload["acquisition"]["function"] == "test acquisition"
    assert payload["acquisition"]["selection_rule"] == ("choose the second candidate row")
    assert set(payload["diagnostics"]["posterior"]) == {
        "objective_mean",
        "objective_sd",
        "uncertainty_kind",
        "incumbent_objective_value",
        "probability_of_improvement",
    }
    assert payload["diagnostics"]["posterior"]["uncertainty_kind"] == "latent_function"
    assert np.array_equal(persisted_pool, custom_pool)
    assert np.array_equal(persisted_weights, custom_pool[1])
    assert payload["candidate_id"] == candidate_id(custom_pool[1])
    assert payload["model"]["objective_sha256"] == expected_objective_hash
    assert sha256(output_dir / POOL_ARTIFACT) == payload["acquisition"]["pool_artifact_sha256"]
    assert sha256(output_dir / ACQUISITION_ARTIFACT) == payload["acquisition"]["values_artifact_sha256"]
    assert read_record(output_dir / CANDIDATE_ARTIFACT) == payload

    campaign_uri = "hf://datasets/test/campaign@0000000000000000000000000000000000000000/transfer_campaign.parquet"
    bundle_manifest_path = write_bundle_manifest(output_dir, campaign_uri)
    bundle_manifest = read_record(bundle_manifest_path)
    assert bundle_manifest_path == output_dir / BUNDLE_MANIFEST_ARTIFACT
    assert bundle_manifest["schema_version"] == 4
    assert bundle_manifest["candidate_id"] == payload["candidate_id"]
    assert bundle_manifest["generation"] == {
        "acquisition_function": "test acquisition",
        "selection_rule": "choose the second candidate row",
        "pool_size": 2,
        "pool_seeds": [17, 19],
        "acquisition_seed": 23,
    }
    assert bundle_manifest["artifact_sha256"] == {
        CANDIDATE_ARTIFACT: sha256(output_dir / CANDIDATE_ARTIFACT),
        POOL_ARTIFACT: sha256(output_dir / POOL_ARTIFACT),
        ACQUISITION_ARTIFACT: sha256(output_dir / ACQUISITION_ARTIFACT),
    }


def test_candidate_bundle_rejects_weights_from_another_pool_row(
    tmp_path: Path,
    campaign_bundle: tuple[Path, Path],
    objective: VarianceNormalizedObjective,
) -> None:
    manifest_path, _ = campaign_bundle
    inputs = load_campaign_inputs(manifest_path)
    campaign = build_campaign(inputs, objective, {"kind": "test"})
    pool = np.asarray(
        [
            [[0.75, 0.25], [0.625, 0.375]],
            [[0.25, 0.75], [0.375, 0.625]],
        ]
    )
    acquired = AcquiredCandidate(1, 1.0, np.asarray([0.0, 1.0]))
    selection = build_candidate_selection(
        campaign,
        ConstantPredictor(),
        pool,
        acquired,
        acquisition=POSTERIOR_MEAN._replace(
            name="test acquisition",
            selection_rule="choose second row",
            seed=23,
        ),
    )._replace(weights=pool[0])

    with pytest.raises(ValueError, match="selected pool row"):
        write_candidate_bundle(
            campaign_manifest=manifest_path,
            campaign=campaign,
            decision=CandidateDecision(
                model_metadata={
                    "kind": "test",
                    "device": "cpu",
                    "details": {"fixture": "mismatch"},
                },
                pool=pool,
                selection=selection,
                diagnostics={"fixture": "mismatch"},
                pool_seeds=(17,),
                proposal={"kind": "hand-authored", "parameters": {"fixture": "mismatch"}},
            ),
            output_dir=tmp_path / "invalid",
            dependency_lock=manifest_path,
        )
