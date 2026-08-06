# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.grug.moe.expert_merge import (
    AssignmentMode,
    ExpertCostMatrix,
    ExpertProbeSet,
    ExpertReservoirCollection,
)
from experiments.grug.moe.merge_artifacts import (
    CalibrationArtifactManifest,
    MatchingArtifactManifest,
    read_calibration_manifest,
    read_cost_matrix,
    read_expert_calibration,
    read_expert_probe,
    read_matching_manifest,
    read_matching_metrics,
    write_calibration_artifact,
    write_matching_artifact,
)


def test_calibration_artifact_round_trip_preserves_weighted_train_and_heldout_states(tmp_path) -> None:
    reservoirs = ExpertReservoirCollection(
        num_experts=2,
        state_dim=3,
        capacity_per_expert=8,
        heldout_fraction=0.5,
        seed=7,
    )
    states = np.arange(36, dtype=np.float32).reshape(12, 3) / 7
    selected = np.tile(np.asarray([[0, 1]], dtype=np.int32), (12, 1))
    combine = np.tile(np.asarray([[0.25, 0.75]], dtype=np.float32), (12, 1))
    reservoirs.add_routes(states, selected, combine)
    manifest = CalibrationArtifactManifest(
        source_checkpoint="gs://marin-us-central1/teacher/checkpoints",
        source_commit="abc123",
        layers=(2,),
        num_experts=2,
        state_dim=3,
        capacity_per_expert=8,
        heldout_fraction=0.5,
        calibration_tokens=12,
    )

    write_calibration_artifact(str(tmp_path), {2: reservoirs}, manifest)

    restored_manifest = read_calibration_manifest(str(tmp_path))
    restored = read_expert_calibration(str(tmp_path), 2, 1)
    expected = reservoirs.calibration(1)
    assert restored_manifest == manifest
    np.testing.assert_allclose(restored.train.states, expected.train.states, rtol=5e-3, atol=2e-2)
    np.testing.assert_allclose(restored.train.weights, expected.train.weights)
    np.testing.assert_allclose(restored.heldout.states, expected.heldout.states, rtol=5e-3, atol=2e-2)
    np.testing.assert_allclose(restored.heldout.weights, expected.heldout.weights)


def _probe(offset: float) -> ExpertProbeSet:
    return ExpertProbeSet(
        ordinary_inputs=np.full((2, 3), offset, dtype=np.float32),
        ordinary_weights=np.asarray([0.25, 0.75], dtype=np.float32),
        centers=np.full((1, 3), offset + 1, dtype=np.float32),
        spectral_pairs=np.full((2, 2, 3), offset + 2, dtype=np.float32),
        input_directions=np.full((3, 1), offset + 3, dtype=np.float32),
        sensitivity_eigenvalues=np.asarray([offset + 4], dtype=np.float32),
    )


def test_matching_artifact_round_trip_preserves_ablation_assignments_and_numeric_evidence(tmp_path) -> None:
    costs = ExpertCostMatrix(
        native=np.asarray([[3.0, 1.0], [1.0, 3.0]]),
        tangent=np.asarray([[0.0, 2.0], [2.0, 0.0]]),
        total=np.asarray([[3.0, 2.0], [2.0, 3.0]]),
    )
    manifest = MatchingArtifactManifest(
        calibration_path="gs://marin-us-central1/calibration",
        representative_layer=2,
        source_layer=3,
        num_experts=2,
        eta=0.5,
        assignments={
            AssignmentMode.IDENTITY: (0, 1),
            AssignmentMode.NATIVE: (1, 0),
            AssignmentMode.SPECTRAL: (1, 0),
        },
    )

    write_matching_artifact(str(tmp_path), (_probe(0), _probe(10)), costs, manifest)

    assert read_matching_manifest(str(tmp_path)) == manifest
    restored_costs = read_cost_matrix(str(tmp_path))
    np.testing.assert_array_equal(restored_costs.native, costs.native)
    np.testing.assert_array_equal(restored_costs.tangent, costs.tangent)
    np.testing.assert_array_equal(restored_costs.total, costs.total)
    metrics = read_matching_metrics(str(tmp_path))
    assert metrics["merge/native_cost_mean"] == np.mean(costs.native)
    assert "spectral" in metrics["merge/assignment_cost_histogram"]
    restored_probe = read_expert_probe(str(tmp_path), 1)
    np.testing.assert_array_equal(restored_probe.spectral_pairs, _probe(10).spectral_pairs)
    np.testing.assert_array_equal(restored_probe.sensitivity_eigenvalues, _probe(10).sensitivity_eigenvalues)
