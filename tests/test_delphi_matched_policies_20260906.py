# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import audit_delphi_matched_policies_20260906 as audit


def test_matched_caps_and_kl_change_order_without_changing_central_prediction():
    inventory = np.array([2.0, 8.0])
    natural = audit.natural_proportions(inventory)
    np.testing.assert_allclose(natural, [0.8, 0.2], atol=1e-15, rtol=0)
    weights = np.array([[0.0, 1.0], [0.5, 0.5], [0.8, 0.2]])
    central = np.array([0.9, 1.0, 1.001])
    indices, scores, kl = audit.policy_order(central, weights, weights * inventory, natural, 4, 0)
    np.testing.assert_array_equal(indices, [1, 2])
    np.testing.assert_array_equal(scores, [1.0, 1.001])
    np.testing.assert_allclose(kl, [np.log(1.25), 0], atol=1e-15, rtol=0)
    indices, scores, _ = audit.policy_order(central, weights, weights * inventory, natural, 4, 0.02)
    np.testing.assert_array_equal(indices, [2, 1])
    np.testing.assert_allclose(scores, [1.001, 1 + 0.02 * np.log(1.25)], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(central, [0.9, 1.0, 1.001])


def test_directional_errors_ignore_common_offsets_and_filter_unseparated_pairs():
    measured = np.array([1.0, 1.03, 1.035])
    predicted = np.array([2.0, 1.98, 2.01])
    result = audit.pairwise_changes(measured, predicted, 0.01)
    assert result["pairs"] == 2
    assert result["direction_accuracy"] == 0.5
    np.testing.assert_allclose(result["delta_rmse"], np.sqrt((0.05**2 + 0.025**2) / 2), atol=1e-15, rtol=0)
    shifted = audit.pairwise_changes(measured, predicted + 7, 0.01)
    np.testing.assert_allclose(shifted["delta_rmse"], result["delta_rmse"], atol=1e-15, rtol=0)
    assert shifted["direction_accuracy"] == result["direction_accuracy"]


def test_policy_rankings_can_be_completed_without_any_bank_label_file(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    inputs = source / "inputs"
    inputs.mkdir(parents=True)
    output.mkdir()
    weights = np.array([[0.0, 1.0], [0.5, 0.5], [0.8, 0.2]])
    inventory = np.array([2.0, 8.0])
    np.savez(
        inputs / "panel.npz",
        weights=weights,
        inventory=inventory,
        exposures=weights * inventory,
        buckets=["a", "b"],
    )
    predictions = []
    for target in audit.benchmark.TARGETS:
        np.savez(
            inputs / f"{target}_bank_features.npz",
            coordinate_id=["a", "b", "c"],
            weights=weights,
            exposures=weights * inventory,
        )
        for method in audit.benchmark.BASELINES:
            predictions.extend(
                {
                    "target": target,
                    "method": method,
                    "population": "external_development",
                    "row_id": coordinate,
                    "prediction": value,
                }
                for coordinate, value in zip(["a", "b", "c"], [0.9, 1, 1.001], strict=True)
            )
    pd.DataFrame(predictions).to_csv(source / "predictions.csv", index=False)
    ranks = audit.rank_policies(source, output)
    winners = ranks[ranks.policy_rank.eq(1)]
    assert set(winners[winners.epoch_cap.eq(4) & winners.kl_penalty.eq(0)].coordinate_id) == {"b"}
    assert set(winners[winners.epoch_cap.eq(4) & winners.kl_penalty.eq(0.02)].coordinate_id) == {"c"}
    assert set(winners[winners.epoch_cap.eq(8) & winners.kl_penalty.eq(0)].coordinate_id) == {"a"}
    assert (output / "ranking_hash.json").exists()
