# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioural checks for the round-5 OLMix-gap scripts: family map, credit rules, calibration, and the DP."""

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_round3_heldout_selection_20260903 as selection,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_round3_proposals_20260903 as proposals,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_round5_candidates_20260904 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_round5_olmix_gap_20260904 as gap,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_round5_remedies_20260904 as remedies,
)

TABLE9_SHORT_NAMES = (
    "minerva_math_algebra codex_humaneval mbpp mt_mbpp_bash arc_easy arc_challenge mmlu_stem csqa hellaswag "
    "winogrande socialiqa piqa coqa drop jeopardy naturalqs squad sciq basic_skills_arithmetic lambada medmcqa"
).split()


def test_family_map_assigns_representative_table9_names_and_rejects_unknown_ones():
    families = {name: gap.family(f"olmo_base_eval/easy_bpb/{name}/bpb") for name in TABLE9_SHORT_NAMES}
    assert families["minerva_math_algebra"] == "math"
    assert families["mt_mbpp_bash"] == "code" and families["codex_humaneval"] == "code"
    assert families["arc_challenge"] == "arc" and families["mmlu_stem"] == "mmlu"
    assert families["csqa"] == "commonsense" and families["drop"] == "qa_reading"
    assert families["basic_skills_arithmetic"] == "basic_skills"
    with pytest.raises(ValueError, match="unmapped"):
        gap.family("olmo_base_eval/easy_bpb/not_a_task/bpb")


def test_olmix_weights_rejects_mass_outside_the_panel_buckets(tmp_path):
    path = tmp_path / "weights.csv"
    pd.Series({"a": 0.6, "b": 0.3, "paloma/x": 0.1}, name="weight").rename_axis("bucket").to_csv(path)
    with pytest.raises(ValueError, match="outside the panel buckets"):
        gap.olmix_weights(path, ("a", "b"))
    pd.Series({"a": 0.6, "b": 0.4, "paloma/x": 0.0}, name="weight").rename_axis("bucket").to_csv(path)
    np.testing.assert_allclose(gap.olmix_weights(path, ("b", "a")), [0.4, 0.6])


def _two_bucket_curves() -> remedies.Curves:
    shape = {"rate": 1.0, "power": 1.0, "threshold": 1.0}
    first = proposals.ComponentCurve(0.5, 1.0, np.array([0.2, 0.1]), np.array([0.0, 0.05]), shape)
    second = proposals.ComponentCurve(0.5, 2.0, np.array([0.0, 0.3]), np.array([0.1, 0.0]), shape)
    return remedies.Curves((first, second))


def test_component_matrix_is_additive_and_share_floor_drops_small_buckets():
    curves = _two_bucket_curves()
    exposures = np.array([[2.0, 3.0]])
    matrix = curves.component_matrix(exposures)
    expected = []
    for curve in curves.curves:
        benefit = models.weibull_response(exposures[0], 1.0, 1.0)
        harm = models.softplus_harm(exposures[0], 1.0)
        expected.append(curve.intercept + float((-curve.benefit * benefit + curve.harm * harm).sum()))
    np.testing.assert_allclose(matrix[0], expected)
    floored = curves.component_matrix(exposures, share_floor=0.05, weights=np.array([[0.01, 0.5]]))
    only_second = curves.component_matrix(np.array([[0.0, 3.0]]))
    np.testing.assert_allclose(floored, only_second)


def test_kernel_smooth_returns_the_training_value_at_zero_distance_and_averages_far_away():
    train = np.array([[1.0, 0.0], [0.0, 1.0]])
    values = np.array([1.0, 3.0])
    at_train = remedies.kernel_smooth(train, values, train, bandwidth=0.01)
    np.testing.assert_allclose(at_train, values)
    midpoint = remedies.kernel_smooth(train, values, np.array([[0.5, 0.5]]), bandwidth=10.0)
    np.testing.assert_allclose(midpoint, [2.0])


def test_ridge_fit_recovers_a_linear_relation_with_a_small_penalty():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = 0.5 + x @ np.array([1.0, -2.0, 0.0])
    fit = remedies.ridge_fit(x, y, ridge=1e-6)
    np.testing.assert_allclose(remedies.ridge_predict(fit, x), y, atol=1e-6)


def test_dp_solution_respects_bounds_and_sums_to_the_block_count():
    curves = _two_bucket_curves()
    inventory = np.array([4.0, 4.0])
    lower = np.array([100, 0])
    upper = np.array([candidates.BLOCKS, 900])
    weights = candidates.solve(curves, inventory, lower, upper, "plain", np.array([10.0, 10.0]))
    counts = np.round(weights * candidates.BLOCKS).astype(int)
    assert counts.sum() == candidates.BLOCKS
    assert counts[0] >= 100 and counts[1] <= 900
    clamped = candidates.solve(curves, inventory, lower, upper, "clamp_panel_max", np.array([0.5, 0.5]))
    assert np.isclose(clamped.sum(), 1.0)
    with pytest.raises(ValueError, match="infeasible"):
        candidates.solve(curves, inventory, np.array([1500, 1500]), upper, "plain", np.array([10.0, 10.0]))


def test_drop_coordinates_keeps_order_and_trims_every_array():
    ids = np.array(["a", "b", "c"])
    bank = selection.Bank(
        "table9", ids, np.array([1.0, 2.0, 3.0]), np.array(["s", "s", "t"]), np.array([1, 1, 2]), np.zeros(3), 0.1
    )
    weights = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    features = models.Features(
        exposures=weights * 2.0,
        weights=weights,
        inventory=np.array([2.0, 2.0]),
        early_fraction=np.zeros(3),
        families=models.no_families(2),
        label="bank",
    )
    trimmed, trimmed_features, keep = remedies.drop_coordinates(bank, features, {"b"})
    assert list(trimmed.coordinate_id) == ["a", "c"] and keep.tolist() == [True, False, True]
    np.testing.assert_allclose(trimmed.measured, [1.0, 3.0])
    np.testing.assert_allclose(trimmed_features.weights, weights[[0, 2]])
    assert trimmed.tolerance == bank.tolerance


def test_held_out_mask_covers_secondary_memberships_and_pools_small_sources():
    sources = np.array(
        [
            "archive::big;archive::small_a",
            "archive::big",
            "archive::big",
            "archive::big",
            "archive::big",
            "archive::small_a",
            "archive::small_b;archive::big",
            "conditional_epoch_dose_response",
        ]
    )
    primary, memberships = remedies.source_groups(sources)
    assert list(primary[:5]) == ["archive::big"] * 5
    assert primary[5] == "other_archive_sources" and primary[6] == "other_archive_sources"
    held_big = remedies.held_out_mask("archive::big", primary, memberships)
    assert held_big.tolist() == [True, True, True, True, True, False, True, False]
    held_other = remedies.held_out_mask("other_archive_sources", primary, memberships)
    assert held_other.tolist() == [True, False, False, False, False, True, True, False]
    values = np.arange(len(sources), dtype=float)
    out = remedies.loso_apply(primary, memberships, lambda train, test: np.full(test.sum(), float(train.sum())))
    # the small-source group is predicted by a fit that excludes every coordinate touching a small source
    assert out[5] == 5.0 and out[6] == 5.0
    assert np.isfinite(out).all() and len(values) == len(out)
