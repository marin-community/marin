# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Invented tie menus separate official selection from set-valued ambiguity."""

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import diagnose_tpp40_menu_minimizers_20261015 as d


def test_exact_tie_can_hide_large_worst_and_uniform_regret():
    frame = pd.DataFrame(
        {
            "role": ["matched_tied", "asymmetric", "actual_tied"],
            "row_id": ["a", "a", "b"],
            "source_index": [1, 1, 2],
            "predicted": [1.0, 1.0, 2.0],
            "truth": [1.5, 1.0, 2.0],
        }
    )
    records, members, pairs = d.evaluate_sets(frame)
    for row in records:
        assert row["official_selected_role"] == "asymmetric" and row["official_regret_at_1"] == 0
        assert row["best_true_regret"] == 0 and row["worst_true_regret"] == 0.5 and row["uniform_true_regret"] == 0.25
        assert row["asymmetric_and_matched_tied_pairs"] == 1
    assert len(members) == 4 and all(r["true_gap_tied_minus_asymmetric"] == 0.5 for r in pairs)


def test_roundoff_set_is_separate_from_exact_minimum_and_official_selection():
    eps = np.finfo(float).eps
    frame = pd.DataFrame(
        {
            "role": ["asymmetric", "matched_tied", "actual_tied"],
            "row_id": ["a", "a", "b"],
            "source_index": [1, 1, 2],
            "predicted": [1.0, 1.0 + 32 * eps, 1.0 + 128 * eps],
            "truth": [2.0, 1.0, 3.0],
        }
    )
    records, _, pairs = d.evaluate_sets(frame)
    assert [r["set_size"] for r in records] == [1, 2]
    assert [r["uniform_true_regret"] for r in records] == [1.0, 0.5]
    assert all(r["official_regret_at_1"] == 1 for r in records)
    assert len(pairs) == 1 and pairs[0]["set"] == "roundoff"


def test_query_failure_retained_and_mismatched_pair_rejected():
    failed = d.unavailable_sets({"error": "overflow encountered in exp"})
    assert len(failed) == 2 and all(r["status"] == "original_query_failure" for r in failed)
    frame = pd.DataFrame(
        {
            "role": ["asymmetric", "matched_tied"],
            "row_id": ["a", "b"],
            "source_index": [1, 1],
            "predicted": [1.0, 1.0],
            "truth": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="row identities"):
        d.evaluate_sets(frame)
