# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from experiments.post_training.pivotrl.mixed_rlvr_analysis import (
    exact_mcnemar_p,
    harbor_scores,
    paired_delta,
    sample_scores,
    sign_flip_p,
)


def test_exact_mcnemar_matches_the_binomial_tail():
    # 85 arm-only against 41 reference-only successes (B4 vs REF on SWE-bench Verified).
    assert exact_mcnemar_p(85, 41) == pytest.approx(0.00011021, abs=1e-7)
    assert exact_mcnemar_p(41, 85) == exact_mcnemar_p(85, 41)
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(3, 3) == 1.0


def test_sign_flip_is_extreme_for_consistent_differences_and_null_for_balanced_ones():
    rng = np.random.default_rng(0)
    assert sign_flip_p([0.3] * 20, rng, permutations=5000) < 0.002
    assert sign_flip_p([0.5, -0.5, 0.5, -0.5], rng, permutations=5000) > 0.5
    assert sign_flip_p([0.0, 0.0], rng) == 1.0


def test_swebench_is_strict_on_the_first_trial_and_tblite_needs_two_of_three():
    swe = harbor_scores(
        {"run": {"solved": [1.0], "timed-out": [None], "retried": [0.0, 1.0], "empty": []}}, "swebench-verified"
    )
    assert swe == {"run": {"solved": 1.0, "timed-out": 0.0, "retried": 0.0}}
    tblite = harbor_scores(
        {"run": {"two": [1.0, 1.0, 0.0], "one": [1.0, 0.0, 0.0], "gap": [1.0, None, 1.0]}}, "ot-tblite-recovery"
    )
    assert tblite == {"run": {"two": 1.0, "one": 0.0, "gap": 1.0}}


def test_sample_scores_average_repeated_trials_and_skip_unscored_ones():
    scores = sample_scores({"run": {"q1\t0": True, "q1\t1": False, "q1\t2": None, "q2\t": True}})
    assert scores == {"run": {"q1": 0.5, "q2": 1.0}}


def test_paired_delta_uses_shared_items_only():
    rng = np.random.default_rng(0)
    ref = {"a": 0.0, "b": 1.0, "c": 1.0, "d": 0.0}
    arm = {"a": 1.0, "b": 1.0, "c": 0.0, "e": 1.0}
    mean, low, high, n = paired_delta(arm, ref, rng)
    assert n == 3
    assert mean == pytest.approx(0.0)
    assert low <= mean <= high
