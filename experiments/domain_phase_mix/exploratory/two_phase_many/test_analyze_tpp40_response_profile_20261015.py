# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Paired descriptive output retains unresolved and empty query strata."""

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import analyze_tpp40_response_profile_20261015 as analysis


def test_paired_readout_keeps_failed_heads_and_empty_tied_strata():
    spec = {"world": {"name": "toy"}, "noise": "clean", "outer_fold": 0}
    frame = pd.DataFrame({"role": ["asymmetric", "matched_tied"], "truth": [1.0, 1.1]})
    records = []
    for arm, bias in zip(analysis.runner.ARMS, (0.1, 0.2, 0.3), strict=True):
        frame["predicted"] = frame.truth + bias
        records.extend(analysis.role_records(frame, spec, arm, arm != analysis.runner.ARMS[0]))
    pairs = analysis.paired_roles(pd.DataFrame(records))
    assert len(pairs) == 9
    assert pairs.either_unresolved.all()
    actual_ties = pairs.loc[pairs.role == "actual_tied"]
    assert (actual_ties["count"] == 0).all()
    assert actual_ties.mse_difference.isna().all()
    comparison = f"{analysis.runner.ARMS[1]}_minus_{analysis.runner.ARMS[0]}"
    ordinary = pairs.loc[(pairs.comparison == comparison) & (pairs.role == "asymmetric")].iloc[0]
    assert ordinary.mse_difference == pytest.approx(0.03)
    assert ordinary.bias_difference == pytest.approx(0.1)


def test_saved_response_replay_uses_normalized_scale_at_large_common_logit():
    fit = {"selected": {"floor": 0.6, "scale": 2.4, "normalization": 1e16, "amplitudes": [1.0]}}
    result = analysis.replay(np.full((3, 1), 1e16), fit)
    np.testing.assert_allclose(result, 3.0, atol=5e-16, rtol=0)
