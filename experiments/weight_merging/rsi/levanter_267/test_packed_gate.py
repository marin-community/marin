# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical-gate artifact validation, independent of an accelerator allocation."""

import math
from types import SimpleNamespace

import pytest

from experiments.weight_merging.rsi.levanter_267.packed_gate import _artifact, compare


@pytest.mark.parametrize("logprob,top", [(math.nan, -0.1), (-0.2, math.inf)])
def test_gate_refuses_nonfinite_selected_or_top_logprobs(logprob, top):
    scores = SimpleNamespace(token_logprobs=[0.0, logprob], top_token_logprobs=[{1: 0.0}, {2: top}])
    with pytest.raises(ValueError, match="Nonfinite"):
        _artifact([1, 2], scores, "fixture", 0)


def test_gate_comparison_reports_known_deltas_and_top1_flips():
    ref = SimpleNamespace(token_logprobs=[0.0, -0.5, -0.8], top_token_logprobs=[{1: 0.0}, {2: -0.1}, {3: -0.2}])
    candidate = SimpleNamespace(token_logprobs=[0.0, -0.51, -0.82], top_token_logprobs=[{1: 0.0}, {4: -0.1}, {3: -0.2}])
    stats = compare(_artifact([1, 2, 3], candidate, "fixture", 0), _artifact([1, 2, 3], ref, "fixture", 0))
    assert stats["overall"]["mean_absolute_delta"] == pytest.approx(0.015)
    assert stats["overall"]["argmax_disagreement_fraction"] == 0.5
    assert stats["overall"]["positions_above_one_nat"] == 0
    assert stats["position_bins"]["1-1024"] == stats["overall"]
