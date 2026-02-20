# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from marin.rl.metrics import pass_at_k_estimator


def test_pass_at_k_matches_combinatorial_estimator():
    correct_list = [True] * 2 + [False] * 8
    expected = 1.0 - (math.comb(8, 3) / math.comb(10, 3))
    assert pass_at_k_estimator(correct_list, 3) == pytest.approx(expected)


def test_pass_at_k_edge_cases():
    assert pass_at_k_estimator([False, False], 1) == 0.0
    assert pass_at_k_estimator([True, False], 1) == pytest.approx(0.5)
    assert pass_at_k_estimator([True, False], 2) == 1.0
