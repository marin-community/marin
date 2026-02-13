# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
