# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""Tests for execution-guided reranking."""

from experiments.kelp.tree.beam_search import BeamCandidate
from experiments.kelp.tree.reranking import (
    filter_passing,
    rerank_candidates,
    score_candidate,
)


def _make_candidate(source: str, score: float = 0.0) -> BeamCandidate:
    return BeamCandidate(source=source, score=score, depth=0, edits=())


# --- score_candidate ---


def test_score_candidate_all_pass():
    candidate = _make_candidate("def add(a, b):\n    return a + b\n")
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    result = score_candidate(candidate, tests)
    assert result.tests_passed == 2
    assert result.tests_total == 2
    assert result.test_pass_rate == 1.0
    assert result.combined_score > 0


def test_score_candidate_none_pass():
    candidate = _make_candidate("def add(a, b):\n    return a - b\n")
    tests = ["assert add(1, 2) == 3", "assert add(3, 4) == 7"]
    result = score_candidate(candidate, tests)
    assert result.tests_passed == 0
    assert result.test_pass_rate == 0.0


def test_score_candidate_partial_pass():
    candidate = _make_candidate("def f(x):\n    return x\n")
    tests = ["assert f(1) == 1", "assert f(2) == 3"]
    result = score_candidate(candidate, tests)
    assert result.tests_passed == 1
    assert result.tests_total == 2
    assert 0.0 < result.test_pass_rate < 1.0


def test_score_candidate_syntax_error():
    candidate = _make_candidate("def f(:\n")
    tests = ["assert f(1) == 1"]
    result = score_candidate(candidate, tests)
    assert result.tests_passed == 0


def test_score_candidate_empty_tests():
    candidate = _make_candidate("x = 1\n")
    result = score_candidate(candidate, [])
    assert result.tests_passed == 0
    assert result.tests_total == 0
    assert result.test_pass_rate == 0.0


def test_score_candidate_model_weight():
    """Higher model score should improve combined_score when execution is equal."""
    c_high = _make_candidate("def f(x):\n    return x\n", score=0.0)
    c_low = _make_candidate("def f(x):\n    return x\n", score=-100.0)
    tests = ["assert f(1) == 1"]
    r_high = score_candidate(c_high, tests, model_weight=1.0)
    r_low = score_candidate(c_low, tests, model_weight=1.0)
    # Same execution score but different model scores.
    assert r_high.combined_score > r_low.combined_score


# --- rerank_candidates ---


def test_rerank_puts_passing_first():
    correct = _make_candidate("def add(a, b):\n    return a + b\n", score=-5.0)
    wrong = _make_candidate("def add(a, b):\n    return a - b\n", score=0.0)
    tests = ["assert add(1, 2) == 3"]

    ranked = rerank_candidates([wrong, correct], tests)
    assert len(ranked) == 2
    # Correct program should be first despite lower model score.
    assert ranked[0].candidate.source == correct.source
    assert ranked[0].tests_passed == 1


def test_rerank_tiebreak_by_model_score():
    """When execution scores are equal, model score breaks the tie."""
    c1 = _make_candidate("def f(x):\n    return x\n", score=-1.0)
    c2 = _make_candidate("def g(x):\n    return x\n", score=-2.0)
    tests = ["assert f(1) == 1"]  # Only c1 passes.
    # But with no tests that both pass, let's use a test both fail.
    tests_both_fail = ["assert False"]
    ranked = rerank_candidates([c2, c1], tests_both_fail)
    # Both fail all tests, so tiebreak by model score (c1 is higher).
    assert ranked[0].candidate.source == c1.source


def test_rerank_empty_candidates():
    ranked = rerank_candidates([], ["assert True"])
    assert ranked == []


def test_rerank_preserves_all_candidates():
    candidates = [_make_candidate(f"x = {i}\n", score=-float(i)) for i in range(5)]
    ranked = rerank_candidates(candidates, ["assert True"])
    assert len(ranked) == 5


# --- filter_passing ---


def test_filter_passing_returns_correct():
    correct = _make_candidate("def add(a, b):\n    return a + b\n")
    wrong = _make_candidate("def add(a, b):\n    return 0\n")
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]

    passing = filter_passing([wrong, correct], tests)
    assert len(passing) == 1
    assert passing[0].source == correct.source


def test_filter_passing_none_pass():
    candidates = [
        _make_candidate("def f(x):\n    return 0\n"),
        _make_candidate("def f(x):\n    return -1\n"),
    ]
    tests = ["assert f(1) == 1"]
    passing = filter_passing(candidates, tests)
    assert passing == []


def test_filter_passing_sorted_by_model_score():
    c1 = _make_candidate("def f(x):\n    return x\n", score=-1.0)
    c2 = _make_candidate("def f(x):\n    return x\n", score=-0.5)
    tests = ["assert f(1) == 1"]
    passing = filter_passing([c1, c2], tests)
    assert len(passing) == 2
    # Higher model score first.
    assert passing[0].score >= passing[1].score


def test_filter_passing_empty_input():
    passing = filter_passing([], ["assert True"])
    assert passing == []


def test_filter_passing_partial_fail_excluded():
    """A candidate that passes only some tests should be excluded."""
    candidate = _make_candidate("def f(x):\n    return 1\n")
    tests = ["assert f(1) == 1", "assert f(2) == 2"]
    passing = filter_passing([candidate], tests)
    assert passing == []
