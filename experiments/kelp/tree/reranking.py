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

"""Execution-guided reranking for tree diffusion inference.

Following Tree Diffusion (Kapur et al., 2024), which uses REPL feedback
to guide the search: after beam search generates candidate programs, we
execute them against test cases and use the results to rerank.

The paper's approach uses a value network trained on execution outcomes.
We start with a simpler strategy: directly score candidates by the fraction
of tests they pass, combined with the model's log-probability score.
"""

import logging
from dataclasses import dataclass

from experiments.kelp.eval.metrics import execute_python_with_tests
from experiments.kelp.tree.beam_search import BeamCandidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankedCandidate:
    """A candidate program with execution-based scoring."""

    candidate: BeamCandidate
    """The original beam search candidate."""

    tests_passed: int
    """Number of test cases passed."""

    tests_total: int
    """Total number of test cases."""

    test_pass_rate: float
    """Fraction of tests passed (0.0 to 1.0)."""

    combined_score: float
    """Combined score: weighted sum of test_pass_rate and model log-prob."""


def score_candidate(
    candidate: BeamCandidate,
    test_cases: list[str],
    timeout: float = 5.0,
    model_weight: float = 0.1,
    execution_weight: float = 1.0,
) -> RankedCandidate:
    """Score a candidate program by executing it against test cases.

    The combined score weights execution success heavily over model
    log-probability, since a program that passes tests is almost always
    better than one with higher probability but wrong output.

    Args:
        candidate: Beam search candidate.
        test_cases: List of test assertion strings.
        timeout: Timeout per test in seconds.
        model_weight: Weight for normalized model score.
        execution_weight: Weight for test pass rate.

    Returns:
        RankedCandidate with execution results and combined score.
    """
    results = execute_python_with_tests(candidate.source, test_cases, timeout=timeout)
    tests_passed = sum(results)
    tests_total = len(results)
    test_pass_rate = tests_passed / max(tests_total, 1)

    # Normalize model score to roughly [0, 1] range. Log-probs are negative,
    # so we use a sigmoid-like transform. A score of 0 maps to 0.5, very
    # negative scores approach 0.
    import math

    normalized_model_score = 1.0 / (1.0 + math.exp(-candidate.score / 10.0))

    combined_score = execution_weight * test_pass_rate + model_weight * normalized_model_score

    return RankedCandidate(
        candidate=candidate,
        tests_passed=tests_passed,
        tests_total=tests_total,
        test_pass_rate=test_pass_rate,
        combined_score=combined_score,
    )


def rerank_candidates(
    candidates: list[BeamCandidate],
    test_cases: list[str],
    timeout: float = 5.0,
    model_weight: float = 0.1,
    execution_weight: float = 1.0,
) -> list[RankedCandidate]:
    """Rerank beam search candidates using execution feedback.

    Executes each candidate against the provided test cases and sorts
    by combined score (execution success + model probability).

    Args:
        candidates: List of beam search candidates.
        test_cases: Test assertion strings to execute.
        timeout: Timeout per test in seconds.
        model_weight: Weight for model log-prob in combined score.
        execution_weight: Weight for test pass rate in combined score.

    Returns:
        Candidates sorted by combined score (best first).
    """
    ranked = [
        score_candidate(c, test_cases, timeout, model_weight, execution_weight)
        for c in candidates
    ]

    ranked.sort(key=lambda r: (r.combined_score, r.candidate.score), reverse=True)

    if ranked:
        logger.debug(
            f"Reranked {len(ranked)} candidates: "
            f"best={ranked[0].tests_passed}/{ranked[0].tests_total} tests, "
            f"score={ranked[0].combined_score:.4f}"
        )

    return ranked


def filter_passing(
    candidates: list[BeamCandidate],
    test_cases: list[str],
    timeout: float = 5.0,
) -> list[BeamCandidate]:
    """Filter candidates to only those that pass all test cases.

    A stricter alternative to reranking: only return candidates that
    achieve 100% test pass rate.

    Args:
        candidates: List of beam search candidates.
        test_cases: Test assertion strings.
        timeout: Timeout per test in seconds.

    Returns:
        Candidates that pass all tests, sorted by model score (best first).
    """
    passing = []
    for candidate in candidates:
        results = execute_python_with_tests(candidate.source, test_cases, timeout=timeout)
        if all(results):
            passing.append(candidate)

    passing.sort(key=lambda c: c.score, reverse=True)
    return passing
