# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Framework-independent next-token parity against frozen scores."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TokenScore:
    logprob: float
    token_id: int


@dataclass(frozen=True)
class NextTokenObservation:
    """One backend's exact next-token response, independent of its serving API."""

    case_id: str
    backend_rank: int
    greedy_token_id: int
    top_logprobs: tuple[TokenScore, ...]

    def parity_against(self, expected_top_logprobs: tuple[TokenScore, ...]) -> "NextTokenParity":
        return _parity_from_token_scores(
            self.case_id,
            expected_top_logprobs,
            self.greedy_token_id,
            {score.token_id: score.logprob for score in self.top_logprobs},
            backend_rank=self.backend_rank,
        )


@dataclass(frozen=True)
class CrossRankDiagnostic:
    """A non-gating summary of variation between backend ranks."""

    case_id: str
    greedy_token_ids: tuple[tuple[int, int], ...]
    shared_top_token_count: int
    max_probability_spread: float


@dataclass(frozen=True)
class NextTokenParity:
    """One backend observation against one frozen next-token distribution."""

    case_id: str
    backend_rank: int
    greedy_token_id: int
    golden_top_token_ids: tuple[int, ...]
    golden_probability_gap_to_greedy: float
    max_probability_error: float
    top_probability_l1_error: float

    def assert_matches(self, *, max_probability_error: float) -> None:
        """Require golden-token coverage and a probability-supported winner."""
        assert self.greedy_token_id in self.golden_top_token_ids, self
        assert self.golden_probability_gap_to_greedy <= 2 * self.max_probability_error, self
        assert self.max_probability_error <= max_probability_error, self


def observation_from_logprob_map(
    case_id: str,
    greedy_token_id: int,
    actual_logprobs: dict[int, float],
    *,
    backend_rank: int,
) -> NextTokenObservation:
    """Normalize a serving response so exact repeats compare as plain values."""
    _assert_valid_greedy_response(case_id, greedy_token_id, actual_logprobs, backend_rank=backend_rank)
    return NextTokenObservation(
        case_id=case_id,
        backend_rank=backend_rank,
        greedy_token_id=greedy_token_id,
        top_logprobs=tuple(
            TokenScore(token_id=token_id, logprob=logprob) for token_id, logprob in sorted(actual_logprobs.items())
        ),
    )


def _assert_valid_greedy_response(
    case_id: str,
    greedy_token_id: int,
    actual_logprobs: dict[int, float],
    *,
    backend_rank: int,
) -> None:
    assert greedy_token_id in actual_logprobs, f"{case_id} rank {backend_rank}: greedy token missing from logprobs"
    maximum_actual_logprob = max(actual_logprobs.values())
    assert (
        actual_logprobs[greedy_token_id] == maximum_actual_logprob
    ), f"{case_id} rank {backend_rank}: greedy token does not have maximum returned logprob"


def assert_same_rank_repeatability(
    first: Sequence[NextTokenObservation],
    second: Sequence[NextTokenObservation],
) -> None:
    """Require exact repeatability per rank without requiring ranks to agree."""
    first_by_rank = _observations_by_rank(first)
    second_by_rank = _observations_by_rank(second)
    assert (
        first_by_rank.keys() == second_by_rank.keys()
    ), f"repeat waves covered different ranks: {sorted(first_by_rank)} != {sorted(second_by_rank)}"
    for rank, first_observation in first_by_rank.items():
        second_observation = second_by_rank[rank]
        assert first_observation == second_observation, (
            f"{first_observation.case_id} rank {rank} was not exactly repeatable: "
            f"first={first_observation!r}, second={second_observation!r}"
        )


def cross_rank_diagnostic(observations: Sequence[NextTokenObservation]) -> CrossRankDiagnostic:
    """Summarize cross-rank spread without imposing a correctness gate."""
    observations_by_rank = _observations_by_rank(observations)
    case_ids = {observation.case_id for observation in observations_by_rank.values()}
    assert len(case_ids) == 1, f"cross-rank diagnostic requires one case, got {sorted(case_ids)}"
    logprobs_by_rank = [
        {score.token_id: score.logprob for score in observation.top_logprobs}
        for observation in observations_by_rank.values()
    ]
    shared_tokens = set.intersection(*(set(scores) for scores in logprobs_by_rank))
    maximum_spread = max(
        (
            max(math.exp(scores[token_id]) for scores in logprobs_by_rank)
            - min(math.exp(scores[token_id]) for scores in logprobs_by_rank)
            for token_id in shared_tokens
        ),
        default=float("nan"),
    )
    return CrossRankDiagnostic(
        case_id=case_ids.pop(),
        greedy_token_ids=tuple(
            (rank, observation.greedy_token_id) for rank, observation in sorted(observations_by_rank.items())
        ),
        shared_top_token_count=len(shared_tokens),
        max_probability_spread=maximum_spread,
    )


def parity_from_logprob_row(
    case_id: str,
    expected_top_logprobs: tuple[TokenScore, ...],
    logprobs_row: np.ndarray,
    *,
    backend_rank: int,
) -> NextTokenParity:
    """Score a full Levanter ``[vocab]`` log-softmax row."""
    greedy_token_id = int(logprobs_row.argmax())
    actual_logprobs = {score.token_id: float(logprobs_row[score.token_id]) for score in expected_top_logprobs}
    return _parity_from_token_scores(
        case_id,
        expected_top_logprobs,
        greedy_token_id,
        actual_logprobs,
        backend_rank=backend_rank,
    )


def _observations_by_rank(
    observations: Sequence[NextTokenObservation],
) -> dict[int, NextTokenObservation]:
    assert observations, "at least one backend observation is required"
    observations_by_rank = {observation.backend_rank: observation for observation in observations}
    assert len(observations_by_rank) == len(observations), "backend ranks must be unique within one wave"
    return observations_by_rank


def _parity_from_token_scores(
    case_id: str,
    expected_top_logprobs: tuple[TokenScore, ...],
    greedy_token_id: int,
    actual_logprobs: dict[int, float],
    *,
    backend_rank: int,
) -> NextTokenParity:
    golden_logprobs = {entry.token_id: entry.logprob for entry in expected_top_logprobs}
    missing = golden_logprobs.keys() - actual_logprobs.keys()
    assert not missing, f"{case_id} rank {backend_rank}: golden tokens missing from backend logprobs: {sorted(missing)}"
    probability_errors = tuple(
        abs(math.exp(actual_logprobs[token_id]) - math.exp(golden_logprob))
        for token_id, golden_logprob in golden_logprobs.items()
    )
    maximum_golden_logprob = max(golden_logprobs.values())
    selected_golden_logprob = golden_logprobs.get(greedy_token_id, -math.inf)
    return NextTokenParity(
        case_id=case_id,
        backend_rank=backend_rank,
        greedy_token_id=greedy_token_id,
        golden_top_token_ids=tuple(golden_logprobs),
        golden_probability_gap_to_greedy=math.exp(maximum_golden_logprob) - math.exp(selected_golden_logprob),
        max_probability_error=max(probability_errors),
        top_probability_l1_error=sum(probability_errors),
    )
