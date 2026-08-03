# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from tests.cluster.vllm.backend_parity import (
    NextTokenObservation,
    NextTokenParity,
    TokenScore,
    assert_same_rank_repeatability,
    cross_rank_diagnostic,
)


def _parity(*, greedy_token_id: int, gap: float, error: float) -> NextTokenParity:
    return NextTokenParity(
        case_id="case",
        backend_rank=0,
        greedy_token_id=greedy_token_id,
        golden_top_token_ids=(2, 3),
        golden_probability_gap_to_greedy=gap,
        max_probability_error=error,
        top_probability_l1_error=error,
    )


def _observation(rank: int, *, greedy_token_id: int = 2, logprob_offset: float = 0.0) -> NextTokenObservation:
    return NextTokenObservation(
        case_id="case",
        backend_rank=rank,
        greedy_token_id=greedy_token_id,
        top_logprobs=(
            TokenScore(token_id=2, logprob=-0.1 + logprob_offset),
            TokenScore(token_id=3, logprob=-2.0),
        ),
    )


@pytest.mark.parametrize(
    "parity",
    [
        _parity(greedy_token_id=2, gap=0.0, error=0.0),
        _parity(greedy_token_id=3, gap=0.01, error=0.005),
    ],
)
def test_backend_distribution_contract_accepts_exact_and_error_explained_winners(parity: NextTokenParity) -> None:
    parity.assert_matches(max_probability_error=0.075)


@pytest.mark.parametrize(
    "parity",
    [
        _parity(greedy_token_id=3, gap=0.011, error=0.005),
        _parity(greedy_token_id=9, gap=0.01, error=0.01),
        _parity(greedy_token_id=2, gap=0.0, error=0.076),
    ],
)
def test_backend_distribution_contract_rejects_unexplained_outside_or_over_bound_winners(
    parity: NextTokenParity,
) -> None:
    with pytest.raises(AssertionError):
        parity.assert_matches(max_probability_error=0.075)


def test_same_rank_repeatability_compares_every_rank_without_cross_rank_equality() -> None:
    first = (_observation(0, greedy_token_id=2), _observation(1, greedy_token_id=3))
    second = tuple(reversed(first))

    assert_same_rank_repeatability(first, second)

    changed_rank_one = (_observation(0, greedy_token_id=2), _observation(1, greedy_token_id=2))
    with pytest.raises(AssertionError, match="rank 1 was not exactly repeatable"):
        assert_same_rank_repeatability(first, changed_rank_one)


def test_cross_rank_spread_is_reported_without_becoming_a_gate() -> None:
    observations = (_observation(0), _observation(1, greedy_token_id=3, logprob_offset=-0.25))

    diagnostic = cross_rank_diagnostic(observations)

    assert diagnostic.case_id == "case"
    assert diagnostic.greedy_token_ids == ((0, 2), (1, 3))
    assert diagnostic.shared_top_token_count == 2
    assert diagnostic.max_probability_spread == pytest.approx(math.exp(-0.1) - math.exp(-0.35))
