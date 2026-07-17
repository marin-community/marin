# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest

from tests.cluster.vllm.backend_parity import parity_from_logprob_map, parity_from_logprob_row
from tests.cluster.vllm.representative_eval import TokenScore
from tests.cluster.vllm.test_snowball_backend_parity import _vllm_job


def test_backend_parity_distinguishes_exact_winner_and_distribution_contracts() -> None:
    exact_tie = (
        TokenScore(logprob=-1.0, token_id=2),
        TokenScore(logprob=-1.0, token_id=3),
        TokenScore(logprob=-2.0, token_id=4),
    )
    tied = parity_from_logprob_map(
        "exact-tie",
        exact_tie,
        greedy_token_id=3,
        actual_logprobs={score.token_id: score.logprob for score in exact_tie},
        backend_rank=7,
    )
    tied.assert_matches(max_probability_error=0.0)
    tied.assert_distribution_matches(max_probability_error=0.0)
    assert tied.golden_greedy_token_ids == (2, 3)
    assert tied.golden_top1_probability_margin == 0.0
    assert tied.golden_probability_gap_to_greedy == 0.0

    unique = (
        TokenScore(logprob=-1.0, token_id=2),
        TokenScore(logprob=-1.01, token_id=3),
        TokenScore(logprob=-2.0, token_id=4),
    )
    near_tie = parity_from_logprob_map(
        "near-tie",
        unique,
        greedy_token_id=3,
        actual_logprobs={2: -1.01, 3: -1.0, 4: -2.0},
        backend_rank=0,
    )
    assert near_tie.golden_top1_probability_margin == pytest.approx(math.exp(-1.0) - math.exp(-1.01))
    with pytest.raises(AssertionError):
        near_tie.assert_matches(max_probability_error=1.0)
    near_tie.assert_distribution_matches(max_probability_error=0.01)


def test_backend_parity_distribution_contract_rejects_greedy_outside_golden_top_scores() -> None:
    expected = (
        TokenScore(logprob=-1.0, token_id=2),
        TokenScore(logprob=-2.0, token_id=3),
    )
    parity = parity_from_logprob_map(
        "outside-top-scores",
        expected,
        greedy_token_id=9,
        actual_logprobs={2: -1.0, 3: -2.0, 9: -0.5},
        backend_rank=0,
    )

    with pytest.raises(AssertionError):
        parity.assert_distribution_matches(max_probability_error=1.0)


def test_backend_parity_scores_map_and_full_row_with_the_same_probability_metric() -> None:
    expected = (
        TokenScore(logprob=-1.0, token_id=1),
        TokenScore(logprob=-2.0, token_id=4),
    )
    actual = {1: -1.1, 4: -1.9}
    row = np.full(6, -100.0)
    for token_id, logprob in actual.items():
        row[token_id] = logprob

    from_map = parity_from_logprob_map("case", expected, 1, actual, backend_rank=0)
    from_row = parity_from_logprob_row("case", expected, row, backend_rank=0)

    assert from_row.greedy_token_id == from_map.greedy_token_id
    assert from_row.max_probability_error == pytest.approx(from_map.max_probability_error)
    assert from_row.top_probability_l1_error == pytest.approx(from_map.top_probability_l1_error)


def test_backend_parity_rejects_missing_golden_tokens() -> None:
    expected = (TokenScore(logprob=-1.0, token_id=1), TokenScore(logprob=-2.0, token_id=4))

    with pytest.raises(AssertionError):
        parity_from_logprob_map("case", expected, 1, {1: -1.0}, backend_rank=3)


def test_vllm_parity_job_enables_batch_invariant_execution() -> None:
    request = _vllm_job((), "FLASH_ATTN")

    assert request.environment.env_vars["VLLM_BATCH_INVARIANT"] == "1"
    assert request.environment.env_vars["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
