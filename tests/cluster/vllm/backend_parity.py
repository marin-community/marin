# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Framework-independent next-token parity against frozen Grug scores.

All comparisons cover the golden top-25 and gate on maximum single-token
probability error. The legacy short-prompt comparison requires an exact golden
winner. Representative prompts additionally allow a golden top-25 token when
the measured probability error is large enough to explain the winner change.
"""

import math
from dataclasses import dataclass

import numpy as np

from tests.cluster.vllm.representative_eval import TokenScore

# Preserve the existing short-prompt Snowball contract independently of the
# representative-prompt calibration.
LEVANTER_MAX_PROBABILITY_ERROR = 0.008

# Across the 64 representative prompts, the observed maxima were 0.06828 for
# Levanter and 0.06447 for vLLM. This shared bound retains measurable headroom.
REPRESENTATIVE_MAX_PROBABILITY_ERROR = 0.075


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
        assert self.golden_probability_gap_to_greedy == 0.0, self
        assert self.max_probability_error <= max_probability_error, self

    def assert_distribution_matches(self, *, max_probability_error: float) -> None:
        """Require top-25 coverage and a probability-supported greedy token."""
        assert self.greedy_token_id in self.golden_top_token_ids, self
        assert self.golden_probability_gap_to_greedy <= 2 * self.max_probability_error, self
        assert self.max_probability_error <= max_probability_error, self


def parity_from_logprob_map(
    case_id: str,
    expected_top_logprobs: tuple[TokenScore, ...],
    greedy_token_id: int,
    actual_logprobs: dict[int, float],
    *,
    backend_rank: int,
) -> NextTokenParity:
    """Score a vLLM-style ``{token_id: logprob}`` response."""
    golden_logprobs = _golden_logprob_map(case_id, expected_top_logprobs)
    assert greedy_token_id in actual_logprobs, f"{case_id} rank {backend_rank}: greedy token missing from logprobs"
    maximum_actual_logprob = max(actual_logprobs.values())
    assert (
        actual_logprobs[greedy_token_id] == maximum_actual_logprob
    ), f"{case_id} rank {backend_rank}: greedy token does not have maximum returned logprob"
    return _parity_from_golden_logprobs(
        case_id,
        golden_logprobs,
        greedy_token_id,
        actual_logprobs,
        backend_rank=backend_rank,
    )


def parity_from_logprob_row(
    case_id: str,
    expected_top_logprobs: tuple[TokenScore, ...],
    logprobs_row: np.ndarray,
    *,
    backend_rank: int,
) -> NextTokenParity:
    """Score a full Levanter ``[vocab]`` log-softmax row."""
    golden_logprobs = _golden_logprob_map(case_id, expected_top_logprobs)
    greedy_token_id = int(logprobs_row.argmax())
    actual_logprobs = {token_id: float(logprobs_row[token_id]) for token_id in golden_logprobs}
    return _parity_from_golden_logprobs(
        case_id,
        golden_logprobs,
        greedy_token_id,
        actual_logprobs,
        backend_rank=backend_rank,
    )


def _golden_logprob_map(case_id: str, expected_top_logprobs: tuple[TokenScore, ...]) -> dict[int, float]:
    if not expected_top_logprobs:
        raise ValueError(f"{case_id}: expected token scores must not be empty")
    golden_logprobs = {entry.token_id: entry.logprob for entry in expected_top_logprobs}
    if len(golden_logprobs) != len(expected_top_logprobs):
        raise ValueError(f"{case_id}: expected token scores contain duplicate token IDs")
    return golden_logprobs


def _parity_from_golden_logprobs(
    case_id: str,
    golden_logprobs: dict[int, float],
    greedy_token_id: int,
    actual_logprobs: dict[int, float],
    *,
    backend_rank: int,
) -> NextTokenParity:
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
