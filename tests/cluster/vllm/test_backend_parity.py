# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.cluster.vllm.backend_parity import NextTokenParity


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
