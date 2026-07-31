# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from tests.cluster.vllm.grug_exact_reference_check import _score_logprobs


def test_fixture_logprob_parity_allows_error_explained_near_tie() -> None:
    expected_probabilities = np.ones(64, dtype=np.float64)
    expected_probabilities[22] = 1.01
    expected_probabilities[24] = 1.0099
    expected_probabilities /= expected_probabilities.sum()
    actual_probabilities = expected_probabilities.copy()
    actual_probabilities[22], actual_probabilities[24] = (
        actual_probabilities[24],
        actual_probabilities[22],
    )
    payload = {
        "choices": [
            {
                "token_ids": [24],
                "logprobs": {
                    "top_logprobs": [
                        {
                            f"token_id:{token_id}": float(np.log(probability))
                            for token_id, probability in enumerate(actual_probabilities)
                        }
                    ]
                },
            }
        ]
    }

    result = _score_logprobs(np.log(expected_probabilities), payload)

    assert result["expected_greedy_token_id"] == 22
    assert result["greedy_token_id"] == 24
    assert not result["greedy_token_agrees"]
    assert result["golden_probability_gap_to_greedy"] <= 2 * result["max_probability_error"]
