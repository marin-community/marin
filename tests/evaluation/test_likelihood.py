# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.evaluation.likelihood import completion_score


@pytest.mark.parametrize("internal_tokens", [0, 1])
def test_echo_scoring_excludes_prefix_and_requires_exact_token_alignment(internal_tokens):
    response = {
        "choices": [
            {
                "prompt_logprobs": [
                    None,
                    {"8": {"decoded_token": "prefix"}},
                    {"9": {"decoded_token": "AC"}},
                    {"10": {"decoded_token": "GT"}},
                ],
                "logprobs": {
                    "tokens": ["token_id:1", "token_id:8", "token_id:9", "token_id:10"],
                    "token_logprobs": [None, -100.0, -0.4, -0.7],
                },
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": internal_tokens},
    }
    assert completion_score(response, [1, 8, 9, 10], 2, "ACGT") == pytest.approx(-1.1)
    assert completion_score(response, [1, 8, 9, 10], 3, "GT") == pytest.approx(-0.7)
    with pytest.raises(AssertionError):
        completion_score(response, [1, 8, 10, 9], 2, "ACGT")

    with pytest.raises(AssertionError, match="exactly the target"):
        completion_score(response, [1, 8, 9, 10], 2, "prefixACGT")
