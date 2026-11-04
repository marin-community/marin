# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.testing.inference.snowball import SNOWBALL, read_representative_goldens
from marin.testing.inference.snowball_backend_parity_jobs import VLLM_MAX_PROBABILITY_ERROR

from experiments.evaluation.serving_parity.nightly import classify, validate_result


@pytest.mark.parametrize(
    "states,expected",
    [
        (["succeeded", "succeeded"], "pass"),
        (["succeeded", "no_capacity"], "no_capacity"),
        (["worker_failed", "no_capacity"], "crash"),
        (["succeeded", "unschedulable"], "crash"),
    ],
)
def test_nightly_outcome(states, expected):
    assert classify(states) == expected


@pytest.mark.parametrize(
    "damage", [None, "missing", "duplicate", "wrong_export", "probability", "unexplained", "repeat"]
)
def test_nightly_rechecks_complete_probability_evidence(damage):
    rows = []
    for golden in read_representative_goldens():
        winner = max(golden.top_logprobs, key=lambda score: score.logprob).token_id
        rows.append(
            dict(
                case_id=golden.id,
                backend_rank=0,
                greedy_token_id=winner,
                golden_top_token_ids=[score.token_id for score in golden.top_logprobs],
                golden_probability_gap_to_greedy=0.0,
                max_probability_error=0.0,
                top_probability_l1_error=0.0,
            )
        )
    result = dict(
        hardware="v4",
        export_sha256=SNOWBALL.export_sha256,
        same_rank_repeatability=True,
        max_probability_error=VLLM_MAX_PROBABILITY_ERROR,
        parities=rows,
    )
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows[-1] = rows[0]
    elif damage == "wrong_export":
        result["export_sha256"] = "wrong"
    elif damage == "probability":
        rows[0]["max_probability_error"] = 0.9
    elif damage == "unexplained":
        rows[0]["golden_probability_gap_to_greedy"] = 0.1
    elif damage == "repeat":
        result["same_rank_repeatability"] = False
    if damage:
        with pytest.raises(AssertionError):
            validate_result(result, "v4")
    else:
        assert validate_result(result, "v4") is None  # Complete valid evidence must permit nightly success.
