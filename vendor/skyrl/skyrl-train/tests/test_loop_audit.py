# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from skyrl_gym.verification import VerificationResult
from skyrl_train.config.reward_shaping import LoopCreditConfig
from skyrl_train.ftpo import ftpo_evaluation_metrics
from vendor.skyrl.scripts.loop_audit import audit_cohort, prompt_hash


def cohort_and_row(tokens):
    cohort = {"budget": 32768, "max_period_tokens": 4096, "rows": [{"task_id": "a", "prompt_hash": prompt_hash([42])}]}
    row = {"task_id": "a", "prompt_token_ids": [42], "response_token_ids": tokens,
           "verification": {"status": "verified", "passed": False}}
    return cohort, row


@pytest.mark.parametrize("tokens,expected", [([7] * 8, 1), ([7] * 7, 0), (list(range(4096)) * 8, 1), ([7] * 80 + [9], 0)])
def test_audit_detects_only_exact_suffix_with_eight_copies(tokens, expected):
    cohort, row = cohort_and_row(tokens)
    result = audit_cohort(cohort, [row])
    assert (result.confirmed, result.unassessed, result.UB) == (expected, 0, expected)


@pytest.mark.parametrize("change", [{"response_token_ids": None}, {"server_error": "failed"},
                                    {"error_treatment": "zero"}, {"exception_type": "TimeoutError"},
                                    {"prompt_token_ids": [99]}])
def test_audit_damaged_generation_enters_upper_bound_and_fails_closed(change):
    cohort, row = cohort_and_row([7] * 8)
    row.update(change)
    row["verification"] = {"status": "verified", "passed": True}
    result = audit_cohort(cohort, [row])
    assert (result.confirmed, result.unassessed, result.UB, result.passes) == (0, 1, 1, 0)


def test_audit_missing_generation_is_unassessed():
    cohort, _ = cohort_and_row([7] * 8)
    result = audit_cohort(cohort, [])
    assert (result.N, result.unassessed, result.unavailable, result.UB) == (1, 1, 1, 1)


@pytest.mark.parametrize("response,expected", [("still reasoning", 1), ("<|end_think|>answer", 0)])
def test_open_thinking_at_cap_includes_the_rendered_prompt(response, expected):
    cohort, row = cohort_and_row(list(range(32)))
    row.update(input_prompt="<|start_think|>", output_response=response, stop_reason="length")
    result = audit_cohort(cohort, [row])
    assert (result.length_stop, result.open_think) == (1, expected)


def test_native_pilot_ranks_fewer_loops_above_correctness_and_counts_errors():
    detector = LoopCreditConfig(max_period_tokens=2048, tail_tokens=16384, minimum_occurrences=8)
    correct = VerificationResult.verified(1.0, passed=True)
    wrong = VerificationResult.verified(0.0, passed=False)
    damaged = ftpo_evaluation_metrics({
        "response_ids": [[7] * 8, list(range(10)), [], [9] * 8],
        "verification_results": [correct, wrong, correct, correct],
        "server_errors": [None, None, None, {"error": "timeout"}],
    }, detector)
    assert (damaged["eval/all/loop_confirmed"], damaged["eval/all/loop_unassessed"],
            damaged["eval/all/loop_upper_bound"], damaged["eval/all/fail_closed_pass_rate"]) == (1, 2, 3, 0.25)
    fewer_loops = ftpo_evaluation_metrics({
        "response_ids": [[7] * 8, list(range(10)), list(range(10)), list(range(10))],
        "verification_results": [wrong] * 4,
    }, detector)
    more_correct = ftpo_evaluation_metrics({
        "response_ids": [[7] * 8, [8] * 8, list(range(10)), list(range(10))],
        "verification_results": [correct] * 4,
    }, detector)
    assert fewer_loops["eval/all/ftpo_selection_score"] > more_correct["eval/all/ftpo_selection_score"]
