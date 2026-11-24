# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
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
