# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source descriptor grading at the verifyit boundary."""

import pytest
from verifyit.adapters.open_instruct_ifeval import compute_score, normalize_ground_truth

CASES = [
    (
        {"func_name": "verify_paragraph_count", "N": 2},
        "First paragraph.\n\n***\n\nSecond paragraph.",
        "First paragraph.\n\nSecond paragraph.",
    ),
    (
        {"func_name": "verify_bullet_points", "N": 2},
        "**Key points:**\n* One\n* Two\n\n* * *\n\nNext paragraph.",
        "**Key points:**\n* One\n\n* * *\n\nNext paragraph.",
    ),
    (
        {"func_name": "validate_sections", "N": 2, "section_splitter": "SECTION"},
        "<<Guide to SECTION 1>>\n\nSECTION 1\nFirst body.\n\nSECTION 2\nSecond body.",
        "<<Guide to SECTION 1>>\n\nSECTION 1\nFirst body.",
    ),
    (
        {"func_name": "validate_highlighted_sections", "N": 2},
        "**First** and **second** are highlighted.",
        "**Only one** is highlighted.",
    ),
]


@pytest.mark.parametrize(("descriptor", "passing", "failing"), CASES)
def test_open_instruct_descriptors_accept_compliant_replies_and_reject_ignored_instructions(
    descriptor, passing, failing
):
    normalized = normalize_ground_truth(descriptor)
    assert compute_score(passing, normalized)["score"] == 1.0
    assert compute_score(failing, normalized)["score"] == 0.0


def test_open_instruct_descriptors_keep_fractional_rewards_and_fail_closed():
    constraints = [{"func_name": "validate_lowercase"}, {"func_name": "validate_no_commas"}]
    assert compute_score("lowercase, with a comma", constraints)["score"] == 0.5
    assert compute_score("", constraints)["score"] == 0.0
    invalid_choice = compute_score("anything", {"func_name": "validate_choice", "options": []})
    invalid_minimum = compute_score("Plain text", {"func_name": "validate_highlighted_sections", "N": 0})
    assert (invalid_choice["score"], invalid_choice["error_type"]) == (0.0, "schema_error")
    assert (invalid_minimum["score"], invalid_minimum["error_type"]) == (0.0, "schema_error")
    assert compute_score("No z here", {"func_name": "verify_letter_frequency", "letter": "q", "N": 0})["score"] == 1.0
