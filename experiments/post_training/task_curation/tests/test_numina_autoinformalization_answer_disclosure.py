# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for disclosure evidence validation."""

import json

import pytest

from experiments.post_training.task_curation.datasets.lean import (
    numina_autoinformalization_classify_answer_disclosure as classifier,
)


@pytest.mark.parametrize(
    "label,status",
    [
        ({"answer_disclosure": "yes", "problem_lines": [1], "lean_lines": [2]}, "valid"),
        ({"answer_disclosure": "no", "problem_lines": [1], "lean_lines": [2]}, "valid"),
        ({"answer_disclosure": "uncertain", "problem_lines": [], "lean_lines": []}, "valid"),
        ({"answer_disclosure": "no", "problem_lines": [], "lean_lines": []}, "invalid_evidence"),
        ({"answer_disclosure": "yes", "problem_lines": [1], "lean_lines": [True]}, "invalid_response"),
        ({"answer_disclosure": "yes", "problem_lines": [1], "lean_lines": [0]}, "invalid_response"),
    ],
)
def test_line_citations_preserve_unicode_and_distinguish_abstention_from_failure(label, status):
    row = {"problem": "Find all n ∈ \u2115.", "formal": "import Mathlib\ntheorem t : n = 12"}
    payload = {**label, "reason": "A short explanation"}
    response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(payload)}]}}]}
    reviewed = classifier.reviewed_response(response, row)
    assert reviewed.status == status
    assert (reviewed.classification is not None) == (status == "valid")
    if reviewed.classification:
        assert reviewed.classification.answer_disclosure == label["answer_disclosure"]


@pytest.mark.parametrize("response", [{}, {"candidates": [{"finishReason": "MAX_TOKENS"}]}])
def test_blocked_and_truncated_outputs_do_not_become_semantic_labels(response):
    reviewed = classifier.reviewed_response(response, {"problem": "Find n", "formal": "n = 12"})
    assert reviewed.status == "incomplete_response"
    assert reviewed.classification is None


def test_long_explanation_does_not_discard_a_valid_decision():
    label = {"answer_disclosure": "yes", "problem_lines": [1], "lean_lines": [1], "reason": "Explanation. " * 30}
    response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(label)}]}}]}
    reviewed = classifier.reviewed_response(response, {"problem": "Find n", "formal": "n = 12"})
    assert reviewed.status == "valid"
    assert reviewed.classification.reason == label["reason"]
