# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import asdict

import pytest
from harbor_config.errors import ErrorCategory, error_category
from verifyit.adapters.harness_native import exact_match
from verifyit.grade import InvalidTask, Status, finalize_preparation_failure
from verifyit.preparation.errors import PreparationError, PreparationFailure
from verifyit.preparation.text import (
    TextInputs,
    TextNormalization,
    TextPolicy,
    normalize_text,
    structure_text,
)
from verifyit.spec import EmptyOutputPolicy


def test_structural_snapshot_preserves_raw_values_and_isolates_reference_container():
    references = [" İ ", "", " İ ", "long"]
    inputs = structure_text(" İ ", references)
    references.clear()
    assert inputs == TextInputs(" İ ", (" İ ", "", " İ ", "long"))
    normalized = normalize_text(inputs, TextNormalization(ignore_case=True))
    assert normalized.raw is inputs
    assert normalized.candidate == " i\u0307"
    assert normalized.references == (" i\u0307 ", "", " i\u0307 ", "long")


def test_named_identity_policy_retains_literal_text_and_exposes_effective_policy():
    source = exact_match("İ", ["i"], ignore_case=True)
    literal = exact_match("İ", ["i"], normalization_policy=TextPolicy.IDENTITY)
    assert source.reward == 1
    assert literal.reward == 0
    assert literal.detail["preparation"]["policy"] == TextPolicy.IDENTITY
    empty = exact_match("!", ["!"], ignore_punctuation=True, empty_output=EmptyOutputPolicy.ZERO)
    assert empty.reward == 0
    assert empty.detail["preparation"]["empty_output"] == EmptyOutputPolicy.ZERO
    with pytest.raises(InvalidTask):
        exact_match("İ", ["i"], normalization_policy=TextPolicy.IDENTITY, ignore_case=True)


def test_preparation_raises_at_failure_with_original_metadata_and_minimum_verdict():
    with pytest.raises(PreparationError) as structural:
        structure_text("correct", ["correct", None])
    assert isinstance(structural.value, InvalidTask)
    assert structural.value.failure.stage == "structure"
    assert (structural.value.verdict.status, structural.value.verdict.reward) == (Status.INVALID_TASK, 0)
    with pytest.raises(PreparationError) as caught:
        exact_match("correct", ["correct"], regexes_to_ignore=["["])
    assert isinstance(caught.value, InvalidTask)
    assert caught.value.failure.stage == "normalize"
    assert caught.value.verdict.status == Status.INVALID_TASK
    assert caught.value.verdict.reward == 0


@pytest.mark.parametrize(
    "error_type,status,expected,category",
    [
        ("AgentTimeoutError", Status.SCORED, Status.SCORED, ErrorCategory.AGENT),
        ("SandboxBuildFailedError", Status.SCORED, Status.INFRA_ERROR, ErrorCategory.INFRASTRUCTURE),
        ("OutputLengthExceededError", Status.SCORED, Status.INFRA_ERROR, ErrorCategory.PASSTHROUGH),
        ("UnrecognizedFailure", Status.SCORED, Status.INFRA_ERROR, ErrorCategory.UNKNOWN),
        ("AgentTimeoutError", Status.INFRA_ERROR, Status.INFRA_ERROR, ErrorCategory.AGENT),
    ],
)
def test_failed_preparation_has_no_partial_grade_to_pass_through(error_type, status, expected, category):
    failure = PreparationFailure(status, error_category(error_type), error_type, "interrupted", "tool")
    verdict = finalize_preparation_failure(**asdict(failure))
    assert (verdict.status, verdict.reward) == (expected, 0)
    assert verdict.detail["category"] == category
    assert verdict.detail["error_type"] == error_type
    assert verdict.detail["source_status"] == status


def test_missing_numpy_raises_infrastructure_failure_before_exact_grading(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(PreparationError) as caught:
        exact_match("correct", ["correct"], ignore_case=True)
    assert not isinstance(caught.value, InvalidTask)
    assert caught.value.failure.stage == "normalize"
    assert caught.value.failure.error_type == "ModuleNotFoundError"
    assert (caught.value.verdict.status, caught.value.verdict.reward) == (Status.INFRA_ERROR, 0)
