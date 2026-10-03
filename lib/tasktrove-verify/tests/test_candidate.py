# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared candidate contracts work without TaskCompendium or a filesystem harness."""

import json

import pytest
from tasktrove_verify.candidate import candidate_spec, grade_ejection_candidate, grade_text_candidate
from tasktrove_verify.grade import grade
from tasktrove_verify.modes.grade_predicted_action import grade_predicted_action_candidate
from tasktrove_verify.spec import (
    ExactSpec,
    FunctionCall,
    Mode,
    PredictedActionSpec,
    StructuredExactSpec,
    UnsolvableSpec,
    parse_spec,
    render_spec,
    spec_to_table,
)


@pytest.mark.parametrize(
    "mode,parameters,correct,wrong",
    [
        (Mode.EXACT, {"expected": ["The Answer"], "ignore_case": True}, "the   answer", "other"),
        (Mode.EXACT, {"expected": ["a", "b"], "ordered": True}, "a,b", "b,a"),
        (Mode.NUMERIC, {"expected": 10.0, "tolerance_abs": 0.01, "tolerance_rel": 0.0}, "10.005", "10.02"),
        (Mode.MCQ, {"expected": "C", "options": 4}, "c", "E"),
    ],
)
def test_private_candidate_configuration_roundtrip_grades_content(mode, parameters, correct, wrong):
    spec = candidate_spec(mode, json.loads(json.dumps(parameters)))
    assert not isinstance(spec, (PredictedActionSpec, StructuredExactSpec))
    assert grade_text_candidate(spec, correct).reward == 1.0
    assert grade_text_candidate(spec, wrong).reward == 0.0


@pytest.mark.parametrize(
    "actual,reward",
    [
        ((FunctionCall("lookup", {"id": 1}),), 1.0),
        ((FunctionCall("lookup", {"id": True}),), 0.0),
        ((FunctionCall("lookup", {"id": 1.0}),), 0.0),
        ((FunctionCall("lookup", {"id": 1, "extra": None}),), 0.0),
        ((FunctionCall("lookup", {"id": 1}), FunctionCall("lookup", {"id": 1})), 0.0),
        ((), 0.0),
    ],
)
def test_function_call_candidate_requires_exact_count_and_json_types(actual, reward):
    spec = PredictedActionSpec(expected_calls=(FunctionCall("lookup", {"id": 1}),))
    assert grade_predicted_action_candidate(spec, actual).reward == reward


def test_function_call_multiset_preserves_duplicates_and_finds_nongreedy_tolerance_match():
    expected = (FunctionCall("set", {"value": 1.0}), FunctionCall("set", {"value": 1.1}))
    actual = (FunctionCall("set", {"value": 1.05}), FunctionCall("set", {"value": 0.95}))
    spec = PredictedActionSpec(expected_calls=expected, numeric_tolerance=0.06)
    assert grade_predicted_action_candidate(spec, actual).reward == 1.0
    duplicates = PredictedActionSpec(expected_calls=(expected[0], expected[0]))
    assert grade_predicted_action_candidate(duplicates, (expected[0], expected[1])).reward == 0.0


def test_function_call_private_descriptor_roundtrip_preserves_nested_json_and_tolerance():
    spec = PredictedActionSpec(expected_calls=(FunctionCall("set", {"values": [None, False, {"score": 1.0}]}),))
    table = spec_to_table(spec)
    mode = table.pop("mode")
    restored = candidate_spec(mode, json.loads(json.dumps(table)))
    assert isinstance(restored, PredictedActionSpec)
    assert grade_predicted_action_candidate(restored, spec.expected_calls).reward == 1.0
    nearby = (FunctionCall("set", {"values": [None, False, {"score": 1.005}]}),)
    assert grade_predicted_action_candidate(restored, nearby).reward == 0.0
    tolerant = PredictedActionSpec(expected_calls=restored.expected_calls, numeric_tolerance=0.01)
    assert grade_predicted_action_candidate(tolerant, nearby).reward == 1.0


def test_exact_candidate_does_not_extract_a_different_presentation():
    spec = ExactSpec(expected=("12",))
    assert grade_text_candidate(spec, "\\boxed{12}").reward == 0.0
    assert grade_text_candidate(spec, "12").reward == 1.0


@pytest.mark.parametrize(
    "spec,expected_reward",
    [
        (ExactSpec(expected=("yes",)), 0.0),
        (PredictedActionSpec(expected_calls=(FunctionCall("choose", {}),)), 0.0),
        (UnsolvableSpec(), 1.0),
    ],
)
def test_ejection_candidate_scores_only_explicit_private_unsolvability(spec, expected_reward):
    table = spec_to_table(spec)
    mode = table.pop("mode")
    restored = candidate_spec(mode, json.loads(json.dumps(table)))
    assert grade_ejection_candidate(restored).reward == expected_reward
    if isinstance(restored, UnsolvableSpec):
        assert grade_text_candidate(restored, "The reason sounds convincing.").reward == 0.0
        assert parse_spec(render_spec(restored)) == restored


def test_unsolvable_filesystem_grading_requires_explicit_terminal_evidence(tmp_path):
    (tmp_path / "answer.txt").write_text("eject_button: cannot solve")
    with pytest.raises(NotImplementedError, match="terminal evidence"):
        grade(UnsolvableSpec(), tests_dir=tmp_path, workspace=tmp_path)
