# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-process candidate grading works without TaskCompendium or a filesystem harness."""

import json
from dataclasses import dataclass

import pytest
from verifyit.candidate import IN_PROCESS_MODES, Candidate, candidate_spec, grade_candidate
from verifyit.grade import InvalidTask, Status
from verifyit.json_comparison import NumericTypePolicy
from verifyit.spec import (
    ExactSpec,
    FunctionCall,
    JsonSchemaSpec,
    Mode,
    PredictedActionSpec,
    StructuredExactSpec,
    parse_spec,
    render_spec,
    spec_to_table,
)

SCHEMA = json.dumps({"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}).encode()


@dataclass(frozen=True)
class DispatchCase:
    parameters: dict
    correct: Candidate
    wrong: Candidate
    resources: dict[str, bytes]


DISPATCH_CASES = {
    Mode.EXACT: DispatchCase({"expected": ["The Answer"]}, "the   answer", "other", {}),
    Mode.NUMERIC: DispatchCase(
        {"expected": "10", "tolerance_abs": 0.01, "tolerance_rel": 0.0}, "The answer is 10.005", "10.02", {}
    ),
    Mode.MCQ: DispatchCase({"expected": "C", "options": 4}, "c", "B", {}),
    Mode.MATH: DispatchCase({"expected": "\\frac{1}{2}"}, "Halving gives\n0.5", "1/3", {}),
    Mode.IFEVAL: DispatchCase({"constraints": [{"name": "punctuation:no_comma"}]}, "No commas here.", "One, two.", {}),
    Mode.JSON_SCHEMA: DispatchCase(
        {"schema": "schemas/person.json"},
        '```json\n{"name": "Ada"}\n```',
        '{"name": 1}',
        {"schemas/person.json": SCHEMA},
    ),
    Mode.XML_ELEMENTS: DispatchCase({"required": ["name"]}, "<person><name>Ada</name></person>", "<person/>", {}),
    Mode.CSV_COLUMNS: DispatchCase({"required": ["name"]}, "name,age\nAda,36", "age\n36", {}),
    Mode.STRUCTURED_EXACT: DispatchCase({"expected": {"nested": [1, None]}}, {"nested": [1, None]}, {"nested": [1]}, {}),
    Mode.PREDICTED_ACTION: DispatchCase(
        {"expected_calls": [{"name": "lookup", "arguments": {"id": 1}}]},
        (FunctionCall("lookup", {"id": 1}),),
        (FunctionCall("lookup", {"id": 2}),),
        {},
    ),
}


@pytest.mark.parametrize("mode", sorted(IN_PROCESS_MODES))
def test_every_in_process_mode_grades_a_typed_candidate_from_a_json_configuration(mode):
    case = DISPATCH_CASES[mode]
    spec = candidate_spec(mode, json.loads(json.dumps(case.parameters)))
    correct = grade_candidate(spec, case.correct, case.resources)
    wrong = grade_candidate(spec, case.wrong, case.resources)
    assert (correct.status, correct.reward) == (Status.SCORED, 1.0)
    assert (wrong.status, wrong.reward) == (Status.SCORED, 0.0)


@pytest.mark.parametrize(
    "spec,candidate",
    [
        (ExactSpec(expected=("12",)), {"answer": "12"}),
        (PredictedActionSpec(expected_calls=(FunctionCall("lookup", {}),)), "lookup()"),
    ],
)
def test_candidate_of_the_wrong_shape_is_a_caller_error(spec, candidate):
    with pytest.raises(TypeError):
        grade_candidate(spec, candidate, {})


def test_missing_schema_resource_makes_the_task_invalid():
    verdict = grade_candidate(JsonSchemaSpec(schema="schema.json"), '{"name": "Ada"}', {})
    assert verdict.status == Status.INVALID_TASK


@pytest.mark.parametrize("mode", [Mode.MCQ, Mode.XML_ELEMENTS, Mode.IFEVAL])
def test_blank_text_scores_zero_without_reaching_the_mode(mode):
    spec = candidate_spec(mode, DISPATCH_CASES[mode].parameters)
    verdict = grade_candidate(spec, " \n", {})
    assert (verdict.status, verdict.reward, verdict.detail["reason"]) == (Status.SCORED, 0.0, "empty_output")


def test_numeric_text_without_one_literal_scores_zero_as_an_invalid_candidate():
    spec = candidate_spec(Mode.NUMERIC, {"expected": "12", "tolerance_abs": 0.0, "tolerance_rel": 0.0})
    verdict = grade_candidate(spec, "12 or 13", {})
    assert (verdict.status, verdict.reward, verdict.detail["reason"]) == (
        Status.SCORED,
        0.0,
        "invalid_numeric_candidate",
    )


def test_exact_candidate_does_not_extract_a_different_presentation():
    spec = ExactSpec(expected=("12",))
    assert grade_candidate(spec, "\\boxed{12}", {}).reward == 0.0
    assert grade_candidate(spec, "12", {}).reward == 1.0


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
    assert grade_candidate(spec, actual, {}).reward == reward


def test_function_call_multiset_preserves_duplicates_and_finds_nongreedy_tolerance_match():
    expected = (FunctionCall("set", {"value": 1.0}), FunctionCall("set", {"value": 1.1}))
    actual = (FunctionCall("set", {"value": 1.05}), FunctionCall("set", {"value": 0.95}))
    spec = PredictedActionSpec(expected_calls=expected, numeric_tolerance=0.06)
    assert grade_candidate(spec, actual, {}).reward == 1.0
    duplicates = PredictedActionSpec(expected_calls=(expected[0], expected[0]))
    assert grade_candidate(duplicates, (expected[0], expected[1]), {}).reward == 0.0


def test_function_call_configuration_roundtrip_preserves_nested_json_and_tolerance():
    spec = PredictedActionSpec(expected_calls=(FunctionCall("set", {"values": [None, False, {"score": 1.0}]}),))
    table = spec_to_table(spec)
    mode = table.pop("mode")
    restored = candidate_spec(mode, json.loads(json.dumps(table)))
    assert isinstance(restored, PredictedActionSpec)
    assert grade_candidate(restored, spec.expected_calls, {}).reward == 1.0
    nearby = (FunctionCall("set", {"values": [None, False, {"score": 1.005}]}),)
    assert grade_candidate(restored, nearby, {}).reward == 0.0
    tolerant = PredictedActionSpec(expected_calls=restored.expected_calls, numeric_tolerance=0.01)
    assert grade_candidate(tolerant, nearby, {}).reward == 1.0


@pytest.mark.parametrize("expected", [["Paris", "Lyon"], [" \n"]])
def test_substring_contract_is_rejected_before_candidate_scoring(expected):
    with pytest.raises(InvalidTask):
        candidate_spec(Mode.EXACT, {"expected": expected, "substring": True})
    spec = ExactSpec(expected=tuple(expected), substring=True)
    assert grade_candidate(spec, "Paris", {}).status == Status.INVALID_TASK


def test_predicted_action_overflowing_tolerance_is_invalid_configuration():
    with pytest.raises(ValueError):
        candidate_spec(
            Mode.PREDICTED_ACTION,
            {"expected_calls": [{"name": "lookup", "arguments": {"id": 1}}], "numeric_tolerance": 10**400},
        )


@pytest.mark.parametrize("policy,reward", [(NumericTypePolicy.VALUE, 1.0), (NumericTypePolicy.STRICT, 0.0)])
def test_structured_exact_numeric_policy_survives_json_and_toml_roundtrips(policy, reward):
    spec = StructuredExactSpec(expected={"nested": [16, True]}, numeric_types=policy)
    table = spec_to_table(spec)
    mode = table.pop("mode")
    for restored in (candidate_spec(mode, json.loads(json.dumps(table))), parse_spec(render_spec(spec))):
        assert isinstance(restored, StructuredExactSpec)
        assert grade_candidate(restored, {"nested": [16.0, True]}, {}).reward == reward
        assert grade_candidate(restored, {"nested": [16, 1]}, {}).reward == 0.0
