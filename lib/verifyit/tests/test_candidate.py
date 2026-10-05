# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared candidate contracts work without TaskCompendium or a filesystem harness."""

import json
from decimal import Decimal

import pytest
import reasoning_gym
from verifyit.candidate import candidate_spec, grade_text_candidate
from verifyit.grade import InvalidTask, grade
from verifyit.modes.grade_predicted_action import grade_predicted_action_candidate
from verifyit.spec import ExactSpec, FunctionCall, Mode, PredictedActionSpec, spec_to_table


@pytest.mark.parametrize(
    "mode,parameters,correct,wrong",
    [
        (Mode.EXACT, {"expected": ["The Answer"], "ignore_case": True}, "the   answer", "other"),
        (Mode.EXACT, {"expected": ["a", "b"], "ordered": True}, "a,b", "b,a"),
        (
            Mode.EXACT,
            {
                "expected": [" answer "],
                "ignore_case": False,
                "ignore_whitespace": False,
                "strip_outer_whitespace": False,
            },
            " answer ",
            "answer",
        ),
        (Mode.EXACT, {"expected": [""], "empty_output": "grade"}, "", "answer"),
        (Mode.NUMERIC, {"expected": 10.0, "tolerance_abs": 0.01, "tolerance_rel": 0.0}, "10.005", "10.02"),
        (Mode.MCQ, {"expected": "C", "options": 4}, "c", "E"),
    ],
)
def test_private_candidate_configuration_roundtrip_grades_content(mode, parameters, correct, wrong):
    spec = candidate_spec(mode, json.loads(json.dumps(parameters)))
    assert not isinstance(spec, PredictedActionSpec)
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


@pytest.mark.parametrize("expected", [["Paris", "Lyon"], [" \n"]])
def test_private_substring_contract_is_rejected_before_candidate_scoring(expected):
    with pytest.raises(InvalidTask):
        candidate_spec(Mode.EXACT, {"expected": expected, "substring": True})


def test_private_predicted_action_overflowing_tolerance_is_invalid_configuration():
    with pytest.raises(ValueError):
        candidate_spec(
            Mode.PREDICTED_ACTION,
            {"expected_calls": [{"name": "lookup", "arguments": {"id": 1}}], "numeric_tolerance": 10**400},
        )


PERSON_SCHEMA = json.dumps({"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}).encode()
JUMBLE_ENTRY = json.dumps({"answer": "alpha beta", "metadata": {"source_dataset": "letter_jumble"}}).encode()
DECIMAL_PARAMS = {"seed": 11, "size": 1, "min_num_decimal_places": 4, "max_num_decimal_places": 4}

# One correct and one wrong extracted answer per mode, with the verifier files each spec names.
FILE_BACKED_CASES = [
    (Mode.MATH, {"expected": "\\frac{1}{2}"}, {}, "\\frac{2}{4}", "\\frac{1}{3}"),
    (Mode.MATH, {"expected": "(2, \\infty)", "math_type": "interval"}, {}, "x > 2", "x < 2"),
    (Mode.JSON_SCHEMA, {"schema": "schema.json"}, {"schema.json": PERSON_SCHEMA}, '{"name": "Ada"}', '{"name": 3}'),
    (
        Mode.JSON_SCHEMA,
        {"schema": "schemas/person.json", "format": "yaml"},
        {"schemas/person.json": PERSON_SCHEMA},
        "```yaml\nname: Ada\n```",
        "age: 36",
    ),
    (Mode.XML_ELEMENTS, {"required": ["title", "year"]}, {}, '<book year="1999"><title>T</title></book>', "<book/>"),
    (Mode.XML_ELEMENTS, {"any_of": ["isbn", "doi"]}, {}, "<ref><doi>10.1/x</doi></ref>", "<ref><title>T</title></ref>"),
    (Mode.CSV_COLUMNS, {"required": ["name", "age"]}, {}, "name,age\nAda,36", "name\nAda"),
    (
        Mode.IFEVAL,
        {"constraints": [{"name": "keywords:forbidden_words", "params": {"forbidden_words": ["banana"]}}]},
        {},
        "I like apples.",
        "I like banana.",
    ),
    (Mode.REASONING_GYM, {"dataset": "letter_jumble"}, {"entry.json": JUMBLE_ENTRY}, "alpha beta", "wrong wrong"),
]


@pytest.mark.parametrize("mode,parameters,files,correct,wrong", FILE_BACKED_CASES)
def test_extracted_text_modes_grade_against_in_memory_verifier_files(mode, parameters, files, correct, wrong):
    spec = candidate_spec(mode, json.loads(json.dumps(parameters)), files=files)
    assert not isinstance(spec, PredictedActionSpec)
    assert grade_text_candidate(spec, correct, files=files).reward == 1.0
    assert grade_text_candidate(spec, wrong, files=files).reward == 0.0


@pytest.mark.parametrize("mode,parameters,files,correct,wrong", FILE_BACKED_CASES)
@pytest.mark.parametrize("empty_output", ["zero", "grade"])
def test_candidate_reward_matches_file_grading_of_the_same_answer(
    tmp_path, mode, parameters, files, correct, wrong, empty_output
):
    spec = candidate_spec(mode, {**parameters, "empty_output": empty_output}, files=files)
    assert not isinstance(spec, PredictedActionSpec)
    tests_dir = tmp_path / "tests"
    workspace = tmp_path / "app"
    workspace.mkdir()
    for path, data in files.items():
        (tests_dir / path).parent.mkdir(parents=True, exist_ok=True)
        (tests_dir / path).write_bytes(data)
    for candidate in (correct, wrong, " \n"):
        (workspace / "answer.txt").write_text(candidate)
        direct = grade_text_candidate(spec, candidate, files=files)
        from_file = grade(spec, tests_dir, workspace)
        assert (direct.reward, direct.status) == (from_file.reward, from_file.status)


def test_reasoning_gym_params_configure_the_scorer_from_in_memory_files():
    entry = reasoning_gym.create_dataset("decimal_arithmetic", **DECIMAL_PARAMS)[0]
    files = {"entry.json": json.dumps(entry).encode(), "params.json": json.dumps(DECIMAL_PARAMS).encode()}
    spec = candidate_spec(Mode.REASONING_GYM, {"dataset": "decimal_arithmetic", "params": "params.json"}, files=files)
    assert not isinstance(spec, PredictedActionSpec)
    assert grade_text_candidate(spec, entry["answer"], files=files).reward == 1.0
    near = str(Decimal(entry["answer"]) + Decimal("0.001"))
    assert grade_text_candidate(spec, near, files=files).reward == 0.0


@pytest.mark.parametrize(
    "mode,parameters,files",
    [
        (Mode.JSON_SCHEMA, {"schema": "schema.json"}, {}),
        (Mode.JSON_SCHEMA, {"schema": "schema.json"}, {"schema.json": b'{"type": "nope"}'}),
        (Mode.JSON_SCHEMA, {"schema": "schema.json"}, {"schema.json": b"\xff"}),
        (Mode.REASONING_GYM, {"dataset": "letter_jumble"}, {}),
        (Mode.REASONING_GYM, {"dataset": "letter_jumble", "params": "params.json"}, {"entry.json": JUMBLE_ENTRY}),
        (Mode.REASONING_GYM, {"dataset": "word_sorting"}, {"entry.json": JUMBLE_ENTRY}),
        (Mode.REASONING_GYM, {"dataset": "letter_jumble", "params": ["a"]}, {"entry.json": JUMBLE_ENTRY}),
        (Mode.MATH, {"expected": "\\frac{"}, {}),
        (Mode.XML_ELEMENTS, {}, {}),
        (Mode.CSV_COLUMNS, {}, {}),
        (Mode.IFEVAL, {"constraints": [{"name": "no:such_check"}]}, {}),
    ],
)
def test_invalid_private_configuration_or_missing_verifier_file_is_rejected(mode, parameters, files):
    with pytest.raises(InvalidTask):
        candidate_spec(mode, parameters, files=files)


def test_grading_requires_the_verifier_files_the_spec_names():
    files = {"schema.json": PERSON_SCHEMA}
    spec = candidate_spec(Mode.JSON_SCHEMA, {"schema": "schema.json"}, files=files)
    assert not isinstance(spec, PredictedActionSpec)
    with pytest.raises(InvalidTask):
        grade_text_candidate(spec, '{"name": "Ada"}')
