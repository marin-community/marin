# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from tasktrove_verify.grade import InvalidTask, Status
from tasktrove_verify.modes import grade_json_schema
from tasktrove_verify.spec import JsonSchemaSpec, SchemaFormat

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "email", "quantity"],
    "properties": {
        "name": {"type": "string", "maxLength": 20},
        "email": {"type": "string", "format": "email"},
        "quantity": {"type": "integer", "minimum": 1},
    },
}

ORDER = {"name": "Ada", "email": "ada@example.com", "quantity": 3}


@pytest.fixture
def tests_dir(tmp_path):
    directory = tmp_path / "tests"
    directory.mkdir()
    (directory / "schema.json").write_text(json.dumps(SCHEMA))
    return directory


@pytest.fixture
def workspace(tmp_path):
    directory = tmp_path / "app"
    directory.mkdir()
    return directory


def answer(workspace, text):
    (workspace / "answer.txt").write_text(text)


def test_document_matching_the_schema_scores_one(tests_dir, workspace):
    answer(workspace, json.dumps(ORDER))
    reward = grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace)
    assert (reward.reward, reward.status) == (1.0, Status.SCORED)


def test_decoded_candidate_uses_same_schema_contract_as_file_grade(tests_dir, workspace):
    for candidate, expected_reward in ((ORDER, 1.0), ({**ORDER, "quantity": "three"}, 0.0)):
        answer(workspace, json.dumps(candidate))
        direct = grade_json_schema.grade_json_schema_candidate(SCHEMA, candidate)
        from_file = grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace)
        assert direct.reward == expected_reward
        assert (direct.reward, direct.status, direct.detail) == (
            from_file.reward,
            from_file.status,
            from_file.detail,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_number_cannot_bypass_numeric_bounds(tests_dir, workspace, value):
    schema = {"type": "object", "properties": {"quantity": {"type": "number", "minimum": 0, "maximum": 10}}}
    (tests_dir / "schema.json").write_text(json.dumps(schema))
    candidate = {"quantity": value}
    answer(workspace, json.dumps(candidate))
    direct = grade_json_schema.grade_json_schema_candidate(schema, candidate)
    from_file = grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace)
    assert (direct.reward, direct.status, direct.detail) == (0.0, Status.SCORED, {"reason": "nonfinite_number"})
    assert (direct.reward, direct.status, direct.detail) == (
        from_file.reward,
        from_file.status,
        from_file.detail,
    )


def test_nonfinite_nested_candidate_scores_zero(tests_dir, workspace):
    schema = {"type": "object", "properties": {"values": {"type": "array", "items": {"type": "number"}}}}
    (tests_dir / "schema.json").write_text(json.dumps(schema))
    answer(workspace, '{"values": [1, NaN]}')
    assert grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace).detail == {"reason": "nonfinite_number"}


def test_nonfinite_schema_bound_is_invalid_task(tests_dir, workspace):
    schema = {"type": "number", "minimum": float("nan")}
    (tests_dir / "schema.json").write_text(json.dumps(schema))
    answer(workspace, "1")
    with pytest.raises(InvalidTask, match="nonfinite"):
        grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace)
    with pytest.raises(InvalidTask, match="nonfinite"):
        grade_json_schema.grade_json_schema_candidate(schema, 1)


def test_document_inside_a_code_fence_is_unwrapped(tests_dir, workspace):
    answer(workspace, f"Here is the order:\n\n```json\n{json.dumps(ORDER)}\n```\n")
    assert grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace).reward == 1.0


def test_missing_required_property_scores_zero_and_names_it(tests_dir, workspace):
    answer(workspace, json.dumps({"name": "Ada", "quantity": 3}))
    reward = grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace)
    assert reward.reward == 0.0
    assert reward.status == Status.SCORED
    assert "email" in reward.detail["error"]


def test_wrong_property_type_scores_zero_and_reports_its_path(tests_dir, workspace):
    answer(workspace, json.dumps({**ORDER, "quantity": "three"}))
    reward = grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace)
    assert reward.reward == 0.0
    assert reward.detail["path"] == "quantity"


def test_format_keyword_is_annotation_only(tests_dir, workspace):
    """The graders this mode replaces validated without a format checker; a bad email is still valid."""
    answer(workspace, json.dumps({**ORDER, "email": "ada-at-example-dot-com"}))
    reward = grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace)
    assert reward.reward == 1.0


def test_unparsable_document_scores_zero_without_raising(tests_dir, workspace):
    answer(workspace, "I could not produce the order, sorry.")
    reward = grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace)
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)
    assert reward.detail["reason"] == "parse_error"


@pytest.mark.parametrize("text", [None, "", "   \n\n"])
def test_absent_or_blank_output_scores_zero_with_no_output(tests_dir, workspace, text):
    if text is not None:
        answer(workspace, text)
    reward = grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace)
    assert reward.reward == 0.0
    assert reward.detail == {"reason": "no_output"}


def test_yaml_candidate_validates_against_the_same_schema(tests_dir, workspace):
    answer(workspace, "name: Ada\nemail: ada@example.com\nquantity: 3\n")
    spec = JsonSchemaSpec(format=SchemaFormat.YAML)
    assert grade_json_schema.grade(spec, tests_dir, workspace).reward == 1.0
    answer(workspace, "name: Ada\nquantity: 0\n")
    assert grade_json_schema.grade(spec, tests_dir, workspace).reward == 0.0


def test_yaml_dates_are_stringified_for_string_typed_fields(tmp_path, workspace):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    schema = {"type": "object", "required": ["due"], "properties": {"due": {"type": "string", "format": "date"}}}
    (tests_dir / "schema.json").write_text(json.dumps(schema))
    answer(workspace, "due: 2026-03-01\n")
    reward = grade_json_schema.grade(JsonSchemaSpec(format=SchemaFormat.YAML), tests_dir, workspace)
    assert reward.reward == 1.0


def test_declared_draft_decides_how_the_schema_is_read(tmp_path, workspace):
    # exclusiveMinimum is a boolean modifier in draft-04 and a number in later drafts, so a
    # draft-04 schema is only read correctly when its own $schema selects the validator.
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {"n": {"type": "number", "minimum": 1, "exclusiveMinimum": True}},
    }
    (tests_dir / "schema.json").write_text(json.dumps(schema))
    answer(workspace, json.dumps({"n": 1}))
    assert grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace).reward == 0.0
    answer(workspace, json.dumps({"n": 2}))
    assert grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace).reward == 1.0


def test_missing_schema_file_is_an_invalid_task(tmp_path, workspace):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    answer(workspace, json.dumps(ORDER))
    with pytest.raises(InvalidTask, match="schema file not found"):
        grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace)


@pytest.mark.parametrize(
    "schema_text, message",
    [
        ('{"type": "object",}', "not JSON"),
        ('["not", "a", "schema"]', "must hold a JSON object"),
        ('{"type": "nonesuch"}', "not a valid JSON Schema"),
    ],
)
def test_unusable_schema_is_an_invalid_task(tmp_path, workspace, schema_text, message):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "schema.json").write_text(schema_text)
    answer(workspace, json.dumps(ORDER))
    with pytest.raises(InvalidTask, match=message):
        grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace)


def test_toml_candidate_validates_against_the_same_schema(tests_dir, workspace):
    answer(workspace, 'name = "Ada"\nemail = "ada@example.com"\nquantity = 3\n')
    spec = JsonSchemaSpec(format=SchemaFormat.TOML)
    assert grade_json_schema.grade(spec, tests_dir, workspace).reward == 1.0
    answer(workspace, 'name = "Ada"\nemail = "ada@example.com"\nquantity = 0\n')
    assert grade_json_schema.grade(spec, tests_dir, workspace).reward == 0.0


def test_toml_dates_are_stringified_for_string_typed_fields(tmp_path, workspace):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    schema = {"type": "object", "required": ["due"], "properties": {"due": {"type": "string", "format": "date"}}}
    (tests_dir / "schema.json").write_text(json.dumps(schema))
    answer(workspace, "due = 2026-03-01\n")
    assert grade_json_schema.grade(JsonSchemaSpec(format=SchemaFormat.TOML), tests_dir, workspace).reward == 1.0


def test_unparsable_toml_scores_zero_without_raising(tests_dir, workspace):
    answer(workspace, "name = Ada\n")
    reward = grade_json_schema.grade(JsonSchemaSpec(format=SchemaFormat.TOML), tests_dir, workspace)
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)
    assert reward.detail["reason"] == "parse_error"


def test_grid_schema_rejects_boolean_cells_even_when_python_equality_matches(tests_dir, workspace):
    # ARC's original row comparison accepts False == 0 and True == 1.
    # A typed JSON schema keeps the expected integer-grid contract.
    schema = {
        "type": "array",
        "items": {"type": "array", "items": {"type": "integer"}},
        "const": [[0, 1]],
    }
    (tests_dir / "schema.json").write_text(json.dumps(schema))
    answer(workspace, "[[0, 1]]")
    assert grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace).reward == 1.0
    answer(workspace, "[[false, true]]")
    reward = grade_json_schema.grade(JsonSchemaSpec(), tests_dir, workspace)
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)


@pytest.mark.parametrize("instance", [None, [], {}, 0, False, ""])
def test_present_json_falsy_values_are_data_not_missing_output(tmp_path, instance):
    (tmp_path / "schema.json").write_text("{}")
    (tmp_path / "answer.txt").write_text(json.dumps(instance))
    spec = JsonSchemaSpec()
    assert grade_json_schema.grade(spec, tmp_path, tmp_path).reward == 1
    assert grade_json_schema.grade_json_schema_candidate({}, instance).reward == 1
    (tmp_path / "answer.txt").unlink()
    assert grade_json_schema.grade(spec, tmp_path, tmp_path).reward == 0


def test_deep_candidate_scores_zero_through_direct_and_file_apis(tmp_path):
    candidate = 0
    for _ in range(1200):
        candidate = [candidate]
    verdict = grade_json_schema.grade_json_schema_candidate({}, candidate)
    assert (verdict.reward, verdict.status) == (0.0, Status.SCORED)
    (tmp_path / "schema.json").write_text("{}")
    for candidate_format, text in (
        (SchemaFormat.JSON, "[" * 1200 + "0" + "]" * 1200),
        (SchemaFormat.YAML, "&loop [*loop]"),
    ):
        (tmp_path / "answer.txt").write_text(text)
        verdict = grade_json_schema.grade(JsonSchemaSpec(format=candidate_format), tmp_path, tmp_path)
        assert (verdict.reward, verdict.status) == (0.0, Status.SCORED)


def test_deep_trusted_schema_is_invalid_before_candidate_scoring(tmp_path):
    schema = {}
    for _ in range(1200):
        schema = {"allOf": [schema]}
    with pytest.raises(InvalidTask):
        grade_json_schema.grade_json_schema_candidate(schema, None)
    (tmp_path / "schema.json").write_text('{"allOf":[' * 1200 + "{}" + "]}" * 1200)
    with pytest.raises(InvalidTask):
        grade_json_schema.grade(JsonSchemaSpec(), tmp_path, tmp_path)
