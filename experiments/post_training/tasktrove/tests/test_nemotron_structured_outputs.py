# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the ``nemotron_structured_outputs`` exemplar."""

import json
import tempfile
from pathlib import Path

from tasktrove_verify.modes.json_schema import grade
from tasktrove_verify.reward import Status
from tasktrove_verify.spec import JsonSchemaSpec, parse_spec

from experiments.post_training.tasktrove.contract import INSTALL_MARKER, VERIFIER_TOML, VERIFY_TEST_SH
from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.sources import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    TEST_SH,
    TaskFiles,
    read_task_binary,
    write_task_binary,
)
from experiments.post_training.tasktrove.verify import verify_task

FIXTURES = Path(__file__).parents[1] / "fixtures"
TOOL_REF = "0123abc"

_VALID_CANDIDATE = {
    "dishName": "Grilled Salmon",
    "wineType": "White",
    "flavorProfile": {"primary": "citrus", "secondary": "green apple"},
    "acidityLevel": "Medium",
    "tanninLevel": "Low",
    "servingTemperature": 12,
    "isRecommended": True,
}


def _fixture() -> bytes:
    return (FIXTURES / "nemotron_structured_outputs.tar.gz").read_bytes()


def _info() -> SourceInfo:
    return SourceInfo("laion__nemotron-gym-structured-outputs-v4", SourceVerdict.KEEP, "other", "")


def _convert(blob: bytes | None = None):
    return convert_one(_info(), "t.tar.gz", blob if blob is not None else _fixture(), converter_index(), TOOL_REF)


def _mutate_verifier_data(**overrides) -> bytes:
    task = read_task_binary(_fixture())
    data = json.loads(task.text("tests/verifier_data.json"))
    data.update(overrides)
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    return write_task_binary(task)


def _mutate_metadata(**overrides) -> bytes:
    task = read_task_binary(_fixture())
    meta = json.loads(task.text("metadata.json"))
    meta.update(overrides)
    task.files["metadata.json"] = json.dumps(meta).encode()
    return write_task_binary(task)


def test_exemplar_converts_to_json_schema_spec_with_expected_tags():
    record = _convert()
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "nemotron_structured_outputs"
    assert record.mode == "json-schema"
    assert record.tags == ["structured-outputs", "json-schema", "nemotron", "json"]

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, JsonSchemaSpec)
    assert spec.schema == "schema.json"
    assert "tests/schema.json" in task.files
    schema = json.loads(task.text("tests/schema.json"))
    assert schema["required"] == [
        "dishName",
        "wineType",
        "flavorProfile",
        "acidityLevel",
        "tanninLevel",
        "servingTemperature",
        "isRecommended",
    ]

    assert task.text(TEST_SH) == VERIFY_TEST_SH
    assert "tests/verifier.py" not in task.files, "old grader code must not ship"
    assert "tests/verifier_data.json" not in task.files, "raw grader data must not ship"

    dockerfile = task.text(DOCKERFILE)
    assert dockerfile.startswith("# DO NOT EDIT") and INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile
    assert record.solution_binary is None and not record.has_solution


def test_exemplar_passes_verification():
    record = _convert()
    assert record.status == ConvertStatus.CONVERTED
    assert verify_task(record.task_binary) is None


def test_exemplar_schema_grades_valid_and_invalid_candidates():
    """The shipped schema.json actually enforces the source schema, independent of verify_task's
    probe-less pass (json-schema has no built-in positive/negative candidate in verify.py)."""
    record = _convert()
    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, JsonSchemaSpec)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / spec.schema).write_bytes(task.files[f"tests/{spec.schema}"])
        workspace = tmp_path / "app"
        workspace.mkdir()
        answer = workspace / "answer.txt"

        answer.write_text(json.dumps(_VALID_CANDIDATE))
        reward = grade(spec, tmp_path, workspace)
        assert reward.status == Status.SCORED and reward.reward == 1.0

        answer.write_text(json.dumps({**_VALID_CANDIDATE, "wineType": 42}))
        reward = grade(spec, tmp_path, workspace)
        assert reward.status == Status.SCORED and reward.reward == 0.0

        answer.unlink()
        reward = grade(spec, tmp_path, workspace)
        assert reward.status == Status.SCORED and reward.reward == 0.0


def test_yaml_schema_type_maps_to_yaml_format():
    record = _convert(_mutate_verifier_data(schema_type="yaml"))
    assert record.status == ConvertStatus.CONVERTED
    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, JsonSchemaSpec)
    assert spec.format.value == "yaml"
    assert record.tags == ["structured-outputs", "json-schema", "nemotron", "yaml"]


def test_xml_schema_type_is_rejected_as_unsupported_variant():
    record = _convert(_mutate_verifier_data(schema_type="xml"))
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None
    assert "xml" in record.error


def test_csv_schema_type_is_rejected_as_unsupported_variant():
    record = _convert(_mutate_verifier_data(schema_type="csv"))
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_toml_schema_type_is_rejected_as_unsupported_variant():
    record = _convert(_mutate_verifier_data(schema_type="toml"))
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_non_object_schema_is_rejected_as_null_grader():
    record = _convert(_mutate_verifier_data(schema=["not", "an", "object"]))
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_empty_schema_is_rejected_as_null_grader():
    empty_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    record = _convert(_mutate_verifier_data(schema=empty_schema))
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_grouped_required_dict_is_flattened_to_a_list():
    """The dataset's non-standard ``required: {"group": [...]}`` shape, seen on ~3% of rows."""
    task = read_task_binary(_fixture())
    data = json.loads(task.text("tests/verifier_data.json"))
    schema = data["schema"]
    schema["required"] = {"group": schema["required"][:3]}
    data["schema"] = schema
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.CONVERTED
    converted_task = read_task_binary(record.task_binary)
    converted_schema = json.loads(converted_task.text("tests/schema.json"))
    assert sorted(converted_schema["required"]) == sorted(schema["required"]["group"])


def test_string_additional_properties_is_coerced_to_bool():
    """The dataset's non-standard ``"additionalProperties": "false"`` string, seen on ~19% of rows."""
    task = read_task_binary(_fixture())
    data = json.loads(task.text("tests/verifier_data.json"))
    schema = data["schema"]
    schema["additionalProperties"] = "false"
    data["schema"] = schema
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.CONVERTED
    converted_task = read_task_binary(record.task_binary)
    converted_schema = json.loads(converted_task.text("tests/schema.json"))
    assert converted_schema["additionalProperties"] is False


def test_null_metadata_field_does_not_break_task_toml_rendering():
    """``schema_fields_count`` is JSON ``null`` on ~40% of this source's rows; ``render_task_toml``
    hands the metadata straight to ``tomlkit``, which has no TOML representation for ``None``."""
    record = _convert(_mutate_metadata(schema_fields_count=None))
    assert record.status == ConvertStatus.CONVERTED
    task = read_task_binary(record.task_binary)
    assert "schema_fields_count" not in task.text("task.toml")


def test_schema_failing_metaschema_check_is_rejected_as_unsupported_variant():
    broken_schema = {"type": "object", "properties": {"x": {"type": "string"}}, "exclusiveMinimum": True}
    record = _convert(_mutate_verifier_data(schema=broken_schema))
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_dockerfile_edit_is_idempotent_across_tasks():
    index = converter_index()
    a = convert_one(_info(), "a.tar.gz", _fixture(), index, TOOL_REF)
    b = convert_one(_info(), "b.tar.gz", _fixture(), index, TOOL_REF)
    assert a.dockerfile_id == b.dockerfile_id
    assert TaskFiles(read_task_binary(a.task_binary).files).text(DOCKERFILE) == read_task_binary(b.task_binary).text(
        DOCKERFILE
    )
