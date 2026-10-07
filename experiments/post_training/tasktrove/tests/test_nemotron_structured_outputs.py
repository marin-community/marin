# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the ``nemotron_structured_outputs`` exemplar."""

import json
import tempfile
from pathlib import Path

import pytest
from verifyit.grade import Status, grade
from verifyit.spec import CsvColumnsSpec, JsonSchemaSpec, SchemaFormat, XmlElementsSpec, parse_spec

from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.dataset import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.task_format import INSTALL_MARKER, VERIFIER_TOML, VERIFY_TEST_SH
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    TEST_SH,
    read_task_binary,
    write_task_binary,
)
from experiments.post_training.tasktrove.verify import verify_task

pytest_plugins = ["lib.verifyit.tests.test_judge"]

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


def test_exemplar_converts_to_grounded_script_with_json_schema_gate():
    record = _convert()
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "nemotron_structured_outputs"
    assert record.mode == "script"
    assert record.tags == ["structured-outputs", "grounded", "script", "nemotron", "json"]

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text("tests/format/verifier.toml"))
    assert isinstance(spec, JsonSchemaSpec)
    assert spec.schema == "schema.json"
    assert "tests/format/schema.json" in task.files
    schema = json.loads(task.text("tests/format/schema.json"))
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
    spec = parse_spec(task.text("tests/format/verifier.toml"))
    assert isinstance(spec, JsonSchemaSpec)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / spec.schema).write_bytes(task.files[f"tests/format/{spec.schema}"])
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
    spec = parse_spec(task.text("tests/format/verifier.toml"))
    assert isinstance(spec, JsonSchemaSpec)
    assert spec.format.value == "yaml"
    assert record.tags == ["structured-outputs", "grounded", "script", "nemotron", "yaml"]


def test_xml_schema_type_grades_required_element_or_attribute_names():
    record = _convert(_mutate_verifier_data(schema_type="xml"))
    assert record.status == ConvertStatus.CONVERTED
    assert record.mode == "script"
    assert record.tags == ["structured-outputs", "grounded", "script", "nemotron", "xml"]

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text("tests/format/verifier.toml"))
    assert isinstance(spec, XmlElementsSpec)
    assert spec.required == tuple(_VALID_CANDIDATE)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        task.write_to(root)
        workspace = root / "app"
        workspace.mkdir()
        answer = workspace / "answer.txt"

        answer.write_text(
            '<pairing dishName="Grilled Salmon"><wineType>White</wineType><flavorProfile />'
            "<acidityLevel>Medium</acidityLevel><tanninLevel>Low</tanninLevel>"
            "<servingTemperature>12</servingTemperature><isRecommended>true</isRecommended></pairing>"
        )
        reward = grade(spec, root / "tests/format", workspace)
        assert reward.status == Status.SCORED and reward.reward == 1.0

        answer.write_text("<pairing><dishName>Grilled Salmon</dishName></pairing>")
        reward = grade(spec, root / "tests/format", workspace)
        assert reward.status == Status.SCORED and reward.reward == 0.0


def test_csv_schema_type_grades_required_scalar_columns():
    record = _convert(_mutate_verifier_data(schema_type="csv"))
    assert record.status == ConvertStatus.CONVERTED
    assert record.mode == "script"
    assert record.tags == ["structured-outputs", "grounded", "script", "nemotron", "csv"]

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text("tests/format/verifier.toml"))
    assert isinstance(spec, CsvColumnsSpec)
    assert spec.required == (
        "dishName",
        "wineType",
        "acidityLevel",
        "tanninLevel",
        "servingTemperature",
        "isRecommended",
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        task.write_to(root)
        workspace = root / "app"
        workspace.mkdir()
        answer = workspace / "answer.txt"

        answer.write_text(
            "dishName,wineType,acidityLevel,tanninLevel,servingTemperature,isRecommended\n"
            "Grilled Salmon,White,Medium,Low,12,true\n"
        )
        reward = grade(spec, root / "tests/format", workspace)
        assert reward.status == Status.SCORED and reward.reward == 1.0

        answer.write_text("dishName,wineType\nGrilled Salmon,White\n")
        reward = grade(spec, root / "tests/format", workspace)
        assert reward.status == Status.SCORED and reward.reward == 0.0


def test_toml_schema_type_grades_parsed_document_against_json_schema():
    record = _convert(_mutate_verifier_data(schema_type="toml"))
    assert record.status == ConvertStatus.CONVERTED
    assert record.mode == "script"
    assert record.tags == ["structured-outputs", "grounded", "script", "nemotron", "toml"]

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text("tests/format/verifier.toml"))
    assert isinstance(spec, JsonSchemaSpec)
    assert spec.format is SchemaFormat.TOML
    assert "tests/format/schema.json" in task.files

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        task.write_to(root)
        workspace = root / "app"
        workspace.mkdir()
        answer = workspace / "answer.txt"

        answer.write_text(
            'dishName = "Grilled Salmon"\nwineType = "White"\nacidityLevel = "Medium"\n'
            'tanninLevel = "Low"\nservingTemperature = 12\nisRecommended = true\n'
            '[flavorProfile]\nprimary = "citrus"\nsecondary = "green apple"\n'
        )
        reward = grade(spec, root / "tests/format", workspace)
        assert reward.status == Status.SCORED and reward.reward == 1.0

        answer.write_text('dishName = "Grilled Salmon"\nwineType = 42\n')
        reward = grade(spec, root / "tests/format", workspace)
        assert reward.status == Status.SCORED and reward.reward == 0.0


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
    converted_schema = json.loads(converted_task.text("tests/format/schema.json"))
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
    converted_schema = json.loads(converted_task.text("tests/format/schema.json"))
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


GROUNDING_CANDIDATES = {
    "json": ('{"fullName":"Enrique Lora","birth":"1970"}', '{"fullName":"fabricated","birth":"fabricated"}'),
    "yaml": ('fullName: Enrique Lora\nbirth: "1970"\n', "fullName: fabricated\nbirth: fabricated\n"),
    "toml": ('fullName = "Enrique Lora"\nbirth = "1970"\n', 'fullName = "fabricated"\nbirth = "fabricated"\n'),
    "xml": (
        "<profile><fullName>Enrique Lora</fullName><birth>1970</birth></profile>",
        "<profile><fullName>fabricated</fullName><birth>fabricated</birth></profile>",
    ),
    "csv": ("fullName,birth\nEnrique Lora,1970\n", "fullName,birth\nfabricated,fabricated\n"),
}


def _grounded_task(tmp_path, schema_type, candidate):
    task = read_task_binary(_fixture())
    task.files["instruction.md"] = (
        f"Extract fullName and birth as {schema_type} from this source document: "
        "Enrique Lora was born in 1970. Write your answer to /app/answer.txt."
    ).encode()
    task.files["tests/verifier_data.json"] = json.dumps(
        {
            "schema_type": schema_type,
            "schema": {
                "type": "object",
                "properties": {"fullName": {"type": "string"}, "birth": {"type": "string"}},
                "required": ["fullName", "birth"],
                "additionalProperties": False,
            },
        }
    ).encode()
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.CONVERTED
    converted = read_task_binary(record.task_binary)
    converted.write_to(tmp_path)
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "answer.txt").write_text(candidate)
    return parse_spec(converted.text(VERIFIER_TOML)), workspace


@pytest.mark.parametrize("schema_type", GROUNDING_CANDIDATES)
@pytest.mark.parametrize("fabricated", [False, True])
def test_grounded_composition_uses_content_verdict_after_format_passes(tmp_path, fake_judge, schema_type, fabricated):
    candidate = GROUNDING_CANDIDATES[schema_type][int(fabricated)]
    fake_judge.replies = ["FAIL" if fabricated else "PASS"]
    spec, workspace = _grounded_task(tmp_path, schema_type, candidate)
    verdict = grade(spec, tmp_path / "tests", workspace)
    assert (verdict.status, verdict.reward) == (Status.SCORED, 0.0 if fabricated else 1.0)
    # The remote grading boundary must receive the source document, not just field names.
    assert "Enrique Lora was born in 1970." in fake_judge.prompts[0]
    assert candidate in fake_judge.prompts[0]


def test_bad_format_scores_zero_without_consulting_the_content_judge(tmp_path, fake_judge):
    spec, workspace = _grounded_task(tmp_path, "xml", "<broken")
    verdict = grade(spec, tmp_path / "tests", workspace)
    assert (verdict.status, verdict.reward) == (Status.SCORED, 0.0)
    assert fake_judge.requests == []


def test_missing_content_judge_is_an_infrastructure_error(tmp_path, unconfigured_judge):
    spec, workspace = _grounded_task(tmp_path, "csv", GROUNDING_CANDIDATES["csv"][0])
    verdict = grade(spec, tmp_path / "tests", workspace)
    assert (verdict.status, verdict.reward) == (Status.INFRA_ERROR, 0.0)
