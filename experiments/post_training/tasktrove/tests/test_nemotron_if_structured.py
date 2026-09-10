# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the ``nemotron_if_structured`` exemplar."""

import json
from pathlib import Path

from tasktrove_verify.spec import JsonSchemaSpec, parse_spec

from experiments.post_training.tasktrove.contract import INSTALL_MARKER, VERIFIER_TOML, VERIFY_TEST_SH
from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.sources import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    TEST_SH,
    read_task_binary,
    write_task_binary,
)
from experiments.post_training.tasktrove.verify import verify_task

FIXTURES = Path(__file__).parents[1] / "fixtures"
TOOL_REF = "0123abc"

# One event from the calendar source's verifier_data.json.
CALENDAR_EVENTS = {
    "0": {
        "constraint": "between 11:45am and 3:45pm",
        "duration": 45,
        "event_id": 0,
        "event_name": "Competitive Gaming Strategy Meeting",
        "max_time": "16:00",
        "min_time": "10:00",
    }
}


def _fixture() -> bytes:
    return (FIXTURES / "nemotron_if_structured.tar.gz").read_bytes()


def _info(source: str = "laion__nemotron-gym-instruction-following-structured-v3") -> SourceInfo:
    return SourceInfo(source, SourceVerdict.KEEP, "instruction-following", "")


def test_structured_exemplar_converts_to_json_schema_spec_with_expected_tags():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "nemotron_if_structured"
    assert record.mode == "json-schema"
    assert record.tags == ["instruction-following", "structured-output", "json-schema", "nemotron"]

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, JsonSchemaSpec) and spec.schema == "schema.json"
    schema = json.loads(task.text("tests/schema.json"))
    assert schema["type"] == "object" and "printerModel" in schema["properties"]

    assert task.text(TEST_SH) == VERIFY_TEST_SH
    assert "tests/verifier.py" not in task.files, "old grader code must not ship"
    assert "tests/validate_verifier_data.py" not in task.files, "old grader code must not ship"
    assert "tests/verifier_data.json" not in task.files, "raw grader data must not ship"

    dockerfile = task.text(DOCKERFILE)
    assert dockerfile.startswith("# DO NOT EDIT") and INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile
    assert record.has_solution is False and record.solution_binary is None


def test_structured_exemplar_passes_verification():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert verify_task(record.task_binary) is None


def test_calendar_shaped_task_goes_through_the_calendar_checker():
    task = read_task_binary(_fixture())
    task.files["tests/verifier_data.json"] = json.dumps({"expected_events": CALENDAR_EVENTS}).encode()
    record = convert_one(
        _info("laion__nemotron-gym-instruction-following-calendar-v3"),
        "t.tar.gz",
        write_task_binary(task),
        converter_index(),
        TOOL_REF,
    )
    assert record.status == ConvertStatus.CONVERTED and record.mode == "script"
    assert "calendar" in record.tags and "tests/agent_calendar_checker.py" in read_task_binary(record.task_binary).files


def test_schema_failing_metaschema_check_is_rejected_as_null_grader():
    task = read_task_binary(_fixture())
    data = json.loads(task.text("tests/verifier_data.json"))
    data["schema"]["properties"]["printSpeed"]["minimum"] = True  # not a number: invalid metaschema
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_missing_schema_is_rejected_as_null_grader():
    task = read_task_binary(_fixture())
    task.files["tests/verifier_data.json"] = json.dumps({"schema_type": "json"}).encode()
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None
