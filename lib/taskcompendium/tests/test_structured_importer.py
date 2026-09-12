# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contracts for TaskTrove structured-output import."""

import io
import json
import tarfile
from pathlib import Path

from taskcompendium.importers.tasktrove import read_archive
from taskcompendium.importers.tasktrove_structured import import_task
from taskcompendium.models import Embedded, NoEnvironment, Rejected, ResourceRole

FIXTURES = Path(__file__).parent / "fixtures/structured"


def _archive(schema_type: str, verifier: str, schema: bytes | None = b'{"type":"object","required":["answer"]}'):
    instruction = (
        "You will produce a structured response. Write your final answer to `/app/answer.txt`.\n\n"
        f"Emit a {schema_type.upper()} document representing the task.\n\n"
        "Given the following text:\n\nPlease answer the question.\n\n"
        "## Submitting your answer (IMPORTANT)\n\nWrite the file."
    )
    metadata = (
        f'[metadata]\nconverter = "nemotron_structured_outputs"\nschema_type = "{schema_type}"\n'
        f'tags = ["structured-outputs", "{schema_type}"]\n'
    )
    files = {
        "instruction.md": instruction.encode(),
        "task.toml": metadata.encode(),
        "tests/verifier.toml": verifier.encode(),
    }
    if schema is not None:
        files["tests/schema.json"] = schema
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return read_archive(output.getvalue(), "1", "other")


def test_json_schema_import_keeps_schema_verifier_only():
    result = import_task(_archive("json", 'mode = "json-schema"\nschema = "schema.json"\nformat = "json"\n'))

    assert not isinstance(result, Rejected)
    assert isinstance(result.environment, NoEnvironment)
    assert result.answer_requirements.kind == "json"
    assert result.resources[0].roles == (ResourceRole.VERIFIER,)
    assert isinstance(result.resources[0].content, Embedded)
    assert "/app/answer.txt" not in result.instructions


def test_xml_elements_import_preserves_structural_verifier():
    result = import_task(_archive("xml", 'mode = "xml-elements"\nrequired = ["answer"]\n'))

    assert not isinstance(result, Rejected)
    assert result.answer_requirements.kind == "xml"
    assert result.verifier.mode.value == "xml-elements"
    assert not result.resources


def test_yaml_variant_is_rejected_instead_of_claiming_json_requirements():
    result = import_task(_archive("yaml", 'mode = "json-schema"\nschema = "schema.json"\nformat = "yaml"\n'))

    assert isinstance(result, Rejected)
    assert result.reason.value == "unsupported_verifier"


def test_real_json_archive_preserves_schema_and_source_contract():
    archive = read_archive((FIXTURES / "json-row-16636.tar.gz").read_bytes(), "16636", "other")

    result = import_task(archive)

    assert not isinstance(result, Rejected)
    assert result.metadata.source.row == "16636"
    assert result.verifier.parameters["schema"] == "schema.json"
    assert result.verifier.parameters["format"].value == "json"
    assert "verifier" not in result.instructions.lower()
    assert result.instructions.startswith("Emit a JSON document that validates against the JSON Schema")
    assert json.loads(result.resources[0].content.data) == json.loads(_archive_schema("json-row-16636.tar.gz"))


def _archive_schema(name: str) -> bytes:
    with tarfile.open(FIXTURES / name, "r:gz") as archive:
        member = archive.extractfile("tests/schema.json")
        assert member is not None
        return member.read()


def test_real_json_archive_with_conflicting_source_guidance_is_rejected():
    archive = read_archive((FIXTURES / "json-row-16635.tar.gz").read_bytes(), "16635", "other")

    result = import_task(archive)

    assert isinstance(result, Rejected)
    assert result.reason.value == "unrecoverable_source"
    assert "quoted values" in result.detail


def test_real_xml_archive_preserves_required_element_verifier():
    archive = read_archive((FIXTURES / "xml-row-16634.tar.gz").read_bytes(), "16634", "other")

    result = import_task(archive)

    assert not isinstance(result, Rejected)
    assert "verifier" not in result.instructions.lower()
    assert "Include every top-level required field" in result.instructions
    assert result.metadata.source.row == "16634"
    assert result.answer_requirements.kind == "xml"
    assert result.verifier.parameters["required"] == (
        "aircraft_name",
        "role",
        "manufacturer",
        "designer",
        "first_flight_year",
        "number_built",
    )
    assert "## Submitting your answer" not in result.instructions


def test_malformed_json_schema_is_rejected_instead_of_exported():
    result = import_task(
        _archive("json", 'mode = "json-schema"\nschema = "schema.json"\nformat = "json"\n', b'{"type": 3}')
    )

    assert isinstance(result, Rejected)
    assert result.reason.value == "broken_grader"
    assert "metaschema" in result.detail


def test_json_schema_with_no_constraints_is_rejected_as_null_grader():
    result = import_task(
        _archive("json", 'mode = "json-schema"\nschema = "schema.json"\nformat = "json"\n', b'{"type":"object"}')
    )

    assert isinstance(result, Rejected)
    assert result.reason.value == "null_answer_passes"


def test_json_schema_yaml_format_is_rejected_even_when_mode_matches():
    result = import_task(_archive("json", 'mode = "json-schema"\nschema = "schema.json"\nformat = "yaml"\n'))

    assert isinstance(result, Rejected)
    assert result.reason.value == "unsupported_verifier"
    assert "format is not json" in result.detail


def test_xml_verifier_without_required_names_is_rejected_as_null_grader():
    result = import_task(_archive("xml", 'mode = "xml-elements"\nrequired = []\nany_of = []\n'))

    assert isinstance(result, Rejected)
    assert result.reason.value == "null_answer_passes"
