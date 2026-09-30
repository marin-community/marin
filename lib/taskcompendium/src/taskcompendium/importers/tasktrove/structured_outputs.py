# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import structured XML and CSV tasks from TaskTrove Clean."""

import hashlib
import json
import tomllib

from tasktrove_verify.spec import CsvColumnsSpec, XmlElementsSpec, parse_spec

from taskcompendium.importers.tasktrove.convert import INSTRUCTION_FILE, METADATA_TABLE, TASK_MANIFEST, VERIFIER_FILE
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.verifiers.csv_columns import csv_columns_answer
from taskcompendium.verifiers.xml_elements import xml_elements_answer

FAMILY = "other"
CONVERTER = "nemotron_structured_outputs"
XML_MODE = "xml-elements"
CSV_MODE = "csv-columns"
_PREFIX_INTRO = "You will produce a structured response. Write your final answer to `/app/answer.txt`."
_SUBMISSION_HEADING = "\n## Submitting your answer (IMPORTANT)\n"
_XML_REQUEST = (
    "Create one well-formed XML document that follows the provided JSON Schema and represents the requested data. "
    "Include each top-level required field as an element or attribute."
)
_CSV_REQUEST = (
    "Create CSV data following the provided schema, with a header row and at least one data row. "
    "Include each top-level required scalar field as a column header, without Markdown fences."
)


def _clean_instructions(instructions: str, mode: str, request: str) -> str:
    """Replace a recognized file-output wrapper while keeping the source request body."""
    scaffold, separator, remainder = instructions.partition("\n---\n\n")
    suffix_start = remainder.rfind(_SUBMISSION_HEADING)
    if not separator or suffix_start < 0 or not scaffold.startswith(_PREFIX_INTRO):
        raise ValueError("Unsupported structured-output submission scaffold")
    request_marker = "XML document" if mode == "xml-elements" else "CSV data"
    if request_marker not in scaffold or "verifier checks" not in scaffold.lower():
        raise ValueError("Unsupported structured-output submission scaffold")
    suffix = remainder[suffix_start:]
    answer_marker = "xml document" if mode == "xml-elements" else "csv data"
    if answer_marker not in suffix.lower() or "/app/answer.txt" not in suffix or "An empty or missing" not in suffix:
        raise ValueError("Unsupported structured-output submission scaffold")
    body = remainder[:suffix_start]
    if not body.strip():
        raise ValueError("Structured-output task prompt is empty")
    return f"{request}\n\n{body}"


def _source_metadata(archive: TaskArchive, mode: str) -> tuple[str, ...]:
    metadata = tomllib.loads(archive.files[TASK_MANIFEST].decode())[METADATA_TABLE]
    if metadata.get("family") != FAMILY or metadata.get("converter") != CONVERTER or metadata.get("mode") != mode:
        raise ValueError(f"Unsupported TaskTrove {mode} source")
    tags = metadata.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ValueError("TaskTrove tags must be an ordered list of strings")
    return tuple(tags)


def _specification(archive: TaskArchive, tags: tuple[str, ...], prompt: str, verifier: VerifierSpec) -> TaskSpec:
    identity = json.dumps(
        (archive.source.dataset, archive.source.revision, archive.source.row),
        separators=(",", ":"),
    )
    return TaskSpec(
        id=f"tasktrove-{hashlib.sha256(identity.encode()).hexdigest()}",
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=verifier,
        source=archive.source,
        tags=tags,
    )


def import_xml_task(archive: TaskArchive) -> TaskSpec:
    """Import one XML structured-output task."""
    try:
        tags = _source_metadata(archive, XML_MODE)
        contract = parse_spec(archive.files[VERIFIER_FILE].decode())
        if not isinstance(contract, XmlElementsSpec):
            raise ValueError("TaskTrove XML archive must declare an XML-elements verifier")
        prompt = _clean_instructions(archive.files[INSTRUCTION_FILE].decode(), XML_MODE, _XML_REQUEST)
        verifier = xml_elements_answer(contract.required, contract.any_of)
        return _specification(archive, tags, prompt, verifier)
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove XML archive: {error}") from error


def import_csv_task(archive: TaskArchive) -> TaskSpec:
    """Import one CSV structured-output task."""
    try:
        tags = _source_metadata(archive, CSV_MODE)
        contract = parse_spec(archive.files[VERIFIER_FILE].decode())
        if not isinstance(contract, CsvColumnsSpec):
            raise ValueError("TaskTrove CSV archive must declare a CSV-columns verifier")
        prompt = _clean_instructions(archive.files[INSTRUCTION_FILE].decode(), CSV_MODE, _CSV_REQUEST)
        verifier = csv_columns_answer(contract.required, contract.any_of)
        return _specification(archive, tags, prompt, verifier)
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove CSV archive: {error}") from error
