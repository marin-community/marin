# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode json-schema: the answer file must parse and then validate against a JSON Schema under tests/.

``format`` says how the candidate is read: as JSON, YAML or TOML. The schema is task data, so a
missing file or one the metaschema rejects raises ``InvalidTask``. Everything the candidate
controls -- a document that does not parse, a constraint it violates -- scores 0.0 with the first
validation error in the detail.
"""

import datetime
import json
import tomllib
from pathlib import Path
from typing import Any

import yaml
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

from tasktrove_verify.modes.extract import unwrap_fence
from tasktrove_verify.grade import read_output
from tasktrove_verify.grade import InvalidTask, Reward, scored
from tasktrove_verify.spec import JsonSchemaSpec, SchemaFormat


def stringify_dates(node: Any) -> Any:
    """YAML and TOML parse dates and times into objects; a schema saying ``type: string`` expects text."""
    if isinstance(node, dict):
        return {key: stringify_dates(value) for key, value in node.items()}
    if isinstance(node, list):
        return [stringify_dates(value) for value in node]
    if isinstance(node, datetime.date | datetime.time):
        return node.isoformat()
    return node


def load_schema(path: Path) -> dict:
    """The JSON Schema at ``path``. Raises ``InvalidTask`` when it is absent or not a schema."""
    if not path.is_file():
        raise InvalidTask(f"schema file not found: {path}")
    try:
        schema = json.loads(path.read_text())
    except ValueError as error:
        raise InvalidTask(f"schema file {path} is not JSON: {error}") from error
    if not isinstance(schema, dict):
        raise InvalidTask(f"schema file {path} must hold a JSON object")
    try:
        validator_for(schema).check_schema(schema)
    except SchemaError as error:
        raise InvalidTask(f"schema file {path} is not a valid JSON Schema: {error.message}") from error
    return schema


def parse_candidate(text: str, candidate_format: SchemaFormat) -> Any:
    """The candidate document. Raises ``ValueError`` or ``yaml.YAMLError`` when it does not parse."""
    if candidate_format is SchemaFormat.JSON:
        return json.loads(text)
    if candidate_format is SchemaFormat.TOML:
        return stringify_dates(tomllib.loads(text))
    document = yaml.safe_load(text)
    if document is None:
        raise ValueError("YAML document is empty")
    return stringify_dates(document)


def grade(spec: JsonSchemaSpec, tests_dir: Path, workspace: Path) -> Reward:
    schema = load_schema(tests_dir / spec.schema)
    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    try:
        instance = parse_candidate(unwrap_fence(text), spec.format)
    except (ValueError, yaml.YAMLError) as error:
        return scored(0.0, reason="parse_error", error=str(error))

    validator_class = validator_for(schema)
    # pyrefly: ignore[bad-instantiation, missing-argument]  # validator_for returns a concrete
    # validator class; jsonschema types it as the Validator protocol.
    validator = validator_class(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda error: [str(part) for part in error.path])
    if not errors:
        return scored(1.0, reason="valid")
    first = errors[0]
    return scored(
        0.0,
        reason="schema_violation",
        errors=len(errors),
        path="/".join(str(part) for part in first.path),
        error=first.message,
    )
