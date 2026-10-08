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
import math
import tomllib
from pathlib import Path
from typing import Any

import yaml
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

from verifyit.file_ops.read import read_text
from verifyit.grade import InvalidTask, Reward, read_output, scored
from verifyit.modes.extract import unwrap_fence
from verifyit.spec import JsonSchemaSpec, SchemaFormat


def stringify_dates(node: Any) -> Any:
    """YAML and TOML parse dates and times into objects; a schema saying ``type: string`` expects text."""
    if isinstance(node, dict):
        return {key: stringify_dates(value) for key, value in node.items()}
    if isinstance(node, list):
        return [stringify_dates(value) for value in node]
    if isinstance(node, datetime.date | datetime.time):
        return node.isoformat()
    return node


def has_nonfinite_number(node: Any) -> bool:
    """JSON numbers must be finite, including inside nested candidate values."""
    if isinstance(node, float):
        return not math.isfinite(node)
    if isinstance(node, dict):
        return any(has_nonfinite_number(value) for value in node.values())
    if isinstance(node, list):
        return any(has_nonfinite_number(value) for value in node)
    return False


def load_schema(path: Path) -> dict:
    """The JSON Schema at ``path``. Raises ``InvalidTask`` when it is absent or not a schema."""
    if not path.is_file():
        raise InvalidTask(f"schema file not found: {path}")
    return parse_schema(read_text(path), str(path))


def parse_schema(text: str, name: str) -> dict:
    """The JSON Schema in ``text``, read from the file ``name``. Raises ``InvalidTask`` for a non-schema."""
    try:
        schema = json.loads(text)
    except (ValueError, RecursionError) as error:
        raise InvalidTask(f"schema file {name} is not JSON: {error}") from error
    if not isinstance(schema, dict):
        raise InvalidTask(f"schema file {name} must hold a JSON object")
    try:
        if has_nonfinite_number(schema):
            raise InvalidTask(f"schema file {name} contains a nonfinite number")
        validator_for(schema).check_schema(schema)
    except RecursionError as error:
        raise InvalidTask(f"schema file {name} exceeds nesting limit") from error
    except SchemaError as error:
        raise InvalidTask(f"schema file {name} is not a valid JSON Schema: {error.message}") from error
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


def grade_json_schema_candidate(schema: dict, instance: Any) -> Reward:
    """Validate an already decoded candidate against a JSON Schema."""
    try:
        if has_nonfinite_number(schema):
            raise InvalidTask("schema contains a nonfinite number")
        validator_class = validator_for(schema)
        validator_class.check_schema(schema)
    except RecursionError as error:
        raise InvalidTask("schema exceeds nesting limit") from error
    except SchemaError as error:
        raise InvalidTask(f"invalid JSON Schema: {error.message}") from error
    try:
        if has_nonfinite_number(instance):
            return scored(0.0, reason="nonfinite_number")
        # pyrefly: ignore[bad-instantiation, missing-argument]  # validator_for returns a concrete validator.
        validator = validator_class(schema)
        errors = sorted(validator.iter_errors(instance), key=lambda error: [str(part) for part in error.path])
    except RecursionError:
        return scored(0.0, reason="candidate_nesting_limit")
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


def grade_json_document(schema: dict, candidate_format: SchemaFormat, text: str) -> Reward:
    """Parse a document and return the full schema verdict with diagnostics."""
    try:
        instance = parse_candidate(unwrap_fence(text), candidate_format)
    except (ValueError, yaml.YAMLError, RecursionError) as error:
        return scored(0.0, reason="parse_error", error=str(error))
    return grade_json_schema_candidate(schema, instance)


def grade(spec: JsonSchemaSpec, tests_dir: Path, workspace: Path) -> Reward:
    schema = load_schema(tests_dir / spec.schema)
    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    return grade_json_document(schema, spec.format, text)
