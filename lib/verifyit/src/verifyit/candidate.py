# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade an already extracted answer in process, without a sandbox or answer file.

Callers own extraction: a text mode receives the answer text, ``structured_exact`` a decoded JSON
value and ``predicted_action`` the final function calls. Files a specification names, such as a
JSON Schema, come from ``resources`` keyed by their path relative to the tests directory. Modes
that execute code, call a model, or need a dataset package run only through the file grader.
"""

import importlib
from collections.abc import Mapping
from functools import cache
from pathlib import PurePosixPath
from types import ModuleType
from typing import Any

from verifyit.grade import (
    MODE_MODULES,
    GradingInfraError,
    InvalidTask,
    Reward,
    empty_output_policy,
    infra_error,
    invalid_task,
    numeric_tolerance,
    scored,
)
from verifyit.json_comparison import JsonValue
from verifyit.numeric import NumericCandidateError, extract_numeric_candidate
from verifyit.spec import (
    CsvColumnsSpec,
    EmptyOutputPolicy,
    ExactSpec,
    FunctionCall,
    IfevalSpec,
    JsonSchemaSpec,
    MathSpec,
    McqSpec,
    Mode,
    NumericSpec,
    PredictedActionSpec,
    SchemaFormat,
    Spec,
    StructuredExactSpec,
    XmlElementsSpec,
    mode_of,
    spec_from_table,
)

IN_PROCESS_MODES: frozenset[Mode] = frozenset(
    {
        Mode.EXACT,
        Mode.NUMERIC,
        Mode.MCQ,
        Mode.MATH,
        Mode.IFEVAL,
        Mode.JSON_SCHEMA,
        Mode.XML_ELEMENTS,
        Mode.CSV_COLUMNS,
        Mode.STRUCTURED_EXACT,
        Mode.PREDICTED_ACTION,
    }
)

Candidate = str | JsonValue | tuple[FunctionCall, ...]


def supports_in_process(mode: str) -> bool:
    return mode in IN_PROCESS_MODES


@cache
def _mode_module(mode: Mode) -> ModuleType:
    # Several modes depend on optional extras, so their modules load on first use.
    return importlib.import_module(f"verifyit.modes.{MODE_MODULES[mode]}")


def _validate(spec: Spec) -> None:
    module = _mode_module(mode_of(spec))
    match spec:
        case ExactSpec():
            module.grade_exact_candidate(spec, "")
        case NumericSpec():
            empty_output_policy(spec)
            numeric_tolerance(spec)
        case McqSpec():
            module.grade_mcq_candidate(spec, spec.expected)
        case MathSpec():
            module.validate_math(spec)
        case IfevalSpec():
            empty_output_policy(spec)
            module.resolve_checks(spec.constraints)
        case JsonSchemaSpec():
            empty_output_policy(spec)
            schema = PurePosixPath(spec.schema)
            if not isinstance(spec.format, SchemaFormat):
                raise InvalidTask("json-schema format must be json, yaml or toml")
            if not spec.schema or schema.is_absolute() or ".." in schema.parts:
                raise InvalidTask("json-schema schema must be a path under the tests directory")
        case XmlElementsSpec():
            module.validate_xml_elements(spec)
        case CsvColumnsSpec():
            module.validate_csv_columns(spec)
        case StructuredExactSpec():
            module.validate_structured_exact(spec)
        case PredictedActionSpec():
            module.validate_predicted_action(spec)


def candidate_spec(mode: str, parameters: dict[str, Any]) -> Spec:
    """Build and validate the specification of an in-process mode.

    Raises ``ValueError`` for a mode without an in-process grader or a malformed table, and
    ``InvalidTask`` for a well-formed table the mode rejects.
    """
    if not supports_in_process(mode):
        raise ValueError(f"Mode {mode!r} has no in-process grader")
    if "mode" in parameters:
        raise ValueError("Verifier parameters must not override the mode")
    spec = spec_from_table({"mode": mode, **parameters})
    _validate(spec)
    return spec


def _grade(spec: Spec, candidate: Candidate, resources: Mapping[str, bytes]) -> Reward:
    module = _mode_module(mode_of(spec))
    if isinstance(spec, StructuredExactSpec):
        return module.grade_structured_exact_candidate(spec, candidate)
    if isinstance(spec, PredictedActionSpec):
        if not isinstance(candidate, tuple) or not all(isinstance(call, FunctionCall) for call in candidate):
            raise TypeError("predicted_action grades a tuple of function calls")
        return module.grade_predicted_action_candidate(spec, candidate)
    if not isinstance(candidate, str):
        raise TypeError(f"Mode {mode_of(spec)} grades text, got {type(candidate).__name__}")
    text = candidate
    if not text.strip() and empty_output_policy(spec) is EmptyOutputPolicy.ZERO:
        return scored(0.0, reason="empty_output")
    match spec:
        case ExactSpec():
            return module.grade_exact_candidate(spec, text)
        case NumericSpec():
            try:
                value = extract_numeric_candidate(text)
            except NumericCandidateError as error:
                return scored(0.0, reason="invalid_numeric_candidate", error=str(error))
            return module.grade_numeric_candidate(spec, value)
        case McqSpec():
            return module.grade_mcq_candidate(spec, text)
        case MathSpec():
            return module.grade_math_candidate(spec, module.math_answer(spec, text))
        case IfevalSpec():
            return module.grade_ifeval_candidate(spec, text)
        case JsonSchemaSpec():
            if spec.schema not in resources:
                raise InvalidTask(f"schema file not found: {spec.schema}")
            try:
                schema_text = resources[spec.schema].decode()
            except UnicodeDecodeError as error:
                raise InvalidTask(f"schema file {spec.schema} is not UTF-8") from error
            return module.grade_json_document(module.parse_schema(schema_text, spec.schema), spec.format, text)
        case XmlElementsSpec():
            return module.grade_xml_candidate(spec, text)
        case CsvColumnsSpec():
            return module.grade_csv_candidate(spec, text)
    raise AssertionError(f"Unhandled in-process mode {mode_of(spec)}")


def grade_candidate(spec: Spec, candidate: Candidate, resources: Mapping[str, bytes]) -> Reward:
    """Score one extracted answer.

    A specification the mode rejects yields an invalid-task reward and an exhausted backend deadline
    an infrastructure-error reward. Raises ``ValueError`` for a mode without an in-process grader
    and ``TypeError`` for a candidate of the wrong shape for its mode.
    """
    if not supports_in_process(mode_of(spec)):
        raise ValueError(f"Mode {mode_of(spec)} has no in-process grader")
    try:
        _validate(spec)
        return _grade(spec, candidate, resources)
    except InvalidTask as error:
        return invalid_task(str(error))
    except GradingInfraError as error:
        return infra_error(str(error), **error.detail)
