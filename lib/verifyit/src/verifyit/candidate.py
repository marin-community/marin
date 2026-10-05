# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure candidate grading using the standard verifier specifications.

Callers own submission extraction. These graders neither read files nor inspect a harness trace.
Spec fields that name verifier files (a JSON Schema, a reasoning-gym entry or params) resolve
against ``files``, which maps each spec-relative path to its bytes.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from verifyit.grade import InvalidTask, Reward, answer_text, numeric_tolerance, scored
from verifyit.modes.grade_csv import grade_csv_candidate
from verifyit.modes.grade_csv import validate_names as validate_csv_names
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.modes.grade_ifeval import grade_ifeval_candidate, resolve_checks
from verifyit.modes.grade_math import grade_math_candidate, grade_numeric_candidate
from verifyit.modes.grade_mcq import grade_mcq_candidate
from verifyit.modes.grade_predicted_action import validate_predicted_action
from verifyit.modes.grade_xml import grade_xml_candidate
from verifyit.modes.grade_xml import validate_names as validate_xml_names
from verifyit.spec import (
    CsvColumnsSpec,
    ExactSpec,
    IfevalSpec,
    JsonSchemaSpec,
    MathSpec,
    McqSpec,
    Mode,
    NumericSpec,
    PredictedActionSpec,
    ReasoningGymSpec,
    XmlElementsSpec,
    spec_from_table,
)

TextSpec = (
    ExactSpec
    | NumericSpec
    | McqSpec
    | MathSpec
    | JsonSchemaSpec
    | XmlElementsSpec
    | CsvColumnsSpec
    | IfevalSpec
    | ReasoningGymSpec
)
CandidateSpec = TextSpec | PredictedActionSpec

CANDIDATE_MODES = frozenset(
    {
        Mode.EXACT,
        Mode.NUMERIC,
        Mode.MCQ,
        Mode.PREDICTED_ACTION,
        Mode.MATH,
        Mode.JSON_SCHEMA,
        Mode.XML_ELEMENTS,
        Mode.CSV_COLUMNS,
        Mode.IFEVAL,
        Mode.REASONING_GYM,
    }
)
NO_FILES: Mapping[str, bytes] = MappingProxyType({})


def supports_candidate_mode(mode: str) -> bool:
    return mode in CANDIDATE_MODES


def verifier_file_text(files: Mapping[str, bytes], path: str) -> str:
    """The UTF-8 text of the verifier file a spec names. Raises ``InvalidTask`` when it is absent or not text."""
    if not isinstance(path, str) or not path:
        raise InvalidTask(f"verifier file path must be a non-empty string, got {path!r}")
    if path not in files:
        raise InvalidTask(f"verifier file not supplied: {path}")
    try:
        return files[path].decode("utf-8")
    except UnicodeDecodeError as error:
        raise InvalidTask(f"verifier file {path} is not UTF-8: {error}") from error


def candidate_spec(mode: str, parameters: dict[str, Any], *, files: Mapping[str, bytes] = NO_FILES) -> CandidateSpec:
    """Validate standard private configuration, and the verifier files it names, for an extracted candidate."""
    if not supports_candidate_mode(mode):
        raise NotImplementedError(f"No pure candidate grader for mode {mode!r}")
    if "mode" in parameters:
        raise ValueError("Candidate parameters must not override the verifier mode")
    spec = spec_from_table({"mode": mode, **parameters})
    if isinstance(spec, ExactSpec):
        grade_exact_candidate(spec, "")
    elif isinstance(spec, NumericSpec):
        numeric_tolerance(spec)
    elif isinstance(spec, McqSpec):
        grade_mcq_candidate(spec, spec.expected)
    elif isinstance(spec, PredictedActionSpec):
        validate_predicted_action(spec)
    elif isinstance(spec, MathSpec):
        grade_math_candidate(spec, "")
    elif isinstance(spec, JsonSchemaSpec):
        _json_schema(spec, files)
    elif isinstance(spec, XmlElementsSpec):
        validate_xml_names(spec)
    elif isinstance(spec, CsvColumnsSpec):
        validate_csv_names(spec)
    elif isinstance(spec, IfevalSpec):
        resolve_checks(spec.constraints)
    else:
        assert isinstance(spec, ReasoningGymSpec)
        _grade_reasoning_gym(spec, None, files)
    return spec


def grade_text_candidate(spec: TextSpec, candidate: str, *, files: Mapping[str, bytes] = NO_FILES) -> Reward:
    """Score an extracted candidate under its mode's text contract.

    ``json-schema``, ``xml-elements``, and ``csv-columns`` unwrap the first fenced code block
    before parsing. Every other mode grades the text as given; ``math`` does not select a boxed
    expression or final line.
    """
    if isinstance(spec, ExactSpec):
        return grade_exact_candidate(spec, candidate)
    if isinstance(spec, McqSpec):
        return grade_mcq_candidate(spec, candidate)
    if isinstance(spec, NumericSpec):
        try:
            value = float(candidate.strip())
        except ValueError:
            value = float("nan")
        return grade_numeric_candidate(spec, value)
    if isinstance(spec, MathSpec):
        if answer_text(spec, candidate) is None:
            # Validate the reference even when the candidate scores zero for being blank.
            grade_math_candidate(spec, "")
            return scored(0.0, reason="no_output")
        return grade_math_candidate(spec, candidate)
    if isinstance(spec, JsonSchemaSpec):
        from verifyit.modes.grade_json_schema import grade_json_schema_text  # noqa: PLC0415  # schema extra

        return grade_json_schema_text(spec, _json_schema(spec, files), candidate)
    if isinstance(spec, XmlElementsSpec):
        return grade_xml_candidate(spec, candidate)
    if isinstance(spec, CsvColumnsSpec):
        return grade_csv_candidate(spec, candidate)
    if isinstance(spec, IfevalSpec):
        return grade_ifeval_candidate(spec, candidate)
    return _grade_reasoning_gym(spec, candidate, files)


def _json_schema(spec: JsonSchemaSpec, files: Mapping[str, bytes]) -> dict:
    from verifyit.modes.grade_json_schema import parse_schema  # noqa: PLC0415  # schema extra

    return parse_schema(verifier_file_text(files, spec.schema), spec.schema)


def _grade_reasoning_gym(spec: ReasoningGymSpec, candidate: str | None, files: Mapping[str, bytes]) -> Reward:
    from verifyit.modes.grade_reasoning_gym import (  # noqa: PLC0415  # reasoning-gym extra
        grade_reasoning_gym_candidate,
        parse_entry,
        parse_params,
    )

    entry = parse_entry(verifier_file_text(files, spec.entry), spec.entry)
    params = None if spec.params is None else parse_params(verifier_file_text(files, spec.params))
    return grade_reasoning_gym_candidate(spec, entry, candidate, params=params)
