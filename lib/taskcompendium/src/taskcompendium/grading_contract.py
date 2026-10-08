# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed evidence and the bridge to shared verifier candidate contracts."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import ConfigDict, JsonValue, TypeAdapter
from verifyit.candidate import candidate_spec
from verifyit.grade import InvalidTask
from verifyit.json_objects import unique_object
from verifyit.spec import ExactSpec, Mode, PredictedActionSpec, Spec, StructuredExactSpec, spec_from_table

from taskcompendium.models import (
    AssistantToolCalls,
    ConversationTrace,
    TextMessage,
    VerifierSpec,
)


@dataclass(frozen=True)
class TextSubmission:
    value: str


@dataclass(frozen=True)
class ActionSubmission:
    message: TextMessage | AssistantToolCalls


@dataclass(frozen=True)
class JsonSubmission:
    value: JsonValue


@dataclass(frozen=True)
class StateSubmission:
    value: JsonValue


@dataclass(frozen=True)
class GradingAttempt:
    """Captured trial evidence; state absence is distinct from captured JSON null."""

    conversation: ConversationTrace
    files: Mapping[str, bytes] = field(default_factory=dict)
    state: StateSubmission | None = None


JSON_VALUE = TypeAdapter(JsonValue, config=ConfigDict(strict=True, allow_inf_nan=False))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def decode_json_value(text: str) -> JsonValue:
    """Decode finite JSON evidence with unique object keys at every nesting level."""
    value = json.loads(text, object_pairs_hook=unique_object, parse_constant=_reject_json_constant)
    return JSON_VALUE.validate_python(value)


type Submission = TextSubmission | ActionSubmission | JsonSubmission | StateSubmission


class SubmissionFailure(ValueError):
    """The agent ended the interaction without a valid submission."""


# Modes the submission bridge grades in process. verifyit grades more modes in process (math,
# ifeval, ...), but this bridge still routes those through file grading.
CANDIDATE_MODES: frozenset[str] = frozenset(
    {Mode.EXACT, Mode.NUMERIC, Mode.MCQ, Mode.PREDICTED_ACTION, Mode.STRUCTURED_EXACT}
)


def supports_candidate_mode(mode: str) -> bool:
    return mode in CANDIDATE_MODES


def resolve_verifier(specification: VerifierSpec) -> Spec:
    """Read a candidate or file-based verifier without acquiring runtime evidence."""
    try:
        parameters = json.loads(specification.parameters_json, object_pairs_hook=unique_object)
        if "mode" in parameters:
            raise ValueError("Verifier parameters must not override the mode")
        if supports_candidate_mode(specification.kind):
            return candidate_spec(specification.kind, parameters)
        return spec_from_table({"mode": specification.kind, **parameters})
    except (ValueError, InvalidTask) as error:
        raise ValueError(f"Invalid {specification.kind!r} verifier parameters: {error}") from error


def accepted_submission_types(verifier: Spec) -> tuple[type[Submission], ...]:
    """Declare the evidence envelopes accepted by the TaskCompendium scoring bridge."""
    if isinstance(verifier, StructuredExactSpec):
        return (JsonSubmission, StateSubmission)
    if isinstance(verifier, PredictedActionSpec):
        return (ActionSubmission,)
    if isinstance(verifier, ExactSpec):
        return (TextSubmission, JsonSubmission, StateSubmission)
    return (TextSubmission,)
