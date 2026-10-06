# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed evidence and the bridge to shared verifier candidate contracts."""

import json
from dataclasses import dataclass

from pydantic import JsonValue
from verifyit.candidate import CandidateSpec, candidate_spec
from verifyit.grade import InvalidTask
from verifyit.json_objects import unique_object
from verifyit.spec import ExactSpec, PredictedActionSpec, StructuredExactSpec

from taskcompendium.models import (
    AssistantToolCalls,
    ConversationTrace,
    EnvironmentRequirements,
    TextMessage,
    VerifierSpec,
)


@dataclass(frozen=True)
class GradingAttempt:
    """Trial evidence available to submission conventions and verifiers."""

    conversation: ConversationTrace
    workspace: object


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


type Submission = TextSubmission | ActionSubmission | JsonSubmission | StateSubmission


class SubmissionFailure(ValueError):
    """The agent ended the interaction without a valid submission."""


def resolve_verifier(specification: VerifierSpec) -> CandidateSpec:
    """Read a shared verifier spec without any TaskCompendium registration step."""
    if specification.environment_requirements != EnvironmentRequirements():
        raise NotImplementedError("Pure verifiers cannot satisfy private environment requirements")
    try:
        return candidate_spec(
            specification.kind, json.loads(specification.parameters_json, object_pairs_hook=unique_object)
        )
    except (ValueError, InvalidTask) as error:
        raise ValueError(f"Invalid {specification.kind!r} verifier parameters: {error}") from error


def accepted_submission_types(verifier: CandidateSpec) -> tuple[type[Submission], ...]:
    """Declare the evidence envelopes accepted by the TaskCompendium scoring bridge."""
    if isinstance(verifier, StructuredExactSpec):
        return (JsonSubmission, StateSubmission)
    if isinstance(verifier, PredictedActionSpec):
        return (ActionSubmission,)
    if isinstance(verifier, ExactSpec):
        return (TextSubmission, JsonSubmission, StateSubmission)
    return (TextSubmission,)
