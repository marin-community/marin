# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade an already extracted answer in process, without a sandbox or answer file.

Callers own extraction: a text mode receives the answer text, ``structured_exact`` a decoded JSON
value and ``predicted_action`` the final function calls. Files a specification names, such as a
JSON Schema, come from ``resources`` keyed by their path relative to the tests directory. Modes
that execute code, call a model, or need a dataset package run only through the file grader.
"""

from collections.abc import Mapping
from typing import Any

from verifyit.grade import (
    GradingInfraError,
    InvalidTask,
    Reward,
    empty_output_policy,
    infra_error,
    invalid_task,
    mode_module,
    scored,
)
from verifyit.json_comparison import JsonValue
from verifyit.spec import (
    MODE_DESCRIPTORS,
    EmptyOutputPolicy,
    FunctionCall,
    JsonSchemaSpec,
    Mode,
    PredictedActionSpec,
    Spec,
    StructuredExactSpec,
    mode_of,
    spec_from_table,
)

IN_PROCESS_MODES: frozenset[Mode] = frozenset(
    mode for mode, descriptor in MODE_DESCRIPTORS.items() if descriptor.candidate is not None
)

Candidate = str | JsonValue | tuple[FunctionCall, ...]


def supports_in_process(mode: str) -> bool:
    return mode in IN_PROCESS_MODES


def _validate(spec: Spec) -> None:
    mode = mode_of(spec)
    entrypoints = MODE_DESCRIPTORS[mode].candidate
    assert entrypoints is not None
    getattr(mode_module(mode), entrypoints.validate)(spec)


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
    mode = mode_of(spec)
    entrypoints = MODE_DESCRIPTORS[mode].candidate
    assert entrypoints is not None
    grader = getattr(mode_module(mode), entrypoints.grade)
    if isinstance(spec, StructuredExactSpec):
        return grader(spec, candidate)
    if isinstance(spec, PredictedActionSpec):
        if not isinstance(candidate, tuple) or not all(isinstance(call, FunctionCall) for call in candidate):
            raise TypeError("predicted_action grades a tuple of function calls")
        return grader(spec, candidate)
    if not isinstance(candidate, str):
        raise TypeError(f"Mode {mode_of(spec)} grades text, got {type(candidate).__name__}")
    text = candidate
    # The schema is part of the task, so a missing one is a task defect whatever the answer.
    if isinstance(spec, JsonSchemaSpec) and spec.schema not in resources:
        raise InvalidTask(f"schema file not found: {spec.schema}")
    if not text.strip() and empty_output_policy(spec) is EmptyOutputPolicy.ZERO:
        return scored(0.0, reason="empty_output")
    if isinstance(spec, JsonSchemaSpec):
        return grader(spec, text, resources[spec.schema])
    return grader(spec, text)


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
