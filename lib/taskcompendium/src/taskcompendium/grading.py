# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade canonical submissions using the pinned source verifier contracts."""

import dataclasses
import tempfile
from pathlib import Path
from typing import Any

import tomlkit
from tasktrove_verify.grade import grade
from tasktrove_verify.modes.exact import _matches as exact_matches
from tasktrove_verify.modes.extract import extract_boxed
from tasktrove_verify.reward import Status
from tasktrove_verify.spec import (
    ExactSpec,
    GotestSpec,
    JunitSpec,
    Mode,
    PytestSpec,
    ScriptSpec,
    Spec,
    StdioSpec,
    parse_spec,
)

from taskcompendium.extraction import ExtractionError, extract
from taskcompendium.grading_paths import submission_relative
from taskcompendium.judging import JudgeClient, grade_judge_attempt
from taskcompendium.models import (
    AssistantFinal,
    ContainerRuntime,
    FileSubmission,
    FinalState,
    GradingResult,
    NoEnvironment,
    Outcome,
    Protocol,
    ResourceRole,
    TaskSpecification,
    VerifierSpec,
)
from taskcompendium.resources import contained_path, materialize

EXECUTABLE_MODES = frozenset({Mode.STDIO, Mode.PYTEST, Mode.JUNIT, Mode.GOTEST, Mode.SCRIPT})


def source_verifier(verifier: VerifierSpec) -> Spec:
    """Validate semantic fields against the pinned, typed source ontology."""
    parameters = {key: value for key, value in verifier.parameters.items() if value is not None}
    return parse_spec(tomlkit.dumps({"mode": verifier.mode.value, **parameters}))


def _candidate(
    specification: TaskSpecification, protocol: Protocol, response: str | None, workspace: Path
) -> str | None:
    submission = protocol.submission
    if isinstance(submission, FinalState):
        return None
    if isinstance(submission, AssistantFinal):
        return extract(response or "", submission.extractor)
    if isinstance(submission, FileSubmission):
        environment = specification.environment
        workdir = "/app" if isinstance(environment, NoEnvironment) else environment.workdir
        path = contained_path(
            workspace,
            submission_relative(
                submission.path,
                workdir,
                environment.additional_directories if not isinstance(environment, NoEnvironment) else (),
            ),
        )
        if not path.is_file():
            raise ExtractionError("Submission file is missing")
        return extract(path.read_text(), submission.extractor)
    raise TypeError(f"Unsupported submission: {type(submission)}")


def grade_attempt(
    specification: TaskSpecification,
    protocol: Protocol,
    response: str | None,
    workspace: Path,
    transcript: tuple[dict[str, Any], ...] = (),
    judge_client: JudgeClient | None = None,
) -> GradingResult:
    """Grade trusted answer-only modes on the host; executable modes need isolation."""
    if isinstance(specification.verifier_runtime, ContainerRuntime) or specification.verifier.mode in EXECUTABLE_MODES:
        raise ValueError("Executable verifiers require the isolated-container grading entry point")
    return _grade_attempt(specification, protocol, response, workspace, transcript, judge_client)


def _grade_attempt(
    specification: TaskSpecification,
    protocol: Protocol,
    response: str | None,
    workspace: Path,
    transcript: tuple[dict[str, Any], ...],
    judge_client: JudgeClient | None,
) -> GradingResult:
    try:
        contract = source_verifier(specification.verifier)
    except (ValueError, KeyError, TypeError) as error:
        return GradingResult(Outcome.INVALID_TASK, None, {"error": str(error)})
    try:
        candidate = _candidate(specification, protocol, response, workspace)
    except (ExtractionError, UnicodeError) as error:
        return GradingResult(Outcome.EXTRACTION_ERROR, None, {"error": str(error)})
    with tempfile.TemporaryDirectory(prefix="taskcompendium-verifier-") as temporary:
        root = Path(temporary)
        tests = root / "tests"
        materialize(specification, ResourceRole.VERIFIER, tests)
        if specification.verifier.mode == Mode.JUDGE:
            return grade_judge_attempt(
                specification, contract, candidate or "", workspace, tests, transcript, judge_client
            )
        if isinstance(contract, ExactSpec):
            # The upstream grader combines correctness with a boxed-answer fallback.
            # Reuse its pinned comparison directly, after our explicit extraction.
            if not contract.expected:
                return GradingResult(Outcome.INVALID_TASK, None, {"error": "No exact-match reference"})
            return GradingResult(Outcome.GRADED, float(exact_matches(candidate or "", contract)))
        if candidate is not None:
            if specification.verifier.mode == Mode.MCQ:
                letter = candidate.strip()
                if len(letter) != 1 or not letter.isascii() or not letter.isalpha():
                    return GradingResult(Outcome.GRADED, 0.0, {"reason": "invalid_choice"})
                candidate = f"Answer: {letter}"
            elif specification.verifier.mode in {Mode.MATH, Mode.NUMERIC}:
                # The existing grader reads boxed values. This wrapper carries the
                # entire extracted candidate, without choosing a line or number.
                wrapped = f"\\boxed{{{candidate.strip()}}}"
                if extract_boxed(wrapped) != candidate.strip():
                    return GradingResult(
                        Outcome.EXTRACTION_ERROR,
                        None,
                        {"error": "Candidate contains conflicting math submission wrappers"},
                    )
                candidate = wrapped
            if isinstance(contract, StdioSpec | PytestSpec | JunitSpec | GotestSpec | ScriptSpec):
                return GradingResult(Outcome.INVALID_TASK, None, {"error": "Executable verifier requires final state"})
            output = root / "candidate.txt"
            output.write_text(candidate)
            contract = dataclasses.replace(contract, output=str(output))
        elif isinstance(contract, StdioSpec | PytestSpec | JunitSpec | GotestSpec | ScriptSpec):
            contract = dataclasses.replace(contract, workspace=str(workspace))
        try:
            result = grade(contract, tests, workspace)
        except Exception as error:
            return GradingResult(Outcome.INFRA_ERROR, None, {"error": f"{type(error).__name__}: {error}"})
        status = {
            Status.SCORED: Outcome.GRADED,
            Status.INVALID_TASK: Outcome.INVALID_TASK,
            Status.INFRA_ERROR: Outcome.INFRA_ERROR,
        }[result.status]
        return GradingResult(status, result.reward if status == Outcome.GRADED else None, result.detail)
