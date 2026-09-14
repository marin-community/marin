# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade canonical submissions using the pinned source verifier contracts."""

import dataclasses
import tempfile
from pathlib import Path
from typing import Any

import tomlkit
from tasktrove_verify.grade import Status, grade
from tasktrove_verify.modes.extract import extract_boxed
from tasktrove_verify.modes.grade_exact import _matches as exact_matches
from tasktrove_verify.modes.grade_ifeval import resolve_checks
from tasktrove_verify.spec import (
    Constraint,
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
    EXECUTABLE_MODES,
    AssistantFinal,
    CodeAnswerVerifier,
    ConstraintVerifier,
    FileSubmission,
    FinalActionSubmission,
    FinalState,
    GradingResult,
    Outcome,
    PredictedActionVerifier,
    Rendering,
    ResourceRole,
    TaskSpecification,
    TaskTroveVerifier,
)
from taskcompendium.predicted_action import action_from_transcript, compare
from taskcompendium.resources import contained_path, materialize


def source_verifier(verifier: TaskTroveVerifier) -> Spec:
    """Validate semantic fields against the pinned, typed source ontology."""
    parameters = {key: value for key, value in verifier.parameters.items() if value is not None}
    return parse_spec(tomlkit.dumps({"mode": verifier.mode.value, **parameters}))


def _candidate(
    specification: TaskSpecification, protocol: Rendering, response: str | None, workspace: Path
) -> str | None:
    submission = protocol.submission
    if isinstance(submission, FinalState):
        return None
    if isinstance(submission, AssistantFinal):
        return extract(response or "", submission.extractor)
    if isinstance(submission, FileSubmission):
        environment = specification.requirements.state
        workdir = environment.workdir
        path = contained_path(
            workspace,
            submission_relative(
                submission.path,
                workdir,
                environment.additional_directories,
            ),
        )
        if not path.is_file():
            raise ExtractionError("Submission file is missing")
        return extract(path.read_text(), submission.extractor)
    raise TypeError(f"Unsupported submission: {type(submission)}")


def grade_attempt(
    specification: TaskSpecification,
    protocol: Rendering,
    response: str | None,
    workspace: Path,
    transcript: tuple[dict[str, Any], ...] = (),
    judge_client: JudgeClient | None = None,
    step_index: int = 0,
) -> GradingResult:
    """Grade trusted answer-only modes on the host; executable modes need isolation."""
    verifier = specification.steps[step_index].verifier
    if isinstance(verifier, PredictedActionVerifier):
        if not isinstance(protocol.submission, FinalActionSubmission):
            return GradingResult(
                Outcome.INVALID_TASK, None, {"error": "Predicted actions require final-action submission"}
            )
        reward, category = compare(verifier.expected_action, action_from_transcript(transcript), verifier.comparator)
        return GradingResult(Outcome.GRADED, reward, {"category": category})
    if isinstance(verifier, ConstraintVerifier):
        try:
            candidate = _candidate(specification, protocol, response, workspace)
        except (ExtractionError, UnicodeError) as error:
            return GradingResult(Outcome.EXTRACTION_ERROR, None, {"error": str(error)})
        try:
            checks = resolve_checks(tuple(Constraint(item.name, item.params) for item in verifier.constraints))
        except Exception as error:
            return GradingResult(Outcome.INVALID_TASK, None, {"error": f"{type(error).__name__}: {error}"})
        results: list[dict[str, Any]] = []
        for constraint, check in checks:
            try:
                passed, detail = check(candidate or "", constraint.params)
            except Exception as error:
                return GradingResult(
                    Outcome.INFRA_ERROR,
                    None,
                    {"error": f"{type(error).__name__}: {error}", "constraint": constraint.name},
                )
            results.append({"name": constraint.name, "passed": passed, "detail": detail})
        passed = sum(result["passed"] for result in results)
        reward = float(passed == len(results)) if verifier.aggregation == "binary" else passed / len(results)
        return GradingResult(Outcome.GRADED, reward, {"constraints": results})
    if isinstance(verifier, CodeAnswerVerifier) or (
        isinstance(verifier, TaskTroveVerifier) and (verifier.runtime is not None or verifier.mode in EXECUTABLE_MODES)
    ):
        raise ValueError("Executable verifiers require the isolated-container grading entry point")
    return _grade_attempt(specification, protocol, response, workspace, transcript, judge_client, step_index)


def _grade_attempt(
    specification: TaskSpecification,
    protocol: Rendering,
    response: str | None,
    workspace: Path,
    transcript: tuple[dict[str, Any], ...],
    judge_client: JudgeClient | None,
    step_index: int = 0,
) -> GradingResult:
    verifier = specification.steps[step_index].verifier
    if not isinstance(verifier, TaskTroveVerifier):
        return GradingResult(Outcome.INVALID_TASK, None, {"error": "Source verifier requires its dedicated adapter"})
    try:
        contract = source_verifier(verifier)
    except (ValueError, KeyError, TypeError) as error:
        return GradingResult(Outcome.INVALID_TASK, None, {"error": str(error)})
    try:
        candidate = _candidate(specification, protocol, response, workspace)
    except (ExtractionError, UnicodeError) as error:
        return GradingResult(Outcome.EXTRACTION_ERROR, None, {"error": str(error)})
    with tempfile.TemporaryDirectory(prefix="taskcompendium-verifier-") as temporary:
        root = Path(temporary)
        tests = root / "tests"
        materialize(specification, ResourceRole.VERIFIER, tests, step_index)
        if verifier.mode == Mode.JUDGE:
            return grade_judge_attempt(
                specification, contract, candidate or "", workspace, tests, transcript, judge_client, step_index
            )
        if isinstance(contract, ExactSpec):
            # The upstream grader combines correctness with a boxed-answer fallback.
            # Reuse its pinned comparison directly, after our explicit extraction.
            if not contract.expected:
                return GradingResult(Outcome.INVALID_TASK, None, {"error": "No exact-match reference"})
            return GradingResult(Outcome.GRADED, float(exact_matches(candidate or "", contract)))
        if candidate is not None:
            if verifier.mode == Mode.MCQ:
                letter = candidate.strip()
                if len(letter) != 1 or not letter.isascii() or not letter.isalpha():
                    return GradingResult(Outcome.GRADED, 0.0, {"reason": "invalid_choice"})
                candidate = f"Answer: {letter}"
            elif verifier.mode in {Mode.MATH, Mode.NUMERIC}:
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
