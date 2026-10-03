# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade TaskCompendium submissions with packaged VerifyIT specifications."""

import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from rigging.filesystem.path_validation import validate_relative_file_path
from shellbox.backends.docker.machine import DockerMachineFactory
from verifyit.candidate import (
    candidate_spec,
    grade_text_candidate,
    supports_candidate_mode,
)
from verifyit.grade import InvalidTask, Reward, Status
from verifyit.grade import grade as verifyit_grade
from verifyit.modes.grade_predicted_action import grade_predicted_action_candidate
from verifyit.spec import (
    DEFAULT_OUTPUT,
    DEFAULT_WORKSPACE,
    ExactSpec,
    GotestSpec,
    JsonSchemaSpec,
    JudgeSpec,
    JunitSpec,
    McqSpec,
    Mode,
    NumericSpec,
    PredictedActionSpec,
    PytestSpec,
    ReasoningGymSpec,
    ScriptSpec,
    Spec,
    StdioSpec,
    spec_from_table,
)
from verifyit.spec import FunctionCall as CandidateCall

from taskcompendium.grader import grader_package
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationTrace,
    EnvironmentRequirements,
    FunctionCall,
    TaskResource,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.runtime.grading import grade_submission
from taskcompendium.runtime.models import RuntimeEvidence
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.submission import AnswerFormat, FinalAction, Submission, extract_answer


def resolve_verifier(specification: VerifierSpec) -> Spec:
    """Read a shared verifier spec without any TaskCompendium registration step."""
    try:
        parameters = json.loads(specification.parameters_json)
        if "mode" in parameters:
            raise ValueError("Verifier parameters must not override the mode")
        if supports_candidate_mode(specification.kind):
            return candidate_spec(specification.kind, parameters)
        return spec_from_table({"mode": specification.kind, **parameters})
    except (ValueError, InvalidTask) as error:
        raise ValueError(f"Invalid {specification.kind!r} verifier parameters: {error}") from error


def validate_verifier(specification: VerifierSpec) -> None:
    resolve_verifier(specification)


def supports_verifier(specification: VerifierSpec) -> bool:
    if specification.environment_requirements != EnvironmentRequirements():
        return False
    if specification.kind not in {mode.value for mode in Mode}:
        return False
    verifier = resolve_verifier(specification)
    if isinstance(verifier, ScriptSpec):
        if verifier.verdict_file is None or verifier.workspace != DEFAULT_WORKSPACE:
            return False
        try:
            validate_relative_file_path(verifier.path)
        except ValueError:
            return False
    return not isinstance(verifier, StdioSpec | PytestSpec | JunitSpec | GotestSpec | ReasoningGymSpec)


def grade_task(
    specification: TaskSpec,
    convention: Submission,
    conversation: ConversationTrace,
    evidence: RuntimeEvidence | None = None,
) -> GradeResult:
    """Score terminal evidence and return its grading status and reward."""
    verifier = resolve_verifier(specification.verifier)
    requirements = specification.verifier.environment_requirements
    executable = isinstance(verifier, StdioSpec | PytestSpec | JunitSpec | GotestSpec)
    if requirements != EnvironmentRequirements() and requirements.docker_image is None:
        return GradeResult(Outcome.INFRA_ERROR, None, "Private grading environment is unavailable")
    if requirements.docker_image and isinstance(verifier, PredictedActionSpec | ExactSpec | NumericSpec | McqSpec):
        return GradeResult(Outcome.INVALID_TASK, None, "Direct candidate modes cannot declare an isolated grader")
    final = conversation.events[-1]
    if isinstance(convention, FinalAction):
        try:
            convention.validate_final_message(final)
        except ValueError as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
    if isinstance(verifier, PredictedActionSpec):
        if convention.answer_format != AnswerFormat.FINAL_ACTION:
            return GradeResult(Outcome.INFRA_ERROR, None, "Incompatible final-action convention")
        if not isinstance(final, (TextMessage, AssistantToolCalls)):
            return GradeResult(Outcome.INFRA_ERROR, None, "Missing final assistant message")
        calls = (
            tuple(CandidateCall(call.name, call.arguments) for call in final.calls)
            if isinstance(final, AssistantToolCalls)
            else ()
        )
        return GradeResult(Outcome.GRADED, grade_predicted_action_candidate(verifier, calls).reward)
    if isinstance(verifier, ExactSpec | NumericSpec | McqSpec):
        try:
            candidate = extract_answer(final, convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        if isinstance(verifier, McqSpec):
            letter = candidate.strip()
            if len(letter) != 1 or not "A" <= letter.upper() <= "Z":
                return GradeResult(Outcome.EXTRACTION_ERROR, None, "MCQA response requires one option letter")
        return _grade_result(grade_text_candidate(verifier, candidate))

    if requirements.docker_image:
        if evidence is None:
            return GradeResult(Outcome.INFRA_ERROR, None, "Missing captured submission files")
        files = dict(evidence.files)
        files["/app/state.json"] = evidence.state_json.encode()
        if specification.answer_type in {AnswerType.TEXT, AnswerType.NUMBER}:
            try:
                candidate = extract_answer(final, convention)
            except (ValueError, TypeError) as error:
                return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
            try:
                output = _answer_output(verifier)
            except InvalidTask as error:
                return GradeResult(Outcome.INVALID_TASK, None, str(error))
            if not output.is_relative_to(DEFAULT_WORKSPACE) or ".." in output.parts:
                return GradeResult(Outcome.INVALID_TASK, None, "Answer output must be within /app")
            files[str(output)] = candidate.encode()
        return asyncio.run(grade_submission(specification, files, DockerMachineFactory()))
    if executable:
        return GradeResult(Outcome.INFRA_ERROR, None, "Executable grading requires an isolated image")
    if isinstance(verifier, ReasoningGymSpec):
        return GradeResult(Outcome.INFRA_ERROR, None, "Reasoning-gym grading requires an isolated runner")
    try:
        candidate = (
            extract_answer(final, convention)
            if specification.answer_type in {AnswerType.TEXT, AnswerType.NUMBER}
            else None
        )
    except (ValueError, TypeError) as error:
        if evidence is None:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        candidate = None
    return _grade_files(specification, verifier, candidate, evidence)


def grade_answer(specification: TaskSpec, convention: Submission, conversation: ConversationTrace) -> GradeResult:
    """Grade a final direct-chat answer."""
    return grade_task(specification, convention, conversation)


def _grade_result(verdict: Reward) -> GradeResult:
    if verdict.status == Status.SCORED:
        return GradeResult(Outcome.GRADED, verdict.reward, detail=verdict.detail)
    status = Outcome.INVALID_TASK if verdict.status == Status.INVALID_TASK else Outcome.INFRA_ERROR
    return GradeResult(status, None, verdict.detail.get("error"), detail=verdict.detail)


def _answer_output(verifier: Spec) -> Path:
    if isinstance(verifier, ScriptSpec):
        return Path(DEFAULT_OUTPUT)
    if isinstance(verifier, StdioSpec | PytestSpec | JunitSpec | GotestSpec):
        raise InvalidTask("Executable verifiers do not accept extracted text answers")
    return Path(verifier.output)


def _write_resource(root: Path, resource: TaskResource) -> None:
    path = root / resource.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resource_bytes(resource))
    if resource.mode is not None:
        path.chmod(int(resource.mode, 8))
    if resource.mtime_ns is not None:
        os.utime(path, ns=(resource.mtime_ns, resource.mtime_ns))


def _grade_files(task: TaskSpec, verifier: Spec, candidate: str | None, evidence: RuntimeEvidence | None) -> GradeResult:
    with TemporaryDirectory(prefix="taskcompendium-grader-") as directory:
        root = Path(directory)
        tests = root / "tests"
        workspace = root / "app"
        tests.mkdir()
        workspace.mkdir()
        for resource in task.resources.all:
            _write_resource(workspace, resource)
        for resource in task.resources.verifier:
            _write_resource(tests, resource)
        if evidence is not None:
            for path, data in evidence.files.items():
                source = Path(path)
                if not source.is_absolute() or ".." in source.parts:
                    return GradeResult(Outcome.INFRA_ERROR, None, f"Invalid captured path: {path}")
                if source.is_relative_to(DEFAULT_WORKSPACE):
                    relative = source.relative_to(DEFAULT_WORKSPACE)
                else:
                    relative = Path("captured") / source.relative_to("/")
                destination = workspace / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            (workspace / "state.json").write_text(evidence.state_json)
        if candidate is not None:
            try:
                output = _answer_output(verifier)
            except InvalidTask as error:
                return GradeResult(Outcome.INVALID_TASK, None, str(error))
            if not output.is_relative_to(DEFAULT_WORKSPACE) or ".." in output.parts:
                return GradeResult(Outcome.INVALID_TASK, None, "Answer output must be within /app")
            target = workspace / output.relative_to(DEFAULT_WORKSPACE)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(candidate)
        if isinstance(verifier, ScriptSpec):
            if verifier.verdict_file is None:
                return GradeResult(Outcome.INVALID_TASK, None, "Script graders require a structured verdict file")
            if verifier.workspace != DEFAULT_WORKSPACE:
                return GradeResult(Outcome.INVALID_TASK, None, "Script workspace must be /app")
            private_paths = (verifier.path,)
        elif isinstance(verifier, JsonSchemaSpec):
            private_paths = (verifier.schema,)
        elif isinstance(verifier, ReasoningGymSpec):
            private_paths = (verifier.entry, *((verifier.params,) if verifier.params is not None else ()))
        elif isinstance(verifier, JudgeSpec) and verifier.context:
            private_paths = (verifier.context,)
        else:
            private_paths = ()
        try:
            for path in private_paths:
                validate_relative_file_path(path)
        except ValueError as error:
            return GradeResult(Outcome.INVALID_TASK, None, str(error))
        try:
            return _grade_result(verifyit_grade(verifier, tests, workspace))
        except InvalidTask as error:
            return GradeResult(Outcome.INVALID_TASK, None, str(error))
        except Exception as error:
            return GradeResult(Outcome.INFRA_ERROR, None, f"{type(error).__name__}: {error}")


def verifier_descriptor(spec: Spec) -> VerifierSpec:
    """Store a conversion-selected shared verifier contract in the private task slot."""
    descriptor = grader_package(spec).verifier
    validate_verifier(descriptor)
    return descriptor


def exact_answer(expected: str, ignore_case: bool = True, collapse_whitespace: bool = True) -> VerifierSpec:
    return verifier_descriptor(
        ExactSpec(expected=(expected,), ignore_case=ignore_case, ignore_whitespace=collapse_whitespace)
    )


def numeric_answer(expected: float, tolerance_abs: float, tolerance_rel: float) -> VerifierSpec:
    return verifier_descriptor(NumericSpec(expected=expected, tolerance_abs=tolerance_abs, tolerance_rel=tolerance_rel))


def multiple_choice_answer(expected: str, options: int) -> VerifierSpec:
    return verifier_descriptor(McqSpec(expected=expected.strip().upper(), options=options))


def predicted_action_verifier(expected_calls: tuple[FunctionCall, ...]) -> VerifierSpec:
    return verifier_descriptor(
        PredictedActionSpec(expected_calls=tuple(CandidateCall(call.name, call.arguments) for call in expected_calls))
    )


def skipped_verifier(reason: str) -> VerifierSpec:
    """Describe an explicit rollout-time grading omission."""
    return VerifierSpec(kind="skipped", parameters_json=json.dumps({"reason": reason}))
