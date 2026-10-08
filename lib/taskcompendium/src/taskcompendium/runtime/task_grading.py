# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade acquired runtime evidence with packaged VerifyIT specifications."""

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from rigging.filesystem.path_validation import validate_relative_file_path
from shellbox.backends.docker.machine import DockerMachineFactory
from verifyit.grade import InvalidTask
from verifyit.grade import grade as verifyit_grade
from verifyit.spec import (
    DEFAULT_OUTPUT,
    DEFAULT_WORKSPACE,
    GotestSpec,
    JsonSchemaSpec,
    JudgeSpec,
    JunitSpec,
    PytestSpec,
    ReasoningGymSpec,
    ScriptSpec,
    Spec,
    StdioSpec,
)

from taskcompendium.grading import grade_answer, grade_result
from taskcompendium.grading_contract import (
    GradingAttempt,
    StateSubmission,
    SubmissionFailure,
    TextSubmission,
    decode_json_value,
    resolve_verifier,
    supports_candidate_mode,
)
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationTrace,
    EnvironmentRequirements,
    TaskResource,
    TaskSpec,
)
from taskcompendium.runtime.grading import grade_submission
from taskcompendium.runtime.models import RuntimeEvidence
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.submission import SubmissionConvention


def grade_task(
    specification: TaskSpec,
    convention: SubmissionConvention,
    conversation: ConversationTrace,
    evidence: RuntimeEvidence | None = None,
) -> GradeResult:
    """Score captured evidence through pure candidate or runtime file grading."""
    verifier = resolve_verifier(specification.verifier)
    requirements = specification.verifier.environment_requirements
    if requirements != EnvironmentRequirements() and requirements.docker_image is None:
        return GradeResult(Outcome.INFRA_ERROR, None, "Private grading environment is unavailable")
    candidate_mode = supports_candidate_mode(specification.verifier.kind) and specification.answer_type not in {
        AnswerType.FILE,
        AnswerType.WORKSPACE_STATE,
    }
    if candidate_mode and requirements.docker_image:
        return GradeResult(Outcome.INVALID_TASK, None, "Direct candidate modes cannot declare an isolated grader")
    attempt = GradingAttempt(conversation)
    if evidence is not None and (candidate_mode or specification.answer_type in {AnswerType.TEXT, AnswerType.NUMBER}):
        try:
            state = StateSubmission(decode_json_value(evidence.state_json))
        except ValueError as error:
            return GradeResult(Outcome.INFRA_ERROR, None, f"Invalid captured state: {error}")
        attempt = GradingAttempt(conversation, files=evidence.files, state=state)
    if candidate_mode:
        return grade_answer(specification, convention, attempt)
    executable = isinstance(verifier, StdioSpec | PytestSpec | JunitSpec | GotestSpec)
    candidate = None
    if specification.answer_type in {AnswerType.TEXT, AnswerType.NUMBER}:
        try:
            submission = convention.extract(attempt)
        except SubmissionFailure as error:
            return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
        if not isinstance(submission, TextSubmission):
            raise TypeError("Runtime text grading requires a text submission")
        candidate = submission.value
    if requirements.docker_image:
        if evidence is None:
            return GradeResult(Outcome.INFRA_ERROR, None, "Missing captured submission files")
        files = dict(evidence.files)
        files["/app/state.json"] = evidence.state_json.encode()
        if candidate is not None:
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
    return _grade_files(specification, verifier, candidate, evidence)


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
            return grade_result(verifier, verifyit_grade(verifier, tests, workspace))
        except InvalidTask as error:
            return GradeResult(Outcome.INVALID_TASK, None, str(error))
        except Exception as error:
            return GradeResult(Outcome.INFRA_ERROR, None, f"{type(error).__name__}: {error}")
