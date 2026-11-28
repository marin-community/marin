# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The grading-boundary assessment derived from a task's own declarations."""

import base64
from collections import Counter

from verifyit.spec import ExactSpec, StdioSpec

from taskcompendium.grader import verifyit_package
from taskcompendium.grading_environment import (
    GradingEnvironmentMode,
    GradingEnvironmentStatus,
    assess_grading_environment,
    count_grading_environment,
    grading_environment_census,
)
from taskcompendium.models import (
    AgentMachine,
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    GraderMachine,
    InlineFile,
    NoGrader,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
    VerifierCommand,
)


def _spec() -> ExactSpec:
    return ExactSpec(expected=("3",))


def _task(grader, *, answer_type=AnswerType.TEXT, output_paths=(), verifier=(), capabilities=()) -> TaskSpec:
    return TaskSpec(
        id="grading-boundary",
        context=ConversationInput(events=(TextMessage(role="user", content="Complete the task."),)),
        environment_requirements=EnvironmentRequirements(capabilities=capabilities),
        output_paths=output_paths,
        answer_type=answer_type,
        answer_format=PlainText(),
        grader=grader,
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
        resources=ResourceGroups(verifier=verifier),
    )


def _inline_toml(path: str, document: str) -> TaskResource:
    return TaskResource(
        path=path,
        source=InlineFile(content_base64=base64.b64encode(document.encode()).decode()),
    )


def _grading_machine() -> GraderMachine:
    return GraderMachine(
        environment=EnvironmentRequirements(
            docker_image="private/grader@sha256:" + "a" * 64, command_semantics=CommandSemantics.LINUX_PROCESS
        )
    )


NATIVE_TASK_TOML = """\
version = "1.0"

[environment]
docker_image = "alexgshaw/count-lines:20251031"
cpus = 2

[agent]
timeout_sec = 900.0
"""

NATIVE_SEPARATE_VERIFIER_TOML = (
    NATIVE_TASK_TOML
    + """
[verifier]
timeout_sec = 300.0
environment_mode = "separate"
"""
)


def test_in_process_verifyit_task_requires_separate_grading_and_matches():
    assessment = assess_grading_environment(_task(verifyit_package(_spec()).grader))
    assert assessment.required is GradingEnvironmentMode.SEPARATE
    assert assessment.declared is GradingEnvironmentMode.SEPARATE
    assert assessment.status is GradingEnvironmentStatus.MATCHED


def test_native_workspace_verifier_declares_shared_and_matches():
    grader = NoGrader(reason="Harbor task")
    assessment = assess_grading_environment(_task(grader, verifier=[_inline_toml("task.toml", NATIVE_TASK_TOML)]))
    assert assessment.required is GradingEnvironmentMode.SHARED
    assert assessment.declared is GradingEnvironmentMode.SHARED
    assert assessment.status is GradingEnvironmentStatus.MATCHED


def test_native_verifier_declaring_an_environment_mismatches_the_shared_boundary():
    grader = NoGrader(reason="Harbor task")
    assessment = assess_grading_environment(
        _task(grader, verifier=[_inline_toml("task.toml", NATIVE_SEPARATE_VERIFIER_TOML)])
    )
    assert assessment.required is GradingEnvironmentMode.SHARED
    assert assessment.declared is GradingEnvironmentMode.SEPARATE
    assert assessment.status is GradingEnvironmentStatus.MISMATCH


def test_grader_without_executable_declaration_is_unavailable_and_separate():
    assessment = assess_grading_environment(_task(NoGrader(reason="hand-graded fixture")))
    assert assessment.required is GradingEnvironmentMode.SEPARATE
    assert assessment.declared is None
    assert assessment.status is GradingEnvironmentStatus.UNAVAILABLE
    assert assessment.reason == "No executable grader is declared"


def test_script_grader_collecting_inputs_grades_in_its_own_environment():
    grader = ScriptGrader(
        argv=["python", "grade.py"],
        placement=GraderMachine(
            environment=_grading_machine().environment,
            collect=(VerifierCommand(argv=("cp", "/app/out.txt", "/logs/artifacts/out.txt")),),
        ),
    )
    assessment = assess_grading_environment(_task(grader))
    assert assessment.required is GradingEnvironmentMode.SEPARATE
    assert (
        assessment.reason == "The script grader consumes declared collected inputs in its separate verifier environment"
    )


def test_agent_machine_grader_declares_shared_grading():
    grader = ScriptGrader(argv=["python", "grade.py"], placement=AgentMachine())
    assessment = assess_grading_environment(_task(grader, capabilities=("git_repository",)))
    assert assessment.required is GradingEnvironmentMode.SHARED
    assert assessment.declared is GradingEnvironmentMode.SHARED
    assert assessment.status is GradingEnvironmentStatus.MATCHED


def test_repository_state_grading_requires_the_agent_machine():
    grader = ScriptGrader(argv=["python", "grade.py"], placement=_grading_machine())
    assessment = assess_grading_environment(_task(grader, capabilities=("git_repository",)))
    assert assessment.required is GradingEnvironmentMode.SHARED
    assert "Repository or workspace-state grading" in assessment.reason


def test_stdio_grading_reads_the_program_in_a_separate_environment():
    grader = verifyit_package(StdioSpec(command="python3 /app/solution.py"), placement=_grading_machine()).grader
    assessment = assess_grading_environment(
        _task(grader, answer_type=AnswerType.FILE, output_paths=("/app/solution.py",))
    )
    assert assessment.required is GradingEnvironmentMode.SEPARATE
    assert assessment.declared is GradingEnvironmentMode.SEPARATE
    assert "Stdio grading" in assessment.reason


def test_workspace_output_without_a_file_boundary_requires_the_agent_machine():
    grader = ScriptGrader(argv=["python", "grade.py"], placement=_grading_machine(), answer_path=None)
    assessment = assess_grading_environment(_task(grader, output_paths=("/app/out",)))
    assert assessment.required is GradingEnvironmentMode.SHARED
    assert "does not declare a file-only grading boundary" in assessment.reason


def test_census_counts_join_into_the_published_evidence_fields():
    counts: Counter[str] = Counter()
    for task in (_task(verifyit_package(_spec()).grader), _task(NoGrader(reason="fixture"))):
        count_grading_environment(counts, assess_grading_environment(task).model_dump(mode="json"))
    census = grading_environment_census(counts)
    assert census["assessed_tasks"] == 2
    assert census["required_counts"] == {"separate": 2}
    assert census["declared_counts"] == {"separate": 1, "unavailable": 1}
    assert census["status_counts"] == {"matched": 1, "unavailable": 1}
    assert census["reason_counts"]["No executable grader is declared"] == 1
