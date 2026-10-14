# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import pytest
from shellbox.machine import Command
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerFormat,
    AnswerType,
    EnvironmentRequirements,
    JsonAnswer,
    JsonValueAnswer,
    PlainText,
)
from verifyit.spec import ExactSpec, McqSpec, NumericSpec, StructuredExactSpec

from taskforge.sandbox.images import DockerBuild
from taskforge.spec import draft
from taskforge.stage40_build_with_program.context import BuildFailure
from taskforge.stage40_build_with_program.sandbox_effects import build_machine, grade_candidate, publish_image
from tests.sandbox.fixture_images import shell_grader

SHELLSIM = draft.requirements(image=None)
MACHINE = draft.machine(startup_timeout=60)
PLAIN = PlainText()
IMAGE = f"registry.example/taskforge-tasks/d00-arithmetic-products-1@sha256:{'1' * 64}"

READS_REPORT = """passed=false
if test -f /workspace/report.txt && test "$(cat /workspace/report.txt)" = total=42; then passed=true; fi
jq -n -c --argjson passed "$passed" '{
    reward: (if $passed then 1.0 else 0.0 end), detail: {checks: [{name: "total", passed: $passed, weight: 1.0}]}
}' > /logs/verifier/reward.json
"""


@dataclass
class RecordingImageBuilder:
    published: list[tuple[DockerBuild, str]] = field(default_factory=list)

    async def publish(self, build: DockerBuild, repository: str) -> str:
        self.published.append((build, repository))
        return IMAGE


@pytest.mark.parametrize("requirements", [SHELLSIM, None], ids=["shellsim", "no-machine"])
@pytest.mark.parametrize(
    ("grader", "answer_type", "answer_format", "right", "wrong"),
    [
        (draft.answer_grader(ExactSpec(expected=("42",))), AnswerType.TEXT, PLAIN, "42", "41"),
        (
            draft.answer_grader(NumericSpec(expected="0.5", tolerance_abs=0.0, tolerance_rel=0.0)),
            AnswerType.NUMBER,
            PLAIN,
            "1/2",
            "0.25",
        ),
        (draft.answer_grader(McqSpec(expected="B")), AnswerType.TEXT, PLAIN, "B", "A"),
        (draft.answer_grader(ExactSpec(expected=("42",))), AnswerType.TEXT, JsonAnswer(), '{"answer": "42"}', "42"),
        (
            draft.answer_grader(StructuredExactSpec(expected={"total": 42})),
            AnswerType.JSON,
            JsonValueAnswer(),
            '{"total": 42}',
            '{"total": 41}',
        ),
    ],
)
async def test_grade_candidate_grades_answers_through_the_answer_format(
    context,
    requirements: EnvironmentRequirements | None,
    grader: GraderPackage,
    answer_type: AnswerType,
    answer_format: AnswerFormat,
    right: str,
    wrong: str,
):
    graded = await grade_candidate(context(), requirements, grader, answer_type, answer_format, "question", right)
    rejected = await grade_candidate(context(), requirements, grader, answer_type, answer_format, "question", wrong)

    assert (graded.status, graded.reward) == (Outcome.GRADED, 1.0)
    assert rejected.reward == 0.0


async def test_grade_candidate_grades_the_workspace_a_candidate_leaves(context):
    grader = shell_grader(
        READS_REPORT, {}, environment=draft.grader_environment(None), answer_path=draft.ANSWER_PATH, timeout=60
    )

    async def grade(report: str) -> float | None:
        result = await grade_candidate(
            context(),
            SHELLSIM,
            grader,
            AnswerType.TEXT,
            PLAIN,
            "question",
            "done",
            workspace=(draft.file("workspace/report.txt", report),),
            output_paths=("/workspace/report.txt",),
        )
        return result.reward

    assert await grade("total=42\n") == 1.0
    assert await grade("total=41\n") == 0.0


async def test_grade_candidate_refuses_an_answer_format_that_cannot_carry_the_answer(context):
    grader = draft.answer_grader(ExactSpec(expected=("42",)))
    with pytest.raises(BuildFailure, match="json_value cannot carry a text answer"):
        await grade_candidate(context(), SHELLSIM, grader, AnswerType.TEXT, JsonValueAnswer(), "question", "42")


async def test_machine_installs_the_files_and_runs_the_setup(context):
    requirements = draft.requirements(image=None, setup=("mkdir -p /workspace/out && echo ready > /workspace/out/s",))
    files = (draft.file("workspace/question.txt", "6 * 7\n"),)

    async with build_machine(context().services, requirements, MACHINE, files) as prepared:
        result = await prepared.run(Command(argv=("cat", "/workspace/question.txt", "/workspace/out/s")))

    assert result.stdout == b"6 * 7\nready\n"


async def test_publish_image_pushes_under_the_items_repository(context):
    images = RecordingImageBuilder()
    build = DockerBuild(files=(draft.file("Dockerfile", "FROM busybox\n"),))

    assert await publish_image(context(images=images), build) == IMAGE
    assert images.published == [(build, "taskforge-tasks/d00-arithmetic-products-1")]
