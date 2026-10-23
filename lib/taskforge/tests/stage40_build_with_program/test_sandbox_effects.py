# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import pytest
from shellbox.machine import Backend, Command
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.models import (
    AnswerFormat,
    AnswerType,
    EnvironmentRequirements,
    JsonAnswer,
    JsonValueAnswer,
    PlainText,
)
from verifyit.spec import ExactSpec, McqSpec, NumericSpec, StructuredExactSpec

from taskforge.sandbox.images import GRADER_BASE_IMAGE, DockerBuild
from taskforge.spec import draft
from taskforge.stage40_build_with_program.context import BuildFailure
from taskforge.stage40_build_with_program.sandbox_effects import (
    build_machine,
    calibrate_decision_grader,
    grade_candidate,
    publish_image,
)
from taskforge.stage40_build_with_program.sdk import FoolingKind, fooling_negatives
from tests.sandbox.fixture_images import FixtureImageFactory, PortlessShellSimFactory, SlowGraderFactory, shell_grader
from tests.spec.fake_decision import FakeDecisionClient
from tests.stage40_build_with_program.conftest import (
    DECISION_FLUENT_WRONG,
    DECISION_HEDGED,
    DECISION_REFERENCE,
    DECISION_RUBRIC,
    DECISION_VARIANTS,
    FACTORIES,
)

SHELLSIM = draft.requirements(image=None)
MACHINE = draft.machine(startup_timeout=60)
PLAIN = PlainText()
IMAGE = f"registry.example/taskforge-tasks/d00-arithmetic-products-1@sha256:{'1' * 64}"
FULL_CREDIT = shell_grader(
    """printf '%s' '{"reward": 1.0}' > /logs/verifier/reward.json\n""",
    {},
    environment=draft.grader_environment(None),
    answer_path=draft.ANSWER_PATH,
    timeout=60,
)

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


@pytest.mark.parametrize(
    ("timeouts", "status", "reward", "failure"),
    [(1, Outcome.GRADED, 1.0, None), (2, Outcome.INFRA_ERROR, None, GradingFailure.TIMEOUT)],
    ids=["slow-once", "slow-twice"],
)
async def test_a_grader_timeout_is_graded_once_more_on_fresh_machines(context, timeouts, status, reward, failure):
    """One slow exec cannot be told from a slow grader, so only a second timeout on fresh machines is returned."""
    slow = SlowGraderFactory(timeouts)
    on_host = context(factories={**FACTORIES, Backend.DOCKER.value: slow})

    grade = await grade_candidate(on_host, None, FULL_CREDIT, AnswerType.TEXT, PLAIN, "question", "42")

    assert (grade.status, grade.reward, grade.failure) == (status, reward, failure)
    assert (slow.machines, slow.grader_runs) == (2, 2)


@pytest.mark.parametrize(
    "requirements", [SHELLSIM, draft.NO_MACHINE, None], ids=["shellsim", "no-machine", "no-requirements"]
)
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


async def test_machine_runs_the_grader_base_on_the_local_backend_when_the_host_has_one(context):
    # Without a Docker factory, only the local backend can start the machine.
    factories = {
        Backend.SHELLSIM.value: PortlessShellSimFactory(),
        Backend.LOCAL.value: FixtureImageFactory(Backend.LOCAL),
    }
    requirements = draft.requirements(image=GRADER_BASE_IMAGE, setup=("echo ready > /workspace/s",))

    async with build_machine(context(factories=factories).services, requirements, MACHINE) as prepared:
        result = await prepared.run(Command(argv=("cat", "/workspace/s")))

    assert result.stdout == b"ready\n"


async def test_publish_image_pushes_under_the_workpieces_repository(context):
    images = RecordingImageBuilder()
    build = DockerBuild(files=(draft.file("Dockerfile", "FROM busybox\n"),))

    assert await publish_image(context(images=images), build) == IMAGE
    assert images.published == [(build, "taskforge-tasks/d00-arithmetic-products-1")]


KNOWN_CORRECT_READS = {DECISION_REFERENCE: 3.9, **dict.fromkeys(DECISION_VARIANTS, 3.0)}


async def calibrated(context, reads: dict[str, float]):
    return await calibrate_decision_grader(
        context(decision=FakeDecisionClient(reads)),
        draft.decision_grader(DECISION_RUBRIC, DECISION_REFERENCE),
        DECISION_VARIANTS,
        fluent_wrong_reply=DECISION_FLUENT_WRONG,
        hedged_reply=DECISION_HEDGED,
    )


async def test_a_decision_grader_that_credits_a_hedged_reply_is_refused(context):
    calibration = await calibrated(context, {**KNOWN_CORRECT_READS, DECISION_HEDGED: 3.2})

    assert calibration.problem is not None and "hedged reward 1.0" in calibration.problem
    assert "fluent_wrong" not in calibration.problem
    assert calibration.grader.acceptance_level is None and calibration.grader.fooling == {}


async def test_a_decision_grader_that_rejects_every_fooling_reply_calibrates_with_their_reads(context):
    negatives = fooling_negatives(DECISION_REFERENCE, DECISION_FLUENT_WRONG, DECISION_HEDGED)
    # Just below the 3.0 bar each earns at most the 0.5 ceiling.
    calibration = await calibrated(context, {**KNOWN_CORRECT_READS, **{n.reply: 2.9 for n in negatives}})

    assert calibration.problem is None
    assert calibration.grader.acceptance_level == 3.0
    assert calibration.grader.fooling == {kind.value: 2.9 for kind in FoolingKind}
    assert [c.reply for c in calibration.candidates] == [
        DECISION_REFERENCE,
        *DECISION_VARIANTS,
        "",
        *(n.reply for n in negatives),
    ]
    assert all(c.status is Outcome.GRADED and (c.reward or 0.0) <= 0.5 for c in calibration.candidates[-5:])
