# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace

import pytest
from shellbox.machine import Backend, Command
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, EnvironmentRequirements, Source, TaskSpec
from taskcompendium.submission import JsonAnswer, JsonValueAnswer, PlainText, SubmissionConvention
from verifyit.spec import ExactSpec, McqSpec, NumericSpec, StructuredExactSpec

from taskforge.builder.run import item_id_for
from taskforge.builder.sdk import Build, BuildFailure
from taskforge.builder.step import StepCache
from taskforge.sandbox.images import DockerBuild
from taskforge.spec import draft

SHELLSIM = draft.requirements(image=None)
MACHINE = draft.machine(startup_timeout=60)
PLAIN = PlainText(id="plain_text")
IMAGE = f"registry.example/taskforge-tasks/d00-arithmetic-products--1@sha256:{'1' * 64}"

READS_REPORT = """import json, os, pathlib
report = pathlib.Path(os.environ["VERIFYIT_WORKSPACE"], "captured/workspace/report.txt")
reward = float(report.is_file() and report.read_text().strip() == "total=42")
verdict = {"status": "scored", "reward": reward, "detail": {}}
pathlib.Path(os.environ["VERIFYIT_LOGS_DIR"], "verdict.json").write_text(json.dumps(verdict))
"""


@dataclass
class RecordingImageBuilder:
    published: list[tuple[DockerBuild, str]] = field(default_factory=list)

    async def publish(self, build: DockerBuild, repository: str) -> str:
        self.published.append((build, repository))
        return IMAGE


@pytest.fixture
async def b(proposal, tmp_path, services) -> AsyncIterator[Build]:
    cache = StepCache(root=tmp_path / "cache", item_id=item_id_for(proposal))
    async with services() as s:
        yield Build(proposal, cache.item_id, s, cache, tmp_path / "scratch", 0)


@pytest.mark.parametrize("requirements", [SHELLSIM, None], ids=["shellsim", "no-machine"])
@pytest.mark.parametrize(
    ("grader", "answer_type", "convention", "right", "wrong"),
    [
        (draft.answer_verifier(ExactSpec(expected=("42",))), AnswerType.TEXT, PLAIN, "42", "41"),
        (
            draft.answer_verifier(NumericSpec(expected="0.5", tolerance_abs=0.0, tolerance_rel=0.0)),
            AnswerType.NUMBER,
            PLAIN,
            "1/2",
            "0.25",
        ),
        (draft.answer_verifier(McqSpec(expected="B")), AnswerType.TEXT, PLAIN, "B", "A"),
        (
            draft.answer_verifier(ExactSpec(expected=("42",))),
            AnswerType.TEXT,
            JsonAnswer(id="json_answer"),
            '{"answer": "42"}',
            "42",
        ),
        (
            draft.answer_verifier(StructuredExactSpec(expected={"total": 42})),
            AnswerType.JSON,
            JsonValueAnswer(id="json"),
            '{"total": 42}',
            '{"total": 41}',
        ),
    ],
)
async def test_try_grader_grades_answers_through_the_convention(
    b: Build,
    requirements: EnvironmentRequirements | None,
    grader: GraderPackage,
    answer_type: AnswerType,
    convention: SubmissionConvention,
    right: str,
    wrong: str,
):
    graded = await b.try_grader(requirements, grader, answer_type, convention, "question", right)
    rejected = await b.try_grader(requirements, grader, answer_type, convention, "question", wrong)

    assert (graded.status, graded.reward) == (Outcome.GRADED, 1.0)
    assert rejected.reward == 0.0


async def test_try_grader_grades_the_workspace_a_candidate_leaves(b: Build):
    grader = draft.script_verifier(READS_REPORT, {}, timeout=60)

    async def grade(report: str) -> float | None:
        files = (draft.file("workspace/report.txt", report),)
        result = await b.try_grader(
            SHELLSIM,
            grader,
            AnswerType.TEXT,
            PLAIN,
            "question",
            "done",
            workspace=files,
            output_paths=("/workspace/report.txt",),
        )
        return result.reward

    assert await grade("total=42\n") == 1.0
    assert await grade("total=41\n") == 0.0


async def test_try_grader_rejects_a_convention_that_cannot_carry_the_answer(b: Build):
    grader = draft.answer_verifier(ExactSpec(expected=("42",)))
    with pytest.raises(BuildFailure, match="convention 'json'"):
        await b.try_grader(SHELLSIM, grader, AnswerType.TEXT, JsonValueAnswer(id="json"), "question", "42")


async def test_machine_installs_the_files_and_runs_the_setup(b: Build):
    requirements = draft.requirements(image=None, setup=("mkdir -p /workspace/out && echo ready > /workspace/out/s",))
    files = (draft.file("workspace/question.txt", "6 * 7\n"),)

    async with b.machine(requirements, MACHINE, files) as machine:
        result = await machine.run(Command(argv=("cat", "/workspace/question.txt", "/workspace/out/s")))

    assert result.stdout == b"6 * 7\nready\n"


SESSION = draft.session(
    max_turns=4,
    model_turn_timeout=None,
    command_timeout=None,
    tool_turn_timeout=None,
    total_turn_timeout=None,
    attempt_timeout=None,
    verifier_timeout=60,
    cleanup_timeout=30,
)


def answer_task(requirements: EnvironmentRequirements) -> TaskSpec:
    return draft.assemble(
        "t",
        "What is six times seven?",
        AnswerType.TEXT,
        draft.answer_verifier(ExactSpec(expected=("42",))),
        Source(dataset="test", revision="r", row="0", importer_revision="test"),
        environment=requirements,
    )


@pytest.mark.parametrize(
    ("requirements", "backend"),
    [(SHELLSIM, Backend.SHELLSIM), (draft.requirements(image=IMAGE), Backend.DOCKER)],
    ids=["shellsim", "image"],
)
def test_lower_picks_this_hosts_backend_even_without_its_factory(b: Build, requirements, backend):
    lowered = b.lower(answer_task(requirements), task_machine=MACHINE, verifier_machine=None, session=SESSION)

    assert lowered.runtime.task_machine is not None
    assert lowered.runtime.task_machine.backend == backend


def test_lower_fails_the_build_on_a_machine_the_task_does_not_take(b: Build):
    with pytest.raises(BuildFailure, match="lower: A verifier machine"):
        b.lower(answer_task(SHELLSIM), task_machine=MACHINE, verifier_machine=MACHINE, session=SESSION)


async def test_publish_image_pushes_under_the_items_repository(proposal, tmp_path, services):
    images = RecordingImageBuilder()
    build = DockerBuild(files=(draft.file("Dockerfile", "FROM busybox\n"),))
    cache = StepCache(root=tmp_path / "cache", item_id=item_id_for(proposal))
    async with services() as s:
        b = Build(proposal, cache.item_id, replace(s, images=images), cache, tmp_path / "scratch", 0)
        assert await b.publish_image(build) == IMAGE

    assert images.published == [(build, "taskforge-tasks/d00-arithmetic-products-1")]
