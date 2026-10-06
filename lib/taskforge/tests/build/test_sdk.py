# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator

import pytest
from taskcompendium.environment import EnvironmentKind, StdoutReward
from taskcompendium.grading import structured_exact
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, VerifierSpec
from taskcompendium.submission import JsonAnswer, JsonValueAnswer, PlainText, SubmissionConvention
from verifyit.spec import ExactSpec, McqSpec, NumericSpec

from taskforge.build.run import item_id_for
from taskforge.build.sdk import Build, BuildFailure
from taskforge.build.step import StepCache
from taskforge.spec import draft

SHELLSIM = draft.environment(EnvironmentKind.SHELLSIM)
PLAIN = PlainText(id="plain_text")

READS_REPORT = """
with open("/workspace/report.txt") as report:
    print(1.0 if report.read().strip() == "total=42" else 0.0)
"""


@pytest.fixture
async def b(proposal, tmp_path, services) -> AsyncIterator[Build]:
    cache = StepCache(root=tmp_path / "cache", item_id=item_id_for(proposal))
    async with services() as s:
        yield Build(proposal, cache.item_id, s, cache, tmp_path / "scratch", 0)


@pytest.mark.parametrize(
    ("verifier", "answer_type", "convention", "right", "wrong"),
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
        (structured_exact({"total": 42}), AnswerType.JSON, JsonValueAnswer(id="json"), '{"total": 42}', '{"total": 41}'),
    ],
)
async def test_try_grader_grades_answers_through_the_convention(
    b: Build,
    verifier: VerifierSpec,
    answer_type: AnswerType,
    convention: SubmissionConvention,
    right: str,
    wrong: str,
):
    graded = await b.try_grader(SHELLSIM, verifier, answer_type, convention, "question", right)
    rejected = await b.try_grader(SHELLSIM, verifier, answer_type, convention, "question", wrong)

    assert (graded.status, graded.reward) == (Outcome.GRADED, 1.0)
    assert rejected.reward == 0.0


async def test_try_grader_grades_the_workspace_a_candidate_leaves(b: Build):
    verifier = draft.shell_verifier(
        argv=("python3", "/grader/grade.py"),
        reward=StdoutReward(),
        timeout=60,
        files=(draft.file("/grader/grade.py", READS_REPORT),),
    )

    async def grade(report: str) -> float | None:
        files = (draft.file("/workspace/report.txt", report),)
        result = await b.try_grader(SHELLSIM, verifier, AnswerType.TEXT, PLAIN, "question", "done", files)
        return result.reward

    assert await grade("total=42\n") == 1.0
    assert await grade("total=41\n") == 0.0


async def test_try_grader_rejects_a_convention_that_cannot_carry_the_answer(b: Build):
    verifier = draft.answer_verifier(ExactSpec(expected=("42",)))
    with pytest.raises(BuildFailure, match="convention 'json'"):
        await b.try_grader(SHELLSIM, verifier, AnswerType.TEXT, JsonValueAnswer(id="json"), "question", "42")
