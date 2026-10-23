# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The build checks that a built verifier is the kind its proposal declares."""

import json

import pytest
from taskcompendium.models import AnswerType, PlainText, Source, TaskSpec, VerifyitGrader, verifyit_spec
from taskcompendium.runtime.resources import resource_bytes
from verifyit.spec import ExactSpec, JudgeSpec

from taskforge.spec import draft
from taskforge.spec.judge import DEFAULT_JUDGE_MODEL, JUDGE_SPEC_PATH, JudgeStrength
from taskforge.spec.rubric import Rubric, RubricItem
from taskforge.stage10_proposal.model import Verification
from taskforge.stage40_build_with_program.verification import (
    built_verification,
    unbuildable_problem,
    verification_problem,
)

SOURCE = Source(dataset="taskforge-test", revision="r1", row="0", importer_revision="test")
RUBRIC = Rubric(
    deliverable="A memo recommending a supplier.",
    items=[
        RubricItem(
            id="recommendation",
            intent="The memo recommends a supplier and says why.",
            pass_criterion="The memo recommends supplier B for its lead time.",
            equivalences=[],
            negation_handling="A supplier named only to be rejected is not recommended.",
            weight=1.0,
            role="scored",
            cap_value=None,
        )
    ],
)
REFERENCE = "Choose supplier B: its two-week lead time meets the launch date; A's six weeks does not."
KEYWORD_RUBRIC = """import json, pathlib
answer = pathlib.Path('/app/answer.txt').read_text()
passed = 'supplier' in answer
pathlib.Path('/logs/verifier/reward.json').write_text(json.dumps({'reward': float(passed)}))
"""


def task(package) -> TaskSpec:
    return draft.assemble("t", "Which supplier?", AnswerType.TEXT, PlainText(), package, SOURCE, environment=None)


GRADERS = {
    Verification.ANSWER_IN_BOX: lambda: draft.answer_grader(ExactSpec(expected=("B",))),
    Verification.CODE: lambda: draft.python_grader(
        KEYWORD_RUBRIC, {}, environment=draft.grader_environment(None), answer_path=draft.ANSWER_PATH, timeout=60.0
    ),
    Verification.LLM_AS_JUDGE: lambda: draft.judge_grader(RUBRIC, REFERENCE),
    Verification.DECISION_MODEL: lambda: draft.decision_grader(RUBRIC, REFERENCE),
}


@pytest.mark.parametrize("kind", list(GRADERS), ids=str)
def test_the_built_kind_is_read_off_the_grader(kind):
    assert built_verification(task(GRADERS[kind]())) is kind


@pytest.mark.parametrize(
    ("declared", "built", "refused"),
    [
        (Verification.ANSWER_IN_BOX, Verification.CODE, True),
        (Verification.LLM_AS_JUDGE, Verification.CODE, True),
        (Verification.CODE, Verification.LLM_AS_JUDGE, True),
        (Verification.DECISION_MODEL, Verification.LLM_AS_JUDGE, True),
        (Verification.LLM_AS_JUDGE, Verification.DECISION_MODEL, True),
        (Verification.DECISION_MODEL, Verification.DECISION_MODEL, False),
        (Verification.LLM_AS_JUDGE, Verification.LLM_AS_JUDGE, False),
        (Verification.CODE, Verification.CODE, False),
    ],
    ids=str,
)
def test_the_built_kind_must_be_the_declared_kind(declared, built, refused):
    problem = verification_problem(declared, task(GRADERS[built]()))
    assert (problem is not None) is refused
    if problem is not None:
        assert f"declares {declared} verification but the built grader is {built}" in problem


def test_a_composite_cannot_be_built_until_a_composite_grader_exists():
    problem = unbuildable_problem(Verification.COMPOSITE)
    assert problem is not None and "no composite grader type exists" in problem
    assert unbuildable_problem(Verification.CODE) is None
    assert unbuildable_problem(Verification.DECISION_MODEL) is None


def test_the_judge_grader_names_the_default_judge_and_carries_the_rubric():
    built = TaskSpec.model_validate_json(task(draft.judge_grader(RUBRIC, REFERENCE)).model_dump_json())

    assert isinstance(built.grader, VerifyitGrader)
    judge = verifyit_spec(built.grader)
    assert isinstance(judge, JudgeSpec)
    assert (judge.model, judge.references, judge.context) == (DEFAULT_JUDGE_MODEL, (REFERENCE,), JUDGE_SPEC_PATH)
    (spec_file,) = built.resources.verifier
    spec = json.loads(resource_bytes(spec_file))
    assert (spec["judge_strength"], spec["rubric"]["deliverable"]) == (JudgeStrength.DEFAULT, RUBRIC.deliverable)
