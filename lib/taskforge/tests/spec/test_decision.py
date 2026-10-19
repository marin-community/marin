# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The decision grader type, its TaskSpec form, and the host-side grade of a trial run without it."""

from dataclasses import dataclass

import pytest
from pydantic import ValidationError
from rolloutengine.contracts import RolloutInterrupted, RolloutOperation
from rolloutengine.test_support import ReplayModel, engine, lowered, machine_runtime
from shellbox.machine import MachineSpec
from taskcompendium.chat import chat_conversation
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    GradingAttempt,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    VerifyitGrader,
)
from verifyit.spec import ExactSpec

from taskforge.spec import draft
from taskforge.spec.decision import (
    QUESTION_ID,
    REJECTED_REWARD_MAX,
    SOLVER_TURN_STATE,
    DecisionGrader,
    StateSource,
    decision_grade,
    grade_decision_answer,
    known_correct_check,
    run_decision_trial,
    task_decision_grader,
)
from taskforge.spec.rubric import Rubric, RubricItem
from tests.spec.fake_decision import FakeDecisionClient

SOURCE = Source(dataset="taskforge-test", revision="r1", row="0", importer_revision="test")
REFERENCE = "The approved total is 1,234.56 USD: the two approved invoices, 1,000.00 and 234.56."
INSTRUCTION = "Reconcile the ledger below and tell me the approved total, with the entries behind it."
REPLY = "Approved total: $1,234.56 (invoices of 1,000.00 and 234.56)."
RUBRIC = Rubric(
    deliverable="The approved total of the ledger and the entries behind it.",
    items=[
        RubricItem(
            id="total",
            intent="The answer states the approved total.",
            pass_criterion="The approved total is 1234.56 USD, within 0.01.",
            equivalences=["1234.56", "$1,234.56", "USD 1234.56"],
            negation_handling="A total mentioned as rejected or hypothetical does not count.",
            weight=1.0,
            role="scored",
            cap_value=None,
        )
    ],
)
# A fixed transcript: one assistant turn whose reasoning the decision model must not see.
SOLVER_TURN = {"role": "assistant", "content": REPLY, "reasoning_content": "Sum the approved rows first."}
TRANSCRIPT = [{"role": "user", "content": INSTRUCTION}, SOLVER_TURN]


def task(grader) -> TaskSpec:
    return draft.assemble("ledger", INSTRUCTION, AnswerType.TEXT, PlainText(), grader, SOURCE, environment=None)


async def graded(grader: DecisionGrader, messages: list[dict], client: FakeDecisionClient):
    """The host's grade of ``messages`` on the ledger task carrying ``grader``."""
    return await grade_decision_answer(task(grader), GradingAttempt(chat_conversation(messages)), client)


def uncalibrated() -> DecisionGrader:
    return draft.decision_grader(RUBRIC, REFERENCE)


def calibrated(bar: float) -> DecisionGrader:
    return uncalibrated().model_copy(update={"acceptance_level": bar, "calibration": {"reference": bar}})


def test_the_grader_survives_the_task_round_trip_as_a_no_grader_contract():
    grader = calibrated(2.9)

    restored = TaskSpec.model_validate_json(task(grader).model_dump_json())

    assert task_decision_grader(restored) == grader
    assert isinstance(restored.grader, NoGrader) and restored.resources.verifier == ()
    assert task_decision_grader(task(draft.answer_grader(ExactSpec(expected=("1234.56",))))) is None


def test_the_default_state_is_the_solver_turn_template_and_a_state_must_carry_the_solver_output():
    assert uncalibrated().state == dict(SOLVER_TURN_STATE)
    assert (uncalibrated().acceptance_level, uncalibrated().bar) == (None, 2.0)

    with pytest.raises(ValidationError, match="must include solver_turn_output"):
        draft.decision_grader(RUBRIC, REFERENCE, state={"rubric": StateSource.RUBRIC})


@pytest.mark.parametrize(
    ("levels", "problem", "bar"),
    [
        ({"reference": 3.9, "variant:1": 3.6, "variant:2": 3.2}, None, 3.0),
        ({"reference": 3.9, "variant:1": 3.6, "variant:2": 2.7}, None, 2.7),
        ({"reference": 2.4, "variant:1": 3.6}, None, 2.4),
        ({"reference": 3.9, "variant:1": 1.7}, "'variant:1': 1.7", None),
        ({}, "no known-correct candidate was read", None),
    ],
    ids=["capped-at-equivalent", "lowest-variant", "reference-is-in-the-set", "below-partial", "no-reads"],
)
def test_the_known_correct_check_refuses_or_sets_the_bar(levels, problem, bar):
    check = known_correct_check(levels)
    assert check.acceptance_level == bar
    assert (check.problem is None) == (problem is None)
    if problem is not None:
        assert check.problem is not None and problem in check.problem


def test_below_the_bar_the_reward_stays_under_the_rejection_ceiling():
    assert decision_grade(2.9, 2.9).reward == 1.0
    assert decision_grade(2.5, 2.9).reward < REJECTED_REWARD_MAX
    assert decision_grade(0.4, 2.9).reward == 0.0


async def test_the_host_grade_reads_the_visible_solver_turn_against_the_calibrated_bar():
    client = FakeDecisionClient({REPLY: 2.4})

    lenient = await graded(uncalibrated(), TRANSCRIPT, client)
    strict = await graded(calibrated(2.9), TRANSCRIPT, client)

    assert (lenient.status, lenient.reward, lenient.detail) == (
        Outcome.GRADED,
        1.0,
        {"level": 2.4, "bar": 2.0, "accepted": True},
    )
    assert strict.passed is False and strict.reward is not None and strict.reward < REJECTED_REWARD_MAX
    state, questions = client.requests[0]
    assert list(state) == list(SOLVER_TURN_STATE)
    assert (state["y_reference_answer"], state["z_candidate_answer"]) == (REFERENCE, REPLY)
    assert "Sum the approved rows" not in str(state)
    assert list(questions) == [QUESTION_ID] and len(questions[QUESTION_ID].levels) == 5


async def test_a_program_that_changes_the_state_template_changes_what_the_model_reads():
    state = {"answer": StateSource.SOLVER_TURN_OUTPUT, "gold": StateSource.REFERENCE}
    client = FakeDecisionClient({REPLY: 3.5}, candidate_key="answer")

    result = await graded(draft.decision_grader(RUBRIC, REFERENCE, state=state), TRANSCRIPT, client)

    assert result.passed is True
    assert client.requests[0][0] == {"answer": REPLY, "gold": REFERENCE}


async def test_a_conversation_without_a_final_reply_scores_zero_without_a_decision_call():
    client = FakeDecisionClient({}, default=4.0)

    result = await graded(uncalibrated(), [TRANSCRIPT[0], {"role": "assistant", "content": "  "}], client)

    assert (result.status, result.reward, client.requests) == (Outcome.SUBMISSION_FAILURE, 0.0, [])


async def test_the_trial_runs_without_the_grader_and_the_host_grades_its_final_conversation():
    built = task(calibrated(2.0))
    client = FakeDecisionClient({REPLY: 3.6})
    model = ReplayModel([dict(SOLVER_TURN)])

    rollout = await run_decision_trial(engine(model), lowered(built), client)

    assert (rollout.grade.status, rollout.grade.reward, rollout.grade.passed) == (Outcome.GRADED, 1.0, True)
    assert client.requests[0][0]["z_candidate_answer"] == REPLY
    # Nothing of the grader reached the solver: its prompt is the task's public conversation alone.
    assert REFERENCE not in str(model.requests[0].messages)


@dataclass(frozen=True)
class NoVerifierMachines:
    """A factory whose machines never start, so a verifier-machine grade fails after the conversation ended."""

    backend: str = "local"

    def check_environment(self, requirements, spec: MachineSpec) -> None:
        pass

    async def create(self, spec: MachineSpec):
        raise RuntimeError("no verifier machine in this test")


async def test_the_no_grader_trial_has_the_same_conversation_as_the_judge_spec_rollout():
    """On a fixed transcript, swapping the grader for NoGrader leaves the conversation the solver had unchanged."""
    decision_task = task(uncalibrated())
    judge_task = task(draft.judge_grader(RUBRIC, REFERENCE))
    assert isinstance(judge_task.grader, VerifyitGrader)
    trial = await run_decision_trial(
        engine(ReplayModel([dict(SOLVER_TURN)])),
        lowered(decision_task),
        FakeDecisionClient({REPLY: 3.0}),
    )

    with pytest.raises(RolloutInterrupted) as interrupted:
        await engine(ReplayModel([dict(SOLVER_TURN)]), {"local": NoVerifierMachines()}).run(
            lowered(judge_task, verifier_machine=machine_runtime())
        )

    assert interrupted.value.operation is RolloutOperation.GRADE
    judged = interrupted.value.rollout
    assert trial.messages == judged.messages
    assert trial.messages[-1]["content"] == REPLY
    assert [step.messages for step in trial.steps] == [step.messages for step in judged.steps]
