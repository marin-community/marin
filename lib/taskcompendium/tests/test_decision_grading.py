# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decision-model grading on the host: the state the model reads, the level it returns, and the reward."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest
from pydantic import JsonValue

from taskcompendium.decision import (
    LEVELS,
    QUESTION_ID,
    SOLVER_TURN_STATE,
    DecisionGrader,
    ItemRole,
    Rubric,
    RubricItem,
    ScoreQuestion,
)
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.task_grading import grade_task

REFERENCE = "The approved total is 1,234.56 USD: the two approved invoices, 1,000.00 and 234.56."
REPLY = "Approved total: $1,234.56 (invoices of 1,000.00 and 234.56)."
RUBRIC = Rubric(
    deliverable="The approved total of the ledger and the entries behind it.",
    items=(
        RubricItem(
            id="total",
            intent="The answer states the approved total.",
            pass_criterion="The approved total is 1234.56 USD, within 0.01.",
            equivalences=("1234.56", "$1,234.56", "USD 1234.56"),
            negation_handling="A total mentioned as rejected or hypothetical does not count.",
            weight=1.0,
            role=ItemRole.SCORED,
            cap_value=None,
        ),
    ),
)
PROMPT = TextMessage(role="user", content="Reconcile the ledger and tell me the approved total.")


def probabilities(*, low: int, high_share: float) -> tuple[float, ...]:
    """A read split between level ``low`` and the next level, whose expected level is ``low + high_share``."""
    read = [0.0] * len(LEVELS)
    read[low] = 1 - high_share
    if high_share:
        read[low + 1] = high_share
    return tuple(read)


@dataclass
class FakeDecisionClient:
    """Answers every question with one fixed read and keeps every request."""

    read: tuple[float, ...]
    requests: list[tuple[Mapping[str, JsonValue], Mapping[str, ScoreQuestion]]] = field(default_factory=list)

    async def decide(
        self, state: Mapping[str, JsonValue], questions: Mapping[str, ScoreQuestion]
    ) -> Mapping[str, tuple[float, ...]]:
        self.requests.append((state, questions))
        return {question: self.read for question in questions}


def decision_task(acceptance_level: float | None) -> TaskSpec:
    calibration = {} if acceptance_level is None else {"reference": acceptance_level}
    grader = DecisionGrader(
        rubric=RUBRIC, reference=REFERENCE, acceptance_level=acceptance_level, calibration=calibration
    )
    return TaskSpec(
        id="ledger",
        context=ConversationInput(events=(PROMPT,)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader,
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
    )


def attempt(*events: TextMessage) -> GradingAttempt:
    return GradingAttempt(ConversationTrace(events=(PROMPT, *events)))


def test_decision_state_holds_the_solver_final_visible_turn_and_the_reference():
    client = FakeDecisionClient(probabilities(low=3, high_share=0.0))
    trial = attempt(
        TextMessage(role="assistant", content="A first draft: 1,000.00."),
        TextMessage(role="user", content="Include every approved invoice."),
        TextMessage(role="assistant", content=REPLY),
    )

    grade_task(decision_task(None), trial, decision_client=client)

    [(state, questions)] = client.requests
    assert list(state) == list(SOLVER_TURN_STATE)
    assert state["z_candidate_answer"] == REPLY
    assert state["y_reference_answer"] == REFERENCE
    assert state["task_deliverable"] == RUBRIC.deliverable
    assert list(questions) == [QUESTION_ID]
    assert questions[QUESTION_ID].levels == LEVELS


def test_decision_final_turn_without_text_fails_the_submission_without_a_read():
    client = FakeDecisionClient(probabilities(low=4, high_share=0.0))

    result = grade_task(decision_task(None), attempt(TextMessage(role="assistant", content="  ")), decision_client=client)

    assert (result.status, result.reward, client.requests) == (Outcome.SUBMISSION_FAILURE, 0.0, [])


@pytest.mark.parametrize(
    "client",
    [
        pytest.param(None, id="no-client"),
        pytest.param(FakeDecisionClient((0.0, 0.0, 1.0)), id="malformed-read"),
    ],
)
def test_decision_grading_without_a_usable_client_is_an_infrastructure_error(client):
    result = grade_task(decision_task(2.5), attempt(TextMessage(role="assistant", content=REPLY)), decision_client=client)

    assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None)


@pytest.mark.parametrize(
    ("acceptance_level", "read", "level", "reward", "accepted"),
    [
        # Below the bar the reward is 0.5 * min(1, credit(level) / credit(bar)); credits are 0, 0, 0.5, 1, 1.
        pytest.param(2.5, probabilities(low=3, high_share=0.5), 3.5, 1.0, True, id="above-bar"),
        pytest.param(2.5, probabilities(low=2, high_share=0.5), 2.5, 1.0, True, id="at-bar"),
        pytest.param(2.5, probabilities(low=2, high_share=0.25), 2.25, 0.5 * 0.625 / 0.75, False, id="below-bar"),
        pytest.param(2.5, probabilities(low=1, high_share=0.0), 1.0, 0.0, False, id="contradicts"),
        pytest.param(None, probabilities(low=2, high_share=0.0), 2.0, 1.0, True, id="null-bar-partial"),
        pytest.param(None, probabilities(low=1, high_share=0.5), 1.5, 0.5 * 0.25 / 0.5, False, id="null-bar-below"),
    ],
)
def test_decision_reward_follows_the_expected_level_against_the_bar(acceptance_level, read, level, reward, accepted):
    task = decision_task(acceptance_level)

    result = grade_task(task, attempt(TextMessage(role="assistant", content=REPLY)), decision_client=FakeDecisionClient(read))

    # A null acceptance level is the "partially matches" bar, level 2.
    bar = 2.0 if acceptance_level is None else acceptance_level
    assert (result.status, result.passed) == (Outcome.GRADED, accepted)
    assert result.reward == pytest.approx(reward)
    assert result.detail == {"level": pytest.approx(level), "bar": bar, "accepted": accepted}
