# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from taskcompendium.grading_result import Outcome
from taskcompendium.reasoning_spans import ReasoningSpanRule, ReasoningViolation

from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import ModelRequest, ModelTurn
from rolloutengine.test_support import SHELL, arithmetic_task, rollout_spec


async def _correct_answer_after_a_second_span(request: ModelRequest) -> ModelTurn:
    return ModelTurn(
        {"role": "assistant", "content": "12"},
        (10, 11),
        (20,),
        (-0.5,),
        "stop",
        reasoning_violation=ReasoningViolation.SECOND_SPAN,
    )


@pytest.mark.parametrize(
    "policy,status,reward",
    [(ReasoningSpanRule.GRADE, Outcome.GRADED, 1.0), (ReasoningSpanRule.REJECT, Outcome.SUBMISSION_FAILURE, 0.0)],
)
async def test_the_engine_policy_decides_whether_a_malformed_span_fails_a_correct_answer(policy, status, reward):
    engine = RolloutEngine({}, {SHELL: ShellHarness(command_timeout=None)}, reasoning_span_rule=policy)

    result = await engine.run_task(rollout_spec(arithmetic_task()), _correct_answer_after_a_second_span)

    assert (result.grade.status, result.grade.reward) == (status, reward)
