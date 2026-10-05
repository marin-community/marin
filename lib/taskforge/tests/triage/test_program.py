# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from dataclasses import replace

import pytest

from taskforge.llm.client import FinishReason, ToolCall, Usage
from taskforge.triage.checks import ALL_COMBINATIONS, CHECKS, CheckContext, CheckStatus
from taskforge.triage.program import REVIEW_TOOL, RubricAssessment, evaluate, rubric_decision, rubric_result
from taskforge.triage.verdict import ModelCall, RubricAxis, RubricResult, TriageDecision, Verdict

ACCEPT, REPAIR, REJECT = TriageDecision.ACCEPT, TriageDecision.REPAIR, TriageDecision.REJECT
CTX = CheckContext(allowed_combinations=ALL_COMBINATIONS)
CALL = ModelCall(Usage(100, 50, 40, 0), wall_time=2.0, finish_reason=FinishReason.TOOL_CALLS)


def rubric(scores=4, critical=(), changes=(), recommendation=ACCEPT, **overrides):
    axes = tuple((axis, overrides.get(axis.value, scores)) for axis in RubricAxis)
    return RubricResult(axes, tuple(critical), ("an observation",), tuple(changes), recommendation)


class FakeRubric:
    def __init__(self, *samples):
        self.samples = samples
        self.assessed = []

    async def assess(self, p, structural):
        self.assessed.append((p, tuple(structural)))
        return RubricAssessment(self.samples, (CALL,) * len(self.samples))

    async def repair(self, p, verdict):
        raise AssertionError("evaluate never repairs")


def run(p, fake, ctx=CTX):
    return asyncio.run(evaluate(p, CHECKS, fake, ctx))


def test_fatal_structural_failure_rejects_without_a_model_call(proposal):
    fake = FakeRubric(rubric())
    ctx = CheckContext(allowed_combinations=frozenset())
    verdict = run(proposal, fake, ctx)
    assert fake.assessed == []
    assert verdict.decision is REJECT and verdict.rubric == () and verdict.calls == ()
    assert verdict.reasons == ("combination_allowed: reasoning x code is not an allowed combination",)


def test_null_proposal_rejects_without_a_model_call(proposal):
    header = replace(
        proposal.header, research=(), build=(), resources=(), null_reason="no gradable artifact exists here"
    )
    fake = FakeRubric(rubric())
    verdict = run(replace(proposal, header=header), fake)
    assert fake.assessed == []
    assert verdict.decision is REJECT
    assert verdict.reasons == ("null proposal: no gradable artifact exists here",)


def test_rubric_sees_advisory_failures(proposal, section_replaced):
    p = section_replaced(proposal, "Build plan", "Sessions follow the usual order with acceptance gates. " * 10)
    fake = FakeRubric(rubric())
    verdict = run(p, fake)
    [(_, structural)] = fake.assessed
    failed = [r.name for r in structural if r.status is CheckStatus.FAIL]
    assert failed == ["resources_in_build_plan"]
    assert verdict.decision is ACCEPT and verdict.calls == (CALL,)


@pytest.mark.parametrize(
    ("result", "decision"),
    [
        (rubric(), ACCEPT),
        (rubric(recommendation=REPAIR), ACCEPT),
        (rubric(reward_validity=3), REPAIR),
        (rubric(reward_validity=3, recommendation=REJECT), REJECT),
        (rubric(critical=("answer leaks into the prompt",)), REPAIR),
        (rubric(changes=("pin the tolerance",), recommendation=ACCEPT), REPAIR),
    ],
)
def test_accept_rule(result, decision):
    assert rubric_decision([result])[0] is decision


def test_blocking_reasons_name_each_cause():
    _, reasons = rubric_decision([rubric(specificity=2, critical=("leak",), changes=("fix units",))])
    assert reasons == (
        "0 accept, 1 repair, 0 reject of 1 samples",
        "specificity scored 2 (< 4)",
        "critical failure: leak",
        "required change: fix units",
    )


@pytest.mark.parametrize(
    ("samples", "decision"),
    [
        ((rubric(), rubric(), rubric(reward_validity=3)), ACCEPT),
        ((rubric(), rubric(reward_validity=3), rubric(alignment=2)), REPAIR),
        ((rubric(reward_validity=2, recommendation=REJECT),) * 2 + (rubric(),), REJECT),
        ((rubric(), rubric(reward_validity=3), rubric(reward_validity=2, recommendation=REJECT)), REPAIR),
        ((rubric(), rubric(reward_validity=3)), REPAIR),
    ],
)
def test_decision_is_the_sample_majority(samples, decision):
    assert rubric_decision(samples)[0] is decision


def test_repair_reasons_merge_every_non_accepting_sample():
    samples = (rubric(), rubric(reward_validity=3), rubric(reward_validity=3, changes=("fix units",)))
    assert rubric_decision(samples) == (
        REPAIR,
        ("1 accept, 2 repair, 0 reject of 3 samples", "reward_validity scored 3 (< 4)", "required change: fix units"),
    )


def test_verdict_round_trips_through_json(proposal):
    verdict = run(proposal, FakeRubric(rubric(alignment=3, recommendation=REPAIR), rubric()))
    assert Verdict.from_json(verdict.to_json()) == verdict


def test_tool_arguments_become_axis_scores():
    arguments = {axis.value: 4 for axis in RubricAxis} | {
        "reward_validity": 2,
        "critical_failures": [],
        "issues": ["tolerance unstated"],
        "required_changes": ["state the tolerance"],
        "recommendation": "repair",
    }
    calls = (ToolCall(id="call-0", name="record_review", arguments=json.dumps(arguments)),)
    result = rubric_result(REVIEW_TOOL.parse(calls))
    assert result.score(RubricAxis.REWARD_VALIDITY) == 2 and result.score(RubricAxis.REALISM) == 4
    assert rubric_decision([result]) == (
        REPAIR,
        (
            "0 accept, 1 repair, 0 reject of 1 samples",
            "reward_validity scored 2 (< 4)",
            "required change: state the tolerance",
        ),
    )
