# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import httpx
import pytest
from lib.forge.tests.test_stages import resources
from rolloutengine.models import TraceMode
from rolloutengine.test_support import arithmetic_task
from rolloutengine.validation import InvalidTaskSpec
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import Outcome
from taskcompendium.models import DECISION_LEVELS, DecisionModelGrader, ScriptGrader, TaskSpec
from taskforge.llm.glm_client import FixedEndpoint, GlmEndpoint, Pool
from verifyit.spec import ExactSpec, JudgeSpec, NumericSpec

from forge.effects.context import Context
from forge.effects.issues import DecisionModelUnavailable
from forge.effects.log import EventLog
from forge.lowering import check_verification, lower_with_settings, lowering_settings, requirements, rollout_engine
from forge.model.controls import Candidate
from forge.model.proposal import Verification
from forge.policy import load_policy
from forge.policy.defaults import DEFAULTS_DIR
from forge.resources.decision_model import UNAVAILABLE_MARKER, ClefDecisionModel
from forge.resources.grading import GradeCandidate, GradingResource
from forge.wiring import install, provider_policy

POLICY = load_policy(DEFAULTS_DIR)
GRADER = DecisionModelGrader(
    deliverable="State the sum of 5 and 7.",
    rubric="The answer states 12; any other value, several values or a list of numbers earns nothing.",
    reference="12",
    cut=2.5,
)


def decision_task(grader: DecisionModelGrader = GRADER) -> TaskSpec:
    return arithmetic_task(grader=grader)


class FakeClef:
    """Clef's /v1/systemone: "matches exactly" for "Twelve", "partially matches" for anything else."""

    def __init__(self, statuses: list[int] = []) -> None:
        self.statuses = list(statuses)
        self.reads: list[dict] = []
        self.clock = 0.0

    def respond(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/systemone" and request.headers["authorization"] == "Bearer token"
        if self.statuses:
            return httpx.Response(self.statuses.pop(0), headers={"retry-after": "30"}, json={"error": {}})
        body = json.loads(request.content)
        self.reads.append(body)
        level = 4 if body["state"]["z_candidate_answer"] == "Twelve" else 2
        probabilities = {str(index): float(index == level) for index in range(len(DECISION_LEVELS))}
        return httpx.Response(
            200,
            json={"model": "clef", "answers": {"whole": {"probabilities": probabilities}}, "usage": {"input_tokens": 9}},
        )

    def client(self, unavailable_wait: float = 900.0) -> ClefDecisionModel:
        async def sleep(seconds: float) -> None:
            self.clock += seconds

        return ClefDecisionModel(
            FixedEndpoint(GlmEndpoint("http://gateway/v1", "token", Pool.HIGH)),
            httpx.AsyncClient(transport=httpx.MockTransport(self.respond)),
            request_timeout=60.0,
            unavailable_wait=unavailable_wait,
            max_inflight=2,
            clock=lambda: self.clock,
            sleep=sleep,
        )


async def grade(clef: ClefDecisionModel, task: TaskSpec, replies: tuple[str, ...], history: tuple = ()) -> tuple:
    log = EventLog(replay_from=history)
    ctx = Context(log=log)
    spec = lower_with_settings(task, lowering_settings(POLICY), {}).model_copy(update={"trace": TraceMode.MESSAGES})
    bundle = replace(
        resources(ctx), grading=GradingResource(rollout_engine({}, command_timeout=None), "grading"), decision_model=clef
    )
    install(ctx, bundle)
    try:
        return tuple([await ctx.emit(GradeCandidate(spec, Candidate(reply))) for reply in replies]), log.records
    finally:
        await ctx.drain()


async def test_decision_model_grade_holds_through_unavailability_reports_level_and_replays_across_cuts() -> None:
    fake = FakeClef([503, 404])
    (right, fooled), history = await grade(fake.client(), decision_task(), ("Twelve", "9 or 12"))
    assert (right.status, right.reward, fooled.status, fooled.reward) == (Outcome.GRADED, 1.0, Outcome.GRADED, 0.0)
    assert right.detail == {"decision_model": {"expected_level": 4.0, "cut": 2.5}}
    assert fooled.detail == {"decision_model": {"expected_level": 2.0, "cut": 2.5}}
    assert fake.clock == 60.0 and not fake.statuses  # retry-after honoured twice
    state = fake.reads[0]["state"]
    assert state == {
        "task_deliverable": GRADER.deliverable,
        "rubric": GRADER.rubric,
        "y_reference_answer": "12",
        "z_candidate_answer": "Twelve",
    }
    assert fake.reads[0]["questions"]["whole"]["criteria"] == list(DECISION_LEVELS)
    # The cut is not part of a read: an author who lowers it replays the logged reads without Clef.
    offline = FakeClef()
    (_, lowered), _ = await grade(
        offline.client(), decision_task(replace_cut(2.0)), ("Twelve", "9 or 12"), tuple(history)
    )
    assert lowered.reward == 1.0 and not offline.reads


def replace_cut(cut: float) -> DecisionModelGrader:
    return GRADER.model_copy(update={"cut": cut})


async def test_long_decision_model_outage_holds_the_workpiece_instead_of_failing_it() -> None:
    fake = FakeClef([503] * 100)
    with pytest.raises(DecisionModelUnavailable, match=UNAVAILABLE_MARKER):
        await grade(fake.client(unavailable_wait=600.0), decision_task(), ("Twelve",))
    assert not fake.reads and fake.clock == 600.0


def test_verification_requires_its_own_grader_kind() -> None:
    settings = lowering_settings(POLICY)
    script = arithmetic_task(grader=ScriptGrader(argv=("true",), environment=requirements(image=None)))
    judge = arithmetic_task(grader=verifyit_package(JudgeSpec(references=("12",))).grader)
    exact = arithmetic_task(grader=verifyit_package(ExactSpec(expected=("12",))).grader)
    decision = decision_task()
    numeric = arithmetic_task(
        grader=verifyit_package(NumericSpec(expected="12", tolerance_abs=0.0, tolerance_rel=0.0)).grader
    )
    allowed = {
        Verification.DECISION_MODEL: (decision,),
        Verification.ANSWER_IN_BOX: (exact, numeric),
        Verification.CODE: (script, exact, numeric),
    }
    for verification, accepted in allowed.items():
        for task in (decision, script, judge, exact, numeric):
            if any(task is ok for ok in accepted):
                check_verification(task, verification, settings)
            else:
                with pytest.raises(InvalidTaskSpec, match=f"Verification is {verification}"):
                    check_verification(task, verification, settings)
    disabled = POLICY.overlaid({"decision_model.model": ""})
    with pytest.raises(ValueError, match="decision_model.model"):
        check_verification(decision_task(), Verification.DECISION_MODEL, lowering_settings(disabled))
    with pytest.raises(ValueError, match="disabled"):
        lower_with_settings(decision_task(), lowering_settings(disabled), {})
    with pytest.raises(ValueError, match="includes decision_model"):
        provider_policy(disabled)
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        replace_cut(1.0).model_validate(replace_cut(1.0).model_dump())
