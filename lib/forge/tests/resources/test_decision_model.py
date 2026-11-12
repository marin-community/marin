# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import override

import httpx
import pytest
from lib.forge.tests.test_stages import resources
from rolloutengine.models import TaskRolloutSpec, TraceMode
from rolloutengine.test_support import arithmetic_task
from rolloutengine.validation import InvalidTaskSpec
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    DECISION_LEVELS,
    AnswerType,
    DecisionModelGrader,
    JsonValueAnswer,
    ScriptGrader,
    TaskSpec,
)
from taskforge.llm.glm_client import FixedEndpoint, GlmEndpoint, Pool
from verifyit.spec import ExactSpec, JudgeSpec, NumericSpec

from forge.coordinator import reasons
from forge.driver import decision_model_held
from forge.effects.context import Context
from forge.effects.issues import DecisionModelUnavailable, Hold
from forge.effects.log import EventLog
from forge.harness.protocol import Harness, Tool
from forge.lowering import check_verification, lower_with_settings, lowering_settings, requirements, rollout_engine
from forge.model.controls import Candidate, Control, ControlKind, RewardBand
from forge.model.proposal import TaskProposal, Verification
from forge.model.review import Cause, Incomplete
from forge.model.stage import Exit
from forge.model.workpiece import Conversation, Submission, Terminal, Workpiece
from forge.policy import load_policy
from forge.policy.defaults import DEFAULTS_DIR
from forge.resources.decision_model import UNAVAILABLE_MARKER, ClefDecisionModel, clef_healthy
from forge.resources.grading import GradeCandidate, GradingResource, ScriptedModel, candidate_turns
from forge.resources.inference import InferenceClient
from forge.resources.rollouts import RolloutResource
from forge.stages.classify import grade_cause
from forge.stages.validate import adversary_result, control_result, failure_reasons, solver_result
from forge.wiring import build_config, install, provider_policy

pytest_plugins = ("lib.forge.tests.test_proposal",)

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

    def __init__(self, statuses: tuple[int, ...] = ()) -> None:
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
    fake = FakeClef((503, 404))
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
    fake = FakeClef((503,) * 100)
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
    with pytest.raises(ValueError, match=r"decision_model\.model"):
        check_verification(decision_task(), Verification.DECISION_MODEL, lowering_settings(disabled))
    with pytest.raises(ValueError, match="disabled"):
        lower_with_settings(decision_task(), lowering_settings(disabled), {})
    with pytest.raises(ValueError, match="includes decision_model"):
        provider_policy(disabled)
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        replace_cut(1.0).model_validate(replace_cut(1.0).model_dump())


class GradingAdversary(Harness):
    """An adversary whose first grade-tool call meets the decision-model outage."""

    def __init__(self, spec: TaskRolloutSpec) -> None:
        self.spec = spec

    @override
    async def run(
        self, ctx: Context, session: Conversation, client: InferenceClient, tools: Mapping[str, Tool[object]]
    ) -> Exit:
        await ctx.emit(GradeCandidate(self.spec, Candidate("Twelve")))
        raise AssertionError("the outage must interrupt the adversary")


def acyclic(error: BaseException) -> bool:
    seen: set[int] = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
            continue
        while current is not None:
            if id(current) in seen:
                return False
            seen.add(id(current))
            current = current.__cause__
    return True


# full-s-20261108-dm canary, 2026-11-11 03:06Z: Clef down for 900 s inside an adversary grade; the engine's
# wrapper and the re-raised hold were each other's cause, failure_reasons recursed and the workpiece failed.
@pytest.mark.parametrize("path", ["control", "solver", "adversary"])
async def test_outage_inside_validation_grades_holds_the_workpiece_with_an_acyclic_cause(
    path: str, proposal: TaskProposal
) -> None:
    fake = FakeClef((503,) * 100)
    task = decision_task()
    ctx = Context(log=EventLog())
    spec = lower_with_settings(task, lowering_settings(POLICY), {}).model_copy(update={"trace": TraceMode.MESSAGES})
    solver = ScriptedModel(candidate_turns(Candidate("Twelve"), spec), 0)
    bundle = replace(
        resources(ctx),
        grading=GradingResource(rollout_engine({}, command_timeout=None), "grading"),
        rollouts=RolloutResource(rollout_engine({}, command_timeout=None), solver, "solver"),
        decision_model=fake.client(unavailable_wait=60.0),
    )
    install(ctx, bundle)
    control = Control(Candidate("Twelve"), RewardBand(Outcome.GRADED, reward_min=1), ControlKind.POSITIVE)
    work = Workpiece("dm", proposal, submission=Submission(task, (control,), "authored"))
    setup = replace(build_config(POLICY, "S", {}), adversaries={"finder": GradingAdversary(spec)}, tools={"finder": {}})
    with pytest.raises(BaseException) as caught:
        if path == "control":
            await control_result(ctx, spec, 0, control)
        elif path == "solver":
            await solver_result(ctx, spec, 0, 0)
        else:
            await adversary_result(ctx, work, setup, bundle, spec, "finder")
    await ctx.drain()
    error = caught.value
    held = error.subgroup(DecisionModelUnavailable) if isinstance(error, BaseExceptionGroup) else error
    assert isinstance(held, (DecisionModelUnavailable, BaseExceptionGroup)) and held is not None
    assert acyclic(error)
    assert any(UNAVAILABLE_MARKER in reason for reason in failure_reasons(error))
    assert not fake.reads


def test_failure_reasons_visits_each_exception_once_on_a_cyclic_chain() -> None:
    first, second = RuntimeError("first"), Hold("second")
    first.__cause__, second.__cause__ = second, first
    assert failure_reasons(ExceptionGroup("observer", [first, second])) == (
        "RuntimeError: first",
        "Hold: second",
    )


# full-s-20261108-dm, 2026-11-11: a json_value decision_model task reached the grader and restarted on every resume.
async def test_decision_model_refuses_structured_answers_at_submit_and_returns_them_to_the_author() -> None:
    task = arithmetic_task(grader=GRADER, answer_type=AnswerType.JSON, answer_format=JsonValueAnswer())
    with pytest.raises(InvalidTaskSpec, match="answer_format 'json_value' submits structured data"):
        check_verification(task, Verification.DECISION_MODEL, lowering_settings(POLICY))
    fake = FakeClef()
    (result,), _ = await grade(fake.client(), task, ('{"sum": 12}',))
    assert result.status is Outcome.INVALID_TASK and grade_cause(result, task) is Cause.INVALID_TASK
    assert not fake.reads


def test_driver_reads_the_decision_model_hold_the_coordinator_recorded(proposal: TaskProposal) -> None:
    outage = ExceptionGroup("validate", [DecisionModelUnavailable(f"{UNAVAILABLE_MARKER}: no answer for 900s")])
    wp = Workpiece("dm", proposal)
    held = wp.with_(terminal=Terminal.HELD, decisions=(Incomplete(reasons(outage)),))
    assert decision_model_held(held)
    assert not decision_model_held(wp.with_(terminal=Terminal.HELD, decisions=(Incomplete(reasons(Hold("GLM"))),)))
    assert not decision_model_held(wp)


async def test_clef_health_is_a_get_of_the_gateway_health_route_and_anything_but_200_is_down() -> None:
    requests: list[tuple[str, str]] = []

    def gateway(status: int) -> httpx.AsyncClient:
        def respond(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            return httpx.Response(status, json={})

        return httpx.AsyncClient(transport=httpx.MockTransport(respond))

    async def hang(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("relay down")

    source = FixedEndpoint(GlmEndpoint("http://gateway/v1", "token", Pool.HIGH))
    assert await clef_healthy(source, gateway(200))
    assert not await clef_healthy(source, gateway(503))
    assert requests == [("GET", "/v1/systemone/health")] * 2  # never a Clef read
    assert not await clef_healthy(source, httpx.AsyncClient(transport=httpx.MockTransport(unreachable)))
    hung = httpx.AsyncClient(transport=httpx.MockTransport(hang))
    assert not await asyncio.wait_for(clef_healthy(source, hung, timeout=0.05), 5)
