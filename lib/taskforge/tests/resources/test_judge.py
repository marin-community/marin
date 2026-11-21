# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import pytest
from lib.taskforge.tests.fake_resources import resources
from lib.taskforge.tests.resource_doubles import FakeInference, FakeWeb
from lib.taskforge.tests.resources.inference_fakes import Backend, ScriptedProvider, provider_completion, segment
from rolloutengine.models import GradedRollout, TraceMode
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ScriptGrader
from verifyit.spec import JudgeSpec

from taskforge.effects.context import Context
from taskforge.effects.events import Resolved
from taskforge.effects.issues import ContextExceeded, Restart
from taskforge.effects.log import EventLog, Record
from taskforge.llm.client import ContextExhausted, Progress, Segment
from taskforge.llm.client import FinishReason as ProviderFinish
from taskforge.llm.policy import GLM_5_3, ModelProfile
from taskforge.lowering import InvalidTaskSpec, lower_with_settings, lowering_settings, requirements, rollout_engine
from taskforge.model.controls import Candidate
from taskforge.model.llm import Completion
from taskforge.policy import load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.grading import GradeCandidate, GradingResource
from taskforge.resources.inference import GlmInference, InferenceClient, InferenceRequest
from taskforge.resources.judge import JudgeEngine
from taskforge.resources.rollouts import SolverRollout
from taskforge.wiring import build_resources, install


async def test_cancel_host_judge_cancels_recorded_inference() -> None:
    entered, cancelled = asyncio.Event(), asyncio.Event()

    class BlockingInference(FakeInference):
        async def complete(self, request: InferenceRequest) -> Completion:
            self.requests.append(request)
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
            raise AssertionError("unreachable")

    ctx = Context(log=EventLog())
    client = BlockingInference(())
    task = arithmetic_task(grader=verifyit_package(JudgeSpec(references=("12",), exact_gate=False)).grader)
    spec = lower_with_settings(
        task, lowering_settings(load_policy(DEFAULTS_DIR).overlaid({"judge.model": "glm-5.3"})), {}
    )
    engine = JudgeEngine(rollout_engine({}, command_timeout=None), ctx, client)
    operation = asyncio.create_task(GradingResource(engine, "judge").grade(GradeCandidate(spec, Candidate("Twelve"))))
    try:
        await asyncio.wait_for(entered.wait(), 5)
    finally:
        operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    await asyncio.wait_for(cancelled.wait(), 5)
    await ctx.drain()
    assert len(client.requests) == 1


async def test_host_judge_preserves_provider_exhaustion_without_sdk_retry() -> None:
    ctx = Context(log=EventLog())
    error = Restart("GLM exhausted availability window")
    client = FakeInference((error,))
    task = arithmetic_task(grader=verifyit_package(JudgeSpec(references=("12",), exact_gate=False)).grader)
    spec = lower_with_settings(
        task, lowering_settings(load_policy(DEFAULTS_DIR).overlaid({"judge.model": "glm-5.3"})), {}
    )
    engine = JudgeEngine(rollout_engine({}, command_timeout=None), ctx, client)
    with pytest.raises(Restart) as caught:
        await GradingResource(engine, "judge").grade(GradeCandidate(spec, Candidate("Twelve")))
    await ctx.drain()
    assert caught.value is error
    assert len(client.requests) == 1


@pytest.mark.parametrize("rubric", ["reference", "checklist", "labels", "responses"])
async def test_host_judge_records_inference_and_replays_without_provider(rubric: str) -> None:
    # Full S 20261108: judge endpoint/model omissions held 11 authored tasks.
    labels = rubric in {"labels", "responses"}
    judge = JudgeSpec(
        rubric="labels" if labels else rubric,
        references=("12",),
        criteria=("Correct arithmetic",),
        exact_gate=False,
        api="responses" if rubric == "responses" else "chat_completions",
        label_scores={"PASS": 1.0, "FAIL": 0.0},
        prompt_template="Reference: {reference}; candidate: {candidate}",
    )
    task = arithmetic_task(grader=verifyit_package(judge).grader)
    policy = load_policy(DEFAULTS_DIR).overlaid({"judge.model": "glm-5.3"})
    spec = lower_with_settings(task, lowering_settings(policy), {}).model_copy(update={"trace": TraceMode.MESSAGES})
    history = ()
    for replay in (False, True):
        client = FakeInference(() if replay else (Completion("PASS" if labels else "SCORE: 1"),), identity="judge")
        log = EventLog(replay_from=history)
        ctx = Context(log=log)
        bundle = replace(
            resources(ctx),
            grading=GradingResource(rollout_engine({}, command_timeout=None), "judge"),
            role_inference={"judge": client},
        )
        install(ctx, bundle)
        # Change the outer grading event identity so only the inner inference effect can replay.
        result = await ctx.emit(GradeCandidate(spec, Candidate("Twelve"), fresh_verifier=replay))
        await ctx.drain()
        assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
        assert len(client.requests) == (0 if replay else 1)
        recorded = [
            record.event
            for record in log.records
            if isinstance(record.event, Resolved) and isinstance(record.event.request, InferenceRequest)
        ]
        assert recorded[-1].replayed is replay
        history = log.records


@pytest.mark.parametrize("script", [False, True])
def test_disabled_and_script_judges_return_actionable_authored_rejection(script: bool) -> None:
    judge = JudgeSpec(references=("12",))
    grader = (
        ScriptGrader(argv=("true",), environment=requirements(image=None), judge=judge)
        if script
        else verifyit_package(judge).grader
    )
    policy = load_policy(DEFAULTS_DIR).overlaid({"judge.model": ""})
    with pytest.raises(InvalidTaskSpec, match="host VerifyitGrader" if script else "policy judge.model"):
        lower_with_settings(arithmetic_task(grader=grader), lowering_settings(policy), {})


JUDGE_CAP = 3
CAPPED_JUDGE = replace(GLM_5_3, name="glm-5.3/output-3", max_output_tokens=JUDGE_CAP)


class CapSensitiveJudge(ScriptedProvider):
    """A host judge that passes an uncapped request and fails one capped at ``JUDGE_CAP`` tokens."""

    async def stream(self, body: Mapping[str, Any], progress: Progress) -> Segment:
        self.bodies.append(body)
        return segment("SCORE: 0" if body.get("max_tokens", JUDGE_CAP + 1) <= JUDGE_CAP else "SCORE: 1")


async def judged(
    rollout: bool, judge: ModelProfile, history: Sequence[Record]
) -> tuple[float | None, int, int, Sequence[Record]]:
    """A solver rollout or a candidate grade through ``build_resources`` and ``install``, replaying ``history``."""
    solver, host_judge = ScriptedProvider((segment("12"),)), CapSensitiveJudge(())
    providers = {
        "author": ScriptedProvider(()),
        "adversary": ScriptedProvider(()),
        "solver": solver,
        "judge": host_judge,
    }
    policy = load_policy(DEFAULTS_DIR).overlaid({f"{role}.model": "test-model" for role in providers})
    task = arithmetic_task(grader=verifyit_package(JudgeSpec(references=("12",), exact_gate=False)).grader)
    spec = lower_with_settings(task, lowering_settings(policy), {}).model_copy(update={"trace": TraceMode.MESSAGES})
    log = EventLog(replay_from=history)
    ctx = Context(log=log)
    install(ctx, build_resources(policy, providers, {}, FakeWeb({}, {}), {"judge": judge}))
    if rollout:
        reward = cast(GradedRollout, await ctx.emit(SolverRollout(spec, 0, 0))).grade.reward
    else:
        reward = (await ctx.emit(GradeCandidate(spec, Candidate("12")))).reward
    await ctx.drain()
    return reward, len(solver.bodies), len(host_judge.bodies), log.records


# Review 2026-11-21: under a judge capped at three tokens, a grade and a rollout replayed 1.0 with zero calls.
@pytest.mark.parametrize("rollout", [False, True], ids=["grade", "rollout"])
@pytest.mark.parametrize(
    ("recorded", "replayed"),
    [(GLM_5_3, GLM_5_3), (GLM_5_3, CAPPED_JUDGE), (CAPPED_JUDGE, GLM_5_3), (CAPPED_JUDGE, CAPPED_JUDGE)],
    ids=["default", "default-then-capped", "capped-then-default", "capped"],
)
async def test_an_outer_result_replays_only_under_the_judge_profile_that_shaped_it(
    rollout: bool, recorded: ModelProfile, replayed: ModelProfile
) -> None:
    *_, history = await judged(rollout, recorded, ())
    reward, solver_calls, judge_calls, _ = await judged(rollout, replayed, history)
    clean = 0.0 if replayed is CAPPED_JUDGE else 1.0
    calls = (0, 0) if recorded is replayed else (int(rollout), 1)
    assert (reward, (solver_calls, judge_calls)) == (clean, calls)


# Review 2026-11-21, follow-up 2: a judge provider reporting another model, under the same policy and profile,
# replayed a grade and a rollout of 1.0 with zero calls where a clean run scored 0.0.
@pytest.mark.parametrize("role", ["author", "solver", "adversary", "judge"])
def test_a_provider_serving_another_model_than_its_role_declares_is_refused(role: str) -> None:
    providers = {name: ScriptedProvider((segment("12"),)) for name in ("author", "solver", "adversary", "judge")}
    policy = load_policy(DEFAULTS_DIR).overlaid({f"{name}.model": "test-model" for name in providers})
    providers[role] = ScriptedProvider((segment("12"),), "other-model")
    with pytest.raises(
        ValueError, match=f"{role} provider serves 'other-model', policy declares {role}.model='test-model'"
    ):
        build_resources(policy, providers, {}, FakeWeb({}, {}))
    assert not any(provider.bodies for provider in providers.values())


ANSWER = replace(
    provider_completion(), content="SCORE: 1", tool_calls=(), finish_reason=ProviderFinish.STOP, continuations=0
)


# Review 2026-11-21 (n12 follow-up 3): a context refusal recorded inside the shared inference resource cached the
# judge's refusal, so after the provider recovered the same grade made no call and raised again.
@pytest.mark.parametrize("injected", [True, False], ids=["injected client", "provider"])
async def test_a_judge_context_refusal_is_asked_again_once_the_provider_recovers(injected: bool) -> None:
    refusal = ContextExhausted("maximum context length exceeded", ())
    first: InferenceClient = (
        FakeInference((ContextExceeded(refusal.body),), identity="judge")
        if injected
        else GlmInference(Backend((refusal,)), "judge")
    )
    task = arithmetic_task(grader=verifyit_package(JudgeSpec(references=("12",), exact_gate=False)).grader)
    policy = load_policy(DEFAULTS_DIR).overlaid({"judge.model": "glm-5.3"})
    spec = lower_with_settings(task, lowering_settings(policy), {})
    history: tuple[Record, ...] = ()
    for recovered in (False, True):
        healthy, backend = FakeInference((Completion("SCORE: 1"),), identity="judge"), Backend((ANSWER,))
        client = (healthy if injected else GlmInference(backend, "judge")) if recovered else first
        log = EventLog(replay_from=history)
        ctx = Context(log=log)
        bundle = replace(
            resources(ctx),
            grading=GradingResource(rollout_engine({}, command_timeout=None), "judge"),
            role_inference={"judge": client},
        )
        install(ctx, bundle)
        grade = ctx.emit(GradeCandidate(spec, Candidate("Twelve"), fresh_verifier=recovered))
        if not recovered:
            with pytest.raises(ContextExceeded):
                await grade
            await ctx.drain()
            resolved = [r.event for r in log.records if isinstance(r.event, Resolved)]
            assert not [event for event in resolved if isinstance(event.request, InferenceRequest)]
            history = log.records
            continue
        result = await grade
        await ctx.drain()
        calls = len(healthy.requests) if injected else len(backend.calls)
        assert (result.status, result.reward, calls) == (Outcome.GRADED, 1.0, 1)
