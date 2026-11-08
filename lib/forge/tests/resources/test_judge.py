# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import replace

import pytest
from lib.forge.tests.test_stages import resources
from rolloutengine.models import TraceMode
from rolloutengine.test_support import arithmetic_task
from rolloutengine.validation import InvalidTaskSpec
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ScriptGrader
from verifyit.spec import JudgeSpec

from forge.effects.context import Context
from forge.effects.events import Resolved
from forge.effects.issues import Restart
from forge.effects.log import EventLog
from forge.lowering import lower_with_settings, lowering_settings, requirements, rollout_engine
from forge.model.controls import Candidate
from forge.model.llm import Completion
from forge.policy import load_policy
from forge.policy.defaults import DEFAULTS_DIR
from forge.resources.grading import GradeCandidate, GradingResource
from forge.resources.inference import FakeInference, InferenceRequest
from forge.resources.judge import JudgeEngine
from forge.wiring import install


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
