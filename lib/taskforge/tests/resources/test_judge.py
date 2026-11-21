# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gzip
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from lib.taskforge.tests.fake_resources import resources
from lib.taskforge.tests.resource_doubles import FakeInference, FakeWeb
from lib.taskforge.tests.resources.inference_fakes import ScriptedProvider, segment
from pydantic import TypeAdapter, ValidationError
from rolloutengine.models import GradedRollout, TaskRolloutSpec, TraceMode
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import ScriptGrader
from verifyit.modes.grade_judge import JudgeCompletion, JudgeConnection, JudgeRequest
from verifyit.spec import JudgeCompletionPolicy, JudgeSpec

from taskforge.effects.context import Context
from taskforge.effects.events import Resolved
from taskforge.effects.issues import Restart
from taskforge.effects.log import EventLog, Record, decode
from taskforge.llm.client import Progress, Segment
from taskforge.llm.policy import GLM_5_3, ModelProfile
from taskforge.lowering import InvalidTaskSpec, lower_with_settings, lowering_settings, requirements, rollout_engine
from taskforge.model.controls import Candidate
from taskforge.model.llm import Completion, FinishReason, Usage
from taskforge.policy import load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.grading import GradeCandidate, GradingResource
from taskforge.resources.inference import InferenceClient, InferenceRequest
from taskforge.resources.judge import JudgeEngine
from taskforge.resources.rollouts import SolverRollout
from taskforge.wiring import build_resources, install

LABELS = {"PASS": 1.0, "FAIL": 0.0}
TEMPLATE = "Reference: {reference}; candidate: {candidate}"
ARENA = {"A>>B": 1.0, "A>B": 0.75, "A=B": 0.5, "B>A": 0.25, "B>>A": 0.0}
CUT = Completion("PA", finish_reason=FinishReason.LENGTH)


def done(text: str) -> Completion:
    return Completion(text, usage=Usage(10, 3))


def labels(**fields: Any) -> JudgeSpec:
    return JudgeSpec(rubric="labels", references=("12",), label_scores=LABELS, prompt_template=TEMPLATE, **fields)


# Each case's judge replies, in order. N5_JUDGED recorded these grades through n5's HTTP transport shim (2fc0e475cf).
JUDGED: dict[str, tuple[JudgeSpec, tuple[Completion, ...]]] = {
    "reference-retry": (
        JudgeSpec(references=("12",), exact_gate=False, max_completion_tokens=16, incomplete_retry_tokens=64),
        (CUT, done("ok\nSCORE: 1")),
    ),
    "checklist": (
        JudgeSpec(rubric="checklist", criteria=("Correct", "Terse"), exact_gate=False),
        (done("SCORE: 1"), done("SCORE: 0")),
    ),
    "labels": (labels(), (done("PASS"),)),
    "responses": (labels(api="responses", system_prompt="Strict.", reasoning_effort="low"), (done("FAIL"),)),
    "pairwise": (
        JudgeSpec(
            references=("baseline",),
            question="Q?",
            rubric="pairwise",
            system_prompt="Compare.",
            prompt_template="Q: {question}\nA: {answer_a}\nB: {answer_b}",
            label_scores=ARENA,
            verdict_patterns=(r"\[\[([AB<>=]+)\]\]",),
            label_case="upper",
        ),
        (done("[[B>A]]"), done("[[A=B]]")),
    ),
    "truncated": (labels(system_prompt="Terse."), (CUT,)),
}
OUTCOMES = {
    "reference-retry": (Outcome.GRADED, 1.0),
    "checklist": (Outcome.GRADED, 0.5),
    "labels": (Outcome.GRADED, 1.0),
    "responses": (Outcome.GRADED, 0.0),
    "pairwise": (Outcome.GRADED, 0.625),
    "truncated": (Outcome.INFRA_ERROR, None),
}
N5_JUDGED = Path(__file__).parents[1] / "fixtures" / "n5-judged-grades.jsonl.gz"


def tool(text: str) -> Completion:
    return Completion(text, finish_reason=FinishReason.TOOL_CALLS, usage=Usage(10, 4))


RETRY = {"max_completion_tokens": 16, "incomplete_retry_tokens": 64}
NO_TEXT = "judge completion is incomplete or has no text"
# Review 2026-11-21: n6 read every finish but stop as truncation. Each case's judge replies, and the status, reward,
# judge calls and error n5 (9eb1a96371) produced from them; N5_FINISHES journals those calls, answering with the name.
FINISHES: dict[str, tuple[JudgeSpec, tuple[Completion, ...], tuple[Outcome, float | None, int, str | None]]] = {
    "reference-tool-call": (
        JudgeSpec(references=("12",), exact_gate=False),
        (tool("SCORE: 1"),),
        (Outcome.INFRA_ERROR, None, 1, NO_TEXT),
    ),
    "reference-tool-call-then-answer": (
        JudgeSpec(references=("12",), exact_gate=False, **RETRY),
        (tool("SCORE: 1"), done("SCORE: 1")),
        (Outcome.INFRA_ERROR, None, 1, NO_TEXT),
    ),
    "checklist-tool-call-then-answer": (
        JudgeSpec(rubric="checklist", criteria=("Correct", "Terse"), exact_gate=False, **RETRY),
        (tool("SCORE: 1"), done("SCORE: 1"), done("SCORE: 0")),
        (Outcome.INFRA_ERROR, None, 1, NO_TEXT),
    ),
    "labels-tool-call": (labels(), (tool("PASS"),), (Outcome.INFRA_ERROR, None, 1, NO_TEXT)),
    "labels-tool-call-then-answer": (
        labels(**RETRY),
        (tool("PASS"), done("PASS")),
        (Outcome.INFRA_ERROR, None, 1, NO_TEXT),
    ),
    "labels-abort-then-answer": (
        labels(**RETRY),
        (Completion("PASS", finish_reason=FinishReason.ABORT), done("PASS")),
        (Outcome.INFRA_ERROR, None, 1, NO_TEXT),
    ),
    "labels-length-then-answer": (labels(**RETRY), (CUT, done("PASS")), (Outcome.GRADED, 1.0, 2, None)),
    "content-tool-call": (
        labels(completion_policy=JudgeCompletionPolicy.CONTENT, **RETRY),
        (tool("PASS"), done("FAIL")),
        (Outcome.GRADED, 1.0, 1, None),
    ),
    "content-length-label": (
        labels(completion_policy=JudgeCompletionPolicy.CONTENT, **RETRY),
        (Completion("PASS", finish_reason=FinishReason.LENGTH, usage=Usage(10, 16)), done("FAIL")),
        (Outcome.GRADED, 1.0, 1, None),
    ),
    "responses-tool-call-then-answer": (
        labels(api="responses", **RETRY),
        (tool("PASS"), done("FAIL")),
        (Outcome.GRADED, 0.0, 2, None),
    ),
    "pairwise-tool-call-then-answer": (
        replace(JUDGED["pairwise"][0], **RETRY),
        (tool("[[B>A]]"), done("[[B>A]]"), done("[[A=B]]")),
        (Outcome.INFRA_ERROR, None, 1, NO_TEXT),
    ),
}
N5_FINISHES = Path(__file__).parents[1] / "fixtures" / "n5-judge-finishes.jsonl.gz"


def judged_spec(judge: JudgeSpec) -> TaskRolloutSpec:
    policy = load_policy(DEFAULTS_DIR).overlaid({"judge.model": "glm-5.3"})
    task = arithmetic_task(grader=verifyit_package(judge).grader)
    return lower_with_settings(task, lowering_settings(policy), {}).model_copy(update={"trace": TraceMode.MESSAGES})


async def grade_judged(
    judges: Sequence[JudgeSpec],
    client: InferenceClient,
    history: Sequence[Record] = (),
    *,
    fresh: bool = False,
    answers: Sequence[str] = (),
) -> tuple[list[GradeResult], EventLog]:
    """Grade each answer (default "Twelve") under its judge through the installed judge role, replaying ``history``."""
    log = EventLog(replay_from=history)
    ctx = Context(log=log)
    bundle = replace(
        resources(ctx),
        grading=GradingResource(rollout_engine({}, command_timeout=None), "judge"),
        role_inference={"judge": client},
    )
    install(ctx, bundle)
    # A fresh verifier changes the outer grade identity, so only the judge's inference effects can replay.
    answers = answers or ["Twelve"] * len(judges)
    results = [
        await ctx.emit(GradeCandidate(judged_spec(judge), Candidate(answer), fresh_verifier=fresh))
        for judge, answer in zip(judges, answers, strict=True)
    ]
    await ctx.drain()
    return results, log


def judge_inference(log: EventLog) -> list[Resolved[Any]]:
    return [
        record.event
        for record in log.records
        if isinstance(record.event, Resolved) and isinstance(record.event.request, InferenceRequest)
    ]


async def test_a_cancelled_grade_leaves_no_judge_call_running_or_recorded() -> None:
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

    log = EventLog()
    ctx = Context(log=log)
    client = BlockingInference(())
    engine = JudgeEngine(rollout_engine({}, command_timeout=None), ctx, client)
    # A second criterion would be a second judge sample if the grading thread outlived the cancellation.
    spec = judged_spec(JUDGED["checklist"][0])
    operation = asyncio.create_task(GradingResource(engine, "judge").grade(GradeCandidate(spec, Candidate("Twelve"))))
    try:
        await asyncio.wait_for(entered.wait(), 5)
    finally:
        operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    await asyncio.wait_for(cancelled.wait(), 5)
    # Joins VerifyIt's grading thread: it ended rather than waiting on, or issuing, another judge sample.
    await asyncio.wait_for(asyncio.get_running_loop().shutdown_default_executor(), 5)
    await ctx.drain()
    assert len(client.requests) == 1
    assert not judge_inference(log)


async def test_a_judge_effect_failure_reaches_the_grade_caller_unchanged() -> None:
    error = Restart("GLM exhausted availability window")
    client = FakeInference((error,))
    with pytest.raises(Restart) as caught:
        await grade_judged((JUDGED["labels"][0],), client)
    assert caught.value is error
    assert len(client.requests) == 1


async def test_judge_samples_are_numbered_and_a_truncated_one_retries_at_the_larger_budget() -> None:
    judges = (JUDGED["reference-retry"][0], JUDGED["pairwise"][0])
    client = FakeInference(JUDGED["reference-retry"][1] + JUDGED["pairwise"][1])
    results, _ = await grade_judged(judges, client)
    assert [(result.status, result.reward) for result in results] == [OUTCOMES["reference-retry"], OUTCOMES["pairwise"]]
    sampled = [(request.sample, request.sampling.max_tokens) for request in client.requests]
    assert sampled == [(0, 16), (1, 64), (0, 8192), (1, 8192)]
    games = [request.messages[-1].content.rsplit("\n", 1)[-1] for request in client.requests[2:]]
    assert games == ["B: Twelve", "B: baseline"]


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
async def test_a_judge_reply_truncated_at_its_only_budget_is_an_incomplete_infrastructure_grade(api: str) -> None:
    [result], _ = await grade_judged((labels(api=api),), FakeInference((CUT,)))
    assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None)
    assert result.error == "judge completion is incomplete"


@pytest.mark.parametrize("rubric", ["reference", "checklist", "labels", "responses"])
async def test_host_judge_records_inference_and_replays_without_provider(rubric: str) -> None:
    # Full S 20261108: judge endpoint/model omissions held 11 authored tasks.
    judged = rubric in {"labels", "responses"}
    judge = JudgeSpec(
        rubric="labels" if judged else rubric,
        references=("12",),
        criteria=("Correct arithmetic",),
        exact_gate=False,
        api="responses" if rubric == "responses" else "chat_completions",
        label_scores=LABELS,
        prompt_template=TEMPLATE,
    )
    history: Sequence[Record] = ()
    for replay in (False, True):
        client = FakeInference(() if replay else (Completion("PASS" if judged else "SCORE: 1"),), identity="judge")
        [result], log = await grade_judged((judge,), client, history, fresh=replay)
        assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
        assert len(client.requests) == (0 if replay else 1)
        assert judge_inference(log)[-1].replayed is replay
        history = log.records


async def test_judge_samples_recorded_through_the_n5_transport_shim_replay_with_zero_calls() -> None:
    """The fixture is ``grade_judged`` over ``JUDGED``, scripted with each case's replies, as n5 journaled it.

    It pins the judge's inference keys (messages, sampling, sample index) across the move from the HTTP shim to
    the callback. A deliberate change to judge prompts or sampling invalidates it; record it afresh then.
    """
    recorded = decode(gzip.decompress(N5_JUDGED.read_bytes()))
    client = FakeInference((), identity="judge")
    results, log = await grade_judged([judge for judge, _ in JUDGED.values()], client, recorded, fresh=True)
    assert {name: (result.status, result.reward) for name, result in zip(JUDGED, results, strict=True)} == OUTCOMES
    assert not client.requests
    replayed = [event for event in judge_inference(log) if event.replayed]
    assert len(replayed) == sum(len(script) for _, script in JUDGED.values())


@pytest.mark.parametrize("case", FINISHES)
async def test_a_judge_finish_grades_and_spends_calls_as_it_did_at_n5(case: str) -> None:
    judge, script, expected = FINISHES[case]
    client = FakeInference(script)
    [result], _ = await grade_judged((judge,), client, answers=(f"Twelve ({case})",))
    assert (result.status, result.reward, len(client.requests), result.error) == expected


async def test_judge_finishes_recorded_at_n5_replay_their_n5_grades_with_zero_calls() -> None:
    recorded = decode(gzip.decompress(N5_FINISHES.read_bytes()))
    client = FakeInference((), identity="judge")
    judges, _, expected = zip(*FINISHES.values(), strict=True)
    answers = [f"Twelve ({case})" for case in FINISHES]
    results, log = await grade_judged(judges, client, recorded, fresh=True, answers=answers)
    assert [(result.status, result.reward, result.error) for result in results] == [
        (status, reward, error) for status, reward, _, error in expected
    ]
    assert not client.requests
    assert sum(event.replayed for event in judge_inference(log)) == sum(calls for _, _, calls, _ in expected)


# Review 2026-11-21: the callback field made TaskRolloutSpec.model_json_schema() raise PydanticInvalidForJsonSchema.
def test_a_judged_rollout_spec_has_a_json_schema_and_never_serializes_its_callback() -> None:
    def judge(request: JudgeRequest) -> JudgeCompletion:
        return JudgeCompletion("PASS", "stop")

    assert "callback" not in TaskRolloutSpec.model_json_schema()["$defs"]["JudgeConnection"]["properties"]
    with pytest.raises(ValidationError, match="callable"):
        TypeAdapter(JudgeConnection).validate_python({"base_url": "", "api_key": "", "callback": "judge"})
    connection = TypeAdapter(JudgeConnection).validate_python({"base_url": "", "api_key": "", "callback": judge})
    spec = judged_spec(labels()).model_copy(update={"judge_connection": connection})
    assert "judge_connection" not in spec.model_dump()
    assert TaskRolloutSpec.model_validate_json(spec.model_dump_json()).judge_connection is None
    assert pickle.loads(pickle.dumps(spec)).judge_connection is None


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
