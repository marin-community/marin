# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gzip
import json
import pickle
import threading
from collections.abc import Coroutine, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, cast

import pytest
from lib.taskforge.tests.fake_resources import resources
from lib.taskforge.tests.resource_doubles import CleanReviewer, FakeInference, FakeWeb
from lib.taskforge.tests.resources.inference_fakes import Backend, ScriptedProvider, provider_completion, segment
from lib.taskforge.tests.stage_fakes import SolverEngine, solver_model
from pydantic import TypeAdapter
from rolloutengine import grading
from rolloutengine.models import (
    GradedRollout,
    GradedRolloutInterrupted,
    Model,
    RolloutOperation,
    TaskRolloutSpec,
    TraceMode,
)
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import ScriptGrader
from verifyit.modes.grade_judge import JudgeCompletion, JudgeConnection, JudgeRequest
from verifyit.spec import JudgeCompletionPolicy, JudgeSpec

from taskforge.effects.context import Context
from taskforge.effects.events import Resolved, ScopeExited
from taskforge.effects.issues import ContextExceeded, Restart
from taskforge.effects.log import EventLog, Record, decode
from taskforge.llm.client import ContextExhausted, ModelClient, Progress, Segment
from taskforge.llm.policy import GLM_5_3, ModelProfile, RetryPolicy
from taskforge.lowering import InvalidTaskSpec, lower_with_settings, lowering_settings, requirements, rollout_engine
from taskforge.model.controls import Candidate, Control, ControlKind, RewardBand
from taskforge.model.llm import Completion, FinishReason, Usage
from taskforge.model.workpiece import Submission
from taskforge.policy import load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.grading import GradeCandidate, GradingResource
from taskforge.resources.inference import GlmInference, InferenceClient, InferenceRequest
from taskforge.resources.judge import JudgeEngine
from taskforge.resources.rollouts import Engine, RolloutResource, SolverRollout
from taskforge.stages.author import AuthorReference, author_reference
from taskforge.wiring import build_resources, install, submit_gate

LABELS = {"PASS": 1.0, "FAIL": 0.0}
TEMPLATE = "Reference: {reference}; candidate: {candidate}"
ARENA = {"A>>B": 1.0, "A>B": 0.75, "A=B": 0.5, "B>A": 0.25, "B>>A": 0.0}
CUT = Completion("PA", finish_reason=FinishReason.LENGTH)


def done(text: str) -> Completion:
    return Completion(text, usage=Usage(10, 3))


def labels(**fields: Any) -> JudgeSpec:
    return JudgeSpec(rubric="labels", references=("12",), label_scores=LABELS, prompt_template=TEMPLATE, **fields)


# Each case's judge replies, in order. N5_JUDGED recorded these grades through the judge's earlier HTTP transport
# shim (2fc0e475cf).
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
# Regression: the judge callback read every finish but stop as truncation. Each case's judge replies, and the
# status, reward, judge calls and error the HTTP transport shim (9eb1a96371) produced from them; N5_FINISHES journals
# those calls, answering with the name.
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
        role_inference={"solve_review": CleanReviewer(), "judge": client},
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


class HeldJudge(FakeInference):
    """Answers sample 0 from its script and holds every later sample until cancelled, then for any ``cleanup``."""

    def __init__(self, script: Sequence[Completion], cleanup: asyncio.Event | None = None) -> None:
        super().__init__(script)
        self.held, self.cleanup, self.cleaned = asyncio.Event(), cleanup, asyncio.Event()

    async def complete(self, request: InferenceRequest) -> Completion:
        if request.sample == 0:
            return await super().complete(request)
        self.requests.append(request)
        self.held.set()
        try:
            await asyncio.Future()
        finally:
            if self.cleanup is not None:
                await self.cleanup.wait()
            self.cleaned.set()
        raise AssertionError("unreachable")


def grading_thread_exits(monkeypatch: pytest.MonkeyPatch) -> tuple[threading.Event, list[BaseException]]:
    """Signal when each grading invocation's thread returns, and collect what it raised."""
    exited, raised = threading.Event(), []
    invocation = grading.grade_remote_judge_answer

    def graded(*args: Any) -> GradeResult:
        try:
            return invocation(*args)
        except BaseException as error:
            raised.append(error)
            raise
        finally:
            exited.set()

    monkeypatch.setattr(grading, "grade_remote_judge_answer", graded)
    return exited, raised


# Regression: holding the grading thread before or during submission let a cancelled, drained grade issue or
# record a judge sample afterwards. The thread may still be unwinding when the grade returns; it adds nothing then.
@pytest.mark.parametrize("hold", ["submission", "publication", "registration", "provider-cleanup"])
async def test_a_cancelled_grade_returns_with_its_journal_final_and_starts_no_later_sample(
    hold: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    client = HeldJudge((done("SCORE: 1"),), release if hold == "provider-cleanup" else None)
    submit, submissions = asyncio.run_coroutine_threadsafe, 0

    def submitted(coro: Coroutine[Any, Any, Completion], loop: asyncio.AbstractEventLoop) -> Future[Completion]:
        # Runs on the grading thread; the second judge sample is the one held.
        nonlocal submissions
        submissions += 1
        if submissions != 2 or hold == "provider-cleanup":
            return submit(coro, loop)
        if hold == "registration":

            async def gated() -> Completion:
                client.held.set()
                await release.wait()
                return await coro

            return submit(gated(), loop)
        if hold == "submission":
            loop.call_soon_threadsafe(client.held.set)
            submit(release.wait(), loop).result()
            return submit(coro, loop)
        future = submit(coro, loop)
        submit(release.wait(), loop).result()
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submitted)
    exited, raised = grading_thread_exits(monkeypatch)
    log = EventLog()
    ctx = Context(log=log)
    engine = JudgeEngine(rollout_engine({}, command_timeout=None), ctx, client)
    spec = judged_spec(JUDGED["checklist"][0])
    operation = asyncio.create_task(GradingResource(engine, "judge").grade(GradeCandidate(spec, Candidate("Twelve"))))
    await client.held.wait()
    operation.cancel()
    if hold == "provider-cleanup":
        for _ in range(10):
            await asyncio.sleep(0)
        assert not operation.done()
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await operation
    returned = (len(log.records), len(client.requests))
    assert client.cleaned.is_set() is (hold in {"publication", "provider-cleanup"})
    # Released only now, the thread meets the closed grade: it starts no sample, journals nothing and exits.
    release.set()
    assert await asyncio.to_thread(exited.wait, 5)
    await ctx.drain()
    assert [type(error) for error in raised] == [asyncio.CancelledError]
    calls = 2 if hold in {"publication", "provider-cleanup"} else 1
    assert returned == (len(log.records), calls)
    assert [cast(InferenceRequest, event.request).sample for event in judge_inference(log)] == [0]
    # The sample that completed before the cancellation stays a valid journal entry: a regrade asks only for the next.
    resumed = FakeInference((done("SCORE: 0"),))
    [result], _ = await grade_judged((JUDGED["checklist"][0],), resumed, log.records)
    assert (result.reward, [request.sample for request in resumed.requests]) == (0.5, [1])


def judge_provider(provider: ScriptedProvider) -> InferenceClient:
    return GlmInference(ModelClient(provider, GLM_5_3, RetryPolicy(0.1, 0.001, 0.001, 1.0), role="judge"), "judge")


# Regression: a provider that had parsed its segment, swallowed the grade's cancellation in cleanup
# and returned it after a second cancellation released the caller journalled a late Resolved; a regrade then scored 0.5.
async def test_a_judge_segment_returned_after_the_grade_is_cancelled_twice_is_never_journalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held, cleanup = asyncio.Event(), asyncio.Event()

    async def swallows_cancellation(progress: Progress) -> Segment:
        parsed = segment("SCORE: 1")
        held.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await cleanup.wait()
        return parsed

    exited, raised = grading_thread_exits(monkeypatch)
    log = EventLog()
    ctx = Context(log=log)
    client = judge_provider(ScriptedProvider((swallows_cancellation,)))
    engine = JudgeEngine(rollout_engine({}, command_timeout=None), ctx, client)
    judge = JUDGED["checklist"][0]
    operation = asyncio.create_task(
        GradingResource(engine, "judge").grade(GradeCandidate(judged_spec(judge), Candidate("Twelve")))
    )
    await held.wait()
    operation.cancel()
    for _ in range(10):
        await asyncio.sleep(0)
    assert not operation.done()
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    returned = len(log.records)
    cleanup.set()
    assert await asyncio.to_thread(exited.wait, 5)
    await ctx.drain()
    assert [type(error) for error in raised] == [asyncio.CancelledError]
    # The disclosed late row is the sample's failed scope exit; its returned segment is never resolved.
    assert [record.event for record in log.records[returned:]] == [ScopeExited("judge", failed=True)]
    assert not judge_inference(log)
    graded = []
    for history in (log.records, ()):
        zeros = ScriptedProvider((segment("SCORE: 0"), segment("SCORE: 0")))
        [result], _ = await grade_judged((judge,), judge_provider(zeros), history)
        graded.append((result.reward, len(zeros.bodies)))
    assert graded == [(0.0, 2), (0.0, 2)]


# Regression: waiting for the grading thread at the verifier deadline turned a grade timeout
# into an attempt failure and left the sample running. The HTTP transport shim (9eb1a96371) raised this Restart from
# this grade.
async def test_a_verifier_deadline_during_a_judge_sample_is_the_n5_grade_timeout_and_cancels_the_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exited, raised = grading_thread_exits(monkeypatch)
    client = HeldJudge((done("SCORE: 1"),))
    log = EventLog()
    ctx = Context(log=log)
    engine = JudgeEngine(rollout_engine({}, command_timeout=None), ctx, client)
    spec = judged_spec(JUDGED["checklist"][0])
    spec = spec.model_copy(update={"limits": spec.limits.model_copy(update={"verifier_timeout": 0.2})})
    with pytest.raises(Restart, match=r"^Rollout interrupted during grade: TimeoutError <- CancelledError$") as caught:
        await GradingResource(engine, "judge").grade(GradeCandidate(spec, Candidate("Twelve")))
    returned = (len(log.records), len(client.requests))
    interrupted = cast(GradedRolloutInterrupted, caught.value.__cause__)
    grade = interrupted.grade
    assert (interrupted.operation, grade.status, grade.reward, grade.error, grade.failure) == (
        RolloutOperation.GRADE,
        Outcome.INFRA_ERROR,
        None,
        "Verifier deadline expired",
        GradingFailure.TIMEOUT,
    )
    assert client.cleaned.is_set()
    assert await asyncio.to_thread(exited.wait, 5)
    await ctx.drain()
    assert [type(error) for error in raised] == [asyncio.CancelledError]
    assert returned == (len(log.records), 2)
    assert [cast(InferenceRequest, event.request).sample for event in judge_inference(log)] == [0]


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
    # Regression: judge endpoint and model omissions held authored tasks.
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
    """The fixture is ``grade_judged`` over ``JUDGED``, scripted with each case's replies, as the HTTP shim journaled it.

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


# Regression: the callback field made TaskRolloutSpec.model_json_schema() raise PydanticInvalidForJsonSchema.
def test_a_judged_rollout_spec_has_a_json_schema_and_never_serializes_its_callback() -> None:
    def judge(request: JudgeRequest) -> JudgeCompletion:
        return JudgeCompletion("PASS", "stop")

    # Typed ``Any`` like the HTTP shim's transport, so VerifyIt needs no Pydantic 2 import for the schema to generate.
    schema = TaskRolloutSpec.model_json_schema()["$defs"]["JudgeConnection"]["properties"]["callback"]
    assert schema == {"default": None, "title": "Callback"}
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


# Each judge field the author reference omits, and a value an author could still type for it.
UNSHOWN = {"temperature": 0.7, "chat_template_kwargs": {"enable_thinking": "false"}}


async def test_the_author_reference_omits_only_the_judge_fields_the_taskforge_judge_fixes() -> None:
    shown = json.loads(await author_reference(AuthorReference(False)))["TaskSpec"]["$defs"]["JudgeSpec"]["properties"]
    assert {"parse_attempts", "context_char_limit"} <= shown.keys()
    assert {f.name for f in fields(JudgeSpec)} - shown.keys() == UNSHOWN.keys()


@pytest.mark.parametrize(
    "judge, refusal",
    [
        *(
            (labels(**{name: value}), f"Leave judge {name} unset: the Taskforge judge samples greedily")
            for name, value in UNSHOWN.items()
        ),
        (
            JudgeSpec(
                rubric="answers", references=("12",), prompt_template="{candidate} {references}", exact_gate=False
            ),
            "The Taskforge judge cannot grade the answers rubric",
        ),
    ],
)
def test_a_task_the_taskforge_judge_cannot_grade_is_returned_to_the_author_at_submit(
    judge: JudgeSpec, refusal: str
) -> None:
    # Regression: the callback refused these only at grading, after a validation round, without naming the cause.
    task = arithmetic_task(grader=verifyit_package(judge).grader)
    control = Control(Candidate("12"), RewardBand(Outcome.GRADED, 1, 1), ControlKind.POSITIVE)
    settings = lowering_settings(load_policy(DEFAULTS_DIR).overlaid({"judge.model": "glm-5.3"}))
    (message,) = submit_gate(Submission(task, (control,), ""), settings=settings, factories={})
    assert message.startswith(refusal)


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


# Regression: under a judge capped at three tokens, a grade and a rollout replayed 1.0 with zero calls.
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


# Regression: a judge provider reporting another model, under the same policy and profile,
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
    provider_completion(), content="SCORE: 1", tool_calls=(), finish_reason=FinishReason.STOP, continuations=0
)


# Regression: a context refusal recorded inside the shared inference resource cached the
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


class Falsey:
    """``engine``, but false: an optional engine that is present though it tests false."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def __bool__(self) -> bool:
        return False

    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        return await self.engine.run_task(spec, model, run_id=run_id)


# Regression: wrapping each engine as ``fresh and JudgeEngine(fresh, ...)`` kept a fresh engine that tests
# false unwrapped, and grading called it directly: a fresh grade the installed judge passes scored 0.0 with no judge
# row. A false fresh engine is dropped, and the fresh grade runs on the wrapped engine.
async def test_a_fresh_engine_that_tests_false_still_grades_through_the_installed_judge() -> None:
    log = EventLog()
    ctx = Context(log=log)
    engine = rollout_engine({}, command_timeout=None)
    client = FakeInference((done("PASS"),), identity="judge")
    bundle = replace(
        resources(ctx),
        grading=GradingResource(engine, "judge", fresh_engine=Falsey(engine)),
        role_inference={"solve_review": CleanReviewer(), "judge": client},
    )
    install(ctx, bundle)
    grade = await ctx.emit(GradeCandidate(judged_spec(labels()), Candidate("Twelve"), fresh_verifier=True))
    await ctx.drain()
    assert (grade.status, grade.reward, len(client.requests), len(judge_inference(log))) == (Outcome.GRADED, 1.0, 1, 1)


class Rebuilt(RolloutResource):
    """A rollout resource an extension builds from another, whose constructor takes only that one."""

    ran: list[int]

    def __init__(self, inner: RolloutResource) -> None:
        super().__init__(inner.engine, inner.model, inner.config_digest)
        object.__setattr__(self, "ran", [])

    async def run(self, event: SolverRollout) -> GradedRollout:
        self.ran.append(event.trial)
        return await super().run(event)


# Regression: with no judge and no decision model, installation still rebuilt the rollout resource around its
# engine, so an injected subclass with its own constructor raised TypeError before any handler was registered.
async def test_an_injected_rollout_resource_is_installed_as_given_when_no_engine_is_wrapped() -> None:
    ctx = Context(log=EventLog())
    bundle = resources(ctx)
    rollouts = Rebuilt(RolloutResource(SolverEngine(1), solver_model, "solver"))  # a wave of one
    install(ctx, replace(bundle, rollouts=rollouts, role_inference={"solve_review": CleanReviewer()}))
    spec = lower_with_settings(arithmetic_task(), lowering_settings(load_policy(DEFAULTS_DIR)), {})
    await ctx.emit(SolverRollout(spec, 0, 0))
    await ctx.drain()
    assert rollouts.ran == [0]
