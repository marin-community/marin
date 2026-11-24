# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, cast, override

import httpx
import pytest
from iris.client.client import IrisContext, iris_ctx_scope
from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.resource_doubles import CleanReviewer, FakeWeb
from lib.taskforge.tests.resources.inference_fakes import LoopTime, ScriptedProvider, segment
from lib.taskforge.tests.resources.test_judge import labels
from rolloutengine.models import LENGTH_STOP_REASON, GradedRollout, TaskRolloutSpec, TraceMode
from rolloutengine.test_support import arithmetic_task
from taskcompendium.decision_model import DecisionModelRequest
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import Outcome
from taskcompendium.models import DECISION_LEVELS, DECISION_QUESTION, DecisionModelGrader

from taskforge.effects.context import Context
from taskforge.effects.events import Resolved, Retrying
from taskforge.effects.issues import DecisionModelUnavailable, Hold
from taskforge.effects.log import BufferSink, EventLog, Record
from taskforge.llm.client import Progress, Segment
from taskforge.llm.glm_provider import FixedEndpoint, GlmEndpoint, Pool
from taskforge.lowering import IN_PROCESS_GRADER_TIMEOUT, InvalidTaskSpec, lower_with_settings, lowering_settings
from taskforge.model.content_hash import digest
from taskforge.model.controls import Candidate, Control, ControlKind, RewardBand
from taskforge.model.llm import FinishReason
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import AdversaryResult, AdversaryStatus, Cause, ControlResult, TrialResult
from taskforge.model.workpiece import Submission, Workpiece
from taskforge.policy import Policy, load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.bundle import Resources
from taskforge.resources.decision_model import (
    DIAGNOSTIC_LIMIT,
    ClefReader,
    DecisionModel,
    DecisionModelDown,
    DecisionModelHealth,
    DecisionModelRead,
    DecisionModelReader,
    read_level,
)
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.rollouts import SolverRollout
from taskforge.stages.author import AuthorReference, author_reference, system_prompt, uses_decision_model
from taskforge.stages.classify import grade_cause, replaced_refusals, replayable_grade, trial_cause
from taskforge.stages.exits import Validated
from taskforge.stages.validate import solver_batch, validate
from taskforge.wiring import (
    RelayEndpoint,
    build_config,
    build_resources,
    decision_model_policy,
    install,
    submit_gate,
)

MODEL = "clef@fixture-1"
ROLES = ("author", "solver", "adversary")
validate_stage = importlib.import_module("taskforge.stages.validate")  # the package re-exports the stage
GRADER = DecisionModelGrader(deliverable="Six plus six.", rubric="The sum, as a number.", reference="12", cut=3.0)


def overlaid(model: str = MODEL) -> Policy:
    roles = {f"{role}.model": "test-model" for role in ROLES}
    overlay = {**roles, "decision_model.model": model, "solver.k": 3, "adversary.roster": []}
    return load_policy(DEFAULTS_DIR).overlaid(overlay)


def decision_spec(policy: Policy) -> TaskRolloutSpec:
    task = arithmetic_task(grader=GRADER)
    return lower_with_settings(task, lowering_settings(policy), {}).model_copy(update={"trace": TraceMode.MESSAGES})


def request(candidate: str, model: str = MODEL) -> DecisionModelRequest:
    return DecisionModelRequest(
        model, DECISION_QUESTION, DECISION_LEVELS, GRADER.deliverable, GRADER.rubric, GRADER.reference, candidate
    )


@dataclass
class FakeReader(DecisionModelReader):
    """Reads ``levels[candidate]``; a candidate in ``down`` is unavailable while it stays in the set."""

    levels: Mapping[str, object] = field(default_factory=lambda: {"12": 4.0, "13": 1.0})
    down: set[str] = field(default_factory=set)
    reads: list[DecisionModelRequest] = field(default_factory=list)

    @override
    async def read(self, request: DecisionModelRequest) -> float:
        self.reads.append(request)
        if request.candidate in self.down:
            raise DecisionModelDown("HTTP 503: clef_unavailable", 30.0)
        return cast(float, self.levels[request.candidate])

    @override
    async def health(self) -> DecisionModelHealth:
        raise AssertionError("grading never reads health")


BACKOFF_STEPS = 10


class VirtualTime(LoopTime):
    """The loop's clock, which only a backoff advances; ``recovered`` is cleared at the first backoff, as an outage
    ends. Each backoff first lets other tasks run a few steps, as a real wait would."""

    def __init__(self, recovered: set[str] | None = None) -> None:
        super().__init__()
        self.slept: list[float] = []
        self.recovered = recovered

    @override
    async def sleep(self, delay: float) -> None:
        self.slept.append(delay)
        if self.recovered is not None:
            self.recovered.clear()
        for _ in range(BACKOFF_STEPS):
            await asyncio.sleep(0)
        await super().sleep(delay)


def installed(
    policy: Policy, reader: DecisionModelReader, solver: ScriptedProvider, ctx: Context, time: VirtualTime
) -> Resources:
    providers = {"author": ScriptedProvider(()), "adversary": ScriptedProvider(()), "solver": solver}
    built = build_resources(policy, providers, {}, FakeWeb({}, {}), decision_model=reader)
    assert built.decision_model is not None
    reviewer = {**built.role_inference, "solve_review": CleanReviewer()}
    built = replace(built, decision_model=replace(built.decision_model, sleep=time.sleep), role_inference=reviewer)
    install(ctx, built)
    return built


async def decided(
    policy: Policy, rollout: bool, history: Sequence[Record] = (), *, fresh: bool = False
) -> tuple[float | None, int, list[DecisionModelRequest], Sequence[Record]]:
    """A solver rollout answering 12, or a grade of the candidate 12, replaying ``history``."""
    reader, solver = FakeReader(), ScriptedProvider((segment("12"),))
    log = EventLog(replay_from=history)
    ctx = Context(log=log)
    installed(policy, reader, solver, ctx, VirtualTime())
    spec = decision_spec(policy)
    if rollout:
        reward = cast(GradedRollout, await ctx.emit(SolverRollout(spec, 0, 0))).grade.reward
    else:
        reward = (await ctx.emit(GradeCandidate(spec, Candidate("12"), fresh_verifier=fresh))).reward
    await ctx.drain()
    return reward, len(solver.bodies), reader.reads, log.records


@pytest.mark.parametrize("rollout", [False, True], ids=["grade", "rollout"])
async def test_an_equivalent_read_replays_without_a_call_and_another_model_identity_reads_again(rollout: bool) -> None:
    reward, solved, paid, history = await decided(overlaid(), rollout)
    assert (reward, solved, paid) == (1.0, int(rollout), [request("12")])
    assert (await decided(overlaid(), rollout, history))[:3] == (1.0, 0, [])
    reward, solved, paid, _ = await decided(overlaid("clef@fixture-2"), rollout, history)
    assert (reward, solved, paid) == (1.0, int(rollout), [request("12", "clef@fixture-2")])


async def test_another_grade_of_the_same_answer_pays_no_second_read() -> None:
    """A fresh verifier changes the outer grade's identity, never the read's."""
    *_, history = await decided(overlaid(), rollout=False)
    reward, _, paid, _ = await decided(overlaid(), rollout=False, history=history, fresh=True)
    assert (reward, paid) == (1.0, [])


async def test_a_read_waits_out_an_outage_with_backoff_then_raises_the_typed_hold() -> None:
    """Never TimeoutError, which RolloutEngine would report as an expired verifier deadline."""
    policy = overlaid().overlaid({"decision_model.unavailable_wait": 100.0, "decision_model.backoff_initial": 40.0})
    reader, time, ctx = FakeReader(down={"12"}), VirtualTime(), Context(log=EventLog())
    installed(policy, reader, ScriptedProvider(()), ctx, time)
    with pytest.raises(DecisionModelUnavailable, match=f"{MODEL} gave no read within 100s: last HTTP 503"):
        await ctx.emit(GradeCandidate(decision_spec(policy), Candidate("12")))
    await ctx.drain()
    assert (time.slept, len(reader.reads)) == ([40.0, 60.0, 60.0], 3)  # retry-after 30 never shortens the backoff
    events = [record.event for record in ctx._log.records]
    assert [event.attempt for event in events if isinstance(event, Retrying)] == [1, 2, 3]
    assert not [event for event in events if isinstance(event, Resolved)]  # nothing deferred is recorded paid


async def test_a_read_that_recovers_inside_its_wait_resolves_as_one_read() -> None:
    reader, policy, ctx = FakeReader(down={"12"}), overlaid(), Context(log=EventLog())
    installed(policy, reader, ScriptedProvider(()), ctx, VirtualTime(recovered=reader.down))
    grade = await ctx.emit(GradeCandidate(decision_spec(policy), Candidate("12")))
    await ctx.drain()
    assert (grade.reward, len(reader.reads)) == (1.0, 2)
    assert len([r for r in ctx._log.records if isinstance(r.event, Resolved)]) == 2  # one read, one grade


class BarrierReader(FakeReader):
    """Counts reads in flight; each waits until ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.active = self.peak = 0
        self.release, self.full = asyncio.Event(), asyncio.Event()

    @override
    async def read(self, request: DecisionModelRequest) -> float:
        self.active += 1
        self.peak = max(self.peak, self.active)
        if self.active == 2:
            self.full.set()
        await self.release.wait()
        self.active -= 1
        return 4.0


async def test_reads_in_flight_never_exceed_the_configured_concurrency() -> None:
    policy = overlaid().overlaid({"decision_model.concurrency": 2})
    reader, ctx = BarrierReader(), Context(log=EventLog())
    installed(policy, reader, ScriptedProvider(()), ctx, VirtualTime())
    spec = decision_spec(policy)
    grades = [asyncio.create_task(ctx.emit(GradeCandidate(spec, Candidate(str(n))))) for n in range(5)]
    await reader.full.wait()
    for _ in range(20):  # every other grade gets as far as it can
        await asyncio.sleep(0)
    assert (reader.active, reader.peak) == (2, 2)
    reader.release.set()
    assert [grade.reward for grade in await asyncio.gather(*grades)] == [1.0] * 5
    await ctx.drain()
    assert reader.peak == 2


class Gateway(FakeReader):
    """Each read takes ``seconds`` on the loop's clock, then refuses while ``down`` and otherwise answers level 4."""

    def __init__(self, seconds: float, down: bool) -> None:
        super().__init__()
        self.seconds, self.is_down = seconds, down

    @override
    async def read(self, request: DecisionModelRequest) -> float:
        self.reads.append(request)
        await asyncio.sleep(self.seconds)
        if self.is_down:
            raise DecisionModelDown("HTTP 503: clef_unavailable")
        return 4.0


NO_READ = f"{MODEL} gave no read within 100s: "


# Regression: reads queued behind the concurrency cap waited outside the availability window, so
# a slow unavailable gateway outlived the verifier deadline and the driver published an ordinary held state. One bound
# now covers the wait for a read slot and every attempt; queued reads report the outage another read observed, and a
# queue on an answering model says so instead.
@pytest.mark.parametrize(
    "down, outcomes",
    [
        (True, [NO_READ + "last HTTP 503: clef_unavailable"] * 3),
        (False, [4.0, NO_READ + "its read went unanswered", NO_READ + "none of its 1 read slots came free"]),
    ],
    ids=["down", "answering slowly"],
)
async def test_a_read_ends_within_the_wait_including_its_wait_for_a_read_slot(
    down: bool, outcomes: list[object]
) -> None:
    time = LoopTime()
    policy = decision_model_policy(overlaid().overlaid({"decision_model.concurrency": 1}))
    assert policy is not None
    model = DecisionModel(Gateway(30.0 if down else 60.0, down), replace(policy, unavailable_wait=100.0))
    start = time.clock()
    reads = asyncio.gather(
        *(read_level(DecisionModelRead(request(str(n))), model, ctx=Context(log=EventLog())) for n in range(3)),
        return_exceptions=True,
    )
    while not reads.done() and time.clock() - start < 150.0:  # half a virtual second at a time
        await time.sleep(0.5)
    assert reads.done() and time.clock() - start < 105.0, "a read outlived its availability window"
    ended = reads.result()
    assert all(isinstance(read, float | DecisionModelUnavailable) for read in ended)
    assert [read if isinstance(read, float) else str(read) for read in ended] == outcomes


HUGE = 10**5000  # 5,001 digits: no repr within the default limit


# Regression: an override returning "4" was graded INFRA_ERROR, then replayed from the journal as
# the level 4.0 and passed with no new read; an unconvertible string made the journal unreadable. A level is checked
# before it is recorded, so nothing off-scale is ever persisted and a repaired reader reads it on resume.
# A 5,001-digit integer was formatted before X2 saw it, raising a rollout restart; X2 bounds the diagnostic.
@pytest.mark.parametrize(
    "level, shown",
    [(7, "7"), ("4", "a value of type str"), ("four", "a value of type str"), (HUGE, "a value of type int")],
    ids=["7", "str", "word", "huge"],
)
async def test_an_off_scale_level_is_an_infrastructure_error_recorded_nowhere_and_read_again(
    level: object, shown: str
) -> None:
    policy, sink = overlaid(), BufferSink()
    ctx = Context(log=EventLog(sink=sink))
    installed(policy, FakeReader(levels={"12": level}), ScriptedProvider(()), ctx, VirtualTime())
    grade = await ctx.emit(GradeCandidate(decision_spec(policy), Candidate("12")))
    await ctx.drain()
    assert (grade.status, grade.reward) == (Outcome.INFRA_ERROR, None)
    assert grade.error == f"Decision model {MODEL!r} returned {shown}, not a level in 0..4"
    events = [record.event for record in ctx._log.records]
    assert not [
        event for event in events if isinstance(event, Resolved) and isinstance(event.request, DecisionModelRead)
    ]
    history = local_records(sink)  # the journal decodes
    repaired, resumed = FakeReader(), Context(log=EventLog(replay_from=history, replayable=replayable_grade))
    installed(policy, repaired, ScriptedProvider(()), resumed, VirtualTime())
    assert (await resumed.emit(GradeCandidate(decision_spec(policy), Candidate("12")))).reward == 1.0
    await resumed.drain()
    assert repaired.reads == [request("12")]


# Regression: three solver answers and two controls bought five reads for two requests, and the
# round's verifier and a fresh one could see different rewards for one key. Equal requests in flight share one read;
# a waiter that is cancelled leaves it to the others, and a read that fails is not shared with the next request.
async def test_equal_reads_in_flight_share_one_paid_read_and_survive_a_cancelled_waiter() -> None:
    policy = overlaid().overlaid({"decision_model.concurrency": 4})
    reader, ctx = BarrierReader(), Context(log=EventLog())
    installed(policy, reader, ScriptedProvider(()), ctx, VirtualTime())
    spec = decision_spec(policy)
    grades = [
        asyncio.create_task(ctx.emit(GradeCandidate(spec, Candidate("12"), fresh_verifier=fresh)))
        for fresh in (False, False, True, True)
    ]
    while not reader.active:
        await asyncio.sleep(0)
    for _ in range(20):  # every grade reaches its read
        await asyncio.sleep(0)
    grades[0].cancel()
    reader.release.set()
    rewards = await asyncio.gather(*grades[1:])
    await ctx.drain()
    assert (reader.peak, [grade.reward for grade in rewards]) == (1, [1.0] * 3)


class Answers(FakeReader):
    """Each read waits for ``release``, then answers the next of ``answers``: a second paid read would show."""

    def __init__(self, *answers: float) -> None:
        super().__init__()
        self.answers, self.release = list(answers), asyncio.Event()

    @override
    async def read(self, request: DecisionModelRequest) -> float:
        self.reads.append(request)
        await self.release.wait()
        return self.answers.pop(0)


# Regression: an equal request arriving after the shared read left the table but
# before its waiter recorded the level paid again, and the two read 1 and 4 for one key. The flight is owned until a
# waiter has recorded its result: at every interleaving around completion, one call and one level.
@pytest.mark.parametrize("steps", range(6))
async def test_an_equal_request_around_a_reads_completion_shares_its_one_recorded_level(steps: int) -> None:
    reader, ctx = Answers(4.0, 1.0), Context(log=EventLog())
    installed(overlaid(), reader, ScriptedProvider(()), ctx, VirtualTime())
    first = asyncio.create_task(ctx.emit(DecisionModelRead(request("12"))))
    while not reader.reads:
        await asyncio.sleep(0)
    reader.release.set()
    for _ in range(steps):
        await asyncio.sleep(0)
    loop = asyncio.get_running_loop()
    second = asyncio.Task(ctx.emit(DecisionModelRead(request("12"))), loop=loop, eager_start=True)  # asks at this step
    levels = (await first, await second)
    await ctx.drain()
    recorded = {r.event.value for r in ctx._log.records if isinstance(r.event, Resolved)}
    assert (levels, len(reader.reads), recorded) == ((4.0, 4.0), 1, {4.0})


# Regression: a waiter cancelled after the read completed evicted it while another
# waiter had yet to record the level; an equal request then paid again, and the two recorded 1 and 4 for one key. A
# cancelled waiter only leaves: the read stays shared until its last waiter leaves or one records it.
@pytest.mark.parametrize("steps", range(6))
@pytest.mark.parametrize("gap", range(3))
async def test_a_waiter_cancelled_after_the_read_completes_leaves_it_to_the_waiter_that_records_it(
    steps: int, gap: int
) -> None:
    reader, ctx = Answers(4.0, 1.0), Context(log=EventLog())
    installed(overlaid(), reader, ScriptedProvider(()), ctx, VirtualTime())
    cancelled, kept = (asyncio.create_task(ctx.emit(DecisionModelRead(request("12")))) for _ in range(2))
    while not reader.reads:
        await asyncio.sleep(0)
    for _ in range(20):  # both wait on the one read
        await asyncio.sleep(0)
    reader.release.set()
    for _ in range(steps):
        await asyncio.sleep(0)
    cancelled.cancel()
    for _ in range(gap):
        await asyncio.sleep(0)
    loop = asyncio.get_running_loop()
    late = asyncio.Task(ctx.emit(DecisionModelRead(request("12"))), loop=loop, eager_start=True)  # asks at this step
    levels = (await kept, await late)
    await asyncio.gather(cancelled, return_exceptions=True)
    await ctx.drain()
    recorded = {r.event.value for r in ctx._log.records if isinstance(r.event, Resolved)}
    assert (levels, len(reader.reads), recorded) == ((4.0, 4.0), 1, {4.0})


# Regression: with every waiter cancelled, the shared read kept retrying after the
# drain. The last waiter to leave cancels and joins it: nothing runs on, and nothing is recorded resolved.
async def test_a_read_every_waiter_left_stops_before_the_drain_and_records_nothing() -> None:
    reader, ctx = FakeReader(down={"12"}), Context(log=EventLog())
    installed(overlaid(), reader, ScriptedProvider(()), ctx, VirtualTime())
    waiters = [asyncio.create_task(ctx.emit(DecisionModelRead(request("12")))) for _ in range(2)]
    while len(reader.reads) < 2:  # retrying
        await asyncio.sleep(0)
    for waiter in waiters:
        waiter.cancel()
    await asyncio.gather(*waiters, return_exceptions=True)
    await ctx.drain()
    calls, rows = len(reader.reads), len(ctx._log.records)
    for _ in range(100):
        await asyncio.sleep(0)
    assert (len(reader.reads), len(ctx._log.records)) == (calls, rows)
    assert not [r for r in ctx._log.records if isinstance(r.event, Resolved)]


async def test_without_a_decision_model_its_grader_is_returned_to_the_author_before_any_validation() -> None:
    policy = overlaid("")
    task = arithmetic_task(grader=GRADER)
    with pytest.raises(InvalidTaskSpec, match="Decision-model grading is disabled"):
        lower_with_settings(task, lowering_settings(policy), {})
    submission = Submission(
        task, (Control(Candidate("12"), RewardBand(Outcome.GRADED, 1, 1), ControlKind.POSITIVE),), ""
    )
    settings = lowering_settings(policy)
    assert submit_gate(submission, settings=settings, factories={})[0].startswith("Decision-model grading is disabled")


async def test_the_author_schema_offers_the_decision_model_kind_only_where_one_is_configured() -> None:
    without, with_model = [
        system_prompt(policy, await author_reference(AuthorReference(uses_decision_model(policy))))
        for policy in (overlaid(""), overlaid())
    ]
    assert "## Decision-model graders" not in without and "DecisionModelGrader" not in without
    assert "## Decision-model graders" in with_model and "DecisionModelGrader" in with_model
    assert digest(overlaid("")) != digest(overlaid())  # the stage completions record which mode ran


@pytest.mark.parametrize(
    "overlay, error",
    [
        ({"decision_model.concurrency": 0}, "decision_model.concurrency must be positive and finite"),
        (
            {"decision_model.request_timeout": float("inf"), "decision_model.backoff_initial": -1.0},
            "decision_model.request_timeout, decision_model.backoff_initial must be positive and finite",
        ),
    ],
)
def test_decision_model_settings_are_checked_at_wiring(overlay: Mapping[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        decision_model_policy(overlaid().overlaid(overlay))


# Regression: both model graders took the solver's multi-hour turn wall as their verifier deadline.
@pytest.mark.parametrize("time_limit", [1800.0, 12000.0])
def test_a_model_grader_has_its_own_deadline_whatever_the_solver_time_limit(time_limit: float) -> None:
    solver = {"solver.time_limit": time_limit, "solver.attempt_timeout": 2 * time_limit, "judge.model": "glm-5.3"}
    policy = overlaid().overlaid(solver)
    judged = arithmetic_task(grader=verifyit_package(labels()).grader)
    judge = lower_with_settings(judged, lowering_settings(policy), {})
    assert decision_spec(policy).limits.verifier_timeout == IN_PROCESS_GRADER_TIMEOUT + 900  # the read's own wait
    assert judge.limits.verifier_timeout == 3600 + 600  # the judge call's availability window and a stall


def test_a_missing_decision_model_setting_fails_at_wiring_naming_it() -> None:
    entries = {key: value for key, value in overlaid().entries.items() if key != "decision_model.backoff_maximum"}
    with pytest.raises(KeyError, match=re.escape("decision_model.backoff_maximum")):
        decision_model_policy(Policy(entries))


EXACTLY = re.escape("supplied exactly when policy decision_model.model is set")


def test_a_reader_is_supplied_exactly_when_a_decision_model_is_configured() -> None:
    providers = {role: ScriptedProvider(()) for role in ROLES}
    with pytest.raises(ValueError, match=EXACTLY):
        build_resources(overlaid(""), providers, {}, FakeWeb({}, {}), decision_model=FakeReader())
    with pytest.raises(ValueError, match=EXACTLY):
        build_resources(overlaid(), providers, {}, FakeWeb({}, {}))


def controls() -> tuple[Control, ...]:
    return (
        Control(Candidate("12"), RewardBand(Outcome.GRADED, 1, 1), ControlKind.POSITIVE),
        Control(Candidate("13"), RewardBand(Outcome.GRADED, reward_max=0), ControlKind.NEGATIVE),
    )


async def late_twelve(_progress: Progress) -> Segment:
    """A solver turn still in flight while a sibling's decision-model wait runs out: an outage cancelling its siblings
    would cancel this paid turn."""
    for _ in range(200):
        await asyncio.sleep(0)
    return segment("12")


async def validated(
    proposal: TaskProposal, reader: FakeReader, history: Sequence[Record] = (), concurrency: int = 3, roster: int = 0
) -> tuple[Validated | BaseExceptionGroup, int, Sequence[Record]]:
    """Validate a decision-model task: three solver trials answering 12, and the controls 12 and 13."""
    overlay = {"decision_model.unavailable_wait": 10.0, "solver.concurrency": concurrency}
    policy = overlaid().overlaid({**overlay, "adversary.roster": ["false_accept"] * roster})
    solver = ScriptedProvider((late_twelve,) * 3)
    log = EventLog(replay_from=history)
    ctx = Context(log=log)
    workpiece = Workpiece("w", proposal, submission=Submission(arithmetic_task(grader=GRADER), controls(), ""))
    outcome: Validated | BaseExceptionGroup
    try:
        async with ctx.scope("workpiece"):
            built = installed(policy, reader, solver, ctx, VirtualTime())
            outcome = await validate(ctx, workpiece, build_config(policy, {}), built)
    except* (DecisionModelUnavailable, Hold) as error:
        outcome = error
    await ctx.drain()
    return outcome, len(solver.bodies), log.records


def human(outcome: object) -> str:
    """The reasons that keep ``outcome`` a human hold, empty when it awaits the decision model alone."""
    rest = outcome.split(DecisionModelUnavailable)[1] if isinstance(outcome, BaseExceptionGroup) else None
    return "" if rest is None else repr(rest)


# Regression: the negative control's outage was raised before the grade observer
# joined, so the off-scale grades' hold was dropped and the run deferred as if only the decision model were missing.
async def test_a_held_grade_beside_an_outage_keeps_the_wave_a_human_hold(proposal: TaskProposal) -> None:
    held, _, _ = await validated(proposal, FakeReader(levels={"12": 7}, down={"13"}))
    assert "returned 7, not a level in 0..4" in human(held)


# Regression: a failed adversary, or a solver refusal over the allowance that
# returned before a control's outage, disappeared beside it, and the automatic resume bought the adversary again or
# reused the refusal on a second launch. Every finished sibling's disposition is settled before the wave waits.
@pytest.mark.parametrize("sibling", ["adversary", "solver"])
async def test_a_human_disposition_finished_before_a_late_outage_outranks_it(
    proposal: TaskProposal, sibling: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = validate_stage.control_result

    async def control(ctx: Context, spec: TaskRolloutSpec, index: int, control: Control) -> ControlResult:
        if index == 0:
            return await real(ctx, spec, index, control)
        for _ in range(1000):  # after every sibling has finished
            await asyncio.sleep(0)
        raise DecisionModelUnavailable(f"{MODEL} gave no read")

    async def refused(*_args: object, **_kwargs: object) -> tuple[TrialResult, ...]:
        return (TrialResult(0, None, Cause.HELD, "the model server refused"),)

    async def failed(*args: object) -> AdversaryResult:
        return AdversaryResult(str(args[5]), AdversaryStatus.FAILED, (), ("the search lost its machine",))

    monkeypatch.setattr(validate_stage, "control_result", control)
    monkeypatch.setattr(validate_stage, "adversary_result", failed)
    if sibling == "solver":
        monkeypatch.setattr(validate_stage, "solver_batch", refused)
    held, _, _ = await validated(proposal, FakeReader(), roster=int(sibling == "adversary"))
    reason = "the search lost its machine" if sibling == "adversary" else "the model server refused"
    assert reason in human(held)


# The decision model is down for the negative control's answer only, so its grade waits while the solver wave and
# the positive control finish; resumed, it pays the deferred read alone and reaches the uninterrupted result.
async def test_an_outage_mid_wave_keeps_finished_siblings_and_resumes_paying_only_the_deferred_read(
    proposal: TaskProposal,
) -> None:
    clean, _, _ = await validated(proposal, FakeReader())
    reader = FakeReader(down={"13"})
    held, solved, history = await validated(proposal, reader)
    assert isinstance(held, BaseExceptionGroup) and held.split(DecisionModelUnavailable)[1] is None
    assert solved == 3
    healthy = FakeReader()
    resumed, solved, _ = await validated(proposal, healthy, history)
    assert isinstance(resumed, Validated) and isinstance(clean, Validated)
    assert (solved, healthy.reads) == (0, [request("13")])
    assert resumed == clean and all(control.met for control in resumed.result.controls)
    assert [trial.grade.reward for trial in resumed.result.solver if trial.grade] == [1.0] * 3


# Regression: with one solver slot at a time and the negative control already waiting for the
# decision model, every solver slot still started. One admission stop serves the batch and its validation siblings:
# the running slot finishes and keeps its row, and no other slot is bought to wait beside the outage.
async def test_a_control_outage_admits_no_further_solver_slot_and_resumes_only_what_never_ran(
    proposal: TaskProposal,
) -> None:
    clean, _, cleaned = await validated(proposal, FakeReader())
    held, solved, history = await validated(proposal, FakeReader(down={"13"}), concurrency=1)
    assert isinstance(held, BaseExceptionGroup) and held.split(DecisionModelUnavailable)[1] is None
    assert solved == 1
    resumed, solved, both = await validated(proposal, FakeReader(), history, concurrency=1)
    assert isinstance(resumed, Validated) and isinstance(clean, Validated) and solved == 2
    assert replace(resumed.result, solver=()) == replace(clean.result, solver=())
    # Each history names its own rows; the evidence they hold is the same.
    for result, records in ((clean.result, cleaned), (resumed.result, both)):
        assert [trial.trial for trial in result.solver] == [0, 1, 2]
        for trial in result.solver:
            assert trial.receipt is not None
            row = records[trial.receipt - 1].event
            assert isinstance(row, Resolved) and not row.replayed
            assert isinstance(request := row.request, SolverRollout) and (request.trial, request.attempt) == (
                trial.trial,
                0,
            )
    assert [replace(trial, receipt=None) for trial in resumed.result.solver] == [
        replace(trial, receipt=None) for trial in clean.result.solver
    ]


async def test_an_outage_in_a_solver_slot_starts_no_waiting_slot_and_resumes_only_unfinished_slots(
    proposal: TaskProposal,
) -> None:
    """With one slot at a time, the first slot's outage leaves the other two unstarted: no rollout is bought to wait."""
    reader = FakeReader(down={"12"})
    policy = overlaid().overlaid({"decision_model.unavailable_wait": 10.0, "solver.concurrency": 1})
    solver, ctx = ScriptedProvider((segment("12"),) * 3), Context(log=EventLog())
    spec = decision_spec(policy)
    async with ctx.scope("workpiece"):
        installed(policy, reader, solver, ctx, VirtualTime())
        with pytest.raises(BaseExceptionGroup) as raised:
            await solver_batch(ctx, spec, "validation", 0, (), 0, 3, concurrency=1, heartbeat=3600.0)
    await ctx.drain()
    assert raised.group_contains(DecisionModelUnavailable) and len(solver.bodies) == 1


def gateway(handler: Callable[[httpx.Request], httpx.Response]) -> ClefReader:
    source = FixedEndpoint(GlmEndpoint("http://gateway/v1", "secret", Pool("high")))
    return ClefReader(source, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_the_default_reader_asks_clef_the_whole_request_and_returns_the_expected_level() -> None:
    sent: list[httpx.Request] = []

    def answer(call: httpx.Request) -> httpx.Response:
        sent.append(call)
        probabilities = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.5, "4": 0.5}
        return httpx.Response(200, json={"answers": {"whole": {"probabilities": probabilities}}})

    assert await gateway(answer).read(request("12")) == 3.5
    (call,) = sent
    assert (str(call.url), call.headers["authorization"]) == ("http://gateway/v1/systemone", "Bearer secret")
    assert json.loads(call.content) == {
        "model": "clef",
        "state": {
            "task_deliverable": GRADER.deliverable,
            "rubric": GRADER.rubric,
            "y_reference_answer": "12",
            "z_candidate_answer": "12",
        },
        "questions": {"whole": {"type": "score", "instructions": DECISION_QUESTION, "criteria": list(DECISION_LEVELS)}},
    }


# Regression: the shared read ran in an empty context, so a caller's scoped Iris
# client was gone and the installed grade raised Restart before any POST, though the same reader called directly read
# 4.0. The read keeps the caller's context and leaves only Taskforge's effect scope.
async def test_the_installed_grade_reads_clef_through_the_callers_iris_client() -> None:
    sent: list[httpx.Request] = []

    def answer(call: httpx.Request) -> httpx.Response:
        sent.append(call)
        probabilities = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0}
        return httpx.Response(200, json={"answers": {"whole": {"probabilities": probabilities}}})

    relay = SimpleNamespace(resolve=lambda _model: SimpleNamespace(endpoints=[SimpleNamespace(url="http://gateway")]))
    client = SimpleNamespace(resolver_for_job=lambda _job: relay)
    source = RelayEndpoint("/muchanem/glm53-relay", "secret", Pool("high"), "glm-5.3")
    reader = ClefReader(source, httpx.AsyncClient(transport=httpx.MockTransport(answer)))
    policy, ctx = overlaid(), Context(log=EventLog())
    with iris_ctx_scope(IrisContext(job_id=None, client=cast(Any, client))):
        installed(policy, reader, ScriptedProvider(()), ctx, VirtualTime())
        grade = await ctx.emit(GradeCandidate(decision_spec(policy), Candidate("12")))
    await ctx.drain()
    assert (grade.reward, [str(call.url) for call in sent]) == (1.0, ["http://gateway/v1/systemone"])


async def test_the_default_reader_reads_unavailability_as_down_and_other_refusals_as_holds() -> None:
    with pytest.raises(DecisionModelDown, match="HTTP 503: clef_unavailable") as raised:
        await gateway(lambda _: httpx.Response(503, text="clef_unavailable", headers={"retry-after": "30"})).read(
            request("12")
        )
    assert raised.value.retry_after == 30.0
    with pytest.raises(DecisionModelDown, match="unreachable: ConnectError"):
        await gateway(lambda call: (_ for _ in ()).throw(httpx.ConnectError("refused", request=call))).read(
            request("12")
        )
    with pytest.raises(Hold, match="refused the read with HTTP 400") as refused:
        await gateway(lambda _: httpx.Response(400, text="bad state")).read(request("12"))
    assert not isinstance(refused.value, DecisionModelUnavailable)  # a refusal is no outage: nothing defers on it


def weighted(*pairs: object) -> str:
    """A Clef answer whose probabilities are zero but at the ``level, weight`` pairs."""
    probabilities: dict[str, object] = dict.fromkeys("01234", 0.0)
    probabilities.update({str(level): weight for level, weight in zip(pairs[::2], pairs[1::2], strict=True)})
    return json.dumps({"answers": {"whole": {"probabilities": probabilities}}})


@pytest.mark.parametrize(
    "body, reason",
    [
        ("{", "answered unusably: JSONDecodeError"),
        ('{"answers": {}}', "answered unusably: KeyError: 'whole'"),
        ('{"answers": {"whole": {"probabilities": [0.5, 0.5]}}}', "answered unusably: TypeError"),
        (weighted(0, "high"), "answered unusably: ValueError"),
        ('{"answers": {"whole": {"probabilities": {"0": 1}}}}', "answered unusably: KeyError: '1'"),
        (json.dumps({"answers": {"whole": {"probabilities": dict.fromkeys("01234", 0)}}}), "are not a distribution"),
        (json.dumps({"answers": {"whole": {"probabilities": dict.fromkeys("01234", -1)}}}), "are not a distribution"),
        (
            '{"answers": {"whole": {"probabilities": {"0": 1, "1": NaN, "2": 0, "3": 0, "4": 0}}}}',
            "are not a distribution",
        ),
        (weighted(4, 10**309), "answered unusably: OverflowError"),
        (weighted(4, True), "answered unusably: TypeError: a boolean is no probability"),
        (weighted(0, 1e308, 1, 1e308), "are not a distribution"),  # finite weights whose sum is not
        (weighted(4, "x" * 2000), "answered unusably: ValueError: could not convert string to float"),
    ],
)
async def test_the_default_reader_refuses_an_answer_that_is_no_distribution_over_the_levels(
    body: str, reason: str
) -> None:
    with pytest.raises(Hold, match=re.escape(reason)) as refused:
        await gateway(lambda _: httpx.Response(200, text=body)).read(request("12"))
    assert not isinstance(refused.value, DecisionModelUnavailable)
    assert len(str(refused.value)) <= len("Decision model answered unusably: ") + DIAGNOSTIC_LIMIT


# The control for the refusals above: every answer that is a distribution reads the level it read before, integer
# and unnormalised weights included.
@pytest.mark.parametrize(
    "body, level",
    [(weighted(3, 0.5, 4, 0.5), 3.5), (weighted(4, 1), 4.0), (weighted(0, 1), 0.0), (weighted(1, 2, 3, 2), 2.0)],
)
async def test_the_default_reader_reads_the_expected_level_of_any_distribution(body: str, level: float) -> None:
    assert await gateway(lambda _: httpx.Response(200, text=body)).read(request("12")) == level


@pytest.mark.parametrize(
    "status, health",
    [
        (200, DecisionModelHealth.HEALTHY),
        (404, DecisionModelHealth.UNSUPPORTED),
        (405, DecisionModelHealth.UNSUPPORTED),
        (503, DecisionModelHealth.DOWN),
    ],
)
async def test_health_is_the_gateways_own_route_and_never_a_read(status: int, health: DecisionModelHealth) -> None:
    paths: list[str] = []

    def answer(call: httpx.Request) -> httpx.Response:
        paths.append(call.url.path)
        return httpx.Response(status)

    assert await gateway(answer).health() is health
    assert paths == ["/v1/systemone/health"]


async def length_stopped(
    reader: FakeReader, sink: BufferSink, history: Sequence[Record] = (), *, resume: bool = False
) -> tuple[GradedRollout | DecisionModelUnavailable, int]:
    """One solver slot whose answer, 12, stops at the generation limit, graded through ``reader``; replays ``history``
    as a launch with or without ``--resume-held``."""
    policy = overlaid().overlaid({"decision_model.unavailable_wait": 10.0})
    solver = ScriptedProvider((segment("12", finish=FinishReason.LENGTH),))
    replaces = replaced_refusals(history, resume)  # as the driver builds its log
    ctx = Context(log=EventLog(sink=sink, replay_from=history, replayable=replayable_grade, replaces=replaces))
    installed(policy, reader, solver, ctx, VirtualTime())
    outcome: GradedRollout | DecisionModelUnavailable
    try:
        outcome = await ctx.emit(SolverRollout(decision_spec(policy), 0, 0))
    except DecisionModelUnavailable as error:
        outcome = error
    await ctx.drain()
    return outcome, len(solver.bodies)


# Regression: an answer cut at its length limit is still graded, and an off-scale level is a grader
# infrastructure error; the cutoff's replay exemption then replayed it forever. The limit decides the trial whatever
# its grade, so the explicit resume replays it and reads nothing.
async def test_an_off_scale_level_on_a_length_stopped_answer_is_a_settled_loss_that_replays() -> None:
    sink = BufferSink()
    failed, _ = await length_stopped(FakeReader(levels={"12": 7}), sink)
    assert isinstance(failed, GradedRollout) and failed.rollout.stop_reason == LENGTH_STOP_REASON
    assert grade_cause(failed.grade) is Cause.GRADER_INFRA
    healthy = FakeReader()
    replayed, solved = await length_stopped(healthy, sink, local_records(sink), resume=True)
    assert (replayed, solved, healthy.reads) == (failed, 0, [])
    assert trial_cause(failed, arithmetic_task(grader=GRADER)) is Cause.GENERATION_LIMIT


async def test_a_503_on_a_length_stopped_answer_leaves_no_slot_and_an_unflagged_launch_reads_it() -> None:
    sink = BufferSink()
    deferred, _ = await length_stopped(FakeReader(down={"12"}), sink)
    assert isinstance(deferred, DecisionModelUnavailable)
    history = local_records(sink)
    assert not [r for r in history if isinstance(r.event, Resolved) and isinstance(r.event.request, SolverRollout)]
    healthy = FakeReader()
    graded, _ = await length_stopped(healthy, sink, history)
    assert isinstance(graded, GradedRollout) and graded.grade.reward == 1.0 and healthy.reads == [request("12")]
