# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The solver batch: slot identities, scheduling outside every replay key, heartbeats and per-slot outcomes."""

import asyncio
import contextlib
import math
import re
from collections.abc import Mapping, Sequence
from functools import partial
from typing import cast, override

import pytest
from lib.taskforge.tests.resources.rollout_fakes import twelve
from rolloutengine.models import (
    LENGTH_STOP_REASON,
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    ConversationRecord,
    GradedRollout,
    Model,
    Rollout,
    RolloutInterrupted,
    RolloutOperation,
    TaskRolloutSpec,
)
from rolloutengine.test_support import arithmetic_task, rollout_spec
from shellbox.machine import MachineTerminated
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome

from taskforge.effects.context import Context
from taskforge.effects.events import Invalidated, Note, Resolved, event_digest
from taskforge.effects.issues import DecisionModelUnavailable, Hold, Restart
from taskforge.effects.log import EventLog, Record
from taskforge.llm.client import ModelRequestRejected
from taskforge.model.review import Cause, TrialResult
from taskforge.policy import Policy, load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.rollouts import Engine, RolloutResource, SolverRollout, run_rollout
from taskforge.stages.classify import replaced_refusals, replayable_grade
from taskforge.stages.validate import solver_batch
from taskforge.wiring import build_config

SPEC = rollout_spec(arithmetic_task())
POLICY = load_policy(DEFAULTS_DIR)
HEARTBEAT = POLICY.setting("solver.heartbeat_interval", float)


class Slots(Engine):
    """Solve every trial, except that a trial in ``gates`` first waits on its gate and one in ``failures`` raises."""

    def __init__(
        self,
        gates: Mapping[int, asyncio.Event] | None = None,
        failures: Mapping[int, BaseException] | None = None,
        stops: Mapping[int, str] | None = None,
        results: Mapping[int, GradedRollout] | None = None,
    ) -> None:
        self.gates, self.failures, self.stops = gates or {}, failures or {}, stops or {}
        self.results = results or {}
        self.executed: list[str] = []
        self.entered = asyncio.Event()

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        assert run_id is not None
        self.executed.append(run_id)
        self.entered.set()
        trial = int(run_id.split(":")[-2])
        if trial in self.gates:
            await self.gates[trial].wait()
        if trial in self.failures:
            raise self.failures[trial]
        if trial in self.results:
            return self.results[trial]
        return GradedRollout(Rollout(run_id, (), self.stops.get(trial, "stop")), GradeResult(Outcome.GRADED, 1.0))


def context(engine: Slots, recorded: Sequence[Record] = (), *, resume: bool = False) -> tuple[Context, EventLog]:
    log = EventLog(replay_from=recorded, replayable=replayable_grade, replaces=replaced_refusals(recorded, resume))
    ctx = Context(log=log)
    ctx.handle(SolverRollout, partial(run_rollout, notify=ctx.notify), RolloutResource(engine, twelve, "solver"))
    return ctx, log


def resolved(ctx: Context, count: int) -> asyncio.Event:
    """Set once ``count`` solver rollouts have resolved in ``ctx``."""
    done, seen = asyncio.Event(), [0]

    async def observe(event: Resolved[object]) -> None:
        seen[0] += isinstance(event.request, SolverRollout)
        if seen[0] >= count:
            done.set()

    ctx.observe(Resolved, observe)
    return done


def requests(log: EventLog) -> list[SolverRollout]:
    return [record.event for record in log.records if isinstance(record.event, SolverRollout)]


async def test_validation_slots_keep_their_identity_and_a_second_phase_numbers_past_them() -> None:
    engine = Slots()
    ctx, log = context(engine)
    validation = await solver_batch(ctx, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT)
    existing = [trial.trial for trial in validation]
    confirmation = await solver_batch(
        ctx, SPEC, "confirmation", 0, existing, len(existing), 8, concurrency=2, heartbeat=HEARTBEAT
    )
    await ctx.drain()
    issued = [event_digest(event) for event in requests(log)]
    # The identities every validation journal recorded: SolverRollout(spec, trial, len(evidence)).
    assert set(issued[:3]) == {event_digest(SolverRollout(SPEC, trial, 0)) for trial in range(3)}
    assert existing == [0, 1, 2] and [trial.trial for trial in confirmation] == [3, 4, 5, 6, 7]
    assert len(set(issued)) == len(engine.executed) == 8  # no confirmation slot reused or re-ran a validation slot


# Regression: a confirmation batch given the eligible slots reused a cached timed-out slot's id; given every
# occupied slot it subtracted the timeout from its target. Either way it ended one eligible score short.
async def test_new_slots_number_past_every_occupied_one_and_fill_the_eligible_target() -> None:
    engine = Slots(stops={2: TOTAL_TURN_TIMEOUT_STOP_REASON})
    ctx, log = context(engine)
    validation = await solver_batch(ctx, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT)
    eligible = [trial.trial for trial in validation if trial.cause is None]
    assert eligible == [0, 1] and validation[2].cause is Cause.AGENT_TIMEOUT
    occupied = [trial.trial for trial in validation]
    fresh = await solver_batch(
        ctx, SPEC, "confirmation", 0, occupied, len(eligible), 8, concurrency=4, heartbeat=HEARTBEAT
    )
    await ctx.drain()
    assert [trial.trial for trial in fresh] == [3, 4, 5, 6, 7, 8] and all(trial.cause is None for trial in fresh)
    assert (
        len(eligible) + len(fresh) == 8
        and len(engine.executed) == len({event_digest(event) for event in requests(log)}) == 9
    )


# Regression: lowering concurrency on resume must keep completed shards.
async def test_a_resumed_batch_executes_only_its_unfinished_slot_under_other_scheduling() -> None:
    gate = asyncio.Event()
    first = Slots(gates={2: gate})
    ctx, log = context(first)
    two = resolved(ctx, 2)
    run = asyncio.create_task(solver_batch(ctx, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT))
    await two.wait()
    run.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await run
    assert len(first.executed) == 3  # the third was executing when the batch was interrupted
    again = Slots(failures={0: AssertionError("slot 0 executed again"), 1: AssertionError("slot 1 executed again")})
    resumed, _ = context(again, log.records)
    results = await solver_batch(resumed, SPEC, "validation", 0, (), 0, 3, concurrency=1, heartbeat=HEARTBEAT / 7)
    await resumed.drain()
    assert again.executed == ["arithmetic:2:0"]
    assert [(trial.trial, trial.grade and trial.grade.reward) for trial in results] == [(0, 1.0), (1, 1.0), (2, 1.0)]


class VirtualClock:
    """``sleep`` parks until the test advances it, then moves ``now`` by the delay slept."""

    def __init__(self) -> None:
        self.now, self.slept = 0.0, list[float]()
        self.parked, self.advance = asyncio.Event(), asyncio.Event()

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.slept.append(delay)
        self.parked.set()
        await self.advance.wait()
        self.advance.clear()
        self.now += delay

    async def tick(self) -> None:
        """Let one sleep end and wait until the sleeper parks again."""
        await self.parked.wait()
        self.parked.clear()
        self.advance.set()
        await self.parked.wait()


async def test_a_long_slot_heartbeats_at_the_policy_cadence_without_touching_its_replay() -> None:
    gate, clock = asyncio.Event(), VirtualClock()
    engine = Slots(gates={0: gate})
    ctx, log = context(engine)
    run = asyncio.create_task(
        solver_batch(
            ctx, SPEC, "validation", 0, (), 0, 1, concurrency=1, heartbeat=HEARTBEAT, sleep=clock.sleep, clock=clock
        )
    )
    await engine.entered.wait()
    await clock.tick()
    await clock.tick()
    gate.set()
    (result,) = await run
    await ctx.drain()
    beats = [record for record in log.records if isinstance(record.event, Note)]
    assert len(beats) == 2
    assert set(clock.slept) == {HEARTBEAT}
    # Each beat names the still-running trial and how long it has run, so a silent slot is visibly not hung.
    assert all(
        record.event.msg.endswith(f"[trial 0 {n * HEARTBEAT:.0f}s]")
        for n, record in enumerate(beats, 1)
        if isinstance(record.event, Note)
    )
    # A heartbeat is a plain diagnostic row: no outcome answers it.
    assert all(record.answers is None for record in beats)
    replayed = Slots(failures={0: AssertionError("a heartbeat made the slot execute again")})
    resumed, _ = context(replayed, log.records)
    again = await solver_batch(resumed, SPEC, "validation", 0, (), 0, 1, concurrency=4, heartbeat=1.0)
    await resumed.drain()
    assert again == (result,) and not replayed.executed


def rejected() -> RolloutInterrupted:
    """What the engine raises when the model server refuses one slot's request."""
    error = RolloutInterrupted(Rollout("refused", (), "interrupted"), RolloutOperation.MODEL)
    error.__cause__ = ModelRequestRejected(400, "malformed request", ())
    return error


REFUSED = "RequestRejected: model server rejected request with HTTP 400: malformed request"


def bought(log: EventLog, trial: int) -> int:
    """The row of the one physical resolution of ``trial``'s slot: a held slot's receipt."""
    (row,) = (
        record.sequence
        for record in log.records
        if isinstance(event := record.event, Resolved)
        and not event.replayed
        and event.request == SolverRollout(SPEC, trial, 0)
    )
    return row


def lost() -> RolloutInterrupted:
    error = RolloutInterrupted(Rollout("lost", (), "interrupted"), RolloutOperation.ADVANCE)
    error.__cause__ = MachineTerminated("pod lost")
    return error


async def test_a_held_slot_is_typed_missing_evidence_beside_its_finished_siblings() -> None:
    engine = Slots(failures={1: rejected()})
    ctx, log = context(engine)
    results = await solver_batch(ctx, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT)
    await ctx.drain()
    assert results[1] == TrialResult(1, None, Cause.HELD, REFUSED, receipt=bought(log, 1))
    assert [trial.grade and trial.grade.reward for trial in (results[0], results[2])] == [1.0, 1.0]


# Regression: a sibling's restart discarded the held slot and its cause; the next stage attempt ran the held
# slot again and accepted under a zero-missing policy, with the hold reason absent from the whole journal.
async def test_a_held_slot_survives_a_sibling_restart_and_replays_without_running_again() -> None:
    engine = Slots(failures={1: rejected(), 2: lost()})
    ctx, log = context(engine)
    with pytest.raises(ExceptionGroup) as raised:
        await solver_batch(ctx, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT, max_missing=1)
    await ctx.drain()
    assert [type(error) for error in raised.value.exceptions] == [Restart]
    again = Slots(failures={0: AssertionError("slot 0 ran again"), 1: AssertionError("slot 1 ran again")})
    replayed, _ = context(again, log.records)
    results = await solver_batch(replayed, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT)
    await replayed.drain()
    assert again.executed == ["arithmetic:2:0"] and results[1] == TrialResult(
        1, None, Cause.HELD, REFUSED, receipt=bought(log, 1)
    )


# Regression: generation reaching its limit before the first response is a completed, failed
# trial whose grade is UNAVAILABLE. The replay rule took it for unavailable grading and ran the slot again whenever a
# sibling restarted or a held root resumed, and the reviewer was left a grade-only stub without the transcript.
PROMPT = ({"role": "system", "content": "solve"}, {"role": "user", "content": "what is 6 + 6?"})


def cut_off(stop: str) -> GradedRollout:
    conversation = ConversationRecord(PROMPT, (), (), (), None)
    return GradedRollout(Rollout("cut", (conversation,), stop), GradeResult(Outcome.UNAVAILABLE, None))


def kept(log: EventLog, trial: int) -> Rollout:
    """The latest resolution the journal holds for ``trial``."""
    *_, value = [
        record.event.value.rollout
        for record in log.records
        if isinstance(record.event, Resolved)
        and record.event.request == SolverRollout(SPEC, trial, 0)
        and isinstance(record.event.value, GradedRollout)
    ]
    return value


@pytest.mark.parametrize("resume", [False, True])
async def test_a_generation_limited_slot_replays_with_its_transcript_beside_an_interrupted_sibling(
    resume: bool,
) -> None:
    engine = Slots(failures={2: lost()}, results={0: cut_off(LENGTH_STOP_REASON)})
    ctx, log = context(engine)
    with pytest.raises(ExceptionGroup) as raised:
        await solver_batch(ctx, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT)
    await ctx.drain()
    assert [type(error) for error in raised.value.exceptions] == [Restart]
    again = Slots(failures={0: AssertionError("the cut-off slot ran again"), 1: AssertionError("slot 1 ran again")})
    restarted, restarted_log = context(again, log.records, resume=resume)
    results = await solver_batch(restarted, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT)
    await restarted.drain()
    assert again.executed == ["arithmetic:2:0"]
    assert [(trial.trial, trial.cause) for trial in results] == [(0, Cause.GENERATION_LIMIT), (1, None), (2, None)]
    assert kept(restarted_log, 0).conversations[0].messages == PROMPT
    cold = Slots(failures={trial: AssertionError(f"slot {trial} ran again") for trial in range(3)})
    rebuilt, rebuilt_log = context(cold, restarted_log.records, resume=resume)
    assert await solver_batch(rebuilt, SPEC, "validation", 0, (), 0, 3, concurrency=1, heartbeat=HEARTBEAT) == results
    await rebuilt.drain()
    assert not cold.executed and kept(rebuilt_log, 0).conversations[0].messages == PROMPT


# Regression: an answer cut at its length limit is still graded; when its grader failed for infrastructure (here the
# host judge's incomplete completion), the exemption above replayed the failure forever, and retrying it bought the
# slot again so a replacement that finished erased the limit. A limit is a settled loss whatever its grade: it
# replays, with its failed grade kept for inspection.
@pytest.mark.parametrize("resume", [False, True])
async def test_a_length_stopped_answer_whose_grader_failed_is_a_settled_loss_that_replays(resume: bool) -> None:
    answered = ConversationRecord((*PROMPT, {"role": "assistant", "content": "12"}), (), (), (), None)
    infra = GradeResult(
        Outcome.INFRA_ERROR, None, "judge returned an incomplete completion", failure=GradingFailure.EXECUTION
    )
    failed = GradedRollout(Rollout("cut", (answered,), LENGTH_STOP_REASON), infra)
    ctx, log = context(Slots(results={0: failed}))
    (first,) = await solver_batch(ctx, SPEC, "validation", 0, (), 0, 1, concurrency=1, heartbeat=HEARTBEAT)
    await ctx.drain()
    cold = Slots(failures={0: AssertionError("slot 0 ran again")})
    resumed, _ = context(cold, log.records, resume=resume)
    (second,) = await solver_batch(resumed, SPEC, "validation", 0, (), 0, 1, concurrency=1, heartbeat=HEARTBEAT)
    await resumed.drain()
    assert first.cause is Cause.GENERATION_LIMIT and first.grade == infra and not cold.executed
    assert second == first


def refused_at_length() -> RolloutInterrupted:
    """A model-server refusal after the generation had reached its length limit."""
    error = RolloutInterrupted(Rollout("refused", (), LENGTH_STOP_REASON), RolloutOperation.MODEL)
    error.__cause__ = ModelRequestRejected(400, "malformed request", ())
    return error


async def test_a_refusal_at_a_length_stop_stays_held_and_is_replaced_only_on_explicit_resume() -> None:
    ctx, log = context(Slots(failures={0: refused_at_length()}))
    (held,) = await solver_batch(ctx, SPEC, "validation", 0, (), 0, 1, concurrency=1, heartbeat=HEARTBEAT, max_missing=1)
    await ctx.drain()
    assert held == TrialResult(0, None, Cause.HELD, REFUSED, receipt=bought(log, 0))
    unasked = Slots(failures={0: AssertionError("a held slot ran without --resume-held")})
    replayed, _ = context(unasked, log.records)
    assert await solver_batch(
        replayed, SPEC, "validation", 0, (), 0, 1, concurrency=1, heartbeat=HEARTBEAT, max_missing=1
    ) == (held,)
    await replayed.drain()
    replacing = Slots()
    replaced, _ = context(replacing, log.records, resume=True)
    (fresh,) = await solver_batch(replaced, SPEC, "validation", 0, (), 0, 1, concurrency=1, heartbeat=HEARTBEAT)
    await replaced.drain()
    assert not unasked.executed and replacing.executed == ["arithmetic:0:0"] and fresh.cause is None


async def test_an_unavailable_grade_after_an_ordinary_stop_still_runs_again() -> None:
    unavailable = GradedRollout(Rollout("ungraded", (), "stop"), GradeResult(Outcome.UNAVAILABLE, None))
    ctx, log = context(Slots(results={0: unavailable}))
    (first,) = await solver_batch(ctx, SPEC, "validation", 0, (), 0, 1, concurrency=1, heartbeat=HEARTBEAT)
    await ctx.drain()
    again = Slots()
    resumed, _ = context(again, log.records)
    (second,) = await solver_batch(resumed, SPEC, "validation", 0, (), 0, 1, concurrency=1, heartbeat=HEARTBEAT)
    await resumed.drain()
    assert first.cause is Cause.NO_GRADE and again.executed == ["arithmetic:0:0"] and second.cause is None


async def test_holds_beyond_the_missing_allowance_hold_the_batch_before_a_sibling_restart() -> None:
    engine = Slots(failures={1: rejected(), 2: lost()})
    ctx, _ = context(engine)
    with pytest.raises(Hold, match="solver trial 1: " + REFUSED):
        await solver_batch(ctx, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT, max_missing=0)
    await ctx.drain()
    assert sorted(engine.executed) == ["arithmetic:0:0", "arithmetic:1:0", "arithmetic:2:0"]


# Regression: an injected hold became one missing slot and the workpiece published ACCEPTED.
@pytest.mark.parametrize("hold", [Hold("environment terminated")], ids=["other"])
async def test_only_a_slot_local_refusal_is_missing_evidence_and_every_other_hold_holds_the_workpiece(
    hold: Hold,
) -> None:
    ctx, _ = context(Slots(failures={1: hold}))
    with pytest.raises(ExceptionGroup) as raised:
        await solver_batch(ctx, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT, max_missing=3)
    await ctx.drain()
    assert [type(error) for error in raised.value.exceptions] == [type(hold)]


@pytest.mark.parametrize("resume", [False, True])
async def test_a_recorded_slot_refusal_runs_again_only_on_explicit_resume(resume: bool) -> None:
    ctx, log = context(Slots(failures={1: rejected()}))
    await solver_batch(ctx, SPEC, "validation", 0, (), 0, 2, concurrency=2, heartbeat=HEARTBEAT, max_missing=1)
    await ctx.drain()
    again = Slots(failures={0: AssertionError("slot 0 ran again")})
    relaunched, _ = context(again, log.records, resume=resume)
    results = await solver_batch(relaunched, SPEC, "validation", 0, (), 0, 2, concurrency=2, heartbeat=HEARTBEAT)
    await relaunched.drain()
    assert again.executed == (["arithmetic:1:0"] if resume else [])
    assert results[1].cause is (None if resume else Cause.HELD)


async def test_a_generation_limit_is_a_failed_slot_and_an_outage_restarts_after_its_siblings() -> None:
    gate = asyncio.Event()
    engine = Slots(gates={0: gate}, failures={1: lost()}, stops={2: LENGTH_STOP_REASON})
    ctx, log = context(engine)
    run = asyncio.create_task(solver_batch(ctx, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT))
    await resolved(ctx, 1).wait()  # the generation-limited slot; the lost one has already failed
    gate.set()  # its sibling was not cancelled by the outage
    with pytest.raises(ExceptionGroup) as raised:
        await run
    await ctx.drain()
    assert [type(error) for error in raised.value.exceptions] == [Restart]
    assert "pod lost" in str(raised.value.exceptions[0])
    again = Slots(failures={0: AssertionError("slot 0 ran again"), 2: AssertionError("slot 2 ran again")})
    resumed, _ = context(again, log.records)
    results = await solver_batch(resumed, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT)
    await resumed.drain()
    assert again.executed == ["arithmetic:1:0"]  # the outage recorded no slot, so only it executes
    assert results[2].cause is Cause.GENERATION_LIMIT and results[2].grade is not None  # a failed, graded trial


def waits() -> RolloutInterrupted:
    """What the engine raises when a slot's grade gets no decision-model read within the wait."""
    error = RolloutInterrupted(Rollout("waits", (), "interrupted"), RolloutOperation.GRADE)
    error.__cause__ = DecisionModelUnavailable("clef@fixture-1 answered no read for 900s")
    return error


# Regression: a refusal beside a slot waiting for the decision model was dropped, the child
# deferred, and the next launch bought the refused slot again. Holds beyond the allowance outrank the outage; within it
# the outage defers alone and the recorded refusal replays.
@pytest.mark.parametrize("allowance", [0, 1])
async def test_a_refusal_beside_a_decision_model_outage_is_settled_before_the_batch_waits(allowance: int) -> None:
    ctx, log = context(Slots(failures={1: rejected(), 2: waits()}))
    with pytest.raises(BaseException) as raised:
        await solver_batch(
            ctx, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT, max_missing=allowance
        )
    await ctx.drain()
    if allowance == 0:
        assert type(raised.value) is Hold and str(raised.value) == f"solver trial 1: {REFUSED}"
        return
    assert isinstance(raised.value, BaseExceptionGroup)
    assert [type(error) for error in raised.value.exceptions] == [DecisionModelUnavailable]
    again = Slots(failures={0: AssertionError("slot 0 ran again"), 1: AssertionError("slot 1 ran again")})
    resumed, _ = context(again, log.records)  # an ordinary launch, never --resume-held
    results = await solver_batch(resumed, SPEC, "validation", 0, (), 0, 3, concurrency=3, heartbeat=HEARTBEAT)
    await resumed.drain()
    assert again.executed == ["arithmetic:2:0"] and results[1] == TrialResult(
        1, None, Cause.HELD, REFUSED, receipt=bought(log, 1)
    )


# Regression: an operator retried a refused slot, the replacement met a decision-
# model outage, and the healthy automatic resume replayed the superseded refusal, asking for a second approval. The
# replacement's start supersedes the refusal it replaces: it resumes, while a refusal it never reached still replays.
async def test_an_authorised_replacement_cut_short_by_an_outage_resumes_and_never_replays_its_refusal() -> None:
    ctx, log = context(Slots(failures={0: rejected(), 1: rejected()}))
    refused = await solver_batch(ctx, SPEC, "validation", 0, (), 0, 2, concurrency=2, heartbeat=HEARTBEAT)
    await ctx.drain()
    assert [trial.cause for trial in refused] == [Cause.HELD] * 2  # review holds it for the operator
    paid = [r.event for r in log.records if isinstance(r.event, Resolved) and isinstance(r.event.request, SolverRollout)]
    slot0 = next(event for event in paid if cast(SolverRollout, event.request).trial == 0)
    ctx, log = context(Slots(failures={0: waits()}), log.records, resume=True)  # the operator's --resume-held
    with pytest.raises(BaseExceptionGroup):
        await solver_batch(ctx, SPEC, "validation", 0, (), 0, 2, concurrency=1, heartbeat=HEARTBEAT)
    await ctx.drain()
    invalidated = [record.event for record in log.records if isinstance(record.event, Invalidated)]
    assert [(event.request, event.fingerprints) for event in invalidated] == [
        (slot0.request, slot0.fingerprints)  # the started replacement names exactly the refusal it replaces
    ]
    healthy = Slots()
    ctx, log = context(healthy, log.records)  # the automatic resume
    resumed = await solver_batch(ctx, SPEC, "validation", 0, (), 0, 2, concurrency=1, heartbeat=HEARTBEAT)
    await ctx.drain()
    assert healthy.executed == ["arithmetic:0:0"]
    assert [trial.cause for trial in resumed] == [None, Cause.HELD] and resumed[1].detail == REFUSED


# Regression: zero concurrency stalled a pending slot forever and a zero heartbeat wrote 1,088 notes in 25 ms.
@pytest.mark.parametrize(
    "key,value",
    [
        ("solver.concurrency", 0),
        ("solver.heartbeat_interval", 0.0),
        ("solver.heartbeat_interval", math.inf),
        ("solver.heartbeat_interval", math.nan),
        ("budget.max_missing_trials", -1),
    ],
)
def test_a_batch_setting_that_would_stall_or_flood_fails_at_wiring(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=re.escape(key)):
        build_config(POLICY.overlaid({key: value}), {})


@pytest.mark.parametrize("key", ["solver.concurrency", "solver.heartbeat_interval", "budget.max_missing_trials"])
def test_a_complete_policy_without_a_batch_setting_fails_at_wiring_naming_it(key: str) -> None:
    with pytest.raises(KeyError, match=re.escape(key)):
        build_config(Policy({name: value for name, value in POLICY.entries.items() if name != key}), {})
