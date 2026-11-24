# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A solver generation whose decision-model grade an outage or an off-scale level ended is journalled before the slot
holds, so that resuming the slot grades it again without generating it."""

import asyncio
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import cast

import pytest
from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.resources.inference_fakes import HANG, Act, Fails, LoopTime, ScriptedProvider, segment
from lib.taskforge.tests.resources.test_decision_model import (
    GRADER,
    FakeReader,
    VirtualTime,
    decision_spec,
    installed,
    overlaid,
    request,
)
from lib.taskforge.tests.resources.test_provider import GENERATION_BOUND, bounded, generates
from lib.taskforge.tests.test_end_to_end import (
    driver_args,  # noqa: F401 (a fixture)
)
from rolloutengine import engine as engine_module
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import (
    LENGTH_STOP_REASON,
    ConversationRecord,
    GenerationLimitReached,
    GradedRollout,
    Model,
    ModelRequest,
    ModelTurn,
    Rollout,
    RolloutInterrupted,
    RolloutOperation,
    RolloutStep,
    TaskRolloutSpec,
    TraceMode,
    Transition,
)
from rolloutengine.test_support import arithmetic_task, machine_runtime, rollout_spec
from taskcompendium.decision_model import DecisionModelConnection
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import AnswerType, FinalAction, FunctionDefinition, GradingAttempt, TaskSpec

from taskforge.checkpoint import JOURNAL_FORMAT, CheckpointSink
from taskforge.effects.context import Context
from taskforge.effects.events import Invalidated, Resolved, ScopeEntered, ScopeExited
from taskforge.effects.issues import DecisionModelUnavailable
from taskforge.effects.log import Billing, BufferSink, EventLog, Record, Status
from taskforge.llm.client import (
    AttemptOutcome,
    ContextExhausted,
    GenerationTimedOut,
    ModelClient,
    ModelRequestRejected,
    Segment,
)
from taskforge.llm.policy import GLM_5_3, LLMPolicy
from taskforge.lowering import lower_with_settings, lowering_settings, requirements
from taskforge.model.llm import FinishReason, ToolCall
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import Cause
from taskforge.model.workpiece import Terminal, Workpiece
from taskforge.policy import Policy
from taskforge.report import report
from taskforge.resources import decision_model
from taskforge.resources.rollouts import Engine, SolverGeneration, SolverRollout, refused
from taskforge.resources.sandbox.factories import shellsim_factory
from taskforge.stages.classify import replaced_refusals, replayable_grade, trial_cause
from taskforge.wiring import install, retry_policy, solver_turn

WAIT: Mapping[str, object] = {"decision_model.unavailable_wait": 10.0}
NATIVE = '{"arguments":{"value":12},"name":"answer","type":"function_call"}'
"""TaskCompendium's candidate text for the native action ``answer({"value": 12})``."""
LEVELS = {"12": 4.0, NATIVE: 4.0}


def native_spec(policy: Policy) -> TaskRolloutSpec:
    answer = FunctionDefinition(
        name="answer", parameters={"type": "object", "properties": {"value": {"type": "integer"}}}
    )
    task = arithmetic_task(
        grader=GRADER, answer_type=AnswerType.NATIVE_ACTION, answer_format=FinalAction(), final_tools=(answer,)
    )
    return lower_with_settings(task, lowering_settings(policy), {}).model_copy(update={"trace": TraceMode.MESSAGES})


def native_answer() -> Segment:
    return segment(tool_calls=(ToolCall("call-1", "answer", json.dumps({"value": 12})),), finish=FinishReason.TOOL_CALLS)


type Script = Sequence[Segment | Fails | str | Act]


async def slot(
    reader: FakeReader,
    sink: BufferSink,
    script: Script = (segment("12"),),
    *,
    resume: bool = False,
    native: bool = False,
    overlay: Mapping[str, object] = WAIT,
    engine: Engine | None = None,
    model: Model | None = None,
    spec: TaskRolloutSpec | None = None,
) -> tuple[GradedRollout | DecisionModelUnavailable, int, Sequence[Record]]:
    """Slot 0 of a decision-model task (or of ``spec``), one launch replaying ``sink`` as the driver builds its log
    (``resume`` is ``--resume-held``): what the slot came to, the solver requests it dispatched, and the launch's
    records. ``engine`` replaces RolloutEngine and ``model`` the solver transport, under the same fingerprint."""
    history = local_records(sink)
    policy = overlaid().overlaid(overlay)
    solver, replaces = ScriptedProvider(script), replaced_refusals(history, resume)
    log = EventLog(sink=sink, replay_from=history, replayable=replayable_grade, replaces=replaces)
    ctx = Context(log=log)
    built = installed(policy, reader, solver, ctx, VirtualTime())
    if engine is not None or model is not None:
        rollouts = replace(built.rollouts, engine=engine or built.rollouts.engine, model=model or built.rollouts.model)
        install(ctx, replace(built, rollouts=rollouts))  # the later handlers answer
    outcome: GradedRollout | DecisionModelUnavailable
    try:
        async with ctx.scope("validate", bills=Billing.SOLVER):  # as the validation stage runs its slots
            spec = spec or (native_spec(policy) if native else decision_spec(policy))
            outcome = await ctx.emit(SolverRollout(spec, 0, 0))
    except DecisionModelUnavailable as error:
        outcome = error
    await ctx.drain()
    return outcome, len(solver.bodies), log.records[len(history) :]


def generations(records: Sequence[Record]) -> list[Rollout]:
    return [
        r.event.value for r in records if isinstance(r.event, Resolved) and isinstance(r.event.request, SolverGeneration)
    ]


def slot_results(records: Sequence[Record]) -> list[GradedRollout]:
    return [
        r.event.value for r in records if isinstance(r.event, Resolved) and isinstance(r.event.request, SolverRollout)
    ]


@pytest.mark.parametrize("native", [False, True], ids=["text answer", "native action"])
async def test_an_outage_after_a_finished_generation_is_graded_on_resume_without_generating_again(native: bool) -> None:
    sink = BufferSink()
    answer, candidate = (native_answer(), NATIVE) if native else (segment("12"), "12")
    deferred, solved, first = await slot(FakeReader(LEVELS, down={candidate}), sink, (answer,), native=native)
    assert isinstance(deferred, DecisionModelUnavailable) and solved == 1
    [generated] = generations(first)
    assert (
        not slot_results(first)
        and generated.failure is None
        and generated.stop_reason == ("tool_calls" if native else "stop")
    )
    healthy = FakeReader(LEVELS)
    graded, solved, resumed = await slot(healthy, sink, (), native=native)  # an unflagged launch: any call would fail
    assert isinstance(graded, GradedRollout) and (graded.grade.reward, solved) == (1.0, 0)
    assert healthy.reads == [request(candidate)] and graded.rollout == generated and not generations(resumed)
    again = FakeReader(LEVELS)
    replayed, solved, _ = await slot(again, sink, (), native=native)
    assert (replayed, solved, again.reads) == (graded, 0, [])


async def test_a_cutoff_before_any_answer_stays_a_completed_loss_without_a_checkpoint() -> None:
    sink = BufferSink()
    reader = FakeReader(LEVELS, down={"", "12"})
    cut, solved, first = await slot(reader, sink, (segment("", finish=FinishReason.LENGTH),))
    assert isinstance(cut, GradedRollout) and cut.rollout.stop_reason == LENGTH_STOP_REASON and solved == 1
    assert trial_cause(cut, arithmetic_task(grader=GRADER)) is Cause.GENERATION_LIMIT and not reader.reads
    assert not generations(first) and slot_results(first) == [cut]
    replayed, solved, _ = await slot(FakeReader(LEVELS), sink, (), resume=True)
    assert (replayed, solved) == (cut, 0)


async def test_a_refused_slot_stays_held_until_replaced_and_its_replacements_outage_is_checkpointed() -> None:
    sink = BufferSink()
    rejected = Fails(AttemptOutcome.REJECTED, status=400)
    held, solved, first = await slot(FakeReader(LEVELS), sink, (rejected,))
    assert isinstance(held, GradedRollout) and refused(held) and solved == 1 and not generations(first)
    kept, solved, _ = await slot(FakeReader(LEVELS), sink, ())
    assert (kept, solved) == (held, 0)
    replacing = FakeReader(LEVELS, down={"12"})
    deferred, solved, second = await slot(replacing, sink, (segment("12"),), resume=True)
    assert isinstance(deferred, DecisionModelUnavailable) and solved == 1 and len(generations(second)) == 1
    assert [r for r in second if isinstance(r.event, Invalidated)]
    healthy = FakeReader(LEVELS)
    graded, solved, _ = await slot(healthy, sink, ())
    assert isinstance(graded, GradedRollout) and (graded.grade.reward, solved, healthy.reads) == (
        1.0,
        0,
        [request("12")],
    )


async def test_an_off_scale_level_is_read_again_on_an_explicit_resume_from_its_checkpoint() -> None:
    sink = BufferSink()
    off, solved, first = await slot(FakeReader({"12": 7}), sink)
    assert isinstance(off, GradedRollout) and off.grade.status is Outcome.INFRA_ERROR and solved == 1
    assert generations(first) == [off.rollout] and slot_results(first) == [off]
    healthy = FakeReader(LEVELS)
    graded, solved, _ = await slot(healthy, sink, (), resume=True)
    assert isinstance(graded, GradedRollout) and (graded.grade.reward, solved, healthy.reads) == (
        1.0,
        0,
        [request("12")],
    )
    assert graded.rollout == off.rollout


async def test_a_changed_generation_fingerprint_generates_again() -> None:
    sink = BufferSink()
    deferred, _, first = await slot(FakeReader(LEVELS, down={"12"}), sink)
    assert isinstance(deferred, DecisionModelUnavailable) and generations(first)
    changed = {**WAIT, "solver.tool_timeout": 121.0}
    graded, solved, resumed = await slot(FakeReader(LEVELS), sink, (segment("12"),), overlay=changed)
    assert isinstance(graded, GradedRollout) and (graded.grade.reward, solved) == (1.0, 1)
    [before] = [
        r.event for r in first if isinstance(r.event, Resolved) and isinstance(r.event.request, SolverGeneration)
    ]
    [after] = [r.event for r in resumed if isinstance(r.event, Resolved) and isinstance(r.event.request, SolverRollout)]
    assert before.fingerprints != after.fingerprints  # the rollouts fingerprint is the generation's


async def test_a_generation_interrupted_before_it_finished_is_generated_again() -> None:
    sink = BufferSink()
    policy = overlaid().overlaid(WAIT)
    solver, ctx = ScriptedProvider((HANG,)), Context(log=EventLog(sink=sink))
    installed(policy, FakeReader(LEVELS), solver, ctx, VirtualTime())
    running = asyncio.create_task(ctx.emit(SolverRollout(decision_spec(policy), 0, 0)))
    await solver.streaming.wait()
    running.cancel()  # the process stops mid-generation
    await asyncio.gather(running, return_exceptions=True)
    await ctx.drain()
    assert not generations(local_records(sink))
    graded, solved, _ = await slot(FakeReader(LEVELS), sink, (segment("12"),))
    assert isinstance(graded, GradedRollout) and (graded.grade.reward, solved) == (1.0, 1)


class AnsweredThenCut(Engine):
    """RolloutEngine on an attempt that answered 12 in its first turn and whose next turn ``model`` cut: the engine
    stops it with its ``length`` and grades the answer, which meets a decision-model outage."""

    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        with pytest.raises(GenerationLimitReached):
            await model(ModelRequest((), {}))
        turn = ModelTurn({"role": "assistant", "content": "12"}, (), (), None, "stop")
        chat = ({"role": "user", "content": "What is six plus six?"}, turn.message)
        record = ConversationRecord(chat, (), (), (), None, (RolloutStep(turn, Transition(False), 0, ()),))
        cut = Rollout("r", (record,), LENGTH_STOP_REASON)
        raise RolloutInterrupted(cut, RolloutOperation.GRADE) from DecisionModelUnavailable("HTTP 503")


def cut_by(limit: Exception) -> Model:
    async def model(_: ModelRequest) -> ModelTurn:
        raise GenerationLimitReached(()) from limit  # as ``solver_turn`` reports either limit

    return model


# Regression: a time-limited generation's checkpoint recorded the engine's ``length``, so the slot graded after
# the outage was a generation limit; the recovering grader must see what the engine's grader saw.
@pytest.mark.parametrize(
    "limit, stop, cause",
    [
        (GenerationTimedOut("generated for 4200s"), "time_limit", Cause.TIME_LIMIT),
        (ContextExhausted("context window", ()), LENGTH_STOP_REASON, Cause.GENERATION_LIMIT),
    ],
    ids=["time limit", "context limit"],
)
async def test_a_limited_generation_keeps_its_limit_through_a_grading_outage_and_its_regrade(
    monkeypatch: pytest.MonkeyPatch, limit: Exception, stop: str, cause: Cause
) -> None:
    sink = BufferSink()
    deferred, _, first = await slot(FakeReader(LEVELS), sink, (), engine=AnsweredThenCut(), model=cut_by(limit))
    assert isinstance(deferred, DecisionModelUnavailable)
    assert isinstance(deferred.__cause__, RolloutInterrupted) and deferred.__cause__.__cause__ is not None
    [generated] = generations(first)
    assert generated.stop_reason == stop
    presented: list[str | None] = []
    grade = decision_model.grade_decision_model_answer

    async def shown(task: TaskSpec, attempt: GradingAttempt, read: DecisionModelConnection) -> GradeResult:
        presented.append(attempt.stop_reason)
        return await grade(task, attempt, read)

    monkeypatch.setattr(decision_model, "grade_decision_model_answer", shown)
    healthy = FakeReader(LEVELS)
    graded, solved, _ = await slot(healthy, sink, ())  # unflagged: any solver call would fail
    assert isinstance(graded, GradedRollout) and (solved, healthy.reads) == (0, [request("12")])
    assert graded.rollout.stop_reason == stop and trial_cause(graded, arithmetic_task(grader=GRADER)) is cause
    assert presented == [LENGTH_STOP_REASON]  # what RolloutEngine set before its own grade, for either limit


async def unscorable(*_args: object, **_kwargs: object) -> GradeResult:
    return GradeResult(Outcome.INFRA_ERROR, None, "grader crashed", failure=GradingFailure.EXECUTION)


async def hangs(*_args: object, **_kwargs: object) -> GradeResult:
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


# Regression: after a shell action and a confirmed time-limited turn, an ordinary verifier that RAISED at
# its own deadline restarted the paid slot three times per launch, while one that returned its error settled the loss.
# A decision-model outage still defers a limited generation as a checkpoint that recovery grades without generating.
@pytest.mark.parametrize("verifier", ["returned error", "verifier timeout", "decision-model outage"])
async def test_a_limited_slot_settles_whatever_its_ordinary_verifier_does_and_an_outage_regrades_it(
    monkeypatch: pytest.MonkeyPatch, verifier: str
) -> None:
    sink = BufferSink()
    if verifier == "decision-model outage":  # a read needs a text answer, which ends a shell rollout: a cut answer
        first, paid, _ = await slot(FakeReader(LEVELS, down={"12"}), sink, (segment("12", finish=FinishReason.LENGTH),))
        assert isinstance(first, DecisionModelUnavailable) and paid == 1
        healthy = FakeReader(LEVELS)
        first, paid, _ = await slot(healthy, sink, ())
        assert (paid, healthy.reads) == (0, [request("12")]) and isinstance(first, GradedRollout)
        assert trial_cause(first, arithmetic_task(grader=GRADER)) is Cause.GENERATION_LIMIT and first.grade.reward == 1
        again, paid, _ = await slot(FakeReader(LEVELS), sink, ())
        assert (again, paid) == (first, 0)
        return
    monkeypatch.setattr(engine_module, "_grade_rollout", unscorable if verifier == "returned error" else hangs)
    task = arithmetic_task(environment_requirements=requirements(image=None))
    shell = rollout_spec(
        task, machine=machine_runtime(backend="shellsim"), trace=TraceMode.MESSAGES, verifier_timeout=0.1
    )
    engine = RolloutEngine({"shellsim": shellsim_factory()}, {"shell": ShellHarness(command_timeout=None)})
    acted = segment("12", (ToolCall("c1", "shell", json.dumps({"command": "echo 12"})),), FinishReason.TOOL_CALLS)

    async def launch(script: Script) -> tuple[GradedRollout | DecisionModelUnavailable, int]:
        provider = ScriptedProvider(script)
        model = partial(solver_turn, bounded(provider, LoopTime()), LLMPolicy())
        outcome, _, _ = await slot(FakeReader(LEVELS), sink, (), engine=engine, model=model, spec=shell)
        return outcome, len(provider.bodies)

    first, paid = await launch((acted, generates(LoopTime(), 10 * GENERATION_BOUND)))
    assert paid == 2 and isinstance(first, GradedRollout) and trial_cause(first, task) is Cause.TIME_LIMIT
    assert first.rollout.failure is None and len(first.rollout.conversations[0].steps) == 1  # the paid shell action
    assert first.grade.failure is (GradingFailure.EXECUTION if verifier == "returned error" else GradingFailure.TIMEOUT)
    again, paid = await launch(())  # a resume replays the settled slot without a solver call
    assert (again, paid) == (first, 0)


async def bounded_slot(sink: BufferSink, limit: float, reader: FakeReader) -> tuple[object, int]:
    """Slot 0 of the decision-model task at ``solver.time_limit=limit``, through the real model client on a stream that
    makes progress for 2,000 s and then answers 12: what it came to and the solver calls it made."""
    overlay = {**WAIT, "solver.time_limit": limit}
    time = LoopTime()
    provider = ScriptedProvider((generates(time, 2000),))
    retry = retry_policy(overlaid().overlaid(overlay), "solver")
    client = ModelClient(provider, GLM_5_3, retry, role="solver", clock=time.clock, sleep=time.sleep)
    outcome, _, _ = await slot(reader, sink, (), overlay=overlay, model=partial(solver_turn, client, LLMPolicy()))
    return outcome, len(provider.bodies)


# Regression: with no model-turn wall the time limit left the solver's replay identity, so a slot settled
# at one limit replayed at another, in either direction, and a checkpointed generation was regraded under another bound.
@pytest.mark.parametrize(("first", "then"), [(1800.0, 7200.0), (7200.0, 1800.0)], ids=["raised", "lowered"])
async def test_a_changed_time_limit_buys_the_slot_and_its_checkpoint_again_and_an_unchanged_one_replays(
    first: float, then: float
) -> None:
    def settled(outcome: object, limit: float) -> bool:
        cause = Cause.TIME_LIMIT if limit < 2000 else None
        return isinstance(outcome, GradedRollout) and trial_cause(outcome, arithmetic_task(grader=GRADER)) is cause

    sink = BufferSink()
    outcome, calls = await bounded_slot(sink, first, FakeReader(LEVELS))
    assert settled(outcome, first) and calls == 1
    replayed, calls = await bounded_slot(sink, first, FakeReader(LEVELS))
    assert (replayed, calls) == (outcome, 0)  # the same bound replays
    changed, calls = await bounded_slot(sink, then, FakeReader(LEVELS))
    assert settled(changed, then) and calls == 1  # another bound buys the slot again
    fresh, calls = await bounded_slot(BufferSink(), then, FakeReader(LEVELS))
    assert settled(fresh, then) and calls == 1
    checkpointed = BufferSink()
    deferred, calls = await bounded_slot(checkpointed, 7200.0, FakeReader(LEVELS, down={"12"}))
    assert isinstance(deferred, DecisionModelUnavailable) and calls == 1  # a 2,000 s generation, checkpointed
    regraded, calls = await bounded_slot(checkpointed, 1800.0, FakeReader(LEVELS))
    assert settled(regraded, 1800.0) and calls == 1  # never graded under a bound it outran


def metered(prompt: int, completion: int) -> Rollout:
    usage = {"prompt_tokens": prompt, "completion_tokens": completion}
    turn = ModelTurn({}, (), (), None, "stop", metadata={"usage": usage})
    return Rollout(
        "r", (ConversationRecord((), (), (), (), None, (RolloutStep(turn, Transition(True), 0, ()),)),), "stop"
    )


async def test_a_checkpointed_generation_is_charged_once_in_validation_and_confirmation(
    tmp_path: Path, proposal: TaskProposal
) -> None:
    """Slot 0 and slot 2 were graded from their checkpoints; slot 3 still awaits its grade; slot 1 never needed one."""
    spec, rollout, graded = decision_spec(overlaid()), metered(40, 7), GradeResult(Outcome.GRADED, 1.0)
    fp = ()

    def generation(trial: int, attempt: int) -> Resolved[Rollout]:
        return Resolved(SolverGeneration(spec, trial, attempt, rollout), fp, rollout)

    def result(trial: int, attempt: int) -> Resolved[GradedRollout]:
        return Resolved(SolverRollout(spec, trial, attempt), fp, GradedRollout(rollout, graded))

    solver, confirmation = Billing.SOLVER, Billing.CONFIRMATION  # every slot row names its payer, graded ones too
    events = (
        (ScopeEntered("workpiece"), None),
        (ScopeEntered("validate"), None),
        (generation(0, 0), solver),
        (result(0, 0), solver),
        (result(1, 0), solver),
        (ScopeExited("validate", False), None),
        (ScopeEntered("confirm"), None),
        (generation(2, 1), confirmation),
        (result(2, 1), confirmation),
        (generation(3, 1), confirmation),
        (ScopeExited("confirm", True), None),
    )
    records = [Record(i, time.time(), e, billed=payer) for i, (e, payer) in enumerate(events, 1)]
    lines = await reported(tmp_path, proposal, records)
    assert "tokens role=solver prompt=80 cached=0 completion=14" in lines
    assert "tokens role=confirmation prompt=80 cached=0 completion=14" in lines


async def reported(
    tmp_path: Path, proposal: TaskProposal, records: Sequence[Record], since_hours: float | None = None
) -> tuple[str, ...]:
    """The report of a held root whose journal is ``records``."""
    sink = CheckpointSink(str(tmp_path / "root"))
    sink.claim(None)
    await sink.local.write(tuple(records))
    await sink.finish(Status(terminal=Terminal.HELD), Workpiece("w", proposal, terminal=Terminal.HELD))
    return await report(str(tmp_path / "root"), since_hours)


class RejectedAfterATurn(Engine):
    """A rollout whose first model turn was paid (10 prompt, 5 completion tokens) and whose second was refused."""

    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        rejected = ModelRequestRejected(400, "bad request", ())
        raise RolloutInterrupted(metered(10, 5), RolloutOperation.MODEL) from rejected


# Regression: the report suppressed every slot result whose key had a checkpoint anywhere in the journal, so a
# refused slot that had paid for a shell turn lost its charge once its replacement checkpointed: 10/5 instead of 20/10.
async def test_a_paid_refusal_before_its_replacements_checkpoint_keeps_its_charge(
    tmp_path: Path, proposal: TaskProposal
) -> None:
    sink = BufferSink()
    held, _, _ = await slot(FakeReader(LEVELS), sink, (), engine=RejectedAfterATurn())
    assert isinstance(held, GradedRollout) and refused(held)
    off, solved, _ = await slot(FakeReader({"12": 7}), sink, (segment("12"),), resume=True)
    assert isinstance(off, GradedRollout) and off.grade.status is Outcome.INFRA_ERROR and solved == 1
    graded, solved, _ = await slot(FakeReader(LEVELS), sink, (), resume=True)
    assert isinstance(graded, GradedRollout) and (graded.grade.reward, solved) == (1.0, 0)
    assert "tokens role=solver prompt=20 cached=0 completion=10" in await reported(
        tmp_path, proposal, local_records(sink)
    )


# Regression: a format-8 root's off-scale slot result predates any checkpoint, yet the replacement's checkpoint
# suppressed it; and a window opening between a checkpoint and its regrade must not charge the regrade. The rows
# before format 10 name no payer and charge as unattributed; the regrade names one and is still not charged.
@pytest.mark.parametrize("since_hours, charged", [(None, (20, 10)), (1.0, (0, 0))], ids=["all time", "after the gen"])
async def test_a_slot_result_is_charged_unless_it_follows_its_checkpoint(
    tmp_path: Path, proposal: TaskProposal, since_hours: float | None, charged: tuple[int, int]
) -> None:
    spec, rollout = decision_spec(overlaid()), metered(10, 5)
    off, graded = GradeResult(Outcome.INFRA_ERROR, None, "off-scale"), GradeResult(Outcome.GRADED, 1.0)
    old, now = time.time() - 3 * 3600, time.time()
    events = [
        (old, Resolved(SolverRollout(spec, 0, 0), (), GradedRollout(rollout, off))),  # a format-8 writer's result
        (old, Resolved(SolverGeneration(spec, 0, 0, rollout), (), rollout)),  # its replacement, checkpointed
        (old, Resolved(SolverRollout(spec, 0, 0), (), GradedRollout(rollout, off))),
        (now, Resolved(SolverRollout(spec, 0, 0), (), GradedRollout(rollout, graded))),  # graded from the checkpoint
    ]
    records = [
        Record(i, stamp, event, billed=Billing.SOLVER if stamp == now else None)
        for i, (stamp, event) in enumerate(events, 1)
    ]
    prompt, completion = charged
    lines = await reported(tmp_path, proposal, records, since_hours)
    assert "tokens role=solver prompt=0 cached=0 completion=0" in lines
    unattributed = f"tokens role=unattributed prompt={prompt} cached=0 completion={completion}"
    assert unattributed in lines if charged != (0, 0) else not any("role=unattributed" in line for line in lines)


PROBE = [sys.executable, "-m", "lib.taskforge.tests.generation_replay_probe"]


def probed(mode: str, directory: Path) -> Mapping[str, object]:
    done = subprocess.run(
        [*PROBE, mode, str(directory)], cwd=Path(__file__).parents[3], capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_grading_outage_defers_and_a_fresh_process_grades_the_generations_without_buying_them_again(
    tmp_path: Path,
) -> None:
    """Through the public driver: the two validation answers are generated, and their grade meets a decision-model
    outage: exit 75. A second process, with the model back and no flag, generates neither again, reads each answer
    once, buys the five confirmation trials (whose equal answers replay those reads) and accepts; every generation is
    charged once."""
    first = probed("record", tmp_path)
    assert first == {
        "exit": 75,
        "solves": 2,
        "reads": ["12", "9"],
        "terminal": "held",
        "awaits": "decision_model",
        "format": JOURNAL_FORMAT,
    }
    second = probed("resume", tmp_path)
    assert (
        second["format"] == JOURNAL_FORMAT
        and second["exit"] == 0
        and second["terminal"] == "accepted"
        and second["awaits"] is None
    )
    assert (second["solves"], second["reads"], second["generations"]) == (5, ["thirteen", "twelve"], 2)
    assert second["graded"] == ["solver", "solver"]  # a payer on the grade, yet the generations are not charged again
    assert second["solver"] == "tokens role=solver prompt=20 cached=0 completion=10"
    assert second["confirmation"] == "tokens role=confirmation prompt=50 cached=0 completion=25"


# Regression: a gateway refusing a resumed read (HTTP 400), or answering it with all-zero probabilities or
# without a level's probability, or with one too large for a float, escaped its slot as a bug, cancelled the read of a
# sibling slot's fresh answer, failed the child, and so made recovery buy that answer again: six solvers, not five.
@pytest.mark.parametrize(
    "failure, reason",
    [
        ("refused", "Hold: Decision model refused the read with HTTP 400: bad state"),
        ("zero", "Hold: Decision model answered unusably: ValueError: probabilities [0.0, 0.0, 0.0, 0.0, 0.0] are not"),
        ("missing", "Hold: Decision model answered unusably: KeyError: '4'"),
        ("overflow", "Hold: Decision model answered unusably: OverflowError"),
        ("off-scale", "returned 7.0"),
    ],
)
def test_a_resumed_read_that_fails_holds_after_its_siblings_settle_and_recovery_buys_only_confirmation(
    tmp_path: Path, failure: str, reason: str
) -> None:
    """Three processes: slots 0 and 1 are checkpointed in an outage; an unflagged resume grades slot 1 and generates
    slot 2, whose read is in flight when slot 0's read is refused (or answered malformed, or off the scale); the
    explicit recovery reads slot 0 once more and buys only the five confirmation trials."""
    assert probed("sibling-record", tmp_path)["exit"] == 75
    held = dict(probed(f"sibling-{failure}", tmp_path))
    [why] = cast(list[str], held.pop("reasons"))
    del held["format"]
    assert held == {"exit": 0, "solves": 1, "reads": ["a dozen", "thirteen"], "terminal": "held", "awaits": None}
    assert reason in why  # the ordinary hold, never a bug, keeps the gateway's diagnostic
    recovered = probed("sibling-recover", tmp_path)
    assert (recovered["exit"], recovered["terminal"], recovered["solves"], recovered["reads"]) == (
        0,
        "accepted",
        5,
        ["twelve"],
    )
    assert recovered["solver"] == "tokens role=solver prompt=30 cached=0 completion=15"
