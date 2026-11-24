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
from pathlib import Path
from typing import cast

import pytest
from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.resources.inference_fakes import HANG, Act, Fails, ScriptedProvider, segment
from lib.taskforge.tests.resources.test_decision_model import (
    GRADER,
    FakeReader,
    VirtualTime,
    decision_spec,
    installed,
    overlaid,
    request,
)
from lib.taskforge.tests.test_end_to_end import (
    driver_args,  # noqa: F401 (a fixture)
)
from rolloutengine.models import (
    LENGTH_STOP_REASON,
    ConversationRecord,
    GradedRollout,
    Model,
    ModelTurn,
    Rollout,
    RolloutInterrupted,
    RolloutOperation,
    RolloutStep,
    TaskRolloutSpec,
    TraceMode,
    Transition,
)
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import AnswerType, FinalAction, FunctionDefinition

from taskforge.checkpoint import CheckpointSink
from taskforge.effects.context import Context
from taskforge.effects.events import Invalidated, Resolved, ScopeEntered, ScopeExited
from taskforge.effects.issues import DecisionModelUnavailable
from taskforge.effects.log import Billing, EventLog, JsonlSink, Record, Status
from taskforge.llm.client import AttemptOutcome, ModelRequestRejected, Segment
from taskforge.lowering import lower_with_settings, lowering_settings
from taskforge.model.llm import FinishReason, ToolCall
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import Cause
from taskforge.model.workpiece import Terminal, Workpiece
from taskforge.policy import Policy
from taskforge.report import report
from taskforge.resources.rollouts import Engine, SolverGeneration, SolverRollout, refused
from taskforge.stages.classify import replaced_refusals, replayable_grade, trial_cause
from taskforge.wiring import install

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
    sink: JsonlSink,
    script: Script = (segment("12"),),
    *,
    resume: bool = False,
    native: bool = False,
    overlay: Mapping[str, object] = WAIT,
    engine: Engine | None = None,
) -> tuple[GradedRollout | DecisionModelUnavailable, int, Sequence[Record]]:
    """Slot 0 of a decision-model task, one launch replaying ``sink`` as the driver builds its log (``resume`` is
    ``--resume-held``): what the slot came to, the solver requests it dispatched, and the launch's records. ``engine``
    replaces RolloutEngine, under the same fingerprint."""
    history = local_records(sink) if Path(sink.path).exists() else ()
    policy = overlaid().overlaid(overlay)
    solver, replaces = ScriptedProvider(script), replaced_refusals(history, resume)
    log = EventLog(sink=sink, replay_from=history, replayable=replayable_grade, replaces=replaces)
    ctx = Context(log=log)
    built = installed(policy, reader, solver, ctx, VirtualTime())
    if engine is not None:
        install(ctx, replace(built, rollouts=replace(built.rollouts, engine=engine)))  # the later handlers answer
    outcome: GradedRollout | DecisionModelUnavailable
    try:
        async with ctx.scope("validate", bills=Billing.SOLVER):  # as the validation stage runs its slots
            outcome = await ctx.emit(SolverRollout(native_spec(policy) if native else decision_spec(policy), 0, 0))
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
async def test_an_outage_after_a_finished_generation_is_graded_on_resume_without_generating_again(
    tmp_path: Path, native: bool
) -> None:
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
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


async def test_a_cutoff_before_any_answer_stays_a_completed_loss_without_a_checkpoint(tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
    reader = FakeReader(LEVELS, down={"", "12"})
    cut, solved, first = await slot(reader, sink, (segment("", finish=FinishReason.LENGTH),))
    assert isinstance(cut, GradedRollout) and cut.rollout.stop_reason == LENGTH_STOP_REASON and solved == 1
    assert trial_cause(cut, arithmetic_task(grader=GRADER)) is Cause.GENERATION_LIMIT and not reader.reads
    assert not generations(first) and slot_results(first) == [cut]
    replayed, solved, _ = await slot(FakeReader(LEVELS), sink, (), resume=True)
    assert (replayed, solved) == (cut, 0)


async def test_a_refused_slot_stays_held_until_replaced_and_its_replacements_outage_is_checkpointed(
    tmp_path: Path,
) -> None:
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
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


async def test_an_off_scale_level_is_read_again_on_an_explicit_resume_from_its_checkpoint(tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
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


async def test_a_changed_generation_fingerprint_generates_again(tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
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


async def test_a_generation_interrupted_before_it_finished_is_generated_again(tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
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
    sink = CheckpointSink(tmp_path / "work", str(tmp_path / "root"))
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
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
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
    assert first == {"exit": 75, "solves": 2, "reads": ["12", "9"], "terminal": "held", "awaits": "decision_model"}
    second = probed("resume", tmp_path)
    assert second["exit"] == 0 and second["terminal"] == "accepted" and second["awaits"] is None
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
