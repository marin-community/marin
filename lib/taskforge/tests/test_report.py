# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest
from lib.taskforge.tests.test_end_to_end import N6_MODIFY_HELD, plant_root
from rolloutengine.models import ConversationRecord, GradedRollout, ModelTurn, Rollout, RolloutStep, Transition
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.checkpoint import HEAD, JOURNAL_FORMAT, CheckpointSink, read_journal
from taskforge.effects.context import Context
from taskforge.effects.events import (
    ENVIRONMENT,
    Event,
    Fingerprint,
    Invalidated,
    Note,
    ReplayPolicy,
    Resolved,
    ScopeEntered,
    ScopeExited,
    StageCompleted,
)
from taskforge.effects.log import Billing, EventLog, Opened, Record, Status
from taskforge.harness.tools.evidence import ReportTool, evidence_tools
from taskforge.llm.policy import LLMPolicy
from taskforge.lowering import lower_with_settings, lowering_settings
from taskforge.model.llm import Completion, Usage
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import Abandon, Incomplete
from taskforge.model.stage import Stage
from taskforge.model.workpiece import Submission, Terminal, Workpiece
from taskforge.policy import load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.report import reason_class, report
from taskforge.resources.env import CommandResult, RunCommand, WriteFile
from taskforge.resources.inference import InferenceRequest
from taskforge.resources.rollouts import SolverRollout
from taskforge.stages.solve_review import evidence


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("Restart: unavailable <- ConnectError: pool full", "Restart"),
        ("author stopped: generation limit after 4 turns", "author stopped"),
        ("bug: ValueError: bad", "bug:ValueError"),
        ("bug: shellbox.machine.MachineUnavailable: down", "bug:MachineUnavailable"),
        (
            "bug: Traceback\nconnectrpc.errors.ConnectError: pool full\ncaused by\nRuntimeError: wrapped",
            "bug:ConnectError",
        ),
        ("bug: ExceptionGroup: bad\n  | KeyError: missing", "bug:KeyError"),
    ],
)
def test_report_reason_class(reason: str, expected: str) -> None:
    assert reason_class(reason) == expected


async def test_report_reads_journals_counts_usage_once_and_uses_decision_time(
    tmp_path: Path, proposal: TaskProposal, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("taskforge.report.time.time", lambda: 1000000.0)
    inference = InferenceRequest((), (), LLMPolicy())
    completion = Completion("ok", usage=Usage(prompt_tokens=100, cached_tokens=20, completion_tokens=10))
    turn = ModelTurn(
        {}, (), (), None, "stop", metadata={"usage": {"prompt_tokens": 40, "cached_tokens": 15, "completion_tokens": 7}}
    )
    conversation = ConversationRecord((), (), (), (), None, (RolloutStep(turn, Transition(True), 0, ()),))
    result = GradedRollout(Rollout("r", (conversation,), "stop"), GradeResult(Outcome.GRADED, 1.0))
    spec = lower_with_settings(arithmetic_task(), lowering_settings(load_policy(DEFAULTS_DIR)), {})
    sandbox = (Fingerprint(ENVIRONMENT, "definition", ReplayPolicy.REEXECUTE),)
    opened, attempts, answers = {3: Opened("author/0", "definition")}, {7: 3, 9: 3}, {8: 7, 10: 9}  # sandbox rows
    events: list[tuple[float, Event[object]]] = [
        (-2, ScopeEntered("judge")),  # An interrupted previous process left this scope open.
        (-1, ScopeEntered("workpiece")),
        (0, ScopeEntered("author")),
        (10, Resolved(inference, (), completion)),
        (11, Resolved(inference, (), completion, replayed=True)),
        (20, ScopeExited("author", False)),
        (21, RunCommand("true", 30)),
        (21, Resolved(RunCommand("true", 30), sandbox, CommandResult(0, "", "", 1, False))),
        (21, WriteFile(PurePosixPath("x"), b"x")),
        (24, Resolved(WriteFile(PurePosixPath("x"), b"x"), sandbox, None)),
        (25, ScopeEntered("adversary:a")),
        (30, Resolved(replace(inference, sample=1), (), completion)),
        (31, ScopeExited("adversary:a", False)),
        (32, Resolved(SolverRollout(spec, 0, 0), (), result)),
        (33, Resolved(SolverRollout(spec, 0, 0), (), result, replayed=True)),
        (33, Invalidated(SolverRollout(spec, 0, 0), ())),  # an authorised replacement starts: no usage
        (34, StageCompleted(Stage.REVIEW, "", "Accepted", terminal=Terminal.ACCEPTED)),
    ]
    for name, age, terminal, reasons in (
        ("accepted", 1800, Terminal.ACCEPTED, ()),
        ("old", 8 * 3600, Terminal.ACCEPTED, ()),
        ("held", 100, Terminal.HELD, ("Restart: transport",)),
        ("abandoned", 100, Terminal.ABANDONED, ("author stopped: generation limit",)),
    ):
        directory = tmp_path / name
        directory.mkdir()
        decisions = (Abandon(reasons),) if terminal is Terminal.ABANDONED else (Incomplete(reasons),)
        data = events if name == "accepted" else [(0, StageCompleted(Stage.REVIEW, "", "Done", terminal=terminal))]
        records = tuple(
            Record(i, 1000000 - age + offset, event, opened.get(i), attempts.get(i), answers.get(i))
            for i, (offset, event) in enumerate(data, 1)
        )
        sink = CheckpointSink(tmp_path / "work", str(directory))
        sink.claim(None)
        await sink.local.write(records[:1])
        await asyncio.to_thread(sink._publish, Status(), None)
        await sink.local.write(records[1:])
        await sink.finish(Status(terminal=terminal), Workpiece(name, proposal, terminal=terminal, decisions=decisions))
    lines = await report(str(tmp_path))
    assert "terminal=accepted count=2" in lines
    assert "terminal=held reason=Restart count=1" in lines
    assert "terminal=abandoned reason=author stopped count=1" in lines
    assert "accepted_per_hour window_hours=1 count=1 rate=1.000" in lines
    assert "accepted_per_hour window_hours=6 count=1 rate=0.167" in lines
    assert "accepted_per_hour window_hours=24 count=2 rate=0.083" in lines
    assert "tokens role=author prompt=100 cached=20 completion=10" in lines
    assert "tokens role=adversary prompt=100 cached=20 completion=10" in lines
    assert "tokens role=solver prompt=40 cached=15 completion=7" in lines
    assert "stage=author n=1 p50_seconds=20.000 p90_seconds=20.000" in lines
    assert "event_gap=InferenceRequest n=2 p50_seconds=5.000 p90_seconds=10.000" in lines
    assert "event_gap=WriteFile n=1 p50_seconds=3.000 p90_seconds=3.000" in lines
    assert "terminal=accepted count=1" in await report(str(tmp_path), 1)


class Profile:
    """The author model under ``name``: a cached resource whose fingerprint is the profile."""

    def __init__(self, name: str) -> None:
        self.name = name

    def fingerprint(self) -> Fingerprint:
        return Fingerprint("inference", self.name, ReplayPolicy.CACHE)


# Review 2026-11-21: cold launches under author profiles A, then B, then A reported
# cumulative author tokens 40/20, 80/40 and 120/60: the third bought again what A had already paid. It reports 80/40.
async def test_a_launch_under_a_profile_already_paid_adds_no_reported_tokens(
    tmp_path: Path, proposal: TaskProposal
) -> None:
    request = InferenceRequest((), (), LLMPolicy())

    async def answer(_event: InferenceRequest, _profile: Profile) -> Completion:
        return Completion("ok", usage=Usage(prompt_tokens=40, completion_tokens=20))

    history: tuple[Record, ...] = ()
    reported = []
    for launch, profile in enumerate("ABA"):
        ctx = Context(log=EventLog(replay_from=history))
        ctx.handle(InferenceRequest, answer, Profile(profile))
        async with ctx.scope("author"):
            await ctx.emit(request)
        await ctx.drain()
        history = ctx._log.records
        root = tmp_path / f"launch-{launch}"
        (root / "workpiece").mkdir(parents=True)
        sink = CheckpointSink(tmp_path / f"work-{launch}", str(root / "workpiece"))
        sink.claim(None)
        await sink.local.write(history)
        await sink.finish(Status(terminal=Terminal.HELD), Workpiece("w", proposal, terminal=Terminal.HELD))
        reported += [line for line in await report(str(root)) if line.startswith("tokens role=author")]
    assert reported == [f"tokens role=author prompt={40 * n} cached=0 completion={20 * n}" for n in (1, 2, 2)]


async def test_report_empty_root_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No workpiece"):
        await report(str(tmp_path))


async def test_report_names_format_1_roots_and_counts_a_claimed_empty_root_as_running(tmp_path: Path) -> None:
    for name, artefact in (("old", "status.json"), ("parts", "events.jsonl.parts/000000000001.jsonl")):
        (tmp_path / name / artefact).parent.mkdir(parents=True)
        (tmp_path / name / artefact).write_text("{}")
    with pytest.raises(ValueError, match=f"2 journal format 1 roots .*{tmp_path / 'old'}, {tmp_path / 'parts'}"):
        await report(str(tmp_path))
    CheckpointSink(tmp_path / "work", str(tmp_path / "claimed")).claim(None)
    lines = await report(str(tmp_path), 1)
    assert f"skipped journal_format=1 count=2 roots={tmp_path / 'old'}, {tmp_path / 'parts'}" in lines
    assert "workpieces=1 since_hours=1" in lines and "terminal=running count=1" in lines


async def test_a_freshly_reclaimed_root_counts_as_running_in_a_short_window(
    tmp_path: Path, proposal: TaskProposal
) -> None:
    """A held root resumed now is running now, though its newest event is a day old."""
    destination = str(tmp_path / "reclaimed")
    first = CheckpointSink(tmp_path / "work", destination)
    first.claim(None)
    await first.local.write((Record(1, time.time() - 86400, Note("yesterday")),))
    await first.finish(Status(terminal=Terminal.HELD), Workpiece("reclaimed", proposal, terminal=Terminal.HELD))
    CheckpointSink(tmp_path / "work", destination).claim(read_journal(destination))
    lines = await report(str(tmp_path), 1)
    assert "workpieces=1 since_hours=1" in lines and "terminal=running count=1" in lines
    assert "tokens role=author prompt=0 cached=0 completion=0" in lines


# Review 2026-11-21: one root in an earlier journal format failed the whole report.
async def test_a_root_in_another_journal_format_is_named_and_the_report_completes(
    tmp_path: Path, proposal: TaskProposal
) -> None:
    current = CheckpointSink(tmp_path / "work", str(tmp_path / "current"))
    current.claim(None)
    await current.finish(Status(terminal=Terminal.HELD), Workpiece("current", proposal, terminal=Terminal.HELD))
    plant_root(tmp_path / "n6", N6_MODIFY_HELD)
    lines = await report(str(tmp_path))
    assert "workpieces=1 since_hours=all" in lines and "terminal=held count=1" in lines
    unsupported = (
        f"skipped unsupported_format file://{tmp_path / 'n6' / HEAD} has journal format 3, not {JOURNAL_FORMAT}"
    )
    assert unsupported in lines


@pytest.mark.parametrize("review_first", [False, True])
async def test_concurrent_usage_is_attributed_to_the_role_whose_request_incurred_it(
    tmp_path: Path, proposal: TaskProposal, review_first: bool
) -> None:
    """Review 2026-11-21: an adversary request resolved while the solve-review's scope was open was counted
    as solve-review usage; a request's own tools name its role, whatever else is open."""
    index = evidence(Submission(arithmetic_task(), (), ""), (), 4000)
    tools = tuple(tool.schema for tool in evidence_tools(index, ReportTool).values())
    review = Resolved(
        InferenceRequest((), tools, LLMPolicy()), (), Completion("", usage=Usage(prompt_tokens=11, completion_tokens=3))
    )
    adversary = Resolved(
        InferenceRequest((), (), LLMPolicy()), (), Completion("", usage=Usage(prompt_tokens=37, completion_tokens=5))
    )
    resolved = (review, adversary) if review_first else (adversary, review)
    events = (
        ScopeEntered("workpiece"),
        ScopeEntered("adversary:a"),
        ScopeEntered("solve_review"),
        *resolved,
        ScopeExited("solve_review", False),
        ScopeExited("adversary:a", False),
        StageCompleted(Stage.REVIEW, "", "Accepted", terminal=Terminal.ACCEPTED),
    )
    records = tuple(Record(i, time.time(), event) for i, event in enumerate(events, 1))
    sink = CheckpointSink(tmp_path / "work", str(tmp_path / "root"))
    sink.claim(None)
    await sink.local.write(records)
    await sink.finish(Status(terminal=Terminal.ACCEPTED), Workpiece("w", proposal, terminal=Terminal.ACCEPTED))
    lines = await report(str(tmp_path / "root"))
    assert "tokens role=adversary prompt=37 cached=0 completion=5" in lines
    assert "tokens role=solve_review prompt=11 cached=0 completion=3" in lines


async def test_a_judge_call_inside_the_confirmation_is_charged_to_the_judge(
    tmp_path: Path, proposal: TaskProposal
) -> None:
    """Review 2026-11-21: a VerifyIt judge callback under the confirmation scope was reported as
    final-review usage, though no final reviewer ran; the reviewer's own request there is the final review's."""
    index = evidence(Submission(arithmetic_task(), (), ""), (), 4000)
    tools = tuple(tool.schema for tool in evidence_tools(index, ReportTool).values())
    judge = Resolved(
        InferenceRequest((), (), LLMPolicy()), (), Completion("", usage=Usage(prompt_tokens=37, completion_tokens=5))
    )
    review = Resolved(
        InferenceRequest((), tools, LLMPolicy()), (), Completion("", usage=Usage(prompt_tokens=11, completion_tokens=3))
    )
    events = (
        ScopeEntered("workpiece"),
        ScopeEntered("confirm"),
        ScopeEntered("judge"),
        judge,
        ScopeExited("judge", False),
        ScopeEntered("solve_review.final"),
        review,
        ScopeExited("solve_review.final", False),
        ScopeExited("confirm", False),
        StageCompleted(Stage.REVIEW, "", "Accepted", terminal=Terminal.ACCEPTED),
    )
    records = tuple(Record(i, time.time(), event) for i, event in enumerate(events, 1))
    sink = CheckpointSink(tmp_path / "work", str(tmp_path / "root"))
    sink.claim(None)
    await sink.local.write(records)
    await sink.finish(Status(terminal=Terminal.ACCEPTED), Workpiece("w", proposal, terminal=Terminal.ACCEPTED))
    lines = await report(str(tmp_path / "root"))
    assert "tokens role=judge prompt=37 cached=0 completion=5" in lines
    assert "tokens role=final_review prompt=11 cached=0 completion=3" in lines


# Review 2026-11-21: while a judge call was in flight, an adversary call of 37 prompt tokens completed, then the judge's
# of 11; the cold report charged 48 to the judge and none to the adversary.
async def test_overlapping_scopes_charge_each_call_to_the_role_its_scope_bills(
    tmp_path: Path, proposal: TaskProposal
) -> None:
    (tmp_path / "root").mkdir()
    sink = CheckpointSink(tmp_path / "work", str(tmp_path / "root"))
    sink.claim(None)
    ctx = Context(log=EventLog(sink=sink))
    judging, answered = asyncio.Event(), asyncio.Event()

    async def answer(_event: InferenceRequest, profile: Profile) -> Completion:
        if profile.name == "judge":
            judging.set()
            await answered.wait()
            return Completion("", usage=Usage(prompt_tokens=11))
        await judging.wait()
        answered.set()
        return Completion("", usage=Usage(prompt_tokens=37))

    async def visit(name: str, bills: Billing) -> None:
        async with ctx.scope(name, bills=bills):
            ctx.handle(InferenceRequest, answer, Profile(bills))
            await ctx.emit(InferenceRequest((), (), LLMPolicy()))

    async with sink.running():
        async with ctx.scope("workpiece"):
            await asyncio.gather(visit("adversary:a", Billing.ADVERSARY), visit("judge", Billing.JUDGE))
        await ctx.drain()
    await sink.finish(Status(terminal=Terminal.HELD), Workpiece("w", proposal, terminal=Terminal.HELD))
    lines = await report(str(tmp_path / "root"))
    assert "tokens role=adversary prompt=37 cached=0 completion=0" in lines
    assert "tokens role=judge prompt=11 cached=0 completion=0" in lines
