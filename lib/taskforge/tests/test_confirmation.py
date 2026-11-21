# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The confirmation step: five more solves of the shipped submission, the final review over all eight, and the
decisions live Forge makes on them."""

import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import override

import pytest
from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.fake_resources import resources as stage_resources
from lib.taskforge.tests.proposal_fakes import sample_proposal
from lib.taskforge.tests.resource_doubles import CleanReviewer, FakeInference
from lib.taskforge.tests.stage_fakes import transcript
from lib.taskforge.tests.test_end_to_end import SETTINGS, Envs, call, resources, script
from lib.taskforge.tests.test_solve_review import SUBMISSION, TRIALS, lowered, recorded_task
from lib.taskforge.tests.test_solver_batch import rejected
from pydantic import TypeAdapter
from rolloutengine.models import (
    ConversationRecord,
    GradedRollout,
    Model,
    ModelTurn,
    Rollout,
    RolloutInterrupted,
    RolloutOperation,
    RolloutStep,
    TaskRolloutSpec,
    Transition,
)
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.checkpoint import CheckpointSink
from taskforge.config import TaskforgeConfig
from taskforge.coordinator import advance, apply_decision, run_workpiece
from taskforge.effects.context import Context
from taskforge.effects.events import Resolved, ScopeEntered, ScopeExited, StageCompleted
from taskforge.effects.log import EventLog, JsonlSink, Record, Status
from taskforge.harness.protocol import Stopped, StopReason
from taskforge.harness.tools.evidence import FinalReportTool, ReportTool, evidence_tools
from taskforge.llm.policy import LLMPolicy
from taskforge.model.content_hash import digest
from taskforge.model.llm import Completion, Message, Role, Usage
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import (
    VALIDATION_RESULT,
    Abandon,
    Accept,
    BandOutcome,
    ControlResult,
    EvidenceKind,
    FindingKind,
    Incomplete,
    MissingEvidence,
    ReviewEnd,
    ReviewFinding,
    Revise,
    SolveReview,
    TrialResult,
    ValidationResult,
)
from taskforge.model.stage import Stage
from taskforge.model.workpiece import WORKPIECE, Conversation, Fallback, Terminal, Workpiece
from taskforge.policy import load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.report import report
from taskforge.resources.bundle import Resources
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.inference import InferenceRequest
from taskforge.resources.rollouts import Engine, SolverRollout
from taskforge.stages.rules import CONFIRMATION_REVISION_FAILED, default_review, tagged
from taskforge.stages.solve_review import evidence, solve_review
from taskforge.wiring import build_config, install

CONFIRMED = {**SETTINGS, "solver.k": 3, "solver.confirmation_k": 5}
"""Three validation solves and five confirmation solves, no adversary, one shared modification round."""


def setup(**settings: object) -> TaskforgeConfig:
    return build_config(load_policy(DEFAULTS_DIR).overlaid({**CONFIRMED, **settings}), {})


type Slot = tuple[int, int]
"""A solver slot by (trial, attempt)."""


def even(trial: int, attempt: int) -> GradeResult:
    return GradeResult(Outcome.GRADED, float(trial % 2 == 0))


class Trials(Engine):
    """Grades each slot with ``grade``; ``runs`` is every paid execution. A ``restart`` slot is interrupted, a
    ``refuse`` slot refused by the model server, and a ``silent`` slot records no messages."""

    def __init__(
        self,
        grade: Callable[[int, int], GradeResult] = even,
        *,
        restart: frozenset[Slot] = frozenset(),
        refuse: frozenset[Slot] = frozenset(),
        silent: frozenset[Slot] = frozenset(),
    ) -> None:
        self.grade, self.restart, self.refuse, self.silent = grade, restart, refuse, silent
        self.runs: list[Slot] = []

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        assert run_id is not None
        slot = trial, attempt = tuple(int(part) for part in run_id.split(":")[-2:])
        self.runs.append((trial, attempt))
        if slot in self.restart:
            raise RolloutInterrupted(Rollout(run_id, (), "interrupted"), RolloutOperation.MODEL)
        if slot in self.refuse:
            raise rejected()
        said = () if slot in self.silent else (transcript(f"trial {trial} attempt {attempt} answers 12"),)
        return GradedRollout(Rollout(run_id, said, "stop"), self.grade(trial, attempt))


def found(trial: int, kind: FindingKind = FindingKind.GRADER_DEFECT, what: str = "accepts 13") -> dict[str, object]:
    quote = {"source": f"trial/{trial}", "start": 1, "end": 1, "quote": "answers 12"}
    return {"kind": kind, "description": what, "edit": "check equality", "citations": [quote]}


def finding(trial: int, kind: FindingKind = FindingKind.GRADER_DEFECT, what: str = "accepts 13") -> ReviewFinding:
    return TypeAdapter(ReviewFinding).validate_python(found(trial, kind, what))


FINAL = "Earlier solve-review reports for this task"
"""Only the final review's opening says this."""


class Reviewer(CleanReviewer):
    """Each round review reports the next of ``rounds`` as findings and each final review the next of ``finals`` as
    (revise, note), clean once a list runs out. With ``read``, a final review first reads ``trial/0``."""

    def __init__(
        self,
        rounds: Sequence[Sequence[dict[str, object]]] = (),
        finals: Sequence[tuple[Sequence[dict[str, object]], Sequence[dict[str, object]]]] = (),
        read: bool = False,
    ) -> None:
        super().__init__()
        self.rounds, self.finals, self.read = list(rounds), list(finals), read
        self.openings: list[str] = []
        self.replies: list[str] = []

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        clean = await super().complete(request)
        accounts = json.loads(clean.tool_calls[0].arguments)["accounts"]
        opening = next(message.content for message in request.conversation if message.role == Role.USER)
        tools = [message.content for message in request.conversation if message.role == Role.TOOL]
        self.replies += tools
        if not tools:
            self.openings.append(opening)
        if FINAL not in opening:
            return call(
                "report", {"findings": list(self.rounds.pop(0) if self.rounds else ()), "accounts": accounts}, "r"
            )
        if self.read and not tools:
            return call("read_evidence", {"source": "trial/0"}, "read")
        revise, note = self.finals.pop(0) if self.finals else ((), ())
        return call("report", {"revise": list(revise), "note": list(note), "accounts": accounts}, "final")

    @property
    def finals_seen(self) -> list[str]:
        return [opening for opening in self.openings if FINAL in opening]


RESUBMIT = (*script()[:3], call("submit", {"notes": "revised", "declined": []}, "submit"))
"""A revision round's author turns: the same task, new notes, no finding declined."""


def bundle(engine: Trials, reviewer: Reviewer, author: Sequence[Completion] = script()) -> Resources:
    live = resources(Envs(), author)
    return replace(live, rollouts=replace(live.rollouts, engine=engine), role_inference={"solve_review": reviewer})


async def run(
    proposal: TaskProposal,
    engine: Trials,
    reviewer: Reviewer,
    config: TaskforgeConfig | None = None,
    sink: JsonlSink | None = None,
) -> tuple[Workpiece, EventLog]:
    log = EventLog(sink=sink)
    wp = await run_workpiece(proposal, config or setup(), bundle(engine, reviewer), log, install=install)
    assert WORKPIECE.validate_json(WORKPIECE.dump_json(wp)) == wp
    return wp, log


def kinds(wp: Workpiece) -> list[str]:
    return [decision.kind for decision in wp.decisions]


def between(records: Sequence[Record], name: str) -> list[Record]:
    """The records inside the first ``name`` scope."""
    start = next(i for i, r in enumerate(records) if isinstance(r.event, ScopeEntered) and r.event.name == name)
    end = next(i for i, r in enumerate(records) if isinstance(r.event, ScopeExited) and r.event.name == name)
    return list(records[start:end])


async def test_a_first_run_buys_three_validation_solves_then_five_more_and_one_final_review(
    proposal: TaskProposal,
) -> None:
    """Five confirmation slots take indices past the round's under a fresh attempt, so no validation slot's identity
    or recorded result changes; controls and adversaries are not run again."""
    engine, reviewer = Trials(), Reviewer()
    wp, log = await run(proposal, engine, reviewer)
    assert wp.terminal is Terminal.ACCEPTED and kinds(wp) == ["accept", "accept"]
    assert engine.runs[:3] == [(0, 0), (1, 0), (2, 0)] and sorted(engine.runs[3:]) == [(t, 1) for t in range(3, 8)]
    stages = [r.event.stage for r in log.records if isinstance(r.event, StageCompleted)]
    assert stages == ["author", "validate", "review", "confirm", "review"]
    validation, confirmation = wp.evidence
    assert [t.trial for t in validation.solver] == [0, 1, 2] and validation.confirms is None
    assert [t.trial for t in confirmation.solver] == list(range(8)) and confirmation.confirms == 0
    assert confirmation.controls == validation.controls and confirmation.adversary == ()
    inside = [r.event for r in between(log.records, "confirm")]
    assert not any(
        isinstance(e, GradeCandidate) or (isinstance(e, ScopeEntered) and "adversary" in e.name) for e in inside
    )
    assert len(reviewer.openings) == 2 and len(reviewer.finals_seen) == 1
    assert all(f"trial/{t}:" in reviewer.finals_seen[0] for t in range(8))
    final = wp.decisions[-1]
    assert isinstance(final, Accept) and "Confirmation: 4 of 8 trials solved: in_band" in final.reasons
    assert (final.missing, final.notes, wp.accepted_with) == ((), (), None)


async def test_a_partial_resume_replays_seven_solves_and_buys_the_eighth(proposal: TaskProposal, tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
    first, reviewer = Trials(restart=frozenset({(5, 1)})), Reviewer()
    held, _ = await run(proposal, first, reviewer, sink=sink)
    assert held.terminal is Terminal.HELD and len(reviewer.finals_seen) == 0
    assert Counter(first.runs)[(5, 1)] == 3 and len(set(first.runs)) == 8  # three replays of the confirmation stage
    engine, again = Trials(), Reviewer()
    live = bundle(engine, again, ())
    log = EventLog(replay_from=local_records(sink), sink=sink)
    wp = await run_workpiece(proposal, setup(), live, log, install=install)
    assert wp.terminal is Terminal.ACCEPTED and engine.runs == [(5, 1)]
    assert len(again.openings) == 1 and again.finals_seen == again.openings
    assert isinstance(live.inference, FakeInference) and not live.inference.requests


async def test_a_full_resume_repeats_nothing(proposal: TaskProposal, tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
    done, _ = await run(proposal, Trials(), Reviewer(), sink=sink)
    engine, reviewer = Trials(), Reviewer()
    live = bundle(engine, reviewer, ())
    wp = await run_workpiece(proposal, setup(), live, EventLog(replay_from=local_records(sink)), install=install)
    assert wp == done and not engine.runs and not reviewer.requests
    assert isinstance(live.inference, FakeInference) and not live.inference.requests


async def test_a_second_process_replays_the_whole_confirmation_with_zero_calls(tmp_path: Path) -> None:
    probe = [sys.executable, "-m", "lib.taskforge.tests.confirmation_replay_probe"]
    root = Path(__file__).parents[3]
    for mode, expected in (("record", "8 solve(s), 2 review call(s)"), ("replay", "0 solve(s), 0 review call(s)")):
        done = subprocess.run([*probe, mode, str(tmp_path)], cwd=root, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr[-3000:]
        assert done.stdout.strip() == f"{mode}: accepted, {expected}"


def faulted_after_attempt_0(trial: int, attempt: int) -> GradeResult:
    return GradeResult(Outcome.INVALID_TASK, None, "bad task") if attempt == 1 else even(trial, attempt)


async def test_a_restored_fallback_is_confirmed_on_its_own_round(proposal: TaskProposal) -> None:
    """Round 0 has a finding; its same-digest revision faults and round 0 is restored. The selector's choice, not a
    digest, names the round: its own transcripts reach the final review, and the new slots number past its indices."""
    engine, reviewer = Trials(faulted_after_attempt_0), Reviewer(rounds=[[found(1)]], read=True)
    live = bundle(engine, reviewer, (*script(), *RESUBMIT))
    wp = await run_workpiece(proposal, setup(), live, EventLog(), install=install)
    assert kinds(wp) == ["modify", "accept", "accept"] and wp.terminal is Terminal.ACCEPTED
    restore = wp.decisions[1]
    assert isinstance(restore, Accept) and restore.restore == wp.evidence[0].task_digest == wp.evidence[1].task_digest
    assert wp.submission is not None and wp.submission.notes == "done"  # round 0's, though the digest is the same
    confirmation = wp.evidence[-1]
    assert confirmation.confirms == 0 and confirmation.solver[:3] == wp.evidence[0].solver
    assert sorted(slot for slot in engine.runs if slot[1] == 2) == [(t, 2) for t in range(3, 8)]
    assert "trial 0 attempt 0 answers 12" in reviewer.replies[-1]
    (opening,) = reviewer.finals_seen
    assert "Solve-review of validation record 0 (this submission, 3 trials):" in opening
    assert "Outcome: given to the author, who revised the task." in opening
    final = wp.decisions[-1]
    assert isinstance(final, Accept) and final.notes == (finding(1),)  # the restored round's finding


@pytest.mark.parametrize("accept_too_easy", [True, False])
async def test_a_task_out_of_band_after_eight_trials_is_accepted_tagged(
    proposal: TaskProposal, accept_too_easy: bool
) -> None:
    """Two of three solved places in band; seven of eight is too easy, and live Forge tags it whatever
    ``band.accept_too_easy`` says, for a confirmed task is never filtered."""
    engine = Trials(lambda trial, attempt: GradeResult(Outcome.GRADED, float(trial != 1)))
    wp, _ = await run(proposal, engine, Reviewer(), setup(**{"band.accept_too_easy": accept_too_easy}))
    final = wp.decisions[-1]
    assert wp.terminal is Terminal.ACCEPTED and wp.accepted_with is BandOutcome.TOO_EASY
    assert isinstance(final, Accept) and final.band is BandOutcome.TOO_EASY
    assert final.reasons[:2] == ("accepted_with: too_easy", "Confirmation: 7 of 8 trials solved: too_easy")


async def test_a_revise_finding_buys_one_revision_outside_a_spent_budget(proposal: TaskProposal) -> None:
    """Owner's ruling (2026-11-13): "A revision asked for by the final review is outside the 2-round budget and always
    happens (one)." With no modification round at all, the revision is validated and confirmed again; the second
    final review's revise finding is then a note."""
    first, second = found(3, what="first"), found(4, what="second")
    engine, reviewer = Trials(), Reviewer(finals=[([first], []), ([second], [])])
    live = bundle(engine, reviewer, (*script(), *RESUBMIT))
    wp = await run_workpiece(proposal, setup(**{"budget.max_modify_rounds": 0}), live, EventLog(), install=install)
    assert kinds(wp) == ["accept", "revise", "accept", "accept"] and wp.round == 1
    assert wp.terminal is Terminal.ACCEPTED and wp.submission is not None and wp.submission.notes == "revised"
    revise = wp.decisions[1]
    assert isinstance(revise, Revise) and revise.review == (finding(3, what="first"),) and wp.fallback is None
    assert [e.confirms for e in wp.evidence] == [None, 0, None, 2]
    assert len(engine.runs) == 16 and sorted(s for s in engine.runs if s[1] == 3) == [(t, 3) for t in range(3, 8)]
    assert isinstance(live.inference, FakeInference)
    assert "Revise the task and controls after the final review:" in live.inference.requests[4].conversation[1].content
    last = reviewer.finals_seen[-1]
    assert "Final review of validation record 1 (this submission, 8 trials):" in last
    assert "first Edit: check equality Outcome: given to the author, who revised the task." in last
    final = wp.decisions[-1]
    assert isinstance(final, Accept) and final.notes == (finding(4, what="second"),)
    assert "The final review's revision is used: its revise findings are recorded as notes" in final.reasons


async def test_note_findings_alone_accept_with_notes(proposal: TaskProposal) -> None:
    note = found(5, FindingKind.AMBIGUOUS_PROMPT, "reads two ways")
    wp, _ = await run(proposal, Trials(), Reviewer(finals=[([], [note])]))
    final = wp.decisions[-1]
    assert kinds(wp) == ["accept", "accept"] and wp.terminal is Terminal.ACCEPTED
    assert isinstance(final, Accept) and final.notes == (finding(5, FindingKind.AMBIGUOUS_PROMPT, "reads two ways"),)


async def test_a_trial_without_a_transcript_is_named_on_the_acceptance(proposal: TaskProposal) -> None:
    wp, _ = await run(proposal, Trials(silent=frozenset({(4, 1)})), Reviewer())
    final = wp.decisions[-1]
    gone = MissingEvidence(EvidenceKind.TRANSCRIPTS, "solver trials [4] have no complete transcript")
    assert wp.terminal is Terminal.ACCEPTED and isinstance(final, Accept) and final.missing == (gone,)


@pytest.mark.parametrize("allowance", [0, 1])
async def test_a_refused_confirmation_slot_holds_beyond_the_batch_allowance(
    proposal: TaskProposal, allowance: int
) -> None:
    reviewer = Reviewer()
    engine = Trials(refuse=frozenset({(4, 1)}))
    wp, _ = await run(proposal, engine, reviewer, setup(**{"budget.max_missing_trials": allowance}))
    final = wp.decisions[-1]
    if not allowance:
        assert wp.terminal is Terminal.HELD and isinstance(final, Incomplete) and not reviewer.finals_seen
        assert any("solver trial 4" in reason for reason in final.reasons)
        return
    assert wp.terminal is Terminal.ACCEPTED and isinstance(final, Accept)
    assert MissingEvidence(EvidenceKind.CONFIRMATION_TRIALS, "solver trials [4] have no score") in final.missing
    assert "1 of 8 solver trials have no score; placed over 7" in final.reasons


SOLVED = (TrialResult(0, GradeResult(Outcome.GRADED, 1.0)), TrialResult(1, GradeResult(Outcome.GRADED, 0.0)))
CONTROLS = (ControlResult(0, GradeResult(Outcome.GRADED, 1.0), True),)
REVISE, NOTE = finding(0, what="revise me"), finding(1, FindingKind.AMBIGUOUS_PROMPT, "note me")


def confirmed(proposal: TaskProposal, revise: bool, note: bool, used: bool, in_band: bool) -> Workpiece:
    """A round placed in band and accepted, then its confirmation record (two of four solved, or five of six), whose
    final review reports ``revise`` and ``note`` findings; ``used`` puts the final revision in its history."""
    rewards = (1.0, 0.0) if in_band else (1.0,) * 4
    added = tuple(TrialResult(2 + i, GradeResult(Outcome.GRADED, reward)) for i, reward in enumerate(rewards))
    review = SolveReview("v", ReviewEnd.REPORTED, (REVISE,) * revise, notes=(NOTE,) * note)
    record = ValidationResult(SUBMISSION.digest, (*SOLVED, *added), CONTROLS, (), review=review, confirms=0)
    earlier = (Revise(("line",), BandOutcome.IN_BAND, (REVISE,)),) * used
    pending = Accept(BandOutcome.IN_BAND)
    first = ValidationResult(SUBMISSION.digest, SOLVED, CONTROLS, (), review=SolveReview("v0", ReviewEnd.REPORTED))
    return Workpiece("w", proposal, submission=SUBMISSION, evidence=(first, record), decisions=(*earlier, pending))


SOLVES = {True: "2 of 4 trials solved", False: "5 of 6 trials solved"}

LIVE_CONFIRMATION = tuple(
    (revise, note, used, in_band, "revise" if revise and not used else "accept", note + (revise and used))
    for revise in (True, False)
    for note in (True, False)
    for used in (True, False)
    for in_band in (True, False)
)
"""(revise finding, note finding, final revision used, in band over every trial) -> live Forge's decision at
marin-forge-passes ``rules.py`` ``confirmation_review`` and the number of notes it accepts with: placement over every
trial tags the acceptance in any band; a revise finding sends the task back once, and is a note after."""


@pytest.mark.parametrize("revise,note,used,in_band,kind,notes", LIVE_CONFIRMATION)
def test_confirmation_decisions_match_live_forge(
    proposal: TaskProposal, revise: bool, note: bool, used: bool, in_band: bool, kind: str, notes: int
) -> None:
    config = setup()
    decision = default_review(confirmed(proposal, revise, note, used, in_band), config.band, config.budget)
    place = BandOutcome.IN_BAND if in_band else BandOutcome.TOO_EASY
    assert decision.kind == kind
    assert isinstance(decision, Accept | Revise)
    assert (decision.band if isinstance(decision, Accept) else decision.placement) is place
    if isinstance(decision, Revise):
        assert decision.review == (REVISE,) and decision.findings[0].startswith(f"{SOLVES[in_band]}: {place}")
        return
    assert len(decision.notes) == notes and decision.restore is None
    line = f"Confirmation: {SOLVES[in_band]}: {place}"
    assert decision.reasons[: 1 + (not in_band)] == (*tagged(place), line)


def after_revise(proposal: TaskProposal, result: ValidationResult) -> Workpiece:
    """The final review's revision validated as ``result``: the confirmed round, its record, the revision."""
    confirmed_wp = confirmed(proposal, True, False, False, True)
    revise = Revise(("line",), BandOutcome.IN_BAND, (REVISE,))
    session = Conversation((Message(Role.SYSTEM, "author"),), ())
    return confirmed_wp.with_(
        round=1, session=session, evidence=(*confirmed_wp.evidence, result), decisions=(*confirmed_wp.decisions, revise)
    )


@pytest.mark.parametrize(
    "case,kind,reason",
    [
        ("fault", "abandon", f"{CONFIRMATION_REVISION_FAILED}: the revision has a fault"),
        ("too easy", "accept", "Placement is final after the final review's revision"),
        ("round finding", "accept", "Solve-review findings recorded as notes: no revision round left"),
        ("author stop", "abandon", "author stopped"),
    ],
)
def test_the_final_revision_s_round_decides_as_live_forge(
    proposal: TaskProposal, case: str, kind: str, reason: str
) -> None:
    """Live Forge (``placed``, ``with_solve_review``, ``stopped``): the final review's revision keeps no fallback, so a
    fault or an author stop abandons; its placement is final; a round finding is a note, rounds left or not."""
    config = setup(**{"budget.max_modify_rounds": 3})
    rewards = {"too easy": (1.0, 1.0)}.get(case, (1.0, 0.0))
    solver = tuple(TrialResult(i, GradeResult(Outcome.GRADED, reward)) for i, reward in enumerate(rewards))
    controls = (ControlResult(0, GradeResult(Outcome.GRADED, 0.0), False),) if case == "fault" else CONTROLS
    review = SolveReview("v2", ReviewEnd.REPORTED, (finding(0, what="round"),) * (case == "round finding"))
    wp = after_revise(proposal, ValidationResult(SUBMISSION.digest, solver, controls, (), review=review))
    if case == "author stop":
        assert wp.session is not None
        wp, _ = advance(wp, Stage.AUTHOR, Stopped(StopReason.CONTEXT_LIMIT, wp.session, "window full"), config)
        decision = wp.decisions[-1]
    else:
        decision = default_review(wp, config.band, config.budget)
    assert decision.kind == kind and isinstance(decision, Accept | Abandon)
    assert any(line.startswith(reason) for line in decision.reasons)


def test_a_restore_confirms_the_round_the_selector_chose(proposal: TaskProposal) -> None:
    """A restore's acceptance confirms its stored fallback's round, though a later round has the same digest."""
    config = setup()
    revision = replace(SUBMISSION, notes="revised")
    first = ValidationResult(SUBMISSION.digest, SOLVED, CONTROLS, (), review=SolveReview("v0", ReviewEnd.REPORTED))
    faulted = replace(first, controls=(ControlResult(0, GradeResult(Outcome.GRADED, 0.0), False),))
    wp = Workpiece(
        "w",
        proposal,
        round=1,
        submission=revision,
        evidence=(first, faulted),
        fallback=Fallback(SUBMISSION, BandOutcome.IN_BAND, 0),
        session=Conversation((Message(Role.SYSTEM, "author"),), ()),
    )
    decision = default_review(wp, config.band, config.budget)
    pending = apply_decision(wp, decision, config.budget, config.policy)
    assert isinstance(decision, Accept) and decision.restore == SUBMISSION.digest and pending.terminal is None
    assert pending.submission == SUBMISSION


def test_a_round_without_confirmation_fields_keeps_its_bytes() -> None:
    plain = ValidationResult(SUBMISSION.digest, SOLVED, CONTROLS, (), review=SolveReview("v", ReviewEnd.REPORTED))
    assert b"confirms" not in VALIDATION_RESULT.dump_json(plain) and b"notes" not in VALIDATION_RESULT.dump_json(plain)
    record = replace(plain, confirms=0, review=SolveReview("v", ReviewEnd.REPORTED, notes=(NOTE,)))
    assert VALIDATION_RESULT.validate_json(VALIDATION_RESULT.dump_json(record)) == record


async def test_confirmation_usage_is_reported_under_its_own_phase_once(tmp_path: Path, proposal: TaskProposal) -> None:
    turn = ModelTurn({}, (), (), None, "stop", metadata={"usage": {"prompt_tokens": 40, "completion_tokens": 7}})
    conversation = ConversationRecord((), (), (), (), None, (RolloutStep(turn, Transition(True), 0, ()),))
    rollout = GradedRollout(Rollout("r", (conversation,), "stop"), GradeResult(Outcome.GRADED, 1.0))
    spec = lowered()
    tools = tuple(t.schema for t in evidence_tools(evidence(SUBMISSION, (), 4000), ReportTool).values())
    final_tools = tuple(t.schema for t in evidence_tools(evidence(SUBMISSION, (), 4000), FinalReportTool).values())
    review = Resolved(InferenceRequest((), tools, LLMPolicy()), (), Completion("", usage=Usage(prompt_tokens=11)))
    final = Resolved(InferenceRequest((), final_tools, LLMPolicy()), (), Completion("", usage=Usage(prompt_tokens=13)))
    events = (
        ScopeEntered("workpiece"),
        ScopeEntered("validate"),
        Resolved(SolverRollout(spec, 0, 0), (), rollout),
        review,
        ScopeExited("validate", False),
        ScopeEntered("confirm"),
        Resolved(SolverRollout(spec, 0, 0), (), rollout, replayed=True),  # the round's trial, read again: no usage
        Resolved(SolverRollout(spec, 1, 1), (), rollout),
        Resolved(SolverRollout(spec, 2, 1), (), rollout),
        final,
        ScopeExited("confirm", False),
        StageCompleted(Stage.REVIEW, "", "Decided", terminal=Terminal.ACCEPTED),
    )
    records = tuple(Record(i, time.time(), event) for i, event in enumerate(events, 1))
    sink = CheckpointSink(tmp_path / "work", str(tmp_path / "root"))
    sink.claim(None)
    await sink.local.write(records)
    await sink.finish(Status(terminal=Terminal.ACCEPTED), Workpiece("w", proposal, terminal=Terminal.ACCEPTED))
    lines = await report(str(tmp_path / "root"))
    assert "tokens role=solver prompt=40 cached=0 completion=7" in lines
    assert "tokens role=solve_review prompt=11 cached=0 completion=0" in lines
    assert "tokens role=confirmation prompt=80 cached=0 completion=14" in lines
    assert "tokens role=final_review prompt=13 cached=0 completion=0" in lines
    assert any(line.startswith("stage=confirm n=1 ") for line in lines)


async def test_the_first_final_reviewer_request_keeps_its_recorded_identity() -> None:
    """The final review's first request for a fixed task, as ``test_the_first_reviewer_request_keeps_its_recorded_
    identity`` pins the round review's: its prompts, its report tool's schema and the earlier-report rendering."""
    recorded = "e9a5ea18d346e52f5ac0eba770c5bd9cf714f97fa941f7d146862db6396242aa"
    account = {"trial": 1, "attribution": "solver", "reason": "misadded"}
    reviewer = FakeInference((call("report", {"revise": [], "note": [], "accounts": [account]}, "r"),))
    ctx = Context(log=EventLog())
    policy = load_policy(DEFAULTS_DIR).overlaid({"tools.output_chars": 400})
    live = replace(stage_resources(ctx), role_inference={"solve_review": reviewer})
    earlier = "Solve-review of validation record 0 (this submission, 2 trials): no findings."
    submission = recorded_task()
    config = build_config(policy, {})
    await solve_review(
        ctx, config, live, sample_proposal(), submission, TRIALS, "solve_review.final", FinalReportTool, earlier
    )
    assert digest(reviewer.requests[0].identity()) == recorded, (
        "the final review's first request changed: a deliberate prompt change only. Update this recorded identity "
        "knowingly; every recorded final review will miss and buy its calls again"
    )
