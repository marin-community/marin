# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The confirmation step: five more solves of the shipped submission, the final review over all eight, and the
decisions made on them."""

import asyncio
import json
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import override

import pytest
from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.fake_resources import resources as stage_resources
from lib.taskforge.tests.proposal_fakes import sample_proposal
from lib.taskforge.tests.resource_doubles import CleanReviewer, FakeCommand, FakeInference
from lib.taskforge.tests.stage_fakes import transcript
from lib.taskforge.tests.test_end_to_end import SETTINGS, Envs, call, plant_root, resources, script
from lib.taskforge.tests.test_solve_review import SUBMISSION, TRIALS, lowered, recorded_task
from lib.taskforge.tests.test_solver_batch import rejected
from pydantic import TypeAdapter
from rolloutengine.models import (
    MAX_TURNS_STOP_REASON,
    ConversationRecord,
    GradedRollout,
    GradedRolloutInterrupted,
    Model,
    ModelTurn,
    Rollout,
    RolloutFailure,
    RolloutInterrupted,
    RolloutOperation,
    RolloutStep,
    TaskRolloutSpec,
    Transition,
)
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from verifyit.modes.grade_judge import JudgeMessage, JudgeRequest
from verifyit.spec import JudgeSpec

import taskforge.harness.tools.evidence as evidence_tools_module
import taskforge.stages.solve_review as review_stage
from taskforge.checkpoint import CheckpointSink, read_journal
from taskforge.config import TaskforgeConfig
from taskforge.coordinator import STAGE_REPLAY_ATTEMPTS, advance, apply_decision, run_workpiece
from taskforge.effects.context import Context
from taskforge.effects.events import Fingerprint, ReplayPolicy, Resolved, ScopeEntered, ScopeExited, StageCompleted
from taskforge.effects.issues import ContextExceeded, DecisionModelUnavailable
from taskforge.effects.log import Billing, BufferSink, EventLog, Record, Status, decode, fold
from taskforge.harness.protocol import Stopped, StopReason, Submitted, ToolResult
from taskforge.harness.tools.evidence import (
    TRIAL,
    Evidence,
    FinalReportArgs,
    FinalReportTool,
    ReportArgs,
    ReportTool,
    Source,
    evidence_tools,
)
from taskforge.llm.policy import LLMPolicy
from taskforge.model.content_hash import digest
from taskforge.model.llm import Completion, Message, Role, Usage
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import (
    VALIDATION_RESULT,
    Abandon,
    Accept,
    Attribution,
    BandOutcome,
    BandRules,
    Budget,
    Cause,
    ControlResult,
    Decision,
    Decline,
    EvidenceKind,
    FindingKind,
    FindingResponse,
    Incomplete,
    MissingEvidence,
    Modify,
    Response,
    ReviewEnd,
    ReviewFinding,
    Revise,
    SolveReview,
    TrialAccount,
    TrialOutcome,
    TrialResult,
    ValidationResult,
    grouped,
    supports,
)
from taskforge.model.stage import Stage
from taskforge.model.workpiece import WORKPIECE, Conversation, Fallback, Terminal, Workpiece
from taskforge.policy import load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.report import report
from taskforge.resources.bundle import Resources
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.inference import InferenceClient, InferenceRequest
from taskforge.resources.rollouts import Engine, SolverRollout
from taskforge.stages.classify import grade_cause, replaced_refusals, replayable_grade
from taskforge.stages.rules import CONFIRMATION_REVISION_FAILED, band_line, default_review, tagged
from taskforge.stages.solve_review import earlier_reports, evidence, solve_review
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
    ``refuse`` slot refused by the model server, a ``capped`` slot ends at its attempt cap, and a ``silent`` slot
    records no messages."""

    def __init__(
        self,
        grade: Callable[[int, int], GradeResult] = even,
        *,
        restart: frozenset[Slot] = frozenset(),
        refuse: frozenset[Slot] = frozenset(),
        silent: frozenset[Slot] = frozenset(),
        stop: Callable[[int, int], str] = lambda trial, attempt: "stop",
        capped: frozenset[Slot] = frozenset(),
    ) -> None:
        self.grade, self.restart, self.refuse, self.silent, self.stop = grade, restart, refuse, silent, stop
        self.capped = capped
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
        if slot in self.capped:  # past solver.attempt_timeout, at ``stop``: RolloutEngine's record names the attempt
            failure = RolloutFailure("TimeoutError", {"operation": RolloutOperation.ATTEMPT.value})
            capped = Rollout(run_id, (), self.stop(trial, attempt), failure=failure)
            raise GradedRolloutInterrupted(capped, RolloutOperation.ATTEMPT, GradeResult(Outcome.UNAVAILABLE, None))
        said = () if slot in self.silent else (transcript(f"trial {trial} attempt {attempt} answers 12"),)
        return GradedRollout(Rollout(run_id, said, self.stop(trial, attempt)), self.grade(trial, attempt))


def found(
    trial: int, kind: FindingKind = FindingKind.GRADER_DEFECT, what: str = "accepts 13", quote: str = "answers 12"
) -> dict[str, object]:
    cited = {"source": f"trial/{trial}", "start": 1, "end": 1, "quote": quote}
    return {"kind": kind, "description": what, "edit": "check equality", "citations": [cited]}


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
    sink: BufferSink | None = None,
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


async def test_a_partial_resume_replays_seven_solves_and_buys_the_eighth(proposal: TaskProposal) -> None:
    sink = BufferSink()
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


async def test_a_full_resume_repeats_nothing(proposal: TaskProposal) -> None:
    sink = BufferSink()
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
    assert (  # the response to the revision the restore discarded says so
        "Outcome: given to the author, who revised the task; the driver then restored the version before that "
        "attempted fix, so this submission does not carry it." in opening
    )
    final = wp.decisions[-1]
    assert isinstance(final, Accept) and final.notes == (finding(1),)  # the restored round's finding


@pytest.mark.parametrize("accept_too_easy", [True, False])
async def test_a_task_too_easy_after_eight_trials_is_tagged_or_revised_then_abandoned(
    proposal: TaskProposal, accept_too_easy: bool
) -> None:
    """Two of three solved places in band; seven of eight is too easy. Accepting too easy, the confirmation tags it;
    refusing it, the final revision is asked for, and a revision that is still too easy over its eight is abandoned."""
    engine = Trials(lambda trial, attempt: GradeResult(Outcome.GRADED, float(trial != 1)))
    config = setup(**{"band.accept_too_easy": accept_too_easy})
    wp = await run_workpiece(
        proposal, config, bundle(engine, Reviewer(), (*script(), *RESUBMIT)), EventLog(), install=install
    )
    final = wp.decisions[-1]
    if not accept_too_easy:
        assert kinds(wp) == ["accept", "revise", "accept", "abandon"] and wp.terminal is Terminal.ABANDONED
        revise = wp.decisions[1]
        assert isinstance(revise, Revise) and revise.placement is BandOutcome.TOO_EASY
        assert isinstance(final, Abandon) and "band.accept_too_easy is false" in final.reasons[0]
        return
    assert wp.terminal is Terminal.ACCEPTED and wp.accepted_with is BandOutcome.TOO_EASY
    assert isinstance(final, Accept) and final.band is BandOutcome.TOO_EASY
    assert final.reasons[:2] == ("accepted_with: too_easy", "Confirmation: 7 of 8 trials solved: too_easy")


async def test_a_revise_finding_buys_one_revision_outside_a_spent_budget(proposal: TaskProposal) -> None:
    """A revision asked for by the final review is outside the modification budget and happens up to
    ``budget.final_revisions`` times, once under this fixture's default (other tests cover 0 and 2). With no
    modification round at all, the revision is validated and confirmed again; the second final review's revise finding
    is then a note."""
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
    opening = live.inference.requests[4].conversation[1].content
    assert opening.startswith("Revise the task and controls after the final review:")
    assert opening.endswith("\n\nThis is final revision 1 of 1.")
    last = reviewer.finals_seen[-1]
    assert "Final review of validation record 1 (this submission, 8 trials):" in last
    assert "first Edit: check equality Outcome: given to the author, who revised the task." in last
    final = wp.decisions[-1]
    assert isinstance(final, Accept) and final.notes == (finding(4, what="second"),)
    assert "No final revision is left: the final review's revise findings are recorded as notes" in final.reasons


@pytest.mark.parametrize("revisions", [0, 2])
async def test_budget_final_revisions_sets_how_many_revisions_the_final_review_buys(
    proposal: TaskProposal, revisions: int
) -> None:
    """Each final review asks for a new revision; past ``budget.final_revisions`` its revise finding is a note."""
    asked = [found(3 + i, what=f"revision {i}") for i in range(revisions + 1)]
    engine, reviewer = Trials(), Reviewer(finals=[([finding], []) for finding in asked])
    live = bundle(engine, reviewer, (*script(), *(RESUBMIT * revisions)))
    config = setup(**{"budget.final_revisions": revisions})
    wp = await run_workpiece(proposal, config, live, EventLog(), install=install)
    assert kinds(wp) == ["accept", *(["revise", "accept"] * revisions), "accept"] and wp.round == revisions
    final = wp.decisions[-1]
    assert wp.terminal is Terminal.ACCEPTED and isinstance(final, Accept)
    assert final.notes == (finding(3 + revisions, what=f"revision {revisions}"),)


@pytest.mark.parametrize(
    "prompt", [{}, {"author.revision": "Address these findings:\n\n{detail}"}], ids=["stock", "custom"]
)
async def test_each_final_revision_is_a_fresh_author_attempt_that_a_restart_replays(
    proposal: TaskProposal, prompt: dict[str, str]
) -> None:
    asked = found(3, what="still accepts 13")
    engine, reviewer = Trials(), Reviewer(finals=[([asked], []), ([asked], [])])
    live = bundle(engine, reviewer, (*script(), *RESUBMIT, *RESUBMIT))
    log, config = EventLog(), setup(**{"budget.final_revisions": 2, **prompt})
    wp = await run_workpiece(proposal, config, live, log, install=install)
    assert kinds(wp) == ["accept", "revise", "accept", "revise", "accept", "accept"]
    assert isinstance(live.inference, FakeInference)
    assert len(live.inference.requests) == len(script()) + 2 * len(RESUBMIT)
    openings = [r.conversation[1].content for r in live.inference.requests if len(r.conversation) == 2]
    assert [o.rsplit("\n\n", 1)[1] for o in openings[1:]] == [f"This is final revision {n} of 2." for n in (1, 2)]
    cold = bundle(Trials(), Reviewer(), ())
    replayed = await run_workpiece(proposal, config, cold, EventLog(replay_from=log.records), install=install)
    assert isinstance(cold.inference, FakeInference) and not cold.inference.requests and replayed == wp


def test_a_confirmation_fault_without_a_final_revision_abandons(proposal: TaskProposal) -> None:
    config = setup(**{"budget.final_revisions": 0})
    decision = default_review(confirming(proposal, [graded(3, INVALID)]), config.band, config.budget)
    assert isinstance(decision, Abandon) and any("invalid_task" in reason for reason in decision.reasons)


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
    shipped = (*(NOTE,) * note, *(REVISE,) * revise)  # what the step computes when the trials support both
    record = ValidationResult(
        SUBMISSION.digest, (*SOLVED, *added), CONTROLS, (), review=review, confirms=0, notes=shipped
    )
    earlier = (Revise(("line",), (REVISE,), placement=BandOutcome.IN_BAND),) * used
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
"""(revise finding, note finding, final revision used, in band over every trial) -> the confirmation review's decision
and the number of notes it accepts with: placement over every
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
    revise = Revise(("line",), (REVISE,), placement=BandOutcome.IN_BAND)
    session = Conversation((Message(Role.SYSTEM, "author"),), ())
    return replace(
        confirmed_wp,
        round=1,
        session=session,
        evidence=(*confirmed_wp.evidence, result),
        decisions=(*confirmed_wp.decisions, revise),
    )


@pytest.mark.parametrize(
    "case,kind,reason",
    [
        ("fault", "abandon", f"{CONFIRMATION_REVISION_FAILED}: the revision has a fault"),
        ("too easy", "accept", "Placement is final after the final review's revision"),
        ("round finding", "accept", "Solve-review findings recorded as notes: no revision round left"),
        ("author stop", "abandon", f"{CONFIRMATION_REVISION_FAILED}: author stopped: context_limit"),
    ],
)
def test_the_final_revision_s_round_decides_as_live_forge(
    proposal: TaskProposal, case: str, kind: str, reason: str
) -> None:
    """The final review's revision keeps no fallback, so a fault or an author stop abandons; its placement is final;
    a round finding is a note, rounds left or not."""
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
    if kind == "abandon":  # owner's decisions (a, b): the abandonment keeps the finding that asked for the revision
        assert f"Unresolved final-review finding: {REVISE.description}" in decision.reasons


EASY = tuple(TrialResult(i, GradeResult(Outcome.GRADED, 1.0)) for i in range(2))
REFUSED = ("round", "last round", "revision with a fallback", "author stop", "final revision's round", "confirmation")


def refusing(proposal: TaskProposal, path: str, config: TaskforgeConfig) -> Workpiece:
    """The workpiece, after ``path``'s decision, of each path that could accept a too-easy task."""
    session = Conversation((Message(Role.SYSTEM, "author"),), ())
    easy = ValidationResult(SUBMISSION.digest, EASY, CONTROLS, (), review=SolveReview("v", ReviewEnd.REPORTED))
    fresh = Workpiece("w", proposal, submission=SUBMISSION, evidence=(easy,), session=session)
    kept = Fallback(SUBMISSION, BandOutcome.IN_BAND, 0)
    if path == "round":
        return apply_decision(fresh, default_review(fresh, config.band, config.budget), config)
    if path == "last round":
        last = replace(fresh, round=config.budget.max_modify_rounds)
        return apply_decision(last, default_review(last, config.band, config.budget), config)
    revised = replace(fresh, round=1, evidence=(confirmed(proposal, False, False, False, True).evidence[0], easy))
    if path == "revision with a fallback":
        revised = replace(revised, fallback=kept)
        return apply_decision(revised, default_review(revised, config.band, config.budget), config)
    if path == "author stop":
        stopped = Stopped(StopReason.CONTEXT_LIMIT, session, "window full")
        return advance(replace(revised, fallback=kept), Stage.AUTHOR, stopped, config)[0]
    if path == "final revision's round":
        wp = after_revise(proposal, easy)
        return apply_decision(wp, default_review(wp, config.band, config.budget), config)
    wp = replace(confirmed(proposal, False, False, False, False), session=session)
    return apply_decision(wp, default_review(wp, config.band, config.budget), config)


@pytest.mark.parametrize("path", REFUSED)
@pytest.mark.parametrize("accept_too_easy", [True, False])
def test_with_too_easy_refused_no_path_accepts_a_too_easy_task(
    proposal: TaskProposal, path: str, accept_too_easy: bool
) -> None:
    """``band.accept_too_easy`` false holds on every path: a round asks for hardening, a restore keeps an accepted
    fallback, a final revision's round is placed again by its confirmation, and a confirmation asks for the final
    revision. True accepts the confirmation tagged."""
    config = setup(**{"band.accept_too_easy": accept_too_easy})
    wp = refusing(proposal, path, config)
    shipped = wp.decisions[-1]
    if not accept_too_easy and isinstance(shipped, Accept) and shipped.band is BandOutcome.TOO_EASY:
        assert path == "final revision's round" and wp.terminal is None
    expected = {
        "round": "modify",
        "last round": "accept" if accept_too_easy else "abandon",
        "revision with a fallback": "accept",
        "author stop": "accept",
        "final revision's round": "accept",
        "confirmation": "accept" if accept_too_easy else "revise",
    }
    assert shipped.kind == expected[path]
    if path == "confirmation":
        assert (wp.accepted_with is BandOutcome.TOO_EASY) is accept_too_easy


@pytest.mark.parametrize("repeated", [False, True], ids=["repaired, then refused", "reported again"])
async def test_a_refused_confirmation_names_its_placement_setting_and_budget_and_only_open_findings(
    proposal: TaskProposal, repeated: bool
) -> None:
    asked = found(3, what="still accepts 13")
    engine = Trials(lambda trial, attempt: GradeResult(Outcome.GRADED, 1.0 if attempt >= 2 else float(trial % 2 == 0)))
    reviewer = Reviewer(finals=[([asked], []), ([asked] if repeated else [], [])])
    live = bundle(engine, reviewer, (*script(), *RESUBMIT))
    config = setup(**{"band.accept_too_easy": False})
    wp = await run_workpiece(proposal, config, live, EventLog(), install=install)
    assert kinds(wp) == ["accept", "revise", "accept", "abandon"] and wp.terminal is Terminal.ABANDONED
    why = (
        f"{CONFIRMATION_REVISION_FAILED}: confirmation placed too_easy (8 of 8 trials solved); band.accept_too_easy is "
        "false and budget.final_revisions (1) is spent"
    )
    unresolved = ("Unresolved final-review finding: still accepts 13",) * repeated
    abandoned = wp.decisions[-1]
    assert isinstance(abandoned, Abandon) and abandoned.reasons == (why, *unresolved)


@pytest.mark.parametrize("last", [Response.REVISED, Response.DECLINED], ids=["revised last", "declined last"])
def test_a_refused_confirmation_reads_the_latest_answer_to_a_final_revision_finding(
    proposal: TaskProposal, last: Response
) -> None:
    config = setup(**{"band.accept_too_easy": False, "budget.final_revisions": 2})
    wp = confirmed(proposal, False, False, True, False)  # clean final review, 6 of 6 solved: too easy
    answers = (Response.DECLINED, Response.REVISED, last)
    responses = tuple(FindingResponse(review, REVISE, answer) for review, answer in enumerate(answers))
    wp = replace(wp, decisions=(wp.decisions[0], *wp.decisions), responses=responses)  # both final revisions spent
    decision = default_review(wp, config.band, config.budget)
    unresolved = ("Unresolved final-review finding: revise me",) * (last is Response.DECLINED)
    assert isinstance(decision, Abandon) and decision.reasons[1:] == unresolved
    assert "budget.final_revisions (2) is spent" in decision.reasons[0]


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
    pending = apply_decision(wp, decision, config)
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
        (ScopeEntered("workpiece"), None),
        (ScopeEntered("validate"), None),
        (Resolved(SolverRollout(spec, 0, 0), (), rollout), Billing.SOLVER),
        (review, Billing.SOLVE_REVIEW),
        (ScopeExited("validate", False), None),
        (ScopeEntered("confirm"), None),
        (Resolved(SolverRollout(spec, 0, 0), (), rollout, replayed=True), None),  # the round's trial read again
        (Resolved(SolverRollout(spec, 1, 1), (), rollout), Billing.CONFIRMATION),
        (Resolved(SolverRollout(spec, 2, 1), (), rollout), Billing.CONFIRMATION),
        (final, Billing.FINAL_REVIEW),
        (ScopeExited("confirm", False), None),
        (StageCompleted(Stage.REVIEW, "", "Decided", terminal=Terminal.ACCEPTED), None),
    )
    records = tuple(Record(i, time.time(), event, billed=payer) for i, (event, payer) in enumerate(events, 1))
    sink = CheckpointSink(str(tmp_path / "root"))
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
    # Updated knowingly: the report tool lets an account lean on an earlier finding, and the
    # role has its own nudge.
    recorded = "8446ffff2274131fb3417500fa2598bdf2c3805ba13c83e734d1f2078218c990"
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


# Regression: each case below reproduces one fault in the confirmation step or the final review, one test or
# parameter each.

BASE = tuple(TrialResult(i, GradeResult(Outcome.GRADED, float(i % 2 == 0))) for i in range(3))


def graded(trial: int, grade: GradeResult, cause: Cause | None = None) -> TrialResult:
    """A slot as ``solver_result`` records it: the cause ``trial_cause`` gives an ordinary stop."""
    return TrialResult(trial, grade, cause or grade_cause(grade))


def confirming(
    proposal: TaskProposal, added: Sequence[TrialResult], review: SolveReview = SolveReview("v1", ReviewEnd.REPORTED)
) -> Workpiece:
    """``BASE`` accepted in band, then its confirmation record with ``added`` slots and a final ``review``."""
    first = ValidationResult(SUBMISSION.digest, BASE, CONTROLS, (), review=SolveReview("v0", ReviewEnd.REPORTED))
    record = ValidationResult(SUBMISSION.digest, (*BASE, *added), CONTROLS, (), review=review, confirms=0)
    accepted = (Accept(BandOutcome.IN_BAND),)
    return Workpiece("w", proposal, submission=SUBMISSION, evidence=(first, record), decisions=accepted)


INVALID = GradeResult(Outcome.INVALID_TASK, None, "the task's files are missing")
NO_REWARD = GradeResult(Outcome.GRADED, None)
UNGRADED = GradeResult(Outcome.UNAVAILABLE, None)


@pytest.mark.parametrize(
    "added,kind",
    [
        ([graded(t, INVALID) for t in range(3, 8)], "revise"),  # was accepted at "2 of 8"
        ([graded(t, NO_REWARD) for t in range(3, 8)], "revise"),
        (
            [graded(t, GradeResult(Outcome.SUBMISSION_FAILURE, 0.0), Cause.BUDGET_EXHAUSTED) for t in range(3, 8)],
            "reject",
        ),
        # 6 solved, 1 failed, 1 invalid: was "6 of 8" in band
        ([*(graded(t, GradeResult(Outcome.GRADED, 1.0)) for t in range(3, 7)), graded(7, INVALID)], "revise"),
    ],
)
def test_a_confirmation_fault_is_decided_before_placement_even_with_a_clean_final_review(
    proposal: TaskProposal, added: list[TrialResult], kind: str
) -> None:
    config = setup()
    decision = default_review(confirming(proposal, added), config.band, config.budget)
    assert decision.kind == kind
    if isinstance(decision, Revise):  # the fault is the author's to repair, with no band line and no placement
        assert decision.placement is None and not decision.review
        assert not any("trials solved" in line for line in decision.findings)


def test_a_fault_in_the_revision_s_confirmation_abandons(proposal: TaskProposal) -> None:
    """Owner's decision (a): after the final review's revision nothing is kept to restore."""
    config = setup()
    revise = Revise(("line",), (REVISE,), placement=BandOutcome.IN_BAND)
    wp = confirming(proposal, [graded(3, INVALID)])
    wp = replace(wp, decisions=(revise, *wp.decisions))
    decision = default_review(wp, config.band, config.budget)
    assert isinstance(decision, Abandon)
    assert decision.reasons[0] == f"{CONFIRMATION_REVISION_FAILED}: its confirmation has a fault"
    assert f"Unresolved final-review finding: {REVISE.description}" in decision.reasons


def test_an_ungraded_slot_is_named_and_excluded_and_a_generation_limit_counts_unsolved(
    proposal: TaskProposal,
) -> None:
    """The same slot returning no grade places over seven; a generation limit is a failed trial in the denominator,
    and the acceptance says it was counted unsolved."""
    config = setup()
    solved = [graded(t, GradeResult(Outcome.GRADED, 1.0)) for t in range(3, 7)]
    for last, over, absent in ((graded(7, UNGRADED), 7, True), (graded(7, UNGRADED, Cause.GENERATION_LIMIT), 8, False)):
        decision = default_review(confirming(proposal, [*solved, last]), config.band, config.budget)
        assert isinstance(decision, Accept) and f"Confirmation: 6 of {over} trials solved" in "|".join(decision.reasons)
        named = MissingEvidence(EvidenceKind.CONFIRMATION_TRIALS, "solver trials [7] have no score")
        assert (named in decision.missing) is absent
        limited = "Counted unsolved without a gradable answer: 1 generation_limit"
        assert (limited in decision.reasons) is not absent


def invalid_in_confirmation(trial: int, attempt: int) -> GradeResult:
    return INVALID if attempt == 1 else even(trial, attempt)


async def test_an_invalid_confirmation_grade_sends_the_task_back_through_the_driver(proposal: TaskProposal) -> None:
    engine, reviewer = Trials(invalid_in_confirmation), Reviewer()
    live = bundle(engine, reviewer, (*script(), *RESUBMIT))
    wp = await run_workpiece(proposal, setup(), live, EventLog(), install=install)
    assert kinds(wp) == ["accept", "revise", "accept", "accept"] and wp.terminal is Terminal.ACCEPTED
    revise, final = wp.decisions[1], wp.decisions[-1]
    assert isinstance(revise, Revise) and revise.placement is None
    assert any("invalid_task" in line for line in revise.findings)
    assert isinstance(final, Accept) and "Confirmation: 4 of 8 trials solved: in_band" in final.reasons


class Outages(Trials):
    """``Trials``, with an ``outage`` slot interrupted by a decision-model outage."""

    def __init__(self, grade: Callable[[int, int], GradeResult], outage: frozenset[Slot]) -> None:
        super().__init__(grade)
        self.outage = outage

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        assert run_id is not None
        if tuple(int(part) for part in run_id.split(":")[-2:]) in self.outage:
            self.runs.append((0, -1))
            error = RolloutInterrupted(Rollout(run_id, (), "interrupted"), RolloutOperation.GRADE)
            error.__cause__ = DecisionModelUnavailable("clef answered no read for 900s; last HTTP 503")
            raise error
        return await super().run_task(spec, model, run_id=run_id)


def infra_in_slot_4(trial: int, attempt: int) -> GradeResult:
    if (trial, attempt) == (4, 1):
        return GradeResult(Outcome.INFRA_ERROR, None, "grader host lost", failure=GradingFailure.EXECUTION)
    return even(trial, attempt)


@pytest.mark.parametrize("beside", [True, False])
async def test_a_confirmation_outage_beside_another_hold_is_a_manual_hold(proposal: TaskProposal, beside: bool) -> None:
    """Slot 4's grade holds through its observer, then slot 5 meets an outage, at concurrency one: both are kept, so
    the root no longer parks for automatic relaunch. A pure outage still parks."""
    engine = Outages(infra_in_slot_4 if beside else even, frozenset({(5, 1)}))
    wp, _ = await run(proposal, engine, Reviewer(), setup(**{"solver.concurrency": 1}))
    held = wp.decisions[-1]
    assert wp.terminal is Terminal.HELD and isinstance(held, Incomplete)
    assert any("503" in reason for reason in held.reasons)
    assert any("grader host lost" in reason for reason in held.reasons) is beside
    assert held.awaits == (None if beside else "decision_model")


async def test_a_clean_final_revision_reaches_its_confirmation_whatever_the_band_setting(proposal: TaskProposal) -> None:
    """With ``band.accept_too_easy`` false, the final revision's round solves 3 of 3, too easy, and is not refused
    there: its confirmation places it again, in band at 5 of 8."""

    def grade(trial: int, attempt: int) -> GradeResult:
        solved = {2: True, 3: trial in (3, 4)}.get(attempt, trial % 2 == 0)
        return GradeResult(Outcome.GRADED, float(solved))

    engine, reviewer = Trials(grade), Reviewer(finals=[([found(3, what="first")], [])])
    live = bundle(engine, reviewer, (*script(), *RESUBMIT))
    wp = await run_workpiece(proposal, setup(**{"band.accept_too_easy": False}), live, EventLog(), install=install)
    assert kinds(wp) == ["accept", "revise", "accept", "accept"] and wp.terminal is Terminal.ACCEPTED
    assert len(engine.runs) == 16 and wp.accepted_with is None
    revised = wp.decisions[2]
    assert isinstance(revised, Accept) and revised.band is BandOutcome.TOO_EASY
    assert "Placement is final after the final review's revision" in revised.reasons
    final = wp.decisions[-1]
    assert isinstance(final, Accept) and "Confirmation: 5 of 8 trials solved: in_band" in final.reasons


def test_a_final_account_may_blame_a_trial_on_an_earlier_finding_whose_quote_still_holds() -> None:
    """An account blaming a trial on a defect needs a citation of that trial: one of a new finding, or of an earlier
    report's finding rechecked against these trials. A finding quoting text the trial no longer has supports nothing."""
    rollout = GradedRollout(
        Rollout("r", (transcript("trial 1 attempt 0 answers 12"),), "stop"), GradeResult(Outcome.GRADED, 0.0)
    )
    index = evidence(SUBMISSION, [(TrialResult(1, rollout.grade), rollout)], 4000)
    args = FinalReportArgs((), (), (TrialAccount(1, Attribution.DEFECT, "the known defect"),))
    stale = replace(finding(1), citations=(replace(finding(1).citations[0], quote="answers 13"),))
    ctx = Context(log=EventLog())
    for earlier, valid in (((), False), ((finding(1),), True), ((stale,), False)):
        outcome = asyncio.run(FinalReportTool(index, earlier).run(ctx, args))
        assert isinstance(outcome, ToolResult) is not valid, outcome
        if not isinstance(outcome, ToolResult):
            assert not outcome.payload.findings and not outcome.payload.notes  # nothing persisted again


async def test_a_repeated_note_is_kept_once(proposal: TaskProposal) -> None:
    note = found(1)
    reviewer = Reviewer(rounds=[[note]], finals=[([], [note])])
    wp, _ = await run(proposal, Trials(), reviewer, setup(**{"budget.max_modify_rounds": 0}))
    final = wp.decisions[-1]
    assert kinds(wp) == ["accept", "accept"] and isinstance(final, Accept) and final.notes == (finding(1),)


DECLINE = (
    *script()[:3],
    call("submit", {"notes": "revised", "declined": [{"finding": 0, "reason": "the grader is right"}]}, "submit"),
)
"""A revision round's author turns that decline the one finding."""


@pytest.mark.parametrize("declined", [True, False])
async def test_a_repeated_finding_buys_the_revision_only_as_a_failed_fix(proposal: TaskProposal, declined: bool) -> None:
    """The final review repeats a round finding as ``revise``: declined by the author, it is not new and is a note;
    after the author revised for it, its repetition is evidence the fix failed, and buys the one revision."""
    reviewer = Reviewer(rounds=[[found(1)], []], finals=[([found(1)], [])])
    author = (*script(), *(DECLINE if declined else RESUBMIT), *RESUBMIT)
    live = bundle(Trials(), reviewer, author)
    wp = await run_workpiece(proposal, setup(), live, EventLog(), install=install)
    assert wp.terminal is Terminal.ACCEPTED
    if declined:
        assert kinds(wp) == ["modify", "accept", "accept"]
        final = wp.decisions[-1]
        assert isinstance(final, Accept) and final.notes == (finding(1),)
        assert "Outcome: the author declined it: the grader is right." in reviewer.finals_seen[0]
    else:
        assert kinds(wp) == ["modify", "accept", "revise", "accept", "accept"]


class Nudged(Reviewer):
    """``Reviewer``, whose first final-review turn is empty, so the harness nudges it."""

    def __init__(self) -> None:
        super().__init__()
        self.nudged = False

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        opening = next(message.content for message in request.conversation if message.role == Role.USER)
        if FINAL in opening and not self.nudged:
            self.nudged = True
            self.requests.append(request)
            return Completion("")
        return await super().complete(request)


async def test_the_final_review_is_nudged_toward_its_own_fields(proposal: TaskProposal) -> None:
    reviewer = Nudged()
    wp, _ = await run(proposal, Trials(), reviewer)
    nudge = load_policy(DEFAULTS_DIR).prompt("solve_review.final.nudge").template
    assert wp.terminal is Terminal.ACCEPTED and "`revise`" in nudge and "`note`" in nudge
    assert reviewer.requests[-1].conversation[-1].content == nudge


def test_responses_are_kept_by_originating_review_across_a_stop_and_several_rounds(proposal: TaskProposal) -> None:
    """Three revisions: the oldest decline reason survives two later submissions; an author who stops leaves its
    findings stopped, not revised."""
    config = setup(**{"budget.max_modify_rounds": 3})
    found_in = [finding(1, what=f"round {n}") for n in range(3)]
    wp = Workpiece("w", proposal, submission=SUBMISSION, session=Conversation((Message(Role.SYSTEM, "author"),), ()))
    for n, f in enumerate(found_in):
        result = ValidationResult(
            SUBMISSION.digest, BASE, CONTROLS, (), review=SolveReview(f"v{n}", ReviewEnd.REPORTED, (f,))
        )
        wp = apply_decision(replace(wp, evidence=(*wp.evidence, result)), Modify(("x",), review=(f,)), config)
        assert wp.session is not None
        if n < 2:
            declined = (Decline(f, f"reason {n}"),) if n == 0 else ()
            wp, _ = advance(wp, Stage.AUTHOR, Submitted(replace(SUBMISSION, declined=declined), wp.session), config)
        else:
            wp, _ = advance(wp, Stage.AUTHOR, Stopped(StopReason.CONTEXT_LIMIT, wp.session, "full"), config)
    assert [(r.review, r.response, r.reason) for r in wp.responses] == [
        (0, Response.DECLINED, "reason 0"),
        (1, Response.REVISED, ""),
        (2, Response.STOPPED, ""),
    ]
    history = earlier_reports(wp)
    assert "round 0 Edit: check equality Outcome: the author declined it: reason 0." in history
    assert "round 1 Edit: check equality Outcome: given to the author, who revised the task." in history
    assert "round 2 Edit: check equality Outcome: given to the author, who stopped without resubmitting." in history
    assert WORKPIECE.validate_json(WORKPIECE.dump_json(wp)) == wp


async def test_a_mixed_final_report_sends_its_note_to_the_author_as_optional_and_keeps_it(
    proposal: TaskProposal,
) -> None:
    first, note = found(3, what="first"), found(5, FindingKind.AMBIGUOUS_PROMPT, "reads two ways")
    reviewer = Reviewer(finals=[([first], [note])])
    live = bundle(Trials(), reviewer, (*script(), *RESUBMIT))
    wp = await run_workpiece(proposal, setup(), live, EventLog(), install=install)
    assert kinds(wp) == ["accept", "revise", "accept", "accept"] and wp.terminal is Terminal.ACCEPTED
    kept = finding(5, FindingKind.AMBIGUOUS_PROMPT, "reads two ways")
    revise = wp.decisions[1]
    assert isinstance(revise, Revise) and revise.notes == (kept,)
    assert isinstance(live.inference, FakeInference)
    prompt = live.inference.requests[4].conversation[1].content
    assert "also recorded these findings as notes" in prompt and "1. [ambiguous_prompt] reads two ways" in prompt
    assert "Outcome: given to the author as an optional note, who revised the task." in reviewer.finals_seen[-1]
    final = wp.decisions[-1]
    assert isinstance(final, Accept) and final.notes == (kept,)


def test_a_pending_acceptance_publishes_no_placement(proposal: TaskProposal) -> None:
    """A provisional too-easy acceptance sent back by the final review and abandoned carried ``accepted_with``."""
    config = setup()
    wp = replace(confirming(proposal, ()), decisions=(), evidence=confirming(proposal, ()).evidence[:1])
    pending = apply_decision(wp, Accept(BandOutcome.TOO_EASY), config)
    assert pending.terminal is None and pending.accepted_with is None
    abandoned = apply_decision(pending, Abandon(("author stopped",)), config)
    assert abandoned.terminal is Terminal.ABANDONED and abandoned.accepted_with is None
    final = apply_decision(confirming(proposal, ()), Accept(BandOutcome.TOO_EASY), config)
    assert final.terminal is Terminal.ACCEPTED and final.accepted_with is BandOutcome.TOO_EASY


def test_an_omission_both_reviews_name_is_listed_once(proposal: TaskProposal) -> None:
    config = setup()
    gone = MissingEvidence(EvidenceKind.TRANSCRIPTS, "solver trials [0] have no complete transcript")
    wp = confirming(proposal, (), review=SolveReview("v1", ReviewEnd.REPORTED, missing=(gone,)))
    wp = replace(wp, decisions=(Accept(BandOutcome.IN_BAND, missing=(gone,)),))
    decision = default_review(wp, config.band, config.budget)
    assert isinstance(decision, Accept) and decision.missing == (gone,)


# Regression: each case below reproduces one fault in how the final review carried findings and notes across
# revisions.


async def test_a_fix_attempted_in_a_discarded_revision_is_no_failed_fix_on_the_restored_task(
    proposal: TaskProposal,
) -> None:
    """The revision that answered round 0's finding faulted and round 0 was restored; the final review cites the same
    defect on round 0's own trials. That revision never shipped, so the repetition is not a failed fix: 11 solves, not
    19, and the finding is a note. The discarded response stays in the history."""
    engine, reviewer = Trials(faulted_after_attempt_0), Reviewer(rounds=[[found(1)]], finals=[([found(1)], [])])
    live = bundle(engine, reviewer, (*script(), *RESUBMIT, *RESUBMIT))
    wp = await run_workpiece(proposal, setup(), live, EventLog(), install=install)
    assert kinds(wp) == ["modify", "accept", "accept"] and len(engine.runs) == 11
    assert [(r.review, r.response, r.discarded) for r in wp.responses] == [(0, Response.REVISED, True)]
    final = wp.decisions[-1]
    assert isinstance(final, Accept) and final.notes == (finding(1),)


async def test_a_newer_decline_outranks_an_older_revision(proposal: TaskProposal) -> None:
    """Round 0's finding was revised for, round 1 repeated it and the author declined it: the latest response on the
    shipped history decides, so the final review's repetition is a note: 14 solves, not 22. (Round 1 cites another
    trial, so the author's second request differs from its first and the cached inference does not answer it.)"""
    reviewer = Reviewer(rounds=[[found(1)], [found(2)], []], finals=[([found(1)], [])])
    live = bundle(Trials(), reviewer, (*script(), *RESUBMIT, *DECLINE, *RESUBMIT))
    engine = live.rollouts.engine
    wp = await run_workpiece(proposal, setup(**{"budget.max_modify_rounds": 2}), live, EventLog(), install=install)
    assert isinstance(engine, Trials) and kinds(wp) == ["modify", "modify", "accept", "accept"]
    assert len(engine.runs) == 14
    assert [r.response for r in wp.responses] == [Response.REVISED, Response.DECLINED]


async def test_notes_the_shipped_trials_support_survive_the_final_revision(proposal: TaskProposal) -> None:
    """Note A rode the first acceptance; the final review asked for a fix with optional note B; the author resubmitted
    the same task and the next final report was clean. Both notes ship, as the shipped trials still support them."""
    a, b = found(1, what="note A"), found(5, FindingKind.AMBIGUOUS_PROMPT, "note B")
    reviewer = Reviewer(rounds=[[a], []], finals=[([found(3, what="fix")], [b])])
    live = bundle(Trials(), reviewer, (*script(), *RESUBMIT))
    wp = await run_workpiece(proposal, setup(**{"budget.max_modify_rounds": 0}), live, EventLog(), install=install)
    assert kinds(wp) == ["accept", "revise", "accept", "accept"]
    final = wp.decisions[-1]
    assert isinstance(final, Accept)
    assert final.notes == (finding(1, what="note A"), finding(5, FindingKind.AMBIGUOUS_PROMPT, "note B"))


async def test_a_note_ships_in_the_version_its_shipped_trial_quotes(proposal: TaskProposal) -> None:
    """Note B first quoted the first confirmation's trial 3; the revision's confirmation quotes its own trial 3. The
    old quote does not occur in the shipped trial, so the current version ships."""
    old = found(3, FindingKind.AMBIGUOUS_PROMPT, "note B", "trial 3 attempt 1 answers 12")
    new = found(3, FindingKind.AMBIGUOUS_PROMPT, "note B", "trial 3 attempt 3 answers 12")
    reviewer = Reviewer(finals=[([found(4, what="fix")], [old]), ([], [new])])
    live = bundle(Trials(), reviewer, (*script(), *RESUBMIT))
    wp = await run_workpiece(proposal, setup(), live, EventLog(), install=install)
    assert kinds(wp) == ["accept", "revise", "accept", "accept"]
    final = wp.decisions[-1]
    assert isinstance(final, Accept) and [c.quote for f in final.notes for c in f.citations] == [
        "trial 3 attempt 3 answers 12"
    ]


def test_a_finding_in_both_final_lists_is_refused_and_a_repeat_within_one_is_kept_once() -> None:
    rollout = GradedRollout(
        Rollout("r", (transcript("trial 1 attempt 0 answers 12"),), "stop"), GradeResult(Outcome.GRADED, 1.0)
    )
    index = evidence(SUBMISSION, [(TrialResult(1, rollout.grade), rollout)], 4000)
    ctx, twice = Context(log=EventLog()), finding(1)
    conflicting = asyncio.run(FinalReportTool(index).run(ctx, FinalReportArgs((twice,), (twice,), ())))
    assert isinstance(conflicting, ToolResult) and "one disposition" in conflicting.content
    repeated = asyncio.run(FinalReportTool(index).run(ctx, FinalReportArgs((twice, twice), (), ())))
    assert not isinstance(repeated, ToolResult) and repeated.payload.findings == (twice,)


class Unavailable(Reviewer):
    """``Reviewer``, whose final review's provider is down."""

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        if FINAL in next(m.content for m in request.conversation if m.role == Role.USER):
            raise AssertionError("a decided confirmation record asked its final reviewer")
        return await super().complete(request)


async def test_a_budget_rejection_in_the_confirmation_needs_no_final_review(proposal: TaskProposal) -> None:
    engine = Trials(stop=lambda trial, attempt: MAX_TURNS_STOP_REASON if attempt == 1 else "stop")
    wp, _ = await run(proposal, engine, Unavailable())
    assert kinds(wp) == ["accept", "reject"] and wp.terminal is Terminal.REJECTED and len(engine.runs) == 8
    assert wp.evidence[-1].review is None


def test_a_fault_revision_writes_no_placement() -> None:
    adapter = TypeAdapter(Revise)
    fault = Revise(("1 ungraded trial(s): invalid_task",), ())
    assert b"placement" not in adapter.dump_json(fault) and adapter.validate_json(adapter.dump_json(fault)) == fault
    placed = Revise(("line",), (REVISE,), placement=BandOutcome.IN_BAND)
    assert adapter.validate_json(adapter.dump_json(placed)) == placed


# Regression: every submitted version is checked before versions of one finding merge.


def failed_trials(*trials: int) -> Evidence:
    """An evidence index over unsolved trials ``trials``, each answering ``trial N attempt 0 answers 12``."""
    rollouts = [
        GradedRollout(
            Rollout(f"r{t}", (transcript(f"trial {t} attempt 0 answers 12"),), "stop"), GradeResult(Outcome.GRADED, 0.0)
        )
        for t in trials
    ]
    return evidence(SUBMISSION, [(TrialResult(t, r.grade), r) for t, r in zip(trials, rollouts, strict=True)], 4000)


def defect(*trials: int) -> tuple[TrialAccount, ...]:
    return tuple(TrialAccount(t, Attribution.DEFECT, "the grader") for t in trials)


@pytest.mark.parametrize("stale_first", [True, False])
def test_a_stale_version_of_a_required_fix_never_hides_its_valid_one(stale_first: bool) -> None:
    valid = finding(1)
    stale = TypeAdapter(ReviewFinding).validate_python(found(1, quote="answers 13"))
    submitted = (stale, valid) if stale_first else (valid, stale)
    outcome = asyncio.run(
        FinalReportTool(failed_trials(1)).run(Context(log=EventLog()), FinalReportArgs(submitted, (), defect(1)))
    )
    assert not isinstance(outcome, ToolResult)
    assert outcome.payload.findings == (valid,) and outcome.payload.dropped[0][0] == stale


def test_two_valid_versions_citing_different_trials_are_one_finding_carrying_both() -> None:
    one, two, index, ctx = finding(1), finding(2), failed_trials(1, 2), Context(log=EventLog())
    final = asyncio.run(FinalReportTool(index).run(ctx, FinalReportArgs((one, two), (), defect(1, 2))))
    ordinary = asyncio.run(ReportTool(index).run(ctx, ReportArgs((one, two), defect(1, 2))))
    for outcome in (final, ordinary):
        assert not isinstance(outcome, ToolResult), outcome.content
        (merged,) = outcome.payload.findings
        assert merged.citations == (*one.citations, *two.citations) and supports(merged) == (
            one.citations,
            two.citations,
        )


@pytest.mark.parametrize("stale_first", [True, False])
async def test_the_order_of_a_repeated_required_fix_does_not_change_the_run(
    proposal: TaskProposal, stale_first: bool
) -> None:
    valid, stale = found(3, what="fix"), found(3, what="fix", quote="answers 13")
    reviewer = Reviewer(finals=[([stale, valid] if stale_first else [valid, stale], [])])
    live = bundle(Trials(), reviewer, (*script(), *RESUBMIT))
    wp = await run_workpiece(proposal, setup(), live, EventLog(), install=install)
    engine = live.rollouts.engine
    assert isinstance(engine, Trials) and kinds(wp) == ["accept", "revise", "accept", "accept"]
    assert len(engine.runs) == 16


async def test_a_finding_the_round_review_repeats_is_one_numbered_item_declined_once(proposal: TaskProposal) -> None:
    reviewer = Reviewer(rounds=[[found(1), found(1)]])
    live = bundle(Trials(), reviewer, (*script(), *DECLINE))
    wp = await run_workpiece(proposal, setup(), live, EventLog(), install=install)
    assert kinds(wp)[0] == "modify" and wp.evidence[0].review is not None
    assert wp.evidence[0].review.findings == (finding(1),)
    assert [(r.review, r.response, r.reason) for r in wp.responses] == [(0, Response.DECLINED, "the grader is right")]


# Regression: a provider's context refusal of the author's revision was raised
# past the effect, so a cold resume asked again; when the provider then answered, the restored round's response became
# a revision and the run bought a second confirmation (16 solves, not 8).
async def test_a_cold_resume_replays_the_authors_context_refusal(proposal: TaskProposal) -> None:
    refused = ContextExceeded("maximum context length exceeded")  # as the inference resource raises it
    sink, engine = BufferSink(), Trials()
    author = FakeInference((*script(), *script()[:3], refused))
    live = replace(bundle(engine, Reviewer(rounds=[[found(1)]]), ()), inference=author)
    done = await run_workpiece(proposal, setup(), live, EventLog(sink=sink), install=install)
    assert done.terminal is Terminal.ACCEPTED
    assert [(r.response, r.discarded) for r in done.responses] == [(Response.STOPPED, True)] and len(engine.runs) == 8
    again, author = Trials(), (*script(), *RESUBMIT)
    resumed = bundle(again, Reviewer(rounds=[[found(1)]]), author)
    wp = await run_workpiece(proposal, setup(), resumed, EventLog(replay_from=local_records(sink)), install=install)
    assert wp == done and not again.runs
    assert isinstance(resumed.inference, FakeInference) and not resumed.inference.requests


# Regression: one note in two versions, a quote trial 4 keeps and one only trial 6's first
# confirmation attempt holds. Merged into one citation set, the note was dropped after a revision that kept trial 4.
UNITS = "the prompt does not say which units"
STABLE, SPECIFIC = (
    found(4, FindingKind.AMBIGUOUS_PROMPT, UNITS),
    found(6, FindingKind.AMBIGUOUS_PROMPT, UNITS, "attempt 1"),
)


@pytest.mark.parametrize(
    "versions",
    [(STABLE, SPECIFIC), (SPECIFIC, STABLE), (STABLE,)],
    ids=["stable first", "specific first", "stable only"],
)
async def test_a_note_ships_with_the_versions_the_revised_trials_still_support(
    proposal: TaskProposal, versions: tuple[dict[str, object], ...]
) -> None:
    reviewer = Reviewer(finals=[([found(3, what="fix")], list(versions))])
    wp = await run_workpiece(
        proposal, setup(), bundle(Trials(), reviewer, (*script(), *RESUBMIT)), EventLog(), install=install
    )
    assert kinds(wp) == ["accept", "revise", "accept", "accept"]
    first, last = wp.decisions[1], wp.decisions[-1]
    assert isinstance(first, Revise) and isinstance(last, Accept)
    assert [len(supports(note)) for note in first.notes] == [len(versions)]
    assert [(n.description, [(c.source, c.quote) for c in n.citations], n.groups) for n in last.notes] == [
        (UNITS, [("trial/4", "answers 12")], ())
    ]


def test_a_defect_account_citing_a_trial_an_earlier_version_still_supports_is_accepted() -> None:
    stable, specific = (TypeAdapter(ReviewFinding).validate_python(v) for v in (STABLE, SPECIFIC))
    earlier = grouped(stable, (stable.citations, specific.citations))  # as the first final report stored it
    tool = FinalReportTool(failed_trials(4, 6), earlier=(earlier,))
    accounts = (TrialAccount(4, Attribution.DEFECT, UNITS), TrialAccount(6, Attribution.SOLVER, "its own work"))
    outcome = asyncio.run(tool.run(Context(log=EventLog()), FinalReportArgs((), (), accounts)))
    assert not isinstance(outcome, ToolResult), outcome.content
    blamed = (TrialAccount(4, Attribution.SOLVER, "its own work"), TrialAccount(6, Attribution.DEFECT, UNITS))
    refused = asyncio.run(tool.run(Context(log=EventLog()), FinalReportArgs((), (), blamed)))
    assert isinstance(refused, ToolResult) and "trial 6 is attributed to a task defect" in refused.content


def test_one_version_stores_no_groups_and_several_round_trip_outside_the_tool_schema() -> None:
    adapter, single = TypeAdapter(ReviewFinding), finding(1)
    assert "groups" not in json.loads(adapter.dump_json(single))
    merged = grouped(single, (single.citations, finding(2).citations))
    assert adapter.validate_json(adapter.dump_json(merged)) == merged
    assert "groups" not in json.dumps(TypeAdapter(FinalReportArgs).json_schema())


JUDGED_TASK = arithmetic_task(grader=verifyit_package(JudgeSpec(references=("12",), exact_gate=False)).grader)
BILLED = {"adversary.roster": ["false_accept"], "judge.model": "glm-5.3"}
"""One adversary and a judged task beside the three validation and five confirmation solves."""


class JudgedEnvs(Envs):
    """The author's draft writes ``JUDGED_TASK``, whose grader asks the judge."""

    def __init__(self) -> None:
        super().__init__()
        draft, path = self.commands["write-draft"], PurePosixPath("/workspace/task.json")
        self.commands["write-draft"] = FakeCommand(
            draft.result, {**draft.writes, path: JUDGED_TASK.model_dump_json().encode()}
        )


class Replies(InferenceClient):
    """``reply`` to every request, or ``inner``'s, each billed ``prompt`` prompt tokens. A request sets ``starts``,
    waits for ``waits``, and sets ``sets`` once answered."""

    def __init__(
        self,
        identity: str,
        prompt: int,
        reply: Completion | None = None,
        inner: InferenceClient | None = None,
        *,
        starts: asyncio.Event | None = None,
        waits: asyncio.Event | None = None,
        sets: asyncio.Event | None = None,
    ) -> None:
        self.identity, self.prompt, self.reply, self.inner = identity, prompt, reply, inner
        self.starts, self.waits, self.sets = starts, waits, sets
        self.calls = 0

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("inference", self.identity, ReplayPolicy.CACHE)

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        self.calls += 1
        if self.starts is not None:
            self.starts.set()
        if self.waits is not None:
            await self.waits.wait()
        answer = self.reply if self.inner is None else await self.inner.complete(request)
        assert answer is not None
        if self.sets is not None:
            self.sets.set()
        return replace(answer, usage=Usage(prompt_tokens=self.prompt))


class Reviews(Replies):
    """The reviewer's client: the round review's call bills 113 prompt tokens and every later call, the final
    review's, 991, so that charging one review's call to the other shows."""

    def __init__(self) -> None:
        super().__init__("reviewer", 113, inner=CleanReviewer())

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        self.prompt = 991 if self.calls else 113
        return await super().complete(request)


class JudgedTrials(Trials):
    """Each slot asks the judge once from a grading thread, as VerifyIt does, and its turn bills 7 prompt tokens."""

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        connection = spec.judge_connection
        assert connection is not None
        await asyncio.to_thread(connection.callback, JudgeRequest("judge", (JudgeMessage("user", f"{run_id}"),), 64, ""))
        graded = await super().run_task(spec, model, run_id=run_id)
        turn = ModelTurn({}, (), (), None, "stop", metadata={"usage": {"prompt_tokens": 7}})
        steps = (RolloutStep(turn, Transition(True), 0, ()),)
        said = tuple(replace(conversation, steps=steps) for conversation in graded.rollout.conversations)
        return replace(graded, rollout=replace(graded.rollout, conversations=said))


def billed_bundle(author: Sequence[Completion] = script()) -> Resources:
    """Every role's client, each billing its own number of prompt tokens per call. The adversary answers once a judge
    call is in flight, and the judge once the adversary has: the adversary's call completes inside an open judge
    scope."""
    live, asked, answered = resources(JudgedEnvs(), ()), asyncio.Event(), asyncio.Event()
    roles = {
        "solve_review": Reviews(),
        "adversary": Replies("adversary", 100, call("report", {"findings": []}, "report"), waits=asked, sets=answered),
        "judge": Replies("judge", 10000, Completion("PASS"), starts=asked, waits=answered),
    }
    author_client = Replies("author", 1, inner=FakeInference(author))
    engine = JudgedTrials()
    return replace(live, inference=author_client, rollouts=replace(live.rollouts, engine=engine), role_inference=roles)


async def billed_root(
    proposal: TaskProposal, directory: Path, live: Resources | None = None, root: Path | None = None
) -> Path:
    """Run ``live`` (by default ``billed_bundle()``) to a published root, ``root`` resumed or a new one under
    ``directory``, as the driver publishes one."""
    if root is None:
        root = directory / "root"
        root.mkdir(parents=True)
    sink = CheckpointSink(str(root))
    log = EventLog(sink=sink, replay_from=sink.claim(read_journal(str(root))))
    async with sink.running():
        config = setup(**BILLED)
        wp = await run_workpiece(
            proposal, config, live or billed_bundle(), log, install=install, stage_plan=digest((proposal, config.policy))
        )
    await sink.finish(fold(tuple(record.event for record in log.records)), wp)
    return root


CHARGED = {
    "author": 4,
    "solver": 21,
    "adversary": 100,
    "solve_review": 113,
    "judge": 80000,
    "confirmation": 35,
    "final_review": 991,
}
"""``billed_bundle``'s prompt tokens by role: four author calls, three validation and five confirmation solves, one
adversary call, one round and one final review call, and eight judge calls, one per solve."""
FORMAT_9_ROOT = Path(__file__).parent / "fixtures" / "format9-judge-overlap-root.json.gz"
"""``billed_root`` as the format-9 writer (9a3da6e3ef) published it: its rows name no payer."""


def charged(lines: Sequence[str]) -> dict[str, int]:
    """The prompt tokens each ``tokens role=`` line of a report charges."""
    return {
        found["role"]: int(found["prompt"])
        for line in lines
        if (found := re.match(r"tokens role=(?P<role>\w+) prompt=(?P<prompt>\d+) ", line))
    }


PAID = {
    ("inference.request", "author"): 4,
    ("grading.grade_candidate", "author"): 1,
    ("rollouts.solver_rollout", "solver"): 3,
    ("grading.grade_candidate", "solver"): 2,
    ("inference.request", "adversary"): 1,
    ("inference.request", "solve_review"): 1,
    ("inference.request", "judge"): 8,
    ("rollouts.solver_rollout", "confirmation"): 5,
    ("inference.request", "final_review"): 1,
    ("schema.author_reference", "author"): 1,
}
"""``billed_bundle``'s physical resolutions by kind and payer, sandbox commands and ids aside: they name none."""


def payers(root: Path, replayed: bool) -> Counter[tuple[str, str | None]]:
    """The payer of each resolution of ``root``'s journal that ``replayed`` or was paid, by request kind, sandbox
    commands and ids aside."""
    journal = read_journal(str(root))
    assert journal is not None
    return Counter(
        (record.event.request.kind, record.billed)
        for record in decode(journal.events())
        if isinstance(record.event, Resolved)
        and record.event.replayed is replayed
        and not record.event.request.kind.startswith(("env.", "ids."))
    )


# Regression: the report guessed each call's role from the scopes open when it resolved, so an adversary call
# that completed inside an open judge scope was charged to the judge: 80,100 judge and 0 adversary tokens here.
async def test_every_role_is_charged_its_own_calls_when_their_scopes_overlap(
    proposal: TaskProposal, tmp_path: Path
) -> None:
    root = await billed_root(proposal, tmp_path)
    lines = await report(str(root))
    assert charged(lines) == CHARGED
    assert not [line for line in lines if line.startswith("tokens role=unattributed")]
    assert payers(root, replayed=False) == PAID and not payers(root, replayed=True)
    live = billed_bundle(())
    await billed_root(proposal, tmp_path, live, root)
    clients = [live.inference, *live.role_inference.values()]
    assert [client.calls for client in clients if isinstance(client, Replies)] == [0, 0, 0, 0]
    assert payers(root, replayed=False) == PAID  # the cold rerun paid for nothing
    assert charged(await report(str(root))) == CHARGED
    replayed = payers(root, replayed=True)  # the rerun resumed at confirmation: its five slots and final review
    assert sum(replayed.values()) == 6 and {payer for _, payer in replayed} == {None}


async def test_a_format_9_root_charges_its_tokens_as_unattributed(tmp_path: Path) -> None:
    """Rows written before the payer was recorded name none, and the report charges their tokens as unattributed."""
    root = plant_root(tmp_path / "root", FORMAT_9_ROOT)
    journal = read_journal(str(root))
    assert journal is not None and journal.head.format == 9
    assert charged(await report(str(root))) == {**dict.fromkeys(CHARGED, 0), "unattributed": 82160}


# Regression: a clean three-and-five confirmation built its evidence index three times (3, 8 and 8 trials) and
# digested it six times (three times each for 3 and 8 trials).
async def test_each_review_builds_its_evidence_index_and_digests_it_once(
    proposal: TaskProposal, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[int] = []
    digested: list[int] = []

    def trials(sources: Sequence[Source]) -> int:
        return sum(source.name.startswith(TRIAL) for source in sources)

    def build(sources: Mapping[str, Source], unpassed: frozenset[int], output_chars: int) -> Evidence:
        built.append(trials(tuple(sources.values())))
        return Evidence(sources, unpassed, output_chars)

    def hashed(value: tuple[Source, ...]) -> str:
        digested.append(trials(value))
        return digest(value)

    monkeypatch.setattr(review_stage, "Evidence", build)
    monkeypatch.setattr(evidence_tools_module, "digest", hashed)
    wp, _ = await run(proposal, Trials(), Reviewer())
    assert wp.terminal is Terminal.ACCEPTED and kinds(wp) == ["accept", "accept"]
    assert (built, digested) == ([3, 8], [3, 8])


# Regression: an ordinary grader's infrastructure error on a time-limited slot's output held the workpiece, and
# the explicit resume bought that slot again; its replacement solved, so the accepted evidence showed no time limit.
@pytest.mark.parametrize("answered", [True, False], ids=["answer before the cutoff", "no answer before it"])
async def test_a_limited_slot_whose_grade_failed_is_a_settled_loss_and_replays(
    proposal: TaskProposal, answered: bool
) -> None:
    broken = GradeResult(Outcome.INFRA_ERROR, None, "verifier unreachable", failure=GradingFailure.EXECUTION)

    def engine() -> Trials:
        return Trials(
            lambda trial, attempt: broken if trial == 1 else even(trial, attempt),
            stop=lambda trial, attempt: "time_limit" if trial == 1 else "stop",
            silent=frozenset() if answered else frozenset({(1, 0)}),
        )

    first, sink = engine(), BufferSink()
    wp, _ = await run(proposal, first, Reviewer(), sink=sink)
    assert wp.terminal is Terminal.ACCEPTED and first.runs.count((1, 0)) == 1 and len(first.runs) == 8
    [limited] = [t for t in wp.evidence[-1].solver if t.trial == 1]
    assert (limited.cause, limited.outcome, limited.grade) == (Cause.TIME_LIMIT, TrialOutcome.TIME_LIMIT, broken)
    accepted = wp.decisions[-1]
    assert isinstance(accepted, Accept) and accepted.reasons == (
        "Confirmation: 4 of 8 trials solved: in_band",
        "Counted unsolved without a gradable answer: 1 time_limit",
    )
    again = engine()
    log = EventLog(replay_from=local_records(sink), replayable=replayable_grade)
    resumed = await run_workpiece(proposal, setup(), bundle(again, Reviewer()), log, install=install)
    assert again.runs == [] and resumed.evidence == wp.evidence


# Regression: the attempt cap fired while a proven time-limited loss was being graded, and the row read
# as missing evidence: excluded from the denominator with an allowance, bought again on an explicit resume without one.
@pytest.mark.parametrize("allowance", [0, 1])
async def test_the_attempt_cap_after_a_proven_limit_keeps_the_settled_loss(
    proposal: TaskProposal, allowance: int
) -> None:
    engine = Trials(
        stop=lambda trial, attempt: "time_limit" if (trial, attempt) == (0, 0) else "stop",
        capped=frozenset({(0, 0)}),
    )
    config = setup(**{"budget.max_missing_trials": allowance})
    wp, log = await run(proposal, engine, Reviewer(), config)
    first = wp.evidence[0]
    assert first.solver[0].cause is Cause.TIME_LIMIT and kinds(wp)[0] == "accept" and engine.runs.count((0, 0)) == 1
    assert band_line(first, config.band)[1].startswith("1 of 3 trials solved")  # the loss is in the denominator
    grade = first.solver[0].grade
    assert grade is not None and grade.diagnostics["interrupted"]["diagnostics"]["operation"] == "attempt"
    again = Trials()
    replaces = replaced_refusals(log.records, resume=True)  # an explicit resume (``--resume-held``)
    relaunch = EventLog(replay_from=log.records, replayable=replayable_grade, replaces=replaces)
    resumed = await run_workpiece(proposal, config, bundle(again, Reviewer()), relaunch, install=install)
    assert again.runs.count((0, 0)) == 0 and resumed.evidence[0].solver[0] == first.solver[0]


@pytest.mark.parametrize("allowance", [0, 1])
async def test_an_attempt_the_cap_stops_before_any_limit_restarts_within_the_stage_bound_then_holds(
    proposal: TaskProposal, allowance: int
) -> None:
    """The watchdog interrupts slot 0 before any limit is proven, every time: the stage restarts, so the slot is bought
    ``STAGE_REPLAY_ATTEMPTS`` times in all and the workpiece then holds, whatever the missing-trial allowance."""
    engine = Trials(capped=frozenset({(0, 0)}))
    wp, _ = await run(proposal, engine, Reviewer(), setup(**{"budget.max_missing_trials": allowance}))
    assert engine.runs.count((0, 0)) == STAGE_REPLAY_ATTEMPTS and wp.terminal is Terminal.HELD
    assert not wp.evidence and isinstance(wp.decisions[-1], Incomplete)


async def test_a_revision_validates_with_fresh_trials(proposal: TaskProposal) -> None:
    """A real author revision is a new submission: its validation buys solver.k new slots under its own attempt."""
    engine = Trials(lambda trial, attempt: GradeResult(Outcome.GRADED, 1.0 if attempt else float(trial == 0)))
    reviewer = Reviewer(rounds=[[found(0)]])
    live = bundle(engine, reviewer, (*script(), *RESUBMIT))
    wp = await run_workpiece(proposal, setup(**{"budget.final_revisions": 0}), live, EventLog(), install=install)
    assert kinds(wp)[0] == "modify" and engine.runs[:6] == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]


# Regression: the pending acceptance's limit count rode along beside the confirmation's own, so one baseline and
# one new timeout read both "2 time_limit" and "1 time_limit", and a baseline timeout alone read the same line twice.
@pytest.mark.parametrize("confirmed", [(), ((4, 1),)], ids=["baseline only", "baseline and confirmation"])
async def test_a_confirmed_acceptance_counts_its_limits_once_over_the_combined_evidence(
    proposal: TaskProposal, confirmed: tuple[Slot, ...]
) -> None:
    timed_out = {(0, 0), *confirmed}
    engine = Trials(stop=lambda trial, attempt: "time_limit" if (trial, attempt) in timed_out else "stop")
    wp, _ = await run(proposal, engine, Reviewer())
    pending, final = wp.decisions
    assert isinstance(pending, Accept) and isinstance(final, Accept) and wp.terminal is Terminal.ACCEPTED
    assert "Counted unsolved without a gradable answer: 1 time_limit" in pending.reasons
    counts = [reason for reason in final.reasons if reason.startswith("Counted unsolved")]
    assert counts == [f"Counted unsolved without a gradable answer: {len(timed_out)} time_limit"]


# Regression: a max-turns stop read as a grader or task fault, because the outcome followed the review policy's
# decision on its cause, and a fault entered the placement's denominator.
@pytest.mark.parametrize(
    "cause, outcome",
    [
        (None, TrialOutcome.GRADED),
        (Cause.GENERATION_LIMIT, TrialOutcome.GENERATION_LIMIT),
        (Cause.TIME_LIMIT, TrialOutcome.TIME_LIMIT),
        (Cause.BUDGET_EXHAUSTED, TrialOutcome.BUDGET_EXHAUSTED),
        (Cause.GRADER_EXECUTION, TrialOutcome.FAULTED),
        (Cause.INVALID_TASK, TrialOutcome.FAULTED),
        (Cause.AGENT_TIMEOUT, TrialOutcome.UNSCORED),
        (Cause.GRADER_INFRA, TrialOutcome.UNSCORED),
        (Cause.HELD, TrialOutcome.UNSCORED),
    ],
)
def test_a_trial_outcome_is_its_recorded_cause_whatever_the_review_decides(
    cause: Cause | None, outcome: TrialOutcome
) -> None:
    assert TrialResult(0, GradeResult(Outcome.GRADED, 0.0), cause).outcome is outcome


def test_only_graded_and_limited_trials_enter_the_placement() -> None:
    faulted = TrialResult(2, GradeResult(Outcome.INFRA_ERROR, None), Cause.GRADER_EXECUTION)
    exhausted = TrialResult(3, GradeResult(Outcome.GRADED, 0.0), Cause.BUDGET_EXHAUSTED)
    limited = TrialResult(4, GradeResult(Outcome.GRADED, 1.0), Cause.TIME_LIMIT)
    trials = (*SOLVED[:1], TrialResult(1, GradeResult(Outcome.GRADED, 0.0)), faulted, exhausted, limited)
    result = ValidationResult(SUBMISSION.digest, trials, CONTROLS, ())
    assert band_line(result, setup().band)[1].startswith("1 of 3 trials solved")


def always_too_easy(workpiece: Workpiece, band: BandRules, budget: Budget) -> Decision:
    return Accept(BandOutcome.TOO_EASY)


@pytest.mark.parametrize("confirmation_k", [0, 5])
async def test_no_review_rule_publishes_a_terminal_acceptance_at_a_refused_band(
    proposal: TaskProposal, confirmation_k: int
) -> None:
    config = replace(
        setup(**{"band.accept_too_easy": False, "solver.confirmation_k": confirmation_k}), review=always_too_easy
    )
    engine = Trials()
    with pytest.raises(ValueError, match=r"terminal acceptance at too_easy, but band\.accept_too_easy is false"):
        await run(proposal, engine, Reviewer(), config)
    assert len(engine.runs) == 3 + confirmation_k  # a pending acceptance went on to its confirmation
