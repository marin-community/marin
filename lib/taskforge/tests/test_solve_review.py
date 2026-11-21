# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The solve-review: its evidence index and read-only tools, citation checks, replay, and how review acts on it."""

import json
import re
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import override

import pytest
from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.fake_resources import resources as stage_resources
from lib.taskforge.tests.proposal_fakes import sample_proposal
from lib.taskforge.tests.resource_doubles import CleanReviewer, FakeInference
from lib.taskforge.tests.stage_fakes import transcript
from lib.taskforge.tests.test_end_to_end import (
    Envs,
    call,
    config,
    driver_args,  # noqa: F401 (a fixture)
    plant,
    resources,
    script,
    unexpected_runtime,
)
from pydantic import TypeAdapter
from rolloutengine.models import ConversationRecord, GradedRollout, Model, Rollout, RolloutFailure, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import EnvironmentRequirements
from taskcompendium.runtime.resources import inline_resource

from taskforge import checkpoint, driver
from taskforge.coordinator import run_workpiece
from taskforge.effects.context import Context
from taskforge.effects.events import Fingerprint, ReplayPolicy, Resolved
from taskforge.effects.issues import ContextExceeded, LogFailure
from taskforge.effects.log import EventLog, JsonlSink
from taskforge.harness.protocol import ToolResult
from taskforge.harness.role import Visit, sitting
from taskforge.harness.tools.base import Outcome as ToolOutcome
from taskforge.harness.tools.evidence import Evidence, ListTool, ReadTool, SearchTool, Source
from taskforge.harness.tools.evidence import report as checked_report
from taskforge.harness.tools.submit import RevisedSubmitTool, SubmitTool
from taskforge.lowering import lower_with_settings, lowering_settings
from taskforge.model.controls import Candidate, Control, ControlKind, RewardBand
from taskforge.model.llm import Completion, FinishReason, ToolCall
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import (
    VALIDATION_RESULT,
    Accept,
    Attribution,
    BandOutcome,
    ControlResult,
    EvidenceKind,
    FindingKind,
    MissingEvidence,
    Modify,
    ReviewEnd,
    ReviewFinding,
    SolveReview,
    TrialAccount,
    TrialResult,
    ValidationResult,
)
from taskforge.model.workpiece import WORKPIECE, Submission, Workpiece
from taskforge.policy import load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.env import Env, EnvFactory, WriteFile
from taskforge.resources.inference import InferenceRequest
from taskforge.resources.rollouts import Engine, SolverRollout, solver_rollout
from taskforge.stages.rules import default_review
from taskforge.stages.solve_review import evidence, solve_review
from taskforge.wiring import build_config, install

CONTROL = Control(Candidate("12"), RewardBand(Outcome.GRADED, 1, 1), ControlKind.POSITIVE)
SUBMISSION = Submission(arithmetic_task(), (CONTROL,), "notes")
LONG = "".join(f"step {n}: carry the remainder. " for n in range(60))
OUTPUT_CHARS = 400


def trial(index: int, reward: float, *replies: str) -> tuple[TrialResult, GradedRollout | None]:
    grade = GradeResult(Outcome.GRADED, reward)
    return TrialResult(index, grade), GradedRollout(Rollout(f"run-{index}", (transcript(*replies),), "stop"), grade)


TRIALS = (trial(0, 1.0, "The answer is 12."), trial(1, 0.0, LONG, "I answer 9."))
"""trial/1's parts: 0 the task, 1 the long reply, 2 "I answer 9.", 3 its grade."""


def cite(source: str, start: int, end: int, quote: str) -> dict[str, object]:
    return {"source": source, "start": start, "end": end, "quote": quote}


def finding(*citations: dict[str, object], kind: str = "ambiguous_prompt") -> dict[str, object]:
    return {"kind": kind, "description": "six plus six reads two ways", "edit": "say 6 + 6", "citations": citations}


ACCOUNT = {"trial": 1, "attribution": "solver", "reason": "misadded"}


def report(*findings: dict[str, object], accounts: Sequence[object] = (ACCOUNT,), **more: object) -> Completion:
    return call("report", {"findings": list(findings), "accounts": list(accounts), **more}, "report")


class NoMachines(EnvFactory):
    @override
    async def create(self, requirements: EnvironmentRequirements) -> Env:
        raise AssertionError("the solve-review has no machine")


async def reviewed(
    reviewer: FakeInference,
    trials: Sequence[tuple[TrialResult, GradedRollout | None]] = TRIALS,
    log: EventLog | None = None,
    settings: Mapping[str, object] | None = None,
) -> SolveReview:
    ctx = Context(log=log or EventLog())
    policy = load_policy(DEFAULTS_DIR).overlaid({"tools.output_chars": OUTPUT_CHARS, **(settings or {})})
    bundle = replace(stage_resources(ctx), envs=NoMachines(), role_inference={"solve_review": reviewer})
    return await solve_review(ctx, build_config(policy, {}), bundle, sample_proposal(), SUBMISSION, trials)


def text(outcome: ToolOutcome) -> str:
    assert isinstance(outcome, ToolResult)
    return outcome.content


def replies(reviewer: FakeInference) -> list[str]:
    """Every tool reply the reviewer was shown, in order."""
    return [m.content for m in reviewer.requests[-1].conversation if m.role == "tool"]


async def test_a_finding_is_kept_only_when_each_quote_occurs_in_the_evidence_it_addresses() -> None:
    """Forged, unknown and out-of-range citations drop their finding with the reason, and never fail the round. A quote
    from a task file the reviewer never opened is valid: validity is the evidence it was shown, not the pages it read."""
    kept = finding(cite("trial/1", 2, 2, "I answer 9."))
    unopened = finding(cite("task/prompt.md", 1, 1, "What is six plus six?"), kind="grader_defect")
    forged = finding(cite("trial/1", 2, 2, "I answer 12."))
    unknown = finding(cite("trial/7", 0, 0, "The answer"))
    outside = finding(cite("trial/0", 5, 6, "The answer"))
    review = await reviewed(FakeInference((report(kept, unopened, forged, unknown, outside),)))
    assert review.end is ReviewEnd.REPORTED
    assert [f.citations[0].source for f in review.findings] == ["trial/1", "task/prompt.md"]
    assert [(f.citations[0].source, reason) for f, reason in review.dropped] == [
        ("trial/1", "its quote does not occur verbatim in trial/1 #2..2"),
        ("trial/7", "cites 'trial/7', which is not in the evidence"),
        ("trial/0", "cites trial/0 #5..6, outside its parts 0..2"),
    ]


async def test_a_report_must_account_for_every_trial_that_did_not_pass() -> None:
    reviewer = FakeInference((report(accounts=()), report()))
    review = await reviewed(reviewer)
    assert replies(reviewer) == [
        "Invalid report: trial 1 did not pass and needs an account. Fix these and call report again."
    ]
    assert review.end is ReviewEnd.REPORTED and review.accounts[0].trial == 1


async def pages(index: Evidence, name: str, start: int = 0) -> list[str]:
    """Every page of ``name`` from part ``start``, following each continuation; each page must advance."""
    read, out, offset = ReadTool(index), [], 0
    while True:
        out.append(page := text(await read(Context(log=EventLog()), {"source": name, "start": start, "offset": offset})))
        assert len(page) <= index.output_chars
        if (more := re.search(r"\[page ends: read start=(\d+) offset=(\d+) next\]$", page)) is None:
            return out
        assert (int(more.group(1)), int(more.group(2))) > (start, offset)
        start, offset = int(more.group(1)), int(more.group(2))


def body(read: Sequence[str]) -> str:
    """The text of consecutive pages without their part headers and continuation lines."""
    return "".join(
        re.sub(r"(?m)^--- part \d+( from character \d+)? ---\n|\n?\[(page ends|end of).*$", "", p) for p in read
    )


async def test_a_long_transcript_is_paged_whole_and_a_quote_across_a_page_boundary_is_valid() -> None:
    index = evidence(SUBMISSION, TRIALS, OUTPUT_CHARS)
    read = await pages(index, "trial/1", 1)
    assert len(read) > 5 and body(read).startswith(index.sources["trial/1"].parts[1])
    boundary = int(re.findall(r"offset=(\d+)", read[0])[0])
    across = index.sources["trial/1"].parts[1][boundary - 25 : boundary + 25]
    calls = [call("read_evidence", {"source": "trial/1", "start": 1, "offset": o}, f"r{o}") for o in (0, boundary)]
    reviewer = FakeInference((*calls, report(finding(cite("trial/1", 1, 1, across)))))
    review = await reviewed(reviewer)
    assert len(review.findings) == 1 and not review.dropped
    assert all(len(reply) <= OUTPUT_CHARS for reply in replies(reviewer))  # nothing spills to a file


async def test_search_is_bounded_and_names_what_it_left_out() -> None:
    found = text(
        await SearchTool(evidence(SUBMISSION, TRIALS, OUTPUT_CHARS))(Context(log=EventLog()), {"pattern": "carry"})
    )
    assert found.startswith("trial/1 #1: assistant: step 0: carry") and len(found) <= OUTPUT_CHARS
    assert re.search(r"\[\d+ more matches; narrow the pattern or the sources\]$", found)


async def test_the_reviewer_has_no_machine_and_only_the_evidence_tools() -> None:
    reviewer = FakeInference((report(),))
    await reviewed(reviewer)  # NoMachines refuses every machine
    assert [tool.name for tool in reviewer.requests[0].tools] == [
        "list_evidence",
        "read_evidence",
        "search_evidence",
        "report",
    ]


async def test_unchanged_evidence_replays_with_zero_calls_and_one_changed_grade_misses() -> None:
    log = EventLog()
    first = await reviewed(FakeInference((call("list_evidence", {}, "l"), report())), log=log)
    replay = FakeInference(())
    assert await reviewed(replay, log=EventLog(replay_from=log.records)) == first and not replay.requests
    changed = (TRIALS[0], (replace(TRIALS[1][0], grade=GradeResult(Outcome.GRADED, 0.1)), TRIALS[1][1]))
    missed = FakeInference(())
    with pytest.raises(RuntimeError, match="Fake inference script exhausted"):
        await reviewed(missed, changed, log=EventLog(replay_from=log.records))
    assert len(missed.requests) == 1  # the opening names the evidence version, so the first call already misses


async def test_a_trial_without_a_transcript_is_a_named_stub_and_none_at_all_skips_the_review() -> None:
    budget = TrialResult(2, GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "output budget exhausted"))
    reviewer = FakeInference((report(accounts=(ACCOUNT, {**ACCOUNT, "trial": 2})),))
    review = await reviewed(reviewer, (*TRIALS, (budget, None)))
    opening = reviewer.requests[0].conversation[1].content
    assert "trial/2: submission_failure reward=0.0 cause=None no transcript (output budget exhausted)" in opening
    assert review.missing == (
        MissingEvidence(EvidenceKind.TRANSCRIPTS, "solver trials [2] have no complete transcript"),
    )
    skipped = FakeInference(())
    alone = await reviewed(skipped, ((budget, None),))
    assert (alone.end, skipped.requests) == (ReviewEnd.SKIPPED, [])
    assert alone.missing == (MissingEvidence(EvidenceKind.TRANSCRIPTS, "solver trials [2] have no complete transcript"),)


async def test_a_review_without_a_report_is_named_missing_evidence() -> None:
    review = await reviewed(FakeInference((ContextExceeded("window full"),)))
    assert review.end is ReviewEnd.NO_REPORT and review.missing[0].kind is EvidenceKind.SOLVE_REVIEW


async def test_the_entry_point_reviews_any_set_of_trials() -> None:
    """Arbitrary trial indices, as a confirmation step's three plus five would be."""
    trials = (trial(0, 1.0, "12"), trial(5, 0.0, "I answer 9."), trial(9, 1.0, "12"))
    reviewer = FakeInference(
        (report(finding(cite("trial/5", 1, 1, "I answer 9.")), accounts=({**ACCOUNT, "trial": 5},)),)
    )
    review = await reviewed(reviewer, trials)
    opening = reviewer.requests[0].conversation[1].content
    assert [s.split(":")[0] for s in opening.splitlines() if s.startswith("trial/")] == ["trial/0", "trial/5", "trial/9"]
    assert [f.citations[0].source for f in review.findings] == ["trial/5"]


def round_with(review: SolveReview | None) -> ValidationResult:
    solver = (TrialResult(0, GradeResult(Outcome.GRADED, 1.0)), TrialResult(1, GradeResult(Outcome.GRADED, 0.0)))
    controls = (ControlResult(0, GradeResult(Outcome.GRADED, 1.0), True),)
    return ValidationResult(SUBMISSION.digest, solver, controls, (), review=review)


FOUND = ReviewFinding(FindingKind.GRADER_DEFECT, "accepts 13", "check equality", ())


@pytest.mark.parametrize("rounds,expected", [(0, Modify), (2, Accept)])
def test_review_findings_revise_while_a_round_is_left_and_are_notes_after(
    proposal: TaskProposal, rounds: int, expected: type
) -> None:
    setup = config()
    wp = Workpiece("w", proposal, round=rounds, submission=SUBMISSION)
    review = SolveReview("v", ReviewEnd.REPORTED, (FOUND,))
    decision = default_review(wp.with_(evidence=(round_with(review),)), setup.band, setup.budget)
    assert isinstance(decision, expected)
    if isinstance(decision, Modify):  # the reviewed version is the fallback the revision restores on a fault
        assert (decision.review, decision.placement, decision.fallback) == (
            (FOUND,),
            BandOutcome.IN_BAND,
            SUBMISSION.digest,
        )
    else:
        assert decision.notes == (FOUND,) and "no revision round left" in decision.reasons[-1]


@pytest.mark.parametrize(
    "review,missing",
    [
        (None, (MissingEvidence(EvidenceKind.SOLVE_REVIEW, "no solve-review ran"),)),
        (SolveReview("v", ReviewEnd.REPORTED), ()),
        (
            SolveReview("v", ReviewEnd.NO_REPORT, missing=(MissingEvidence(EvidenceKind.SOLVE_REVIEW, "window"),)),
            (MissingEvidence(EvidenceKind.SOLVE_REVIEW, "window"),),
        ),
    ],
    ids=["no review", "reviewed clean", "no report"],
)
def test_an_acceptance_names_the_evidence_it_went_without(
    proposal: TaskProposal, review: SolveReview | None, missing: tuple[MissingEvidence, ...]
) -> None:
    setup = config()
    wp = Workpiece("w", proposal, submission=SUBMISSION, evidence=(round_with(review),))
    decision = default_review(wp, setup.band, setup.budget)
    assert isinstance(decision, Accept) and decision.missing == missing and not decision.notes


class Rounds(Engine):
    """Round 0 solves one trial of two (in band); later rounds grade every trial with ``later``."""

    def __init__(self, later: GradeResult) -> None:
        self.later = later

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        assert run_id is not None
        trial, attempt = (int(part) for part in run_id.split(":")[-2:])
        grade = GradeResult(Outcome.GRADED, float(trial == 0)) if attempt == 0 else self.later
        return GradedRollout(Rollout(run_id, (transcript(f"trial {trial} answers 12"),), "stop"), grade)


class FindingReviewer(CleanReviewer):
    """Reports one quoted finding in its first review, and none after."""

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        clean = await super().complete(request)
        if len(self.requests) > 1:
            return clean
        arguments = {
            **json.loads(clean.tool_calls[0].arguments),
            "findings": [finding(cite("trial/1", 1, 1, "answers 12"))],
        }
        return call("report", arguments, "review")


async def revised_after_review(
    proposal: TaskProposal, later: GradeResult, declined: Sequence[Mapping[str, object]]
) -> Workpiece:
    resubmit = call("submit", {"notes": "revised", "declined": list(declined)}, "submit")
    bundle = resources(Envs(), (*script(), *script()[:3], resubmit))
    bundle = replace(
        bundle,
        rollouts=replace(bundle.rollouts, engine=Rounds(later)),
        role_inference={"solve_review": FindingReviewer()},
    )
    wp = await run_workpiece(proposal, config(), bundle, EventLog(), install=install)
    assert WORKPIECE.validate_json(WORKPIECE.dump_json(wp)) == wp
    first = wp.decisions[0]
    assert isinstance(first, Modify) and len(first.review) == 1 and first.placement is BandOutcome.IN_BAND
    return wp


async def test_a_clean_revision_after_review_ships_at_its_own_placement_and_records_what_the_author_declined(
    proposal: TaskProposal,
) -> None:
    """Owner's ruling (2026-11-13): a band change is never a regression. The in-band reviewed task is revised once;
    the clean revision lands too easy and is accepted as the revision, tagged, not restored."""
    wp = await revised_after_review(
        proposal, GradeResult(Outcome.GRADED, 1.0), [{"finding": 0, "reason": "the reply is right"}]
    )
    final = wp.decisions[-1]
    assert isinstance(final, Accept) and final.restore is None and wp.accepted_with is BandOutcome.TOO_EASY
    assert "Clean revision placed too_easy, not in_band: a placement change is tagged, not reverted" in final.reasons
    assert wp.submission is not None and wp.submission.notes == "revised"
    assert [(d.finding.kind, d.reason) for d in wp.submission.declined] == [
        (FindingKind.AMBIGUOUS_PROMPT, "the reply is right")
    ]


async def test_a_faulted_revision_after_review_restores_the_reviewed_version(proposal: TaskProposal) -> None:
    """The restore goes through the existing fallback, named by digest and band; the round left is spent."""
    wp = await revised_after_review(proposal, GradeResult(Outcome.INVALID_TASK, None, "bad task"), [])
    final, first = wp.decisions[-1], wp.decisions[0]
    assert isinstance(final, Accept) and isinstance(first, Modify) and final.restore == first.fallback
    assert wp.submission is not None and wp.submission.notes == "done" and wp.accepted_with is None


async def test_a_decline_must_name_a_finding_once() -> None:
    tool = RevisedSubmitTool(SubmitTool(lambda _: (), "/t.json", "/c.json"), (FOUND,))
    for declined in ([{"finding": 1, "reason": "no"}], [{"finding": 0, "reason": "a"}, {"finding": 0, "reason": "b"}]):
        outcome = await tool(Context(log=EventLog()), {"declined": declined})
        assert text(outcome) == "declined names each finding at most once, by its number 0..0"


def test_the_recorded_solver_rollouts_are_one_projection_of_the_journal() -> None:
    policy = load_policy(DEFAULTS_DIR)
    request = SolverRollout(lower_with_settings(arithmetic_task(), lowering_settings(policy), {}), 3, 0)
    recorded = Resolved(request, (), TRIALS[0][1])
    assert solver_rollout(recorded) == (request, TRIALS[0][1]) and solver_rollout(Resolved(request, (), None)) is None


async def test_a_format_6_root_is_refused_before_claim_and_left_untouched(
    driver_args: Namespace,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A format-6 acceptance, recorded at the base, names no missing evidence although no solve-review ran, and
    resuming a format-6 round would buy a review its history never had; such a root waits for a migration."""
    destination = plant(driver_args, b"", checkpoint.JOURNAL_FORMAT - 1)
    before = sorted((p, p.read_bytes()) for p in destination.rglob("*") if p.is_file())
    monkeypatch.setattr(driver, "runtime", unexpected_runtime)
    with pytest.raises(LogFailure, match="has journal format 6, not 7"):
        await driver.one(driver_args)
    assert sorted((p, p.read_bytes()) for p in destination.rglob("*") if p.is_file()) == before


def test_a_round_without_review_fields_keeps_its_bytes() -> None:
    """The new fields are omitted when empty, so a record that never had them serializes as it did."""
    plain = round_with(None)
    assert b"review" not in VALIDATION_RESULT.dump_json(plain)
    decision = json.loads(TypeAdapter(Accept).dump_json(Accept(BandOutcome.IN_BAND)))
    assert not {"missing", "notes"} & set(decision)


def lowered() -> TaskRolloutSpec:
    return lower_with_settings(arithmetic_task(), lowering_settings(load_policy(DEFAULTS_DIR)), {})


async def test_a_cold_reload_of_the_journal_replays_the_review_with_zero_calls(tmp_path: Path) -> None:
    """Review 2026-11-21 (n11 P1): a reloaded grade came back with its detail keys sorted, which changed the rendered
    evidence and its version, so a default-policy resume paid the reviewer again."""
    grade = GradeResult(Outcome.GRADED, 0.0, detail={"z": 1, "a": {"y": 2, "b": 3}, "cut": 3.0})  # reloads as 3
    rollout = GradedRollout(Rollout("run-1", (transcript(LONG, "I answer 9."),), "stop"), grade)
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
    log = EventLog(sink=sink)
    await log.record(
        Resolved(SolverRollout(lowered(), 1, 0), (Fingerprint("rollouts", "d", ReplayPolicy.CACHE),), rollout)
    )
    first = await reviewed(
        FakeInference((call("list_evidence", {}, "l"), report())), (TRIALS[0], (TrialResult(1, grade), rollout)), log
    )
    await log.persist()
    records = local_records(sink)
    reloaded = next(pair[1] for record in records if (pair := solver_rollout(record.event)) is not None)
    cold = FakeInference(())
    again = await reviewed(cold, (TRIALS[0], (TrialResult(1, reloaded.grade), reloaded)), EventLog(replay_from=records))
    assert again == first and not cold.requests


def parallel(first: str, second: str) -> GradedRollout:
    """One assistant turn with two tool calls, then their replies naming ``first`` and ``second``."""
    calls = [{"id": name, "type": "function", "function": {"name": "check", "arguments": "{}"}} for name in "ab"]
    messages = (
        {"role": "user", "content": "Solve the task."},
        {"role": "assistant", "content": "", "tool_calls": calls},
        {"role": "tool", "tool_call_id": first, "content": "PASS"},
        {"role": "tool", "tool_call_id": second, "content": "FAIL"},
    )
    grade = GradeResult(Outcome.GRADED, 0.0)
    return GradedRollout(Rollout("run-1", (ConversationRecord(messages, (), (), (), None),), "stop"), grade)


def test_a_tool_reply_keeps_the_call_it_answers() -> None:
    """Review 2026-11-21 (n11): two parallel calls whose replies swapped ids rendered identically."""
    one, other = (evidence(SUBMISSION, ((TRIALS[1][0], parallel(*ids)),), OUTPUT_CHARS) for ids in ("ab", "ba"))
    assert one.version != other.version
    assert "tool (reply to a):\nPASS" in one.sources["trial/1"].parts[2]


async def test_two_parallel_evidence_reads_replay_cold_with_zero_calls(tmp_path: Path) -> None:
    reads = tuple(
        ToolCall(f"r{start}", "read_evidence", json.dumps({"source": "trial/1", "start": start})) for start in (1, 2)
    )
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
    log = EventLog(sink=sink)
    first = await reviewed(FakeInference((Completion("", reads, FinishReason.TOOL_CALLS), report())), log=log)
    await log.persist()
    cold = FakeInference(())
    assert await reviewed(cold, log=EventLog(replay_from=local_records(sink))) == first and not cold.requests


async def test_an_opening_without_the_version_slot_still_misses_on_changed_evidence() -> None:
    """Review 2026-11-21 (n11 P1): a legal user prompt without ``{version}`` replayed a stale review after a solver's
    answer changed from 9 to 8; the evidence version is part of the request whatever the template renders."""
    overlay = {"solve_review.user": "{task}\n{proposal}\n{evidence}"}
    log = EventLog()
    await reviewed(FakeInference((report(),)), log=log, settings=overlay)
    changed = (TRIALS[0], trial(1, 0.0, LONG, "I answer 8."))
    missed = FakeInference(())
    with pytest.raises(RuntimeError, match="Fake inference script exhausted"):
        await reviewed(missed, changed, EventLog(replay_from=log.records), settings=overlay)


def with_resources(**groups: tuple[tuple[str, bytes], ...]) -> Submission:
    task = arithmetic_task()
    files = {group: tuple(inline_resource(path, data) for path, data in entries) for group, entries in groups.items()}
    return replace(SUBMISSION, task=task.model_copy(update={"resources": task.resources.model_copy(update=files)}))


def test_every_source_has_its_own_address_and_the_bytes_it_was_built_from() -> None:
    """Review 2026-11-21 (n11 P1): a worker and a verifier file both named ``same.txt`` collapsed to one source, a
    resource named ``prompt.md`` replaced the task prompt, one named like a trial raised, and distinct non-UTF-8 bytes
    decoded to one text and one version."""

    def index(worker: bytes) -> Evidence:
        named = (("same.txt", worker), ("prompt.md", b"not the prompt"), ("trial-0", b"x"), ("trial/1", b"y"))
        return evidence(with_resources(worker=named, verifier=(("same.txt", b"v"),)), TRIALS, OUTPUT_CHARS)

    sources = index(b"w").sources
    assert sources["resource/worker/same.txt"].parts == ("w",) and sources["resource/verifier/same.txt"].parts == ("v",)
    assert sources["task/prompt.md"].parts == ("User:", "What is six plus six?")
    assert sources["trial/1"].parts[-1].startswith("grader:")
    assert index(b"\xff").version != index(b"\xfe").version
    assert index(b"\xff").sources["resource/worker/same.txt"].parts == ()
    assert "not UTF-8" in index(b"\xff").sources["resource/worker/same.txt"].summary


def test_an_account_must_name_a_trial_of_the_review() -> None:
    """A resource named like a trial does not make a fabricated account valid."""
    index = evidence(with_resources(worker=(("trial-99", b"x"), ("trial/99", b"y"))), TRIALS, OUTPUT_CHARS)
    accounts = (TrialAccount(1, Attribution.SOLVER, "misadded"), TrialAccount(99, Attribution.SOLVER, "invented"))
    outcome = text(checked_report(index, (), accounts))
    assert outcome == "Invalid report: trial 99 is not a solver trial. Fix these and call report again."


async def test_a_page_is_bounded_with_its_headers_and_trailer_and_always_advances() -> None:
    """Review 2026-11-21 (n11 P1): parts of 178 and 900 characters made a 1,143-character page under a 400-character
    cap, continuing at offset -12."""
    parts = ("a" * 178, "b" * 900, "c" * 3)
    index = Evidence({"s": Source("s", "s", parts)}, frozenset(), OUTPUT_CHARS)
    read = await pages(index, "s")
    assert read[-1].endswith("[end of source]") and body(read).replace("\n", "") == "".join(parts)


async def test_no_reply_to_the_machine_less_reviewer_spills_to_a_file() -> None:
    """The real harness journalled a ``WriteFile`` for an oversized reply and pointed the reviewer at a file reader it
    does not have; an unknown source's error listed every source."""
    reads = [
        call("read_evidence", {"source": "trial/1", "start": 1}, "r"),
        call("read_evidence", {"source": "x" * 900}, "u"),
    ]
    log = EventLog()
    reviewer = FakeInference((*reads, call("search_evidence", {"pattern": "."}, "s"), report()))
    await reviewed(reviewer, log=log)
    assert all(len(reply) <= OUTPUT_CHARS for reply in replies(reviewer))
    assert not [record for record in log.records if isinstance(record.event, WriteFile)]


async def test_a_machine_less_visit_refuses_a_tool_that_does_not_bound_its_replies() -> None:
    ctx = Context(log=EventLog())
    unbounded = {"list_evidence": ListTool(evidence(SUBMISSION, TRIALS, OUTPUT_CHARS))}
    with pytest.raises(ValueError, match="bound"):
        async with sitting(ctx, config().solve_review, Visit("v", "v", None, tools=unbounded), stage_resources(ctx)):
            pass


async def test_a_failure_attributed_to_a_defect_needs_a_finding_that_survives_its_citations() -> None:
    """Review 2026-11-21 (n11 P1): an account blaming a grader defect beside no finding shipped as a clean review."""
    defect = {**ACCOUNT, "attribution": "defect", "reason": "the grader rejects 12"}
    forged = finding(cite("trial/1", 2, 2, "I answer 12."))
    reviewer = FakeInference((report(accounts=(defect,)), report(forged, accounts=(defect,)), report()))
    review = await reviewed(reviewer)
    refused = "Invalid report: trial 1 is attributed to a task defect, but no finding survived its citation check"
    assert len(replies(reviewer)) == 2 and all(reply.startswith(refused) for reply in replies(reviewer))
    assert "its quote does not occur verbatim" in replies(reviewer)[1]
    assert review.accounts[0].attribution is Attribution.SOLVER and not review.findings


@pytest.mark.parametrize("field", ["description", "edit"])
async def test_a_finding_or_account_needs_prose_beyond_whitespace(field: str) -> None:
    """A quoted finding with a whitespace-only description and edit bought an author round; the quote stays as
    entered."""
    blank = {**finding(cite("trial/1", 2, 2, "  I answer 9.  ")), field: "   "}
    reviewer = FakeInference((report(blank), report(accounts=({**ACCOUNT, "reason": " \n"},)), report()))
    await reviewed(reviewer)
    shown = replies(reviewer)
    assert len(shown) == 2 and all("report arguments failed validation" in reply for reply in shown)


async def test_a_decline_needs_a_reason_beyond_whitespace() -> None:
    tool = RevisedSubmitTool(SubmitTool(lambda _: (), "/t.json", "/c.json"), (FOUND,))
    outcome = await tool(Context(log=EventLog()), {"declined": [{"finding": 0, "reason": "   "}]})
    assert "submit arguments failed validation" in text(outcome)


async def test_an_interrupted_recording_is_reviewed_by_its_prefix_and_named_incomplete() -> None:
    """Review 2026-11-21 (n11 P2): two interrupted recordings of two and three messages were skipped as missing."""
    cut = RolloutFailure("OutputBudgetExhausted", {"z": 1, "a": 2})

    def interrupted(index: int, *replies: str) -> tuple[TrialResult, GradedRollout]:
        result, rollout = trial(index, 0.0, *replies)
        assert rollout is not None
        return result, replace(rollout, rollout=replace(rollout.rollout, failure=cut))

    trials = (interrupted(0, "I add."), interrupted(1, "I add.", "I answer 9."))
    accounts = [{**ACCOUNT, "trial": index} for index in (0, 1)]
    reviewer = FakeInference((report(finding(cite("trial/1", 2, 2, "I answer 9.")), accounts=accounts),))
    review = await reviewed(reviewer, trials)
    opening = reviewer.requests[0].conversation[1].content
    assert review.end is ReviewEnd.REPORTED and len(review.findings) == 1 and opening.count("interrupted") == 2
    assert review.missing == (
        MissingEvidence(EvidenceKind.TRANSCRIPTS, "solver trials [0, 1] have no complete transcript"),
    )
    assert evidence(SUBMISSION, trials, OUTPUT_CHARS).sources["trial/1"].parts[3] == (
        "[interrupted: the recording ends here (OutputBudgetExhausted)]"
    )
