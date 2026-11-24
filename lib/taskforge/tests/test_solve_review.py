# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The solve-review: its evidence index and read-only tools, citation checks, replay, and how review acts on it."""

import json
import re
import subprocess
import sys
from argparse import Namespace
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import override

import pytest
from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.fake_resources import resources as stage_resources
from lib.taskforge.tests.proposal_fakes import sample_proposal
from lib.taskforge.tests.resource_doubles import CleanReviewer, FakeInference
from lib.taskforge.tests.stage_fakes import SolverEngine, transcript
from lib.taskforge.tests.test_end_to_end import (
    SETTINGS,
    Envs,
    call,
    config,
    driver_args,  # noqa: F401 (a fixture)
    plant,
    published,
    published_workpiece,
    resources,
    script,
    serve,
    unexpected_runtime,
)
from pydantic import TypeAdapter
from rolloutengine.models import ConversationRecord, GradedRollout, Model, Rollout, RolloutFailure, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AssistantToolCalls,
    CommandSemantics,
    ConversationToolCall,
    DockerBuildContext,
    EnvironmentRequirements,
    JsonAnswer,
    TaskSpec,
)
from taskcompendium.models import ToolResult as ConversationToolResult
from taskcompendium.runtime.resources import inline_resource

from taskforge import checkpoint, driver
from taskforge.checkpoint import CheckpointSink
from taskforge.config import TaskforgeConfig
from taskforge.coordinator import advance, apply_decision, run_workpiece
from taskforge.effects.context import Context
from taskforge.effects.events import Fingerprint, ReplayPolicy, Resolved
from taskforge.effects.issues import ContextExceeded, Hold, LogFailure
from taskforge.effects.log import BufferSink, EventLog, Record
from taskforge.harness.protocol import Stopped, StopReason, ToolResult
from taskforge.harness.tools.base import Outcome as ToolOutcome
from taskforge.harness.tools.evidence import PAGE_ROOM, Evidence, ListTool, ReadTool, SearchTool, Source
from taskforge.harness.tools.evidence import report as checked_report
from taskforge.harness.tools.submit import RevisedSubmitTool, SubmitTool
from taskforge.lowering import lower_with_settings, lowering_settings
from taskforge.model.content_hash import digest
from taskforge.model.controls import Candidate, Control, ControlKind, RewardBand
from taskforge.model.llm import Completion, FinishReason, Message, Role, ToolCall
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import (
    VALIDATION_RESULT,
    Accept,
    AdversaryResult,
    AdversaryStatus,
    Attribution,
    BandOutcome,
    Cause,
    ControlResult,
    Decision,
    EvidenceKind,
    FindingKind,
    MissingEvidence,
    Modify,
    ReviewEnd,
    ReviewFinding,
    SearchLimit,
    SolveReview,
    TrialAccount,
    TrialResult,
    ValidationResult,
)
from taskforge.model.stage import Stage
from taskforge.model.workpiece import WORKPIECE, Conversation, Fallback, Submission, Terminal, Workpiece
from taskforge.policy import Policy, load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.bundle import Resources
from taskforge.resources.env import Env, EnvFactory, WriteFile
from taskforge.resources.inference import InferenceRequest
from taskforge.resources.rollouts import Engine, SolverRollout
from taskforge.stages.rules import INSTRUCTIONS, default_review
from taskforge.stages.solve_review import evidence, solve_review
from taskforge.wiring import build_config, install

CONTROL = Control(Candidate("12"), RewardBand(Outcome.GRADED, 1, 1), ControlKind.POSITIVE)
SUBMISSION = Submission(arithmetic_task(), (CONTROL,), "notes")
LONG = "".join(f"step {n}: carry the remainder. " for n in range(60))
OUTPUT_CHARS = 400
EARLIER_USER = (
    "Task manifest (resource paths, byte sizes and roles; payloads omitted; read files with read_evidence):\n{task}\n\n"
    "Proposal the task was built from:\n{proposal}\n\nEvidence version {version}, its sources:\n{evidence}\n"
)
"""The user prompt as checked in before the version moved outside the template."""


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
    submission: Submission = SUBMISSION,
) -> SolveReview:
    ctx = Context(log=log or EventLog())
    policy = load_policy(DEFAULTS_DIR).overlaid({"tools.output_chars": OUTPUT_CHARS, **(settings or {})})
    bundle = replace(stage_resources(ctx), envs=NoMachines(), role_inference={"solve_review": reviewer})
    return await solve_review(ctx, build_config(policy, {}), bundle, sample_proposal(), submission, trials)


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
    assert "trial/2: submission_failure reward=0 cause=None no transcript (output budget exhausted)" in opening
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
    decision = default_review(replace(wp, evidence=(round_with(review),)), setup.band, setup.budget)
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
    """A band change is never a regression. The in-band reviewed task is revised once;
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


@pytest.mark.parametrize("found", [6, 7])
async def test_an_earlier_reviewed_format_root_is_refused_before_claim_and_left_untouched(
    driver_args: Namespace,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    found: int,
) -> None:
    """A format-6 acceptance, recorded at the base, names no missing evidence although no solve-review ran, and
    resuming a format-6 round would buy a review its history never had; such a root waits for a migration. A format-7
    acceptance likewise names no missing confirmation, and its resumed round would buy one."""
    destination = plant(driver_args, b"", found)
    before = sorted((p, p.read_bytes()) for p in destination.rglob("*") if p.is_file())
    monkeypatch.setattr(driver, "runtime", unexpected_runtime)
    with pytest.raises(LogFailure, match=f"has journal format {found}, not {checkpoint.JOURNAL_FORMAT}"):
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


async def test_a_cold_reload_of_the_journal_replays_the_review_with_zero_calls() -> None:
    """Regression: a reloaded grade came back with its detail keys sorted, which changed the rendered
    evidence and its version, so a default-policy resume paid the reviewer again."""
    grade = GradeResult(Outcome.GRADED, 0.0, detail={"z": 1, "a": {"y": 2, "b": 3}, "cut": 3.0})  # reloads as 3
    rollout = GradedRollout(Rollout("run-1", (transcript(LONG, "I answer 9."),), "stop"), grade)
    sink = BufferSink()
    log = EventLog(sink=sink)
    await log.record(
        Resolved(SolverRollout(lowered(), 1, 0), (Fingerprint("rollouts", "d", ReplayPolicy.CACHE),), rollout)
    )
    first = await reviewed(
        FakeInference((call("list_evidence", {}, "l"), report())), (TRIALS[0], (TrialResult(1, grade), rollout)), log
    )
    await log.persist()
    records = local_records(sink)
    reloaded = next(
        r.event.value for r in records if isinstance(r.event, Resolved) and isinstance(r.event.request, SolverRollout)
    )
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
    """Regression: two parallel calls whose replies swapped ids rendered identically."""
    one, other = (evidence(SUBMISSION, ((TRIALS[1][0], parallel(*ids)),), OUTPUT_CHARS) for ids in ("ab", "ba"))
    assert one.version != other.version
    assert "tool (reply to a):\nPASS" in one.sources["trial/1"].parts[2]


async def test_two_parallel_evidence_reads_replay_cold_with_zero_calls() -> None:
    reads = tuple(
        ToolCall(f"r{start}", "read_evidence", json.dumps({"source": "trial/1", "start": start})) for start in (1, 2)
    )
    sink = BufferSink()
    log = EventLog(sink=sink)
    first = await reviewed(FakeInference((Completion("", reads, FinishReason.TOOL_CALLS), report())), log=log)
    await log.persist()
    cold = FakeInference(())
    assert await reviewed(cold, log=EventLog(replay_from=local_records(sink))) == first and not cold.requests


async def test_an_opening_without_the_version_slot_still_misses_on_changed_evidence() -> None:
    """Regression: a legal user prompt without ``{version}`` replayed a stale review after a solver's
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
    """Regression: a worker and a verifier file both named ``same.txt`` collapsed to one source, a
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
    """Regression: parts of 178 and 900 characters made a 1,143-character page under a 400-character
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


async def test_a_failure_attributed_to_a_defect_needs_a_finding_that_survives_its_citations() -> None:
    """Regression: an account blaming a grader defect beside no finding shipped as a clean review."""
    defect = {**ACCOUNT, "attribution": "defect", "reason": "the grader rejects 12"}
    forged = finding(cite("trial/1", 2, 2, "I answer 12."))
    reviewer = FakeInference((report(accounts=(defect,)), report(forged, accounts=(defect,)), report()))
    review = await reviewed(reviewer)
    refused = "Invalid report: trial 1 is attributed to a task defect, but no surviving finding cites trial/1"
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
    """Regression: two interrupted recordings of two and three messages were skipped as missing."""
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


def settings_config(**settings: object) -> TaskforgeConfig:
    return build_config(load_policy(DEFAULTS_DIR).overlaid({**SETTINGS, **settings}), {})


def decided(wp: Workpiece, setup: TaskforgeConfig) -> tuple[Decision, Workpiece]:
    decision = default_review(wp, setup.band, setup.budget)
    return decision, apply_decision(wp, decision, setup)


def reviewed_round(proposal: TaskProposal, setup: TaskforgeConfig) -> Workpiece:
    """Round 0 in band with finding A: revised, its reviewed version stored as the fallback."""
    session = Conversation((Message(Role.SYSTEM, "author"),), ())
    first = round_with(SolveReview("v0", ReviewEnd.REPORTED, (FOUND,)))
    _, wp = decided(Workpiece("w", proposal, submission=SUBMISSION, evidence=(first,), session=session), setup)
    assert wp.round == 1 and wp.fallback is not None
    return wp


def test_a_restore_carries_the_review_of_the_round_it_restores(proposal: TaskProposal) -> None:
    """Regression: a same-digest revision with other author notes faulted with finding B; the restored
    round-0 submission shipped with B, and a variation with the revision's missing transcripts."""
    setup = config()
    wp = reviewed_round(proposal, setup)
    other = ReviewFinding(FindingKind.ANSWER_LEAK, "B", "hide it", ())
    gone = (MissingEvidence(EvidenceKind.TRANSCRIPTS, "solver trials [1] have no complete transcript"),)
    faulted = replace(
        round_with(SolveReview("v1", ReviewEnd.REPORTED, (other,), missing=gone)),
        controls=(ControlResult(0, GradeResult(Outcome.GRADED, 0.0), False),),
    )
    revision = replace(SUBMISSION, notes="revised")
    assert revision.digest == SUBMISSION.digest
    decision, shipped = decided(replace(wp, submission=revision, evidence=(*wp.evidence, faulted)), setup)
    assert isinstance(decision, Accept) and decision.restore == SUBMISSION.digest
    assert shipped.submission == SUBMISSION and (decision.notes, decision.missing) == ((FOUND,), ())


def test_a_clean_same_digest_revision_is_the_occurrence_that_ships(proposal: TaskProposal) -> None:
    setup = settings_config(**{"budget.max_modify_rounds": 1})
    wp = reviewed_round(proposal, setup)
    clean = round_with(SolveReview("v1", ReviewEnd.REPORTED))
    revision = replace(SUBMISSION, notes="revised")
    decision, shipped = decided(replace(wp, submission=revision, evidence=(*wp.evidence, clean)), setup)
    assert isinstance(decision, Accept) and decision.restore is None and shipped.submission == revision
    assert (decision.notes, decision.missing) == ((), ())


def test_an_author_stopping_restores_the_stored_fallback(proposal: TaskProposal) -> None:
    """Regression: the author hit its context limit in a review revision and the task was abandoned
    with a valid fallback. An author stopping reverts to the fallback."""
    setup = settings_config(**{"budget.max_modify_rounds": 2})
    wp = reviewed_round(proposal, setup)
    session = Conversation((Message(Role.SYSTEM, "author"),), ())
    stopped, after = advance(wp, Stage.AUTHOR, Stopped(StopReason.CONTEXT_LIMIT, session, "window full"), setup)
    decision = stopped.decisions[-1]
    assert after is None and stopped.terminal is Terminal.ACCEPTED and stopped.submission == SUBMISSION
    assert isinstance(decision, Accept) and decision.restore == SUBMISSION.digest and decision.notes == (FOUND,)
    assert any("author stopped" in reason for reason in decision.reasons)
    # Regression: the stop, not an exhausted budget, is why the notes are notes.
    assert setup.budget.max_modify_rounds > wp.round and not any("no revision round" in r for r in decision.reasons)


def test_a_stored_fallback_leaves_the_remaining_rounds_for_review(proposal: TaskProposal) -> None:
    """Regression: with three rounds, a fresh finding in round 1 was accepted as "no revision round
    left"; only the shared round count ends review revisions."""
    setup = settings_config(**{"budget.max_modify_rounds": 3})
    wp = reviewed_round(proposal, setup)
    fresh = ReviewFinding(FindingKind.ANSWER_LEAK, "B", "hide it", ())
    again = replace(SUBMISSION, notes="revised")
    decision, after = decided(
        replace(
            wp, submission=again, evidence=(*wp.evidence, round_with(SolveReview("v1", ReviewEnd.REPORTED, (fresh,))))
        ),
        setup,
    )
    assert isinstance(decision, Modify) and decision.review == (fresh,) and after.round == 2


class IntegerRewards(SolverEngine):
    """The engine's own integer rewards, 1 for trial 0 and 0 for trial 1; a reloaded grade holds them as floats."""

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        graded = await super().run_task(spec, model, run_id=run_id)
        assert graded.grade.reward is not None
        return replace(graded, grade=replace(graded.grade, reward=int(graded.grade.reward)))


async def test_a_cold_driver_resume_buys_no_second_review(
    driver_args: Namespace,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: through the default driver, engine rewards 1 and 0 reloaded as 1.0 and
    0.0 in the trial summaries, and task labels written z then a reloaded a then z in the manifest, so a cold resume
    paid the reviewer again. Everything the reviewer sees is rendered canonically."""
    envs = Envs()
    check = inline_resource("check.sh", b"exit 0\n").model_copy(update={"mode": "0755", "mtime_ns": 1_700_000_000_123})
    task = arithmetic_task().model_copy(update={"labels": {"z": "last", "a": "first"}})
    task = task.model_copy(update={"resources": task.resources.model_copy(update={"verifier": (check,)})})
    draft = envs.commands["write-draft"]
    written = {**draft.writes, PurePosixPath("/workspace/task.json"): task.model_dump_json().encode()}
    envs.commands["write-draft"] = replace(draft, writes=written)
    first, again = CleanReviewer(), CleanReviewer()
    live = resources(envs, script())
    live = replace(
        live, rollouts=replace(live.rollouts, engine=IntegerRewards(2)), role_inference={"solve_review": first}
    )
    cold = replace(resources(envs, ()), role_inference={"solve_review": again})
    attempts = iter((live, cold))

    @asynccontextmanager
    async def fake_runtime(
        policy: Policy, *_: object, **_wiring: object
    ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
        yield build_config(policy, {}), next(attempts)

    async def preempted(*_: object) -> None:
        """Only the periodic checkpoints reach storage, so the second attempt resumes from the reloaded journal."""

    monkeypatch.setattr(driver, "runtime", fake_runtime)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    final = CheckpointSink.finish
    monkeypatch.setattr(CheckpointSink, "finish", preempted)
    await driver.one(driver_args)
    monkeypatch.setattr(CheckpointSink, "finish", final)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    assert published(destination).head.status.terminal is None and len(first.requests) == 1
    await driver.one(driver_args)
    assert published_workpiece(destination).terminal is Terminal.ACCEPTED and not again.requests


async def test_the_earlier_template_with_its_version_slot_still_renders() -> None:
    """Regression: the user prompt checked in before the unconditional version suffix
    names ``{version}``; used as an overlay it raised KeyError. It renders, and the suffix is still appended."""
    overlay = {"solve_review.user": EARLIER_USER}
    reviewer = FakeInference((report(),))
    await reviewed(reviewer, settings=overlay)
    opening = reviewer.requests[0].conversation[1].content
    version = evidence(SUBMISSION, TRIALS, OUTPUT_CHARS).version
    assert f"Evidence version {version}, its sources" in opening and opening.endswith(f"Evidence version: {version}")


def test_a_file_s_mode_and_build_inputs_are_evidence() -> None:
    """Regression: making verifier ``check.sh`` executable changed the submission digest
    but not the evidence, so the old report was reused; an equal-length Dockerfile edit was invisible too. Each file
    the task carries is shown with the signature the task review signs, build inputs included."""
    script = inline_resource("check.sh", b"exit 0\n")

    def index(mode: str, dockerfile: bytes, mtime: int | None = None) -> Evidence:
        submission = with_resources(verifier=())
        task = submission.task
        build = DockerBuildContext(files=(inline_resource("Dockerfile", dockerfile),))
        environment = task.environment_requirements.model_copy(
            update={"docker_build": build, "command_semantics": CommandSemantics.LINUX_PROCESS}
        )
        resources = task.resources.model_copy(
            update={"verifier": (script.model_copy(update={"mode": mode, "mtime_ns": mtime}),)}
        )
        task = task.model_copy(update={"resources": resources, "environment_requirements": environment})
        return evidence(replace(submission, task=task), TRIALS, OUTPUT_CHARS)

    plain, executable, rebased = index("0644", b"FROM a:1"), index("0755", b"FROM a:1"), index("0644", b"FROM b:1")
    assert len({plain.version, executable.version, rebased.version}) == 3
    assert '"mode": "0755"' in executable.sources["resource/verifier/check.sh"].summary
    assert rebased.sources["resource/actor_build/Dockerfile"].parts == ("FROM b:1",)
    # Coordinator question C: mtime is the task's declared ``mtime_ns``, never a checkout's stat, so it is the same live,
    # on a fresh resume and in another process (the cold driver and two-process tests carry one); changing it is a
    # change to the task, as its digest says.
    touched = index("0644", b"FROM a:1", 1_700_000_000_123)
    assert (
        touched.version != plain.version
        and '"mtime_ns": 1700000000123' in touched.sources["resource/verifier/check.sh"].summary
    )


async def listing(index: Evidence) -> list[str]:
    """Every listing page from the first, following each continuation; each page must advance and fit a reply."""
    tool, out, start, offset = ListTool(index), [], 0, 0
    while True:
        out.append(page := text(await tool(Context(log=EventLog()), {"start": start, "offset": offset})))
        assert len(page) <= index.output_chars
        if (more := re.search(r"list_evidence start=(\d+)(?: offset=(\d+))?\]$", page)) is None:
            return out
        assert (int(more.group(1)), int(more.group(2) or 0)) > (start, offset)
        start, offset = int(more.group(1)), int(more.group(2) or 0)


async def test_every_listed_address_is_recoverable_at_the_smallest_cap() -> None:
    """Regression: at cap 240 a 187-character resource address was cut to 119 characters
    and the listing's continuation reported zero sources left. A summary longer than a page continues within itself."""
    path = "/".join(["deeply-nested-directory"] * 7) + "/check.sh"
    index = evidence(with_resources(verifier=((path, b"exit 0\n"),)), TRIALS, 2 * PAGE_ROOM)
    joined = "".join(re.sub(r"\n?\[(\d+ more sources|source \d+ continues).*$", "", p) for p in await listing(index))
    for source in index.sources.values():
        assert source.summary in joined
    assert f"resource/verifier/{path}" in joined


def test_a_cap_too_small_for_an_evidence_page_is_refused_when_wired() -> None:
    """Regression: caps under 240 failed only after paid work, even when no trial left
    any evidence. The configuration is refused before a run starts, naming the key; zero trials still skip."""
    with pytest.raises(ValueError, match=r"tools\.output_chars 239 .* at least 240"):
        build_config(load_policy(DEFAULTS_DIR).overlaid({"tools.output_chars": 239}), {})
    build_config(load_policy(DEFAULTS_DIR).overlaid({"tools.output_chars": 240}), {})
    assert evidence(SUBMISSION, (), 1).sources


async def test_a_defect_account_needs_a_surviving_finding_that_cites_its_trial() -> None:
    """Regression: a report blaming failed trial 1 on a defect passed beside a finding
    that cited only successful trial 0. A surviving finding must cite the blamed trial; a task-file
    citation may support it beside the trial's."""
    defect = {**ACCOUNT, "attribution": "defect", "reason": "the prompt reads two ways"}
    elsewhere = finding(cite("trial/0", 1, 1, "The answer is 12."))
    supported = finding(cite("trial/1", 2, 2, "I answer 9."), cite("task/prompt.md", 1, 1, "six plus six"))
    reviewer = FakeInference((report(elsewhere, accounts=(defect,)), report(supported, accounts=(defect,))))
    review = await reviewed(reviewer)
    assert replies(reviewer)[0].startswith("Invalid report: trial 1 is attributed to a task defect, but no surviving")
    assert len(review.findings) == 1 and review.findings[0].citations[0].source == "trial/1"


LIVE_RULES = (
    (True, True, True, True, True, "modify", ""),
    (True, True, True, True, False, "modify", ""),
    (True, True, True, False, True, "accept", "no revision round left"),
    (True, True, True, False, False, "accept", "no revision round left"),
    (True, True, False, True, True, "modify", ""),
    (True, True, False, True, False, "modify", ""),
    (True, True, False, False, True, "accept", "no revision round left"),
    (True, True, False, False, False, "accept", "no revision round left"),
    (True, False, True, True, True, "accept", ""),
    (True, False, True, True, False, "accept", ""),
    (True, False, True, False, True, "accept", ""),
    (True, False, True, False, False, "accept", ""),
    (True, False, False, True, True, "accept", ""),
    (True, False, False, True, False, "accept", ""),
    (True, False, False, False, True, "accept", ""),
    (True, False, False, False, False, "accept", ""),
    (False, True, True, True, True, "modify", ""),
    (False, True, True, True, False, "modify", ""),
    (False, True, True, False, True, "accept", "a placement change is tagged"),
    (False, True, True, False, False, "accept", "The one band-only round is used"),
    (False, True, False, True, True, "modify", ""),
    (False, True, False, True, False, "modify", ""),
    (False, True, False, False, True, "accept", "a placement change is tagged"),
    (False, True, False, False, False, "accept", "Shared modification budget exhausted"),
    (False, False, True, True, True, "accept", "a placement change is tagged"),
    (False, False, True, True, False, "accept", "The one band-only round is used"),
    (False, False, True, False, True, "accept", "a placement change is tagged"),
    (False, False, True, False, False, "accept", "The one band-only round is used"),
    (False, False, False, True, True, "accept", "a placement change is tagged"),
    (False, False, False, True, False, "modify", ""),
    (False, False, False, False, True, "accept", "a placement change is tagged"),
    (False, False, False, False, False, "accept", "Shared modification budget exhausted"),
)
"""(in band, review finding, band-only round used, a round left, stored fallback) -> the review decision (placement,
then the solve-review findings), with a fragment of an acceptance's reasons."""


@pytest.mark.parametrize("in_band,finding,used,left,stored,kind,reason", LIVE_RULES)
def test_review_decisions_match_live_forge(
    proposal: TaskProposal, in_band: bool, finding: bool, used: bool, left: bool, stored: bool, kind: str, reason: str
) -> None:
    """Every combination decides as the table says. A stored fallback was placed in band;
    a used band-only round is an earlier too-easy round with no review finding to revise."""
    setup = config()
    reward = 0.0 if in_band else 1.0
    current = replace(
        round_with(SolveReview("v", ReviewEnd.REPORTED, (FOUND,) * finding)),
        solver=(TrialResult(0, GradeResult(Outcome.GRADED, 1.0)), TrialResult(1, GradeResult(Outcome.GRADED, reward))),
    )
    decisions = (Modify(("2 of 2 trials solved: too_easy",), BandOutcome.TOO_EASY),) * used
    wp = Workpiece("w", proposal, round=len(decisions), submission=SUBMISSION, decisions=decisions)
    wp = replace(
        wp,
        evidence=(round_with(None), current),
        fallback=Fallback(SUBMISSION, BandOutcome.IN_BAND, 0) if stored else None,
    )
    budget = replace(setup.budget, max_modify_rounds=wp.round + left)
    decision = default_review(wp, setup.band, budget)
    placement = BandOutcome.IN_BAND if in_band else BandOutcome.TOO_EASY
    assert type(decision).__name__.lower() == kind
    assert isinstance(decision, Accept | Modify)
    assert (decision.band if isinstance(decision, Accept) else decision.placement) is placement
    if isinstance(decision, Modify):
        assert decision.review == (FOUND,) * finding and decision.fallback == SUBMISSION.digest
    else:
        assert decision.restore is None and decision.notes == (FOUND,) * finding
        assert not reason or any(reason in line for line in decision.reasons)


def test_a_clean_changed_band_after_a_reviewed_round_ships_and_a_defect_round_leaves_the_band_round(
    proposal: TaskProposal,
) -> None:
    """Regression: with two rounds, an in-band task revised for a defect came back clean
    and too easy and bought another round; live ships it at its new placement. Conversely a too-easy round with a
    defect spent the band-only allowance, so the clean too-easy revision was accepted instead of revised for its band."""
    setup = settings_config(**{"budget.max_modify_rounds": 2})
    easy = replace(
        round_with(SolveReview("v1", ReviewEnd.REPORTED)),
        solver=(TrialResult(0, GradeResult(Outcome.GRADED, 1.0)), TrialResult(1, GradeResult(Outcome.GRADED, 1.0))),
    )
    wp = reviewed_round(proposal, setup)
    decision, _ = decided(replace(wp, evidence=(*wp.evidence, easy)), setup)
    assert isinstance(decision, Accept) and decision.band is BandOutcome.TOO_EASY and decision.restore is None
    defect = replace(easy, review=SolveReview("v0", ReviewEnd.REPORTED, (FOUND,)))
    session = Conversation((Message(Role.SYSTEM, "author"),), ())
    start = Workpiece("w", proposal, submission=SUBMISSION, evidence=(defect,), session=session)
    first, wp = decided(start, setup)
    assert isinstance(first, Modify) and first.review == (FOUND,) and first.placement is BandOutcome.TOO_EASY
    decision, _ = decided(replace(wp, evidence=(*wp.evidence, easy)), setup)
    assert isinstance(decision, Modify) and not decision.review and decision.placement is BandOutcome.TOO_EASY


def recorded_task() -> Submission:
    """A task whose prompt carries a historical tool call with unsorted, integral-float arguments, unsorted labels, and
    a verifier file with a declared mode and mtime: everything a JSONL reload reorders or retypes."""
    task = arithmetic_task()
    call = ConversationToolCall(call_id="c1", name="lookup", arguments={"z": 3.0, "a": 5})
    events = (
        task.context.events[0],
        AssistantToolCalls(calls=(call,)),
        ConversationToolResult(call_id="c1", content="8"),
        *task.context.events[1:],
    )
    check = inline_resource("check.sh", b"exit 0\n").model_copy(update={"mode": "0755", "mtime_ns": 1_700_000_000_123})
    task = task.model_copy(
        update={
            "context": task.context.model_copy(update={"events": events}),
            "labels": {"z": "last", "a": "first"},
            "resources": task.resources.model_copy(update={"verifier": (check,)}),
        }
    )
    return replace(SUBMISSION, task=task)


def journalled(submission: Submission, rollout: GradedRollout) -> Resolved[GradedRollout]:
    spec = lower_with_settings(submission.task, lowering_settings(load_policy(DEFAULTS_DIR)), {})
    return Resolved(SolverRollout(spec, 1, 0), (Fingerprint("rollouts", "d", ReplayPolicy.CACHE),), rollout)


def reloaded(records: Sequence[Record]) -> tuple[Submission, GradedRollout]:
    """The task and trial a resumed process holds: as the journal reloads the solver rollout that ran them."""
    event = next(
        r.event for r in records if isinstance(r.event, Resolved) and isinstance(r.event.request, SolverRollout)
    )
    assert isinstance(event.request, SolverRollout)
    return replace(SUBMISSION, task=event.request.spec.task), event.value


@pytest.mark.parametrize("variant", ["plain", "recorded"])
async def test_evidence_is_the_same_before_and_after_a_journal_round_trip(variant: str) -> None:
    """Regression: for a task and trial before and after a JSONL round trip, every source
    and the version are identical."""
    submission = recorded_task() if variant == "recorded" else SUBMISSION
    grade = GradeResult(Outcome.GRADED, 0.0, detail={"z": 1, "a": {"y": 2.0}})
    rollout = GradedRollout(Rollout("run-1", (transcript(LONG, "I answer 9."),), "stop"), grade)
    sink = BufferSink()
    log = EventLog(sink=sink)
    await log.record(journalled(submission, rollout))
    await log.persist()
    task, trial = reloaded(local_records(sink))
    live = evidence(submission, (TRIALS[0], (TrialResult(1, grade), rollout)), OUTPUT_CHARS)
    cold = evidence(task, (TRIALS[0], (TrialResult(1, trial.grade), trial)), OUTPUT_CHARS)
    assert live.sources == cold.sources and live.version == cold.version


async def test_a_second_process_replays_the_review_of_a_recorded_task_with_zero_calls(tmp_path: Path) -> None:
    """Regression: across two processes, a historical tool call in ``task/prompt.md``
    rendered ``lookup({'z': 3.0, 'a': 5})`` fresh and ``lookup({'a': 5, 'z': 3})`` after the reload, and the second
    process paid the reviewer again. The verifier file's declared mtime is the same in both, as on any machine."""
    probe = [sys.executable, "-m", "lib.taskforge.tests.review_replay_probe"]
    root = Path(__file__).parents[3]
    for mode in ("record", "replay"):
        done = subprocess.run([*probe, mode, str(tmp_path)], cwd=root, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr[-3000:]
        assert done.stdout.strip() == f"{mode}: {1 if mode == 'record' else 0} reviewer call(s)"


SPENT = "its context window ran out before a finding; it did not validate the task"


def exhausted_round(review: SolveReview, reason: str = SPENT) -> ValidationResult:
    """An in-band round over two of three scored slots whose adversary ran out of search budget without a finding."""
    solver = (*round_with(None).solver, TrialResult(2, None, Cause.HELD))
    search = AdversaryResult("false_accept", AdversaryStatus.EXHAUSTED, (), (reason,), SearchLimit.CONTEXT_WINDOW)
    return replace(round_with(review), solver=solver, adversary=(search,))


@pytest.mark.parametrize("ending", ["author stop", "same-digest fault"])
def test_a_restore_reports_the_round_that_ships(proposal: TaskProposal, ending: str) -> None:
    """Regression: restoring round 0, whose adversary ran out of search budget and which
    placed over two of three slots, dropped both notes on an author stop, and a same-digest faulted revision reported
    its own adversary reason instead. The acceptance describes the round that ships, and names the fault apart."""
    setup = settings_config(**{"budget.max_missing_trials": 1})
    session = Conversation((Message(Role.SYSTEM, "author"),), ())
    first = exhausted_round(SolveReview("v0", ReviewEnd.REPORTED, (FOUND,)))
    _, wp = decided(Workpiece("w", proposal, submission=SUBMISSION, evidence=(first,), session=session), setup)
    assert wp.fallback is not None and wp.fallback.evidence == 0
    if ending == "author stop":
        stopped, _ = advance(wp, Stage.AUTHOR, Stopped(StopReason.CONTEXT_LIMIT, session, "window full"), setup)
        decision = stopped.decisions[-1]
    else:
        faulted = replace(
            exhausted_round(SolveReview("v1", ReviewEnd.REPORTED), "the discarded round's search"),
            controls=(ControlResult(0, GradeResult(Outcome.GRADED, 0.0), False),),
        )
        revision = replace(SUBMISSION, notes="revised")
        decision, _ = decided(replace(wp, submission=revision, evidence=(*wp.evidence, faulted)), setup)
    assert isinstance(decision, Accept) and decision.restore == SUBMISSION.digest and decision.notes == (FOUND,)
    if ending == "same-digest fault":
        assert "Revision fault: Control 0 violates its expected reward" in decision.reasons
    assert (
        f"false_accept: {SPENT}" in decision.reasons
        and "1 of 3 solver trials have no score; placed over 2" in decision.reasons
    )
    assert not any("discarded round" in reason for reason in decision.reasons)


@pytest.mark.parametrize(
    "rewards,finding,line",
    [
        ((1.0, 0.0), True, "1 of 2 trials solved: in_band. Keep the difficulty."),
        (
            (1.0, 1.0),
            False,
            "2 of 2 trials solved: too_easy. Make the task harder while preserving correct controls and grading.",
        ),
        (
            (0.0, 0.0),
            False,
            "0 of 2 trials solved: too_hard. Make the task more tractable while preserving its objective, correct "
            "controls and grading.",
        ),
    ],
    ids=["in band with a finding", "too easy", "too hard"],
)
def test_a_revision_request_carries_live_forge_s_band_line(
    proposal: TaskProposal, rewards: tuple[float, float], finding: bool, line: str
) -> None:
    """Regression: when findings turned an acceptance into a repair, the author was sent only
    the number of proposed edits; a revision request carries the measured solve rate and what the placement asks for."""
    setup = config()
    solver = tuple(TrialResult(index, GradeResult(Outcome.GRADED, reward)) for index, reward in enumerate(rewards))
    current = replace(round_with(SolveReview("v", ReviewEnd.REPORTED, (FOUND,) * finding)), solver=solver)
    decision = default_review(
        Workpiece("w", proposal, submission=SUBMISSION, evidence=(current,)), setup.band, setup.budget
    )
    assert isinstance(decision, Modify) and decision.findings[0] == line


def test_a_restore_for_an_unaccepted_placement_labels_it_a_placement(proposal: TaskProposal) -> None:
    """Regression: with ``band.accept_too_easy`` false, a clean too-easy revision restores the
    reviewed version, as it should, but its placement instruction was labelled ``Revision fault:``. Only an actual
    fault carries that label."""
    setup = settings_config(**{"band.accept_too_easy": False, "budget.max_modify_rounds": 2})
    wp = reviewed_round(proposal, setup)
    easy = replace(
        round_with(SolveReview("v1", ReviewEnd.REPORTED)),
        solver=tuple(TrialResult(index, GradeResult(Outcome.GRADED, 1.0)) for index in range(3)),
    )
    decision, _ = decided(replace(wp, evidence=(*wp.evidence, easy)), setup)
    assert isinstance(decision, Accept) and decision.restore == SUBMISSION.digest
    line = f"3 of 3 trials solved: too_easy. {INSTRUCTIONS[BandOutcome.TOO_EASY]}"
    assert f"Revision placement: {line}" in decision.reasons
    assert not any(reason.startswith("Revision fault:") for reason in decision.reasons)


@pytest.fixture
def reword() -> Iterator[Callable[[], None]]:
    """A sibling release that rewords ``JsonAnswer``'s description and nothing else, applied when called."""
    original = JsonAnswer.__doc__

    def apply() -> None:
        before = TaskSpec.model_json_schema()
        JsonAnswer.__doc__ = "Reworded by a sibling release, with the same fields."
        JsonAnswer.model_rebuild(force=True)
        TaskSpec.model_rebuild(force=True)
        assert TaskSpec.model_json_schema() != before  # the live schema really changed

    yield apply
    JsonAnswer.__doc__ = original
    JsonAnswer.model_rebuild(force=True)
    TaskSpec.model_rebuild(force=True)


@pytest.mark.parametrize("reworded", [False, True], ids=["installed wording", "reworded sibling"])
async def test_a_held_root_resumes_only_its_unfinished_turns(
    driver_args: Namespace,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    reword: Callable[[], None],
    reworded: bool,
) -> None:
    """Regression: a root this driver held after three author calls re-bought them on resume once the library reworded
    ``JsonAnswer`` (five calls where the earlier reader made two). The root replays the author reference it recorded,
    so only the unfinished turns reach inference: the author's submit and the never-run review, whatever the installed
    library's descriptions say between the hold and the resume."""
    held = FakeInference((*script()[:3], Hold("relay unavailable")))
    first = replace(resources(Envs(), ()), inference=held, role_inference={"solve_review": CleanReviewer()})
    author, reviewer = FakeInference(script()[3:]), CleanReviewer()
    resumed = replace(resources(Envs(), ()), inference=author, role_inference={"solve_review": reviewer})
    serve(monkeypatch, first, resumed)
    await driver.one(driver_args)
    destination = Path(driver_args.output) / driver.proposal_key(driver_args.proposals)
    head = published(destination).head
    assert (head.status.terminal, head.format, len(held.requests)) == (Terminal.HELD, checkpoint.JOURNAL_FORMAT, 4)
    if reworded:
        reword()
    driver_args.resume_held = True
    await driver.one(driver_args)
    assert published_workpiece(destination).terminal is Terminal.ACCEPTED
    assert (len(author.requests), len(reviewer.requests)) == (1, 1)


async def test_the_first_reviewer_request_keeps_its_recorded_identity() -> None:
    """Pins two rendered surfaces that stay on shared live code for v1: the task library's ``format_conversation`` (its
    fixed labels become ``task/prompt.md`` and so the evidence version the opening names) and Taskforge's docstrings in
    ``taskforge.model.review``, which become the report tool's schema. A change to either changes every recorded
    review's first request, so it must be a deliberate prompt change."""
    recorded = "33e49ad87349bd18ce57c1cb3f0421761277a01d30d34e55e61627b93c27f2a3"
    reviewer = FakeInference((report(),))
    await reviewed(reviewer, submission=recorded_task())  # a user turn, a tool call and its result: every label
    assert digest(reviewer.requests[0].identity()) == recorded, (
        "the solve-review's first request changed: a deliberate prompt change only. Update this recorded identity "
        "knowingly; every recorded review will miss and buy its calls again"
    )
