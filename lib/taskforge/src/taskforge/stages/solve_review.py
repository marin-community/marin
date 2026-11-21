# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Forge marin-forge-passes, lib/forge/src/forge/stages/solve_review.py (behaviour and prompts).

"""The solve-review: a machine-less agent that reads recorded solver trials and reports the task defects they show.

A round's review and the confirmation step's final review are one implementation over any set of trials; only their
prompts and report schema (``ROUND`` and ``FINAL``) differ, and their decision rules live with the review rules.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from pydantic import TypeAdapter
from rolloutengine.models import GradedRollout
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import format_conversation
from taskcompendium.runtime.resources import resource_bytes

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.harness.protocol import Returned, Stopped
from taskforge.harness.role import Fresh, Visit, sitting
from taskforge.harness.tools.evidence import (
    FINAL_GUIDE,
    ROUND_GUIDE,
    Evidence,
    FinalReportTool,
    ReportTool,
    Source,
    evidence_tools,
)
from taskforge.model.controls import CONTROLS
from taskforge.model.proposal import TaskProposal, render
from taskforge.model.review import EvidenceKind, MissingEvidence, ReviewEnd, ReviewFinding, SolveReview, TrialResult
from taskforge.model.workpiece import Submission
from taskforge.resources.bundle import Resources
from taskforge.stages.rules import passed
from taskforge.stages.summary import task_manifest


@dataclass(frozen=True, slots=True)
class Kind:
    prompts: str
    report: type[ReportTool]
    guide: str


ROUND, FINAL = Kind("solve_review", ReportTool, ROUND_GUIDE), Kind("solve_review.final", FinalReportTool, FINAL_GUIDE)


def render_message(message: Mapping[str, object]) -> str:
    parts = [f"{message.get('role', '?')}:"]
    for key in ("reasoning_content", "reasoning"):
        if isinstance(reasoning := message.get(key), str) and reasoning:
            parts.append(f"[reasoning] {reasoning}")
    if content := message.get("content"):
        parts.append(content if isinstance(content, str) else json.dumps(content, default=str))
    calls = message.get("tool_calls")
    for call in calls if isinstance(calls, list) else ():
        parts.append(f"[tool call] {json.dumps(call.get('function', call) if isinstance(call, Mapping) else call)}")
    return "\n".join(parts)


def trial_source(trial: TrialResult, rollout: GradedRollout | None) -> Source:
    """A trial's messages, every conversation in order, then its grade; how the recording ended, or why there is no
    transcript, is in its summary."""
    messages = (
        ()
        if rollout is None
        else tuple(map(render_message, (m for c in rollout.rollout.conversations for m in c.messages)))
    )
    grade = TypeAdapter(GradeResult | None).dump_python(trial.grade, mode="json")
    parts = (
        *messages,
        "grader:\n" + json.dumps({"cause": trial.cause, "detail": trial.detail, "grade": grade}, default=str),
    )
    if rollout is None or not messages:
        why = (
            "its recording holds no messages" if rollout else trial.detail or (trial.grade.error if trial.grade else "")
        )
        ended = f"no transcript ({why or trial.cause}); the grade is all there is"
    else:
        failure = rollout.rollout.failure
        ended = f"stop={rollout.rollout.stop_reason}" + (
            f", recorder kept the prefix before {failure}" if failure else ""
        )
    status = "ungraded" if trial.grade is None else f"{trial.grade.status} reward={trial.grade.reward}"
    summary = (
        f"trial-{trial.trial}: {status} cause={trial.cause} {ended}; parts={len(parts)} chars={sum(map(len, parts))}"
    )
    return Source(f"trial-{trial.trial}", summary, parts)


def task_sources(submission: Submission) -> tuple[Source, ...]:
    task, resources = submission.task, submission.task.resources
    files = {
        "task.json": task_manifest(task),
        "prompt.md": format_conversation(task.context.events),
        "controls.json": CONTROLS.dump_json(submission.controls, indent=1).decode(),
        **{
            file.path: resource_bytes(file).decode(errors="replace")
            for file in (*resources.all, *resources.worker, *resources.oracle, *resources.verifier)
        },
    }
    return tuple(
        Source(path, f"{path}: lines={len(text.splitlines())} chars={len(text)}", tuple(text.splitlines()))
        for path, text in files.items()
    )


def transcribed(rollout: GradedRollout | None) -> bool:
    """Whether a trial's recording is whole: its rollout recorded messages and was not cut by an interruption."""
    return (
        rollout is not None
        and rollout.rollout.failure is None
        and any(c.messages for c in rollout.rollout.conversations)
    )


def evidence(
    submission: Submission,
    trials: Sequence[tuple[TrialResult, GradedRollout | None]],
    output_chars: int,
) -> Evidence:
    sources = (*(trial_source(*pair) for pair in trials), *task_sources(submission))
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("A task file is named like a solver trial")
    unpassed = frozenset(t.trial for t, _ in trials if t.grade is None or t.cause is not None or not passed(t.grade))
    return Evidence({source.name: source for source in sources}, unpassed, output_chars)


def finding_lines(findings: Sequence[ReviewFinding]) -> str:
    """Numbered findings with their quoted evidence, for the author and for a later review."""
    return "\n".join(
        f"{number}. [{f.kind}] {f.description}\n   Edit: {f.edit}\n"
        + "\n".join(f"   Evidence {c.source} #{c.start}..{c.end}: {c.quote}" for c in f.citations)
        for number, f in enumerate(findings)
    )


def earlier_lines(earlier: Sequence[SolveReview]) -> str:
    blocks = [
        f"Report {index} ({review.end}): " + (finding_lines((*review.findings, *review.notes)) or "no findings")
        for index, review in enumerate(earlier)
    ]
    return "\n\n".join(blocks) or "No earlier reports."


async def solve_review(
    ctx: Context,
    config: TaskforgeConfig,
    resources: Resources,
    proposal: TaskProposal,
    submission: Submission,
    trials: Sequence[tuple[TrialResult, GradedRollout | None]],
    kind: Kind = ROUND,
    earlier: Sequence[SolveReview] = (),
    visit: str = "solve_review",
) -> SolveReview:
    """Review ``trials`` of ``submission``, shown ``earlier`` reports; emits only the reviewer's inference effects.

    The opening names the index's version, so changed evidence never replays a recorded review.
    """
    policy = config.policy
    index = evidence(submission, trials, policy.setting("tools.output_chars", int))
    cut = [trial.trial for trial, rollout in trials if not transcribed(rollout)]
    missing = (MissingEvidence(EvidenceKind.TRANSCRIPTS, f"solver trials {cut} have no complete transcript"),) * bool(
        cut
    )
    if len(cut) == len(trials):  # owner's ruling: missing evidence is named on acceptance, never a hold
        return SolveReview(index.version, ReviewEnd.SKIPPED, missing=missing)
    listing = "\n".join(source.summary for source in index.sources.values())
    opening = Fresh(
        policy.prompt(f"{kind.prompts}.system").template,
        policy.prompt(f"{kind.prompts}.user").render(
            task=task_manifest(submission.task),
            proposal=render(proposal),
            evidence=listing,
            version=index.version,
            earlier=earlier_lines(earlier),
        ),
    )
    async with sitting(
        ctx,
        config.solve_review,
        Visit(visit, visit, None, tools=evidence_tools(index, kind.report, kind.guide)),
        resources,
    ) as session:
        outcome = await session(opening)
    if isinstance(outcome, Returned) and isinstance(outcome.payload, SolveReview):
        return replace(outcome.payload, missing=missing)
    reason = outcome.summary if isinstance(outcome, Stopped) else f"expected report, got {type(outcome).__name__}"
    reason = f"solve-review ended without a report: {reason}"
    missing = (MissingEvidence(EvidenceKind.SOLVE_REVIEW, reason), *missing)
    return SolveReview(index.version, ReviewEnd.NO_REPORT, reasons=(reason,), missing=missing)
