# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The solve-review: a machine-less agent that reads recorded solver trials and reports the task defects they show.

One implementation over any set of trials: a review names its prompts and its report tool, so a later kind (the
confirmation step's final review) passes its own, not a new path. Decision rules live with the review rules.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from pydantic import TypeAdapter
from rolloutengine.models import GradedRollout
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import AssistantToolCalls, TaskResource, TaskSpec, format_conversation
from taskcompendium.runtime.resources import resource_bytes, resource_inventory, resource_signature

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.effects.log import Billing
from taskforge.harness.loop import AgenticHarness
from taskforge.harness.protocol import Returned, Stopped, Tool
from taskforge.harness.role import Fresh, Visit, sitting
from taskforge.harness.tools.evidence import (
    RESOURCE,
    TASK,
    TRIAL,
    Evidence,
    ReportTool,
    Source,
    consolidated,
    evidence_tools,
    source_listing,
)
from taskforge.model.controls import CONTROLS
from taskforge.model.proposal import TaskProposal, render
from taskforge.model.review import (
    Accept,
    EvidenceKind,
    MissingEvidence,
    Response,
    ReviewEnd,
    ReviewFinding,
    SolveReview,
    TrialResult,
    same,
)
from taskforge.model.workpiece import Submission, Workpiece
from taskforge.resources.bundle import Resources
from taskforge.stages.rules import solved
from taskforge.stages.summary import manifest


def journal_form(value: object) -> object:
    """``value`` as the journal reloads it: an untyped integral float comes back an integer (3.0 as 3)."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, Mapping):
        return {key: journal_form(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [journal_form(item) for item in value]
    return value


def canonical(value: object) -> str:
    """Structured values as sorted JSON in their journal form: the journal reloads mappings key-sorted and integral
    floats as integers, so live and resumed text agree."""
    return json.dumps(journal_form(value), sort_keys=True, default=str)


def render_message(message: Mapping[str, object]) -> str:
    """One recorded message; a tool call keeps its id and a reply names the call it answers, so swapping two
    parallel replies changes the text."""
    reply = message.get("tool_call_id")
    parts = [f"{message.get('role', '?')}" + (f" (reply to {reply})" if reply else "") + ":"]
    for key in ("reasoning_content", "reasoning"):
        if isinstance(reasoning := message.get(key), str) and reasoning:
            parts.append(f"[reasoning] {reasoning}")
    if content := message.get("content"):
        parts.append(content if isinstance(content, str) else canonical(content))
    calls = message.get("tool_calls")
    for call in calls if isinstance(calls, list) else ():
        parts.append(f"[tool call] {canonical(call)}")
    return "\n".join(parts)


def trial_source(trial: TrialResult, rollout: GradedRollout | None) -> Source:
    """A trial's messages, every conversation in order, an interruption's marker, then its grade; how the recording
    ended, or why there is no transcript, is in its summary."""
    messages = (
        ()
        if rollout is None
        else tuple(map(render_message, (m for c in rollout.rollout.conversations for m in c.messages)))
    )
    failure = None if rollout is None else rollout.rollout.failure
    if messages and failure:
        messages = (*messages, f"[interrupted: the recording ends here ({failure.exception_type})]")
    grade = TypeAdapter(GradeResult | None).dump_python(trial.grade, mode="json")
    parts = (
        *messages,
        "grader:\n" + canonical({"cause": trial.cause, "detail": trial.detail, "grade": grade}),
    )
    if rollout is None or not messages:
        why = (
            "its recording holds no messages" if rollout else trial.detail or (trial.grade.error if trial.grade else "")
        )
        ended = f"no transcript ({why or trial.cause}); the grade is all there is"
    else:
        ended = f"stop={rollout.rollout.stop_reason}" + (
            f", interrupted: the recorder kept the prefix before {failure.exception_type}" if failure else ""
        )
    status = "ungraded" if trial.grade is None else f"{trial.grade.status} reward={canonical(trial.grade.reward)}"
    summary = (
        f"{TRIAL}{trial.trial}: {status} cause={trial.cause} {ended}; parts={len(parts)} chars={sum(map(len, parts))}"
    )
    return Source(f"{TRIAL}{trial.trial}", summary, parts)


def text_source(name: str, text: str, about: str = "") -> Source:
    return Source(name, f"{name}: {about}lines={len(text.splitlines())} chars={len(text)}", tuple(text.splitlines()))


def resource_source(role: str, resource: TaskResource) -> Source:
    """A file read by line, its summary the signature the task review signs (role, path, mode, mtime, bytes); bytes
    that are not UTF-8 are named opaque and have no parts, never replacement characters two files could share."""
    name, data = f"{RESOURCE}{role}/{resource.path}", resource_bytes(resource)
    about = f"{canonical(resource_signature(role, resource, data))} "
    try:
        return text_source(name, data.decode(), about)
    except UnicodeDecodeError:
        return Source(name, f"{name}: {about}binary (not UTF-8), not readable as text", ())


def task_prompt(task: TaskSpec) -> str:
    """The task's conversation with each historical tool call's arguments rendered canonically, as the journal reloads
    them (sorted, integral floats as integers), keeping the call and reply ids."""
    events = tuple(
        (
            event.model_copy(
                update={"calls": tuple(c.model_copy(update={"arguments": canonical(c.arguments)}) for c in event.calls)}
            )
            if isinstance(event, AssistantToolCalls)
            else event
        )
        for event in task.context.events
    )
    return format_conversation(events)


def task_document(submission: Submission) -> str:
    """The task manifest the reviewer sees, in the opening and as ``task/task.json`` alike, rendered canonically."""
    return canonical(manifest(submission.task.model_dump(mode="json")))


def task_sources(submission: Submission) -> tuple[Source, ...]:
    """The task's documents under ``task/``, then every file it carries under ``resource/<role>/``, build inputs
    included: no file can take another file's or a trial's address."""
    task = submission.task
    return (
        text_source(f"{TASK}task.json", task_document(submission)),
        text_source(f"{TASK}prompt.md", task_prompt(task)),
        text_source(f"{TASK}controls.json", CONTROLS.dump_json(submission.controls, indent=1).decode()),
        *(resource_source(role, resource) for role, resource in resource_inventory(task)),
    )


def evidence(
    submission: Submission,
    trials: Sequence[tuple[TrialResult, GradedRollout | None]],
    output_chars: int,
) -> Evidence:
    sources = (*(trial_source(*pair) for pair in trials), *task_sources(submission))
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("Two evidence sources share an address")
    unpassed = frozenset(t.trial for t, _ in trials if not solved(t))
    return Evidence({source.name: source for source in sources}, unpassed, output_chars)


def finding_lines(findings: Sequence[ReviewFinding], start: int = 0) -> str:
    """Findings numbered from ``start`` with their quoted evidence, for the author."""
    return "\n".join(
        f"{number}. [{f.kind}] {f.description}\n   Edit: {f.edit}\n"
        + "\n".join(f"   Evidence {c.source} #{c.start}..{c.end}: {c.quote}" for c in f.citations)
        for number, f in enumerate(findings, start)
    )


async def solve_review(
    ctx: Context,
    config: TaskforgeConfig,
    resources: Resources,
    proposal: TaskProposal,
    submission: Submission,
    trials: Sequence[tuple[TrialResult, GradedRollout | None]],
    prompts: str = "solve_review",
    report: Callable[[Evidence], Tool[object]] = ReportTool,
    previous: str = "",
    bills: Billing = Billing.SOLVE_REVIEW,
    index: Evidence | None = None,
) -> SolveReview:
    """Review ``trials`` of ``submission``; emits only the reviewer's inference effects. The final review passes its
    own prompts, its report tool, the earlier reports (``earlier_reports``) as the template's ``{previous}``, the role
    its calls bill (it shares the round reviewer's model, profile and replay keys) and ``index``, the evidence of
    ``trials`` that it also checks shipped notes against; without one the review builds it.

    The opening ends with the index's version whatever the policy's template renders, so changed evidence never
    replays a recorded review.
    """
    policy = config.policy
    index = evidence(submission, trials, policy.setting("tools.output_chars", int)) if index is None else index
    read = {trial.trial: len(index.sources[f"{TRIAL}{trial.trial}"].parts) > 1 for trial, _ in trials}  # a message
    cut = [trial.trial for trial, rollout in trials if not read[trial.trial] or (rollout and rollout.rollout.failure)]
    missing = (MissingEvidence(EvidenceKind.TRANSCRIPTS, f"solver trials {cut} have no complete transcript"),) * bool(
        cut
    )
    if not any(read.values()):  # missing evidence is named on acceptance, never a hold
        return SolveReview(index.version, ReviewEnd.SKIPPED, missing=missing)
    opening = Fresh(
        policy.prompt(f"{prompts}.system").template,
        policy.prompt(f"{prompts}.user").render(
            task=task_document(submission),
            proposal=render(proposal),
            evidence=source_listing(index),
            version=index.version,  # an earlier template's slot
            previous=previous,
        )
        + f"\n\nEvidence version: {index.version}",  # outside the template, so no overlay can drop it
    )
    role = config.solve_review  # each kind of review is nudged toward its own report's fields
    if isinstance(role.harness, AgenticHarness):
        role = replace(role, harness=replace(role.harness, nudge=policy.prompt(f"{prompts}.nudge").template))
    async with sitting(
        ctx,
        role,
        Visit(prompts, None, tools=evidence_tools(index, report), bills=bills),
        resources,
    ) as session:
        outcome = await session(opening)
    if isinstance(outcome, Returned) and isinstance(outcome.payload, SolveReview):
        return replace(outcome.payload, missing=missing)
    reason = outcome.summary if isinstance(outcome, Stopped) else f"expected report, got {type(outcome).__name__}"
    reason = f"solve-review ended without a report: {reason}"
    missing = (MissingEvidence(EvidenceKind.SOLVE_REVIEW, reason), *missing)
    return SolveReview(index.version, ReviewEnd.NO_REPORT, missing=missing)


def earlier_reports(workpiece: Workpiece) -> str:
    """Every solve-review of ``workpiece``, oldest first, each finding with what became of it, for the final review."""
    current = workpiece.submission.digest if workpiece.submission else None
    blocks = []
    for index, result in enumerate(workpiece.evidence):
        if (review := result.review) is None:
            continue
        kind = "Solve-review" if result.confirms is None else "Final review"
        subject = "this submission" if result.task_digest == current else "an earlier submission"
        head = f"{kind} of validation record {index} ({subject}, {len(result.solver)} trials):"
        if review.end is not ReviewEnd.REPORTED:
            blocks.append(f"{head} no report ({'; '.join(m.detail for m in review.missing)})")
            continue
        lines = [
            f"- [{f.kind}] {f.description} Edit: {f.edit} Outcome: {fate(workpiece, index, f, f in review.notes)}."
            for f in (*review.findings, *review.notes)
        ]
        blocks.append("\n".join((head, *lines)) if lines else f"{head} no findings.")
    return "\n\n".join(blocks) or "No earlier solve-review reports."


def shipped_notes(index: Evidence, findings: Sequence[ReviewFinding]) -> tuple[ReviewFinding, ...]:
    """Each (``same``) finding of ``findings`` once, at its first occurrence, carrying the citations of its versions
    that check against ``index`` (``consolidated``): a version quoting text the shipped trials lack is never
    published, and a finding no version of which checks is not carried."""
    chosen = {same(finding): finding for finding in consolidated(index, findings)[0]}
    return tuple(chosen[key] for key in dict.fromkeys(map(same, findings)) if key in chosen)


FATES = {
    Response.REQUESTED: "given to the author, who has not answered yet",
    Response.REVISED: "given to the author, who revised the task",
    Response.DECLINED: "the author declined it: {reason}",
    Response.STOPPED: "given to the author, who stopped without resubmitting",
}


DISCARDED = "; the driver then restored the version before that attempted fix, so this submission does not carry it"


def fate(workpiece: Workpiece, index: int, finding: ReviewFinding, note: bool) -> str:
    """What became of ``finding`` of the review of evidence record ``index``: the author's response recorded against
    that review (a final review's ``note`` was sent as optional), else a note on an acceptance, else nothing."""
    response = next((r for r in reversed(workpiece.responses) if (r.review, r.finding) == (index, finding)), None)
    if response is not None:
        sent = FATES[response.response].format(reason=response.reason) + (DISCARDED if response.discarded else "")
        return sent.replace("given to the author", "given to the author as an optional note") if note else sent
    if any(isinstance(d, Accept) and finding in d.notes for d in workpiece.decisions):
        return "recorded as a note on the acceptance"
    return "not acted on"
