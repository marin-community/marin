# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Forge marin-forge-passes, lib/forge/src/forge/stages/solve_review.py (behaviour and prompts).

"""The solve-review: a machine-less agent that reads recorded solver trials and reports the task defects they show.

One implementation over any set of trials: a review names its prompts and its report tool, so a later kind (the
confirmation step's final review) passes its own, not a new path. Decision rules live with the review rules.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from pydantic import TypeAdapter
from rolloutengine.models import GradedRollout
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import TaskResource, format_conversation
from taskcompendium.runtime.resources import resource_bytes, resource_inventory, resource_signature

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.harness.protocol import Returned, Stopped
from taskforge.harness.role import Fresh, Visit, sitting
from taskforge.harness.tools.evidence import (
    RESOURCE,
    TASK,
    TRIAL,
    Evidence,
    ListTool,
    ReadTool,
    ReportTool,
    SearchTool,
    Source,
    evidence_tools,
    source_listing,
)
from taskforge.model.controls import CONTROLS
from taskforge.model.proposal import TaskProposal, render
from taskforge.model.review import EvidenceKind, MissingEvidence, ReviewEnd, ReviewFinding, SolveReview, TrialResult
from taskforge.model.workpiece import Submission
from taskforge.resources.bundle import Resources
from taskforge.resources.inference import InferenceRequest
from taskforge.stages.rules import passed
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


def task_document(submission: Submission) -> str:
    """The task manifest the reviewer sees, in the opening and as ``task/task.json`` alike, rendered canonically."""
    return canonical(manifest(submission.task.model_dump(mode="json")))


def task_sources(submission: Submission) -> tuple[Source, ...]:
    """The task's documents under ``task/``, then every file it carries under ``resource/<role>/``, build inputs
    included: no file can take another file's or a trial's address."""
    task = submission.task
    return (
        text_source(f"{TASK}task.json", task_document(submission)),
        text_source(f"{TASK}prompt.md", format_conversation(task.context.events)),
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
    unpassed = frozenset(t.trial for t, _ in trials if t.grade is None or t.cause is not None or not passed(t.grade))
    return Evidence({source.name: source for source in sources}, unpassed, output_chars)


def reviewing(request: InferenceRequest) -> bool:
    """Whether ``request`` is the reviewer's: only it is offered the evidence tools."""
    return {tool.name for tool in request.tools} >= {ListTool.name, ReadTool.name, SearchTool.name}


def finding_lines(findings: Sequence[ReviewFinding]) -> str:
    """Numbered findings with their quoted evidence, for the author."""
    return "\n".join(
        f"{number}. [{f.kind}] {f.description}\n   Edit: {f.edit}\n"
        + "\n".join(f"   Evidence {c.source} #{c.start}..{c.end}: {c.quote}" for c in f.citations)
        for number, f in enumerate(findings)
    )


async def solve_review(
    ctx: Context,
    config: TaskforgeConfig,
    resources: Resources,
    proposal: TaskProposal,
    submission: Submission,
    trials: Sequence[tuple[TrialResult, GradedRollout | None]],
    prompts: str = "solve_review",
    report: type[ReportTool] = ReportTool,
) -> SolveReview:
    """Review ``trials`` of ``submission``; emits only the reviewer's inference effects.

    The opening ends with the index's version whatever the policy's template renders, so changed evidence never
    replays a recorded review.
    """
    policy = config.policy
    index = evidence(submission, trials, policy.setting("tools.output_chars", int))
    read = {trial.trial: len(index.sources[f"{TRIAL}{trial.trial}"].parts) > 1 for trial, _ in trials}  # a message
    cut = [trial.trial for trial, rollout in trials if not read[trial.trial] or (rollout and rollout.rollout.failure)]
    missing = (MissingEvidence(EvidenceKind.TRANSCRIPTS, f"solver trials {cut} have no complete transcript"),) * bool(
        cut
    )
    if not any(read.values()):  # owner's ruling: missing evidence is named on acceptance, never a hold
        return SolveReview(index.version, ReviewEnd.SKIPPED, missing=missing)
    opening = Fresh(
        policy.prompt(f"{prompts}.system").template,
        policy.prompt(f"{prompts}.user").render(
            task=task_document(submission),
            proposal=render(proposal),
            evidence=source_listing(index),
            version=index.version,  # an earlier template's slot
        )
        + f"\n\nEvidence version: {index.version}",  # outside the template, so no overlay can drop it
    )
    async with sitting(
        ctx,
        config.solve_review,
        Visit(prompts, prompts, None, tools=evidence_tools(index, report)),
        resources,
    ) as session:
        outcome = await session(opening)
    if isinstance(outcome, Returned) and isinstance(outcome.payload, SolveReview):
        return replace(outcome.payload, missing=missing)
    reason = outcome.summary if isinstance(outcome, Stopped) else f"expected report, got {type(outcome).__name__}"
    reason = f"solve-review ended without a report: {reason}"
    missing = (MissingEvidence(EvidenceKind.SOLVE_REVIEW, reason), *missing)
    return SolveReview(index.version, ReviewEnd.NO_REPORT, missing=missing)
