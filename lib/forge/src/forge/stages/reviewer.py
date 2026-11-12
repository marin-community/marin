# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The rollout reviewer: a GLM agent that reads graded solver transcripts and suggests revisions to the author.

It runs inside validation, after the evidence would be accepted, so its inference resolves through the same effect
log as the solver and adversary and replays with the stage. It is not a grader: it never changes a grade.

Failures use the existing decisions: provider ``Transient`` retries in the inference executor, ``Restart`` replays the
validate stage, and ``Hold`` propagates and holds the workpiece for resume. A reviewer that ends without ``suggest``
(window exhausted, degenerate output) is recorded incomplete with its reason and does not block acceptance: the
reviewer is advisory, and the reason stays on the acceptance.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict

from taskcompendium.models import TaskSpec

from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.harness.protocol import Returned, Stopped
from forge.harness.tools.reviewer import ReviewedTrial, Suggestions, reviewer_tools
from forge.model.llm import Message, Role
from forge.model.proposal import render
from forge.model.review import ReviewerResult, ValidationResult
from forge.model.workpiece import Conversation, Submission, Workpiece
from forge.resources.bundle import Resources
from forge.stages.summary import excerpt, grade_summary, task_manifest


def evidence_summary(submission: Submission, result: ValidationResult) -> str:
    """Controls and adversary findings as the reviewer sees them; the controls are the task's reference."""
    controls = [
        {
            "index": control.index,
            "kind": authored.kind,
            "reply": excerpt(authored.candidate.reply),
            "files": [file.path for file in authored.candidate.files],
            "expected": asdict(authored.expected),
            "met": control.met,
            **grade_summary(control.grade),
        }
        for control, authored in zip(result.controls, submission.controls, strict=True)
    ]
    findings = [
        {"adversary": a.name, "claim": excerpt(f.claim), "reproduced": f.met, **grade_summary(f.grade)}
        for a in result.adversary
        for f in a.findings
    ]
    return json.dumps({"controls": controls, "adversary_findings": findings}, default=str, separators=(",", ":"))


async def run_reviewer(
    ctx: Context,
    workpiece: Workpiece,
    config: ForgeConfig,
    resources: Resources,
    task: TaskSpec,
    result: ValidationResult,
    trials: Sequence[ReviewedTrial],
) -> ReviewerResult:
    assert workpiece.submission is not None
    policy = config.policy
    tools = reviewer_tools(trials, task, policy.setting("tools.output_chars", int))
    session = Conversation(
        (
            Message(Role.SYSTEM, policy.prompt("reviewer.system").template),
            Message(
                Role.USER,
                policy.prompt("reviewer.user").render(
                    task=task_manifest(task),
                    proposal=render(workpiece.proposal),
                    evidence=evidence_summary(workpiece.submission, result),
                    trials="\n".join(f"trial {index}: {trial.summary}" for index, trial in enumerate(trials)),
                ),
            ),
        ),
        tuple(tool.schema for tool in tools.values()),
    )
    async with ctx.scope("reviewer"):
        outcome = await config.reviewer.run(ctx, session, resources.inference_for("reviewer"), tools)
    if isinstance(outcome, Returned) and isinstance(outcome.payload, Suggestions):
        return ReviewerResult(True, outcome.payload.revisions)
    reason = outcome.summary if isinstance(outcome, Stopped) else f"expected suggest, got {type(outcome).__name__}"
    return ReviewerResult(False, (), (f"reviewer stopped without suggest: {reason}",))
