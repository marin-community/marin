# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source of the section check: Taskforge integration 9807c5cfb3, stage20_triage/checks.py.

"""Deterministic preflight with at most one repair, disabled by a zero repair budget.

No transition reaches triage yet. The proposal header's validity is ``parse``'s job; triage checks
what ``parse`` leaves open: every required body section is present, filled and substantive.
"""

import re
from functools import partial

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.model.llm import Message, Role
from taskforge.model.proposal import REQUIRED_HEADINGS, ProposalFormatError, TaskProposal, body_sections, parse, render
from taskforge.model.workpiece import Workpiece
from taskforge.policy import sampling
from taskforge.resources.bundle import Resources
from taskforge.resources.inference import InferenceRequest, complete
from taskforge.stages.exits import Triaged, TriageRejected

MIN_SECTION_WORDS = 40
# The template's section placeholder; angle-bracket answer formats such as "<yes|no>" are legitimate.
SECTION_PLACEHOLDER = re.compile(r"<\s*(?:\.\.\.|\u2026)\s*>")


# Expected false positives: a section that is short because it is genuinely simple (a reasoning
# task's Build plan can be a few lines); MIN_SECTION_WORDS is set well below what GLM writes (its
# documents run 20-30 KB). A template placeholder such as "<...>" is never a false positive.
# Expected false negatives: long but empty prose.
def problems(proposal: TaskProposal) -> tuple[str, ...]:
    sections = body_sections(proposal.body)
    found: list[str] = []
    for heading in REQUIRED_HEADINGS:
        text = sections.get(heading)
        if text is None:
            found.append(f"'## {heading}' missing")
        elif SECTION_PLACEHOLDER.fullmatch(text):
            found.append(f"'## {heading}' is still the template placeholder {text!r}")
        elif len(text.split()) < MIN_SECTION_WORDS:
            found.append(f"'## {heading}' has {len(text.split())} words, fewer than {MIN_SECTION_WORDS}")
    return tuple(found)


async def triage(
    ctx: Context, workpiece: Workpiece, config: TaskforgeConfig, resources: Resources
) -> Triaged | TriageRejected:
    proposal = workpiece.proposal
    if proposal.header.null_reason is not None:
        return TriageRejected((f"Null proposal: {proposal.header.null_reason}",))
    async with ctx.scope("triage"):
        reasons = problems(proposal)
        if not reasons:
            return Triaged(proposal)
        if config.policy.setting("budget.max_triage_repairs", int) < 1:
            return TriageRejected(reasons)
        # Triage repairs use the explicitly selected author role and its sampling policy.
        client = resources.inference_for("author")
        ctx.resource(client, client.fingerprint)
        ctx.handle(InferenceRequest, partial(complete, notify=ctx.emit), uses=(client,))
        completion = await ctx.emit(
            InferenceRequest(
                (
                    Message(Role.SYSTEM, config.policy.prompt("triage.system").render()),
                    Message(
                        Role.USER,
                        config.policy.prompt("triage.user").render(
                            proposal=render(proposal), reasons="\n".join(reasons)
                        ),
                    ),
                ),
                (),
                sampling(config.policy),
            )
        )
        try:
            repaired = parse(completion.content)
        except ProposalFormatError as error:
            return TriageRejected((*reasons, f"Repair is invalid: {error}"))
        if repaired.header != proposal.header:
            return TriageRejected(("Format repair changed the proposal header",))
        remaining = problems(repaired)
        return TriageRejected(remaining) if remaining else Triaged(repaired)
