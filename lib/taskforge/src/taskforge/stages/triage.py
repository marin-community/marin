# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic preflight with at most one repair, disabled by a zero repair budget."""

from functools import partial

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.effects.events import Note
from taskforge.model.llm import Message, Role
from taskforge.model.proposal import ProposalFormatError, TaskProposal, parse, render
from taskforge.model.workpiece import Workpiece
from taskforge.policy import sampling
from taskforge.resources.bundle import Resources
from taskforge.resources.inference import InferenceRequest, complete
from taskforge.stages.checks import CHECKS, run_checks
from taskforge.stages.exits import Triaged, TriageRejected


def problems(proposal: TaskProposal) -> tuple[str, ...]:
    return tuple(f"{check.name}: {check.reason}" for check in run_checks(proposal, CHECKS) if check.blocks)


async def triage(
    ctx: Context, workpiece: Workpiece, config: TaskforgeConfig, resources: Resources
) -> Triaged | TriageRejected:
    proposal = workpiece.proposal
    if proposal.header.null_reason is not None:
        return TriageRejected((f"Null proposal: {proposal.header.null_reason}",))
    async with ctx.scope("triage"):
        checks = run_checks(proposal, CHECKS)
        for check in checks:
            await ctx.emit(Note(f"{check.name}: {check.status}: {check.reason}"))
        reasons = tuple(f"{check.name}: {check.reason}" for check in checks if check.blocks)
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
