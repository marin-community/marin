# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic preflight with at most one repair, disabled by a zero repair budget."""

from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.effects.events import Note
from forge.model.llm import Message, Sampling
from forge.model.proposal import ProposalFormatError, TaskProposal, parse, render
from forge.model.workpiece import Workpiece
from forge.resources.bundle import Resources
from forge.resources.inference import InferenceRequest
from forge.stages.checks import CHECKS, run_checks
from forge.stages.exits import Triaged, TriageRejected


def problems(proposal: TaskProposal) -> tuple[str, ...]:
    return tuple(f"{check.name}: {check.reason}" for check in run_checks(proposal, CHECKS) if check.blocks)


async def triage(
    ctx: Context, workpiece: Workpiece, config: ForgeConfig, _resources: Resources
) -> Triaged | TriageRejected:
    proposal = workpiece.proposal
    if proposal.header.null_reason is not None:
        return TriageRejected((f"Null proposal: {proposal.header.null_reason}",))
    async with ctx.scope("triage"):
        for check in run_checks(proposal, CHECKS):
            await ctx.emit(Note(f"{check.name}: {check.status}: {check.reason}"))
        reasons = problems(proposal)
        if not reasons:
            return Triaged(proposal)
        if config.policy.setting("budget.max_triage_repairs", int) < 1:
            return TriageRejected(reasons)
        completion = await ctx.emit(
            InferenceRequest(
                (
                    Message("system", config.policy.prompt("triage.system").render()),
                    Message(
                        "user",
                        config.policy.prompt("triage.user").render(
                            proposal=render(proposal), reasons="\n".join(reasons)
                        ),
                    ),
                ),
                (),
                Sampling(),
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
