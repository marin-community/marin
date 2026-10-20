# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Triage's default: a deterministic preflight that decides with no model call.

``preflight`` runs ``checks.CHECKS``: the proposal parses back from its canonical rendering, its
header schema holds, its environment and verification pair is allowed, its declared solver machine
fits its environment (``solver_machine_fits``), its required sections have content, and a null
proposal carries only its reason. A FATAL failure rejects the proposal with each failing check as a
reason, and so does a null proposal, which describes no task to build. Anything else is accepted and
goes to build. Advisory failures ride on the verdict's ``structural`` results and decide nothing.

Whether the host can provide the declared machine is the build's question, not triage's: the build
stage checks it before any model call and ends such a workpiece as the host's.
"""

from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage20_triage.checks import ALL_COMBINATIONS, CHECKS, CheckContext, run_checks
from taskforge.stage20_triage.verdict import TriageDecision, Verdict

EVERY_COMBINATION = CheckContext(allowed_combinations=ALL_COMBINATIONS)


async def preflight(proposal: TaskProposal, /, *, ctx: CheckContext = EVERY_COMBINATION) -> Verdict:
    """Accept ``proposal`` unless a fatal check fails or it is a null proposal; ``ctx`` narrows the allowed pairings."""
    structural = run_checks(proposal, CHECKS, ctx)
    reasons = tuple(f"{r.name}: {r.reason}" for r in structural if r.blocks)
    if not reasons and proposal.header.null_reason is not None:
        reasons = (f"null proposal: {proposal.header.null_reason}",)
    decision = TriageDecision.REJECT if reasons else TriageDecision.ACCEPT
    return Verdict(proposal.header.id, proposal.digest, decision, structural, reasons)
