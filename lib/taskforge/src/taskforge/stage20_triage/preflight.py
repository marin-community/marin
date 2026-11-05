# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Triage's default: a deterministic preflight that decides with no model call.

``preflight`` takes a proposal that ``stage10_proposal.model.parse`` accepted, so its header is
already valid, and runs ``checks.CHECKS``. One check is fatal: ``required_sections``, every required
section has content. Three are advisory: ``grader_matches_verification``, the grader section names
what the declared verification needs; ``research_has_purpose``, every research item states a
purpose; and ``resources_in_build_plan``, the build plan names every declared resource. A fatal
failure rejects the proposal with the failing check as its reason. A null proposal is rejected with
its ``null_reason``, since it describes no task to build. Anything else is accepted and goes to
build. Advisory failures ride on the verdict's ``structural`` results and decide nothing.

Whether the host can provide the declared machine is the build's question, not triage's: the build
stage checks it before any model call and ends such a workpiece as the host's.
"""

from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage20_triage.checks import CHECKS, run_checks
from taskforge.stage20_triage.verdict import TriageDecision, Verdict


async def preflight(proposal: TaskProposal) -> Verdict:
    """Accept ``proposal`` unless ``required_sections`` fails or it is a null proposal.

    Every check in ``CHECKS`` is recorded on the verdict; a rejection's reasons name the failing fatal
    check, or read ``null proposal: <null_reason>``.
    """
    structural = run_checks(proposal, CHECKS)
    reasons = tuple(f"{r.name}: {r.reason}" for r in structural if r.blocks)
    if not reasons and proposal.header.null_reason is not None:
        reasons = (f"null proposal: {proposal.header.null_reason}",)
    decision = TriageDecision.REJECT if reasons else TriageDecision.ACCEPT
    return Verdict(proposal.header.id, proposal.digest, decision, structural, reasons)
