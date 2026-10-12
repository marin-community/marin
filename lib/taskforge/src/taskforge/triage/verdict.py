# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The decision triage reaches on one proposal, and the stage function that runs no check."""

from enum import StrEnum

from taskforge.proposal.model import TaskProposal


class TriageDecision(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


def accept(proposal: TaskProposal) -> TriageDecision:
    """Accept every proposal: no structural check and no rubric sample runs on it."""
    return TriageDecision.ACCEPT
