# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The triage decision on a proposal, and ``accept``, a triage function admitting every proposal."""

from enum import StrEnum


class TriageDecision(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


async def accept(proposal: object) -> TriageDecision:
    """Accept ``proposal`` without checking it."""
    return TriageDecision.ACCEPT
