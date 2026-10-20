# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The triage verdict on one proposal: the decision, the structural results, and the reasons.

``accept`` is a triage function admitting every proposal.
"""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import TypeAdapter

from taskforge.stage20_triage.checks import CheckResult


class TriageDecision(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass(frozen=True)
class Verdict:
    """Triage outcome for the proposal with ``proposal_digest``.

    ``structural`` holds every check's result, advisory ones included. ``reasons`` names each fatal
    failure, or the null proposal's reason, of a REJECT, and is empty for an ACCEPT.
    """

    proposal_id: str
    proposal_digest: str
    decision: TriageDecision
    structural: tuple[CheckResult, ...]
    reasons: tuple[str, ...]

    def to_json(self) -> str:
        return _VERDICT.dump_json(self, indent=2).decode()

    @classmethod
    def from_json(cls, text: str) -> "Verdict":
        return _VERDICT.validate_json(text)


_VERDICT = TypeAdapter(Verdict)


async def accept(proposal: object) -> TriageDecision:
    """Accept ``proposal`` without checking it."""
    return TriageDecision.ACCEPT
