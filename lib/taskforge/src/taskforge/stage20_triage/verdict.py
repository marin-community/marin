# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The triage verdict on one proposal: the decision, the structural results, the reasons, and the author state.

``accept`` is a triage function admitting every proposal.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import JsonValue, TypeAdapter

from taskforge.stage20_triage.checks import CheckResult


class TriageDecision(StrEnum):
    """What triage asks of the stage coordinator.

    REPAIR asks for a rewrite of the proposal, which triage judges again, within the workpiece's
    ``max_triage_repairs``. The preflight returns only ACCEPT or REJECT; REPAIR is the surface a
    consumer's triage may use, for a deterministic rewrite such as fixing a proposal header.
    """

    ACCEPT = "accept"
    REPAIR = "repair"
    REJECT = "reject"


@dataclass(frozen=True)
class Verdict:
    """Triage outcome for the proposal with ``proposal_digest``.

    ``structural`` holds every check's result, advisory ones included. ``reasons`` names each fatal
    failure, or the null proposal's reason, of a REJECT, says what a REPAIR asks to change, and is
    empty for an ACCEPT. ``author_state`` is the JSON mapping an ACCEPT hands every author call of
    the workpiece, such as findings or obligations the author should honour; the preflight attaches
    none.
    """

    proposal_id: str
    proposal_digest: str
    decision: TriageDecision
    structural: tuple[CheckResult, ...]
    reasons: tuple[str, ...]
    author_state: Mapping[str, JsonValue] = field(default_factory=dict)

    def to_json(self) -> str:
        return _VERDICT.dump_json(self, indent=2).decode()

    @classmethod
    def from_json(cls, text: str) -> "Verdict":
        return _VERDICT.validate_json(text)


_VERDICT = TypeAdapter(Verdict)


async def accept(proposal: object) -> TriageDecision:
    """Accept ``proposal`` without checking it."""
    return TriageDecision.ACCEPT
