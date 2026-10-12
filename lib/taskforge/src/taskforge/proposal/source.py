# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What a proposal source returns, and the protocol every source implements."""

from dataclasses import dataclass
from typing import Protocol

from taskforge.llm.recording import RecordedModel
from taskforge.proposal.model import TaskProposal


@dataclass(frozen=True)
class SlotFailure:
    """A slot whose document failed the parser."""

    slot: int
    error: str


@dataclass(frozen=True)
class ProposalBatch:
    """Everything one ``propose`` call produced: the parsed proposals and the failed slots."""

    proposals: tuple[TaskProposal, ...]
    failures: tuple[SlotFailure, ...]


class ProposalSource[IdeaT](Protocol):
    """Turns one idea of the source's own type into ``n`` proposal slots."""

    async def propose(self, idea: IdeaT, n: int, model: RecordedModel) -> ProposalBatch: ...
