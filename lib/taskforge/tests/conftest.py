# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for taskforge tests.

``ledger`` is an in-memory ``Ledger`` that keeps every recorded entry in ``entries``. ``proposal`` is
a ShellSim task proposal (multiply the two numbers in ``/workspace/numbers.txt``), and
``proposal_source`` proposes it for any idea without calling its model.
"""

from dataclasses import dataclass

import pytest

from taskforge.ledger.records import LedgerEntry
from taskforge.llm.recording import RecordedModel
from taskforge.proposal.model import TaskProposal, parse
from taskforge.proposal.source import ProposalBatch


class ListLedger:
    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)


@pytest.fixture
def ledger() -> ListLedger:
    return ListLedger()


PROPOSAL = """---
id: "IDEA/0"
source: {kind: capability, ref: "IDEA", hash: "abc123"}
environment: shellsim
verification: simple
grounding: unverified
research: []
build:
  - "numbers": "the two factors"
resources: ["workspace/numbers.txt"]
null_reason: null
---
## Task
Multiply the two numbers in numbers.txt.

## Realism and workflow
A clerk checks a product.

## Research plan
None.

## Build plan
Write the numbers file.

## Grader design and controls
Numeric, zero tolerance: only 42 passes.

## Risks and null conditions
None.
"""


@pytest.fixture
def proposal() -> TaskProposal:
    return parse(PROPOSAL)


@dataclass
class FixedSource:
    """Proposes ``proposal`` for any idea, ignoring its model, and counts its calls."""

    proposal: TaskProposal
    calls: int = 0

    async def propose(self, idea: object, n: int, model: RecordedModel) -> ProposalBatch:
        self.calls += 1
        return ProposalBatch(proposals=(self.proposal,), failures=())


@pytest.fixture
def proposal_source(proposal: TaskProposal) -> FixedSource:
    return FixedSource(proposal)
