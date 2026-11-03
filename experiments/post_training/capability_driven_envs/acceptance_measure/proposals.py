# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read a sampled n=25 proposal into this stack's ``TaskProposal``.

The n=25 run wrote its proposals with ``composite``, a reserved header field that is ``[]`` for every non-composite
proposal; this stack's header carries it too. The reader refuses a non-empty one (the sampler never draws one), since
no grader type authors a composite yet.
"""

import hashlib

from taskforge.stage10_proposal.model import TaskProposal, parse


class ProposalMismatch(ValueError):
    """The proposal bytes are not the ones the sample recorded, or the header cannot be carried to this stack."""


def read_proposal(raw: bytes, expected_sha256: str) -> TaskProposal:
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ProposalMismatch(f"proposal sha256 {actual} is not the sampled {expected_sha256}")
    proposal = parse(raw.decode())
    if proposal.header.composite:
        raise ProposalMismatch(f"proposal {proposal.header.id} has composite parts, which this stack cannot author")
    return proposal
