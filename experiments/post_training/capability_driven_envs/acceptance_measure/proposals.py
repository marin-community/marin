# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read a sampled n=25 proposal into this stack's ``TaskProposal``.

The n=25 run wrote its proposals on the decision-model stack (#10302), whose header carries ``composite``, a
reserved field that is ``[]`` for every non-composite proposal. This branch's header has no such field, so the
reader drops an empty ``composite`` once, here, and refuses a non-empty one (the sampler never draws one).
"""

import hashlib
import json

from taskforge.stage10_proposal.model import TaskProposal, parse


class ProposalMismatch(ValueError):
    """The proposal bytes are not the ones the sample recorded, or the header cannot be carried to this stack."""


def read_proposal(raw: bytes, expected_sha256: str) -> TaskProposal:
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ProposalMismatch(f"proposal sha256 {actual} is not the sampled {expected_sha256}")
    document = json.loads(raw)
    header = document["header"]
    if "composite" in header:
        if header["composite"]:
            raise ProposalMismatch(f"proposal {header['id']} has composite parts, which this stack cannot author")
        del header["composite"]
    return parse(json.dumps(document))
