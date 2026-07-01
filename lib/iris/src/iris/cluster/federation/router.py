# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit-time routing between local execution and a federation peer.

This is a separate layer from the meta-scheduler's static, startup-built backend
index: peer capabilities are dynamic (learned live over the heartbeat), so peer
selection cannot fold into that index. Prefer-local is the rule — a job runs on a
local backend whenever one is feasible. The router selects local execution for
every job, so no job is handed off.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from iris.cluster.constraints import Constraint
from iris.cluster.federation.peer import FederationPeer


@dataclass(frozen=True)
class SubmitRouting:
    """Where a submitted job executes.

    An empty ``peer_id`` means local execution on one of this controller's
    backends; a non-empty ``peer_id`` names the federation peer the whole job is
    handed off to.
    """

    peer_id: str = ""

    @property
    def is_local(self) -> bool:
        return not self.peer_id


_LOCAL = SubmitRouting()


class PeerRouter:
    """Chooses local execution or a peer for each submission.

    Selects local execution for every job.
    """

    def __init__(self, peers: Sequence[FederationPeer]):
        self._peers = {peer.peer_id: peer for peer in peers}

    def decide(self, constraints: Sequence[Constraint], user: str) -> SubmitRouting:
        """Select where ``user``'s job (with ``constraints``) executes.

        Selects local execution for every job.
        """
        del constraints, user
        return _LOCAL
