# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit-time routing between local execution and a federation peer.

A separate layer from the meta-scheduler's static, startup-built backend index:
peer capabilities are dynamic (learned live over the heartbeat), so peer
selection cannot fold into that index. The decision, in order:

1. An explicit ``cluster=<peer>`` pin forces that peer (the caller has already
   rejected a job that also pins a local ``backend``, and validated the peer
   exists).
2. Otherwise **prefer-local**: if any local backend is feasible for the job's
   shape, run it here.
3. Otherwise match the job's routing constraints against each reachable peer's
   *live* advertised capability and hand off to the first that can host it.
4. Otherwise stay local so the caller fails the job as unschedulable — never
   wedge it. The chosen peer may still reject at handoff (its live capacity moved
   between the heartbeat snapshot and delivery); the manager tolerates that.

Routing decides only *where* a job can run. *Who* may run it there is the peer's own
``auth.allowed_submitters``, enforced where the job lands and surfaced from the handoff.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from iris.cluster.constraints import (
    AttributeValue,
    Constraint,
    ConstraintOp,
    evaluate_constraint,
    routing_constraints,
    strip_backend_constraints,
    strip_cluster_constraints,
)
from iris.cluster.federation.peer import FederationPeer


@dataclass(frozen=True)
class RoutingRequest:
    """The submit-time context the router decides on."""

    constraints: Sequence[Constraint]
    # Whether at least one local backend can host the job's shape (computed by the
    # feasibility gate before routing).
    local_feasible: bool
    # An explicit ``cluster=<peer>`` pin, or "" for none. The caller validates the
    # peer exists and that no local ``backend`` pin was also set.
    cluster_pin: str = ""


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


def _backend_satisfies(advertised: Mapping[str, list[str]], constraint: Constraint) -> bool:
    """Whether a peer backend advertising ``advertised`` satisfies one constraint.

    Advertised attributes are multi-valued (a backend may offer several device
    variants); the constraint holds if any advertised value for its key does.
    """
    values = advertised.get(constraint.key, [])
    if constraint.op == ConstraintOp.EXISTS:
        return bool(values)
    if constraint.op == ConstraintOp.NOT_EXISTS:
        return not values
    return any(evaluate_constraint(AttributeValue(v), constraint) for v in values)


def _peer_can_host(peer: FederationPeer, constraints: Sequence[Constraint]) -> bool:
    """Whether ``peer`` currently advertises a backend that can host the job.

    A best-effort match against the last capability heartbeat: an unreachable
    peer can host nothing; a job with no routing constraints (e.g. plain CPU) can
    run on any reachable peer; otherwise some advertised backend must satisfy
    every routing constraint.
    """
    heartbeat = peer.heartbeat()
    if not heartbeat.reachable:
        return False
    routing = routing_constraints(strip_cluster_constraints(strip_backend_constraints(constraints)))
    if not routing:
        return True
    for backend in heartbeat.backends:
        advertised = {key: list(values.values) for key, values in backend.advertised_attributes.items()}
        if all(_backend_satisfies(advertised, c) for c in routing):
            return True
    return False


class PeerRouter:
    """Chooses local execution or a peer for each submission."""

    def __init__(self, peers: Sequence[FederationPeer]):
        self._peers = {peer.peer_id: peer for peer in peers}

    def decide(self, request: RoutingRequest) -> SubmitRouting:
        """Select where ``request``'s job executes (see the module docstring).

        Routing answers *where the job can run*, never *who may run it there*: a peer
        admits or refuses a submitter under its own ``auth.allowed_submitters``, and a
        refusal surfaces from the handoff itself.
        """
        if request.cluster_pin:
            # Validated to exist by the caller; force it even if locally feasible.
            return SubmitRouting(peer_id=request.cluster_pin)
        if request.local_feasible:
            return _LOCAL
        for peer_id, peer in sorted(self._peers.items()):
            if _peer_can_host(peer, request.constraints):
                return SubmitRouting(peer_id=peer_id)
        return _LOCAL
