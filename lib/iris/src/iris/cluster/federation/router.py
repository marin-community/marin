# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit-time classification: local execution, the federation queue, or reject.

A separate layer from the meta-scheduler's static, startup-built backend index:
peer capabilities are dynamic (learned live over the heartbeat), so peer
selection cannot fold into that index. Submit no longer *picks* a peer — that is a
scheduling decision, and all peer scheduling decisions live on the control tick's
federation pass. Submit only classifies, in order:

1. An explicit ``cluster=<peer>`` pin routes the job to the federation **queue**
   pinned to that peer (the caller validated the peer exists and that no local
   ``backend`` pin was also set). The tick waits for that peer's availability.
2. Otherwise **prefer-local**: if any local backend is feasible for the job's
   shape, run it here.
3. Otherwise, if any reachable peer advertises the job's shape, route it to the
   federation **queue** (unpinned); the tick assigns it to a peer that has room.
4. Otherwise reject it as unschedulable now — no local backend and no reachable
   peer can host the shape, so no queue could help (status quo fast-fail).

Classification decides only *where* a job can run. *Who* may run it there is the
peer's own ``auth.allowed_submitters``, enforced where the job lands.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

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
    """The submit-time context the classifier decides on."""

    constraints: Sequence[Constraint]
    # Whether at least one local backend can host the job's shape (computed by the
    # feasibility gate before routing).
    local_feasible: bool
    # An explicit ``cluster=<peer>`` pin, or "" for none. The caller validates the
    # peer exists and that no local ``backend`` pin was also set.
    cluster_pin: str = ""


class SubmitDisposition(StrEnum):
    """What a submit-time classification decided for a job."""

    LOCAL = "local"  # run on one of this controller's own backends
    QUEUE = "queue"  # admit to the federation queue; the tick assigns a peer
    REJECT = "reject"  # unschedulable now — no local backend, no reachable peer


@dataclass(frozen=True)
class SubmitPlan:
    """The classification outcome. ``pinned_peer_id`` is set only for a pinned QUEUE."""

    disposition: SubmitDisposition
    pinned_peer_id: str = ""


def backend_satisfies(advertised: Mapping[str, list[str]], constraint: Constraint) -> bool:
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
        if all(backend_satisfies(advertised, c) for c in routing):
            return True
    return False


class PeerRouter:
    """Classifies each submission as local, federation-queued, or unschedulable."""

    def __init__(self, peers: Sequence[FederationPeer]):
        self._peers = {peer.peer_id: peer for peer in peers}

    def classify(self, request: RoutingRequest) -> SubmitPlan:
        """Classify where ``request``'s job can run (see the module docstring).

        Answers *where the job can run*, never *who may run it there*: a peer admits
        or refuses a submitter under its own ``auth.allowed_submitters``, surfaced
        from the handoff at delivery.
        """
        if request.cluster_pin:
            # Validated to exist by the caller; queue pinned to it even if locally
            # feasible (an explicit pin means "federate this there").
            return SubmitPlan(SubmitDisposition.QUEUE, pinned_peer_id=request.cluster_pin)
        if request.local_feasible:
            return SubmitPlan(SubmitDisposition.LOCAL)
        if any(_peer_can_host(peer, request.constraints) for peer in self._peers.values()):
            return SubmitPlan(SubmitDisposition.QUEUE)
        return SubmitPlan(SubmitDisposition.REJECT)
