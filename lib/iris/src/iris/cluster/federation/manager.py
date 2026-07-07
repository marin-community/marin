# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The federation manager: peer registry, handoff, delta-sync, and cancel.

The controller composes one manager. It owns the peer registry, the submit-time
:class:`~iris.cluster.federation.router.PeerRouter`, and two background loops —
the capability heartbeat and the delta-sync loop that mirrors each peer's handed-
off jobs back into the local projection. Every durable mutation goes through an
injected :class:`~iris.cluster.federation.store.FederationStore`, so the manager
stays a self-contained module.

With no peers configured it is inert: neither loop starts and every view is
empty, so a single-cluster deployment is unchanged. A ``store`` is required only
to hand a job off or run the sync loop; the observability slice (heartbeat,
``ListPeers``) works without one.
"""

import logging
import threading
from collections.abc import Callable, Sequence

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.timing import Duration, Timestamp

from iris.cluster.constraints import BACKEND_CONSTRAINT_KEY, CLUSTER_CONSTRAINT_KEY
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.federation.router import PeerRouter, RoutingRequest, SubmitRouting
from iris.cluster.federation.store import (
    CancelTarget,
    FederationStore,
    HandoffAdmission,
    HandoffSpec,
)
from iris.cluster.types import JobName
from iris.managed_thread import ManagedThread, ThreadContainer
from iris.rpc import controller_pb2, job_pb2

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_INTERVAL = Duration.from_seconds(30)
DEFAULT_SYNC_INTERVAL = Duration.from_seconds(3)
_JOIN_TIMEOUT = Duration.from_seconds(5.0)

_PEER_RPC_ERRORS = (ConnectError, ConnectionError, OSError)


class FederationManager:
    """Owns the federation peer registry, handoff, delta-sync, and cancel."""

    def __init__(
        self,
        peers: Sequence[FederationPeer],
        *,
        threads: ThreadContainer,
        store: FederationStore | None = None,
        cluster_id: str = "",
        heartbeat_interval: Duration = DEFAULT_HEARTBEAT_INTERVAL,
        sync_interval: Duration = DEFAULT_SYNC_INTERVAL,
    ):
        self._peers = {peer.peer_id: peer for peer in peers}
        self._threads = threads
        self._store = store
        self._cluster_id = cluster_id
        self._heartbeat_interval = heartbeat_interval
        self._sync_interval = sync_interval
        self._router = PeerRouter(peers)
        self._heartbeat_thread: ManagedThread | None = None
        self._sync_thread: ManagedThread | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the heartbeat and (when a store is wired) the sync loop.

        A no-op when no peers are configured, so a single-cluster deployment is
        unchanged.
        """
        if not self._peers:
            return
        self._heartbeat_thread = self._threads.spawn(self._run_heartbeat_loop, name="federation-heartbeat")
        if self._store is not None:
            self._sync_thread = self._threads.spawn(self._run_sync_loop, name="federation-sync")

    def stop(self) -> None:
        """Stop both loops and release peer connections. Idempotent."""
        for thread in (self._heartbeat_thread, self._sync_thread):
            if thread is not None:
                thread.stop()
                thread.join(timeout=_JOIN_TIMEOUT)
        self._heartbeat_thread = None
        self._sync_thread = None
        for peer in self._peers.values():
            peer.close()

    # -- routing / views -----------------------------------------------------

    def route_submit(self, request: RoutingRequest) -> SubmitRouting:
        """Route a submission to local execution or a peer."""
        return self._router.decide(request)

    def has_peer(self, peer_id: str) -> bool:
        """Whether ``peer_id`` names a configured federation peer."""
        return peer_id in self._peers

    def peer_summaries(self) -> list[controller_pb2.Controller.PeerSummary]:
        """A ``PeerSummary`` for every configured peer, ordered by peer id."""
        return [self._build_summary(peer) for _, peer in sorted(self._peers.items())]

    # -- handoff (parent side, synchronous) ----------------------------------

    def submit_federated_handle(
        self,
        *,
        local_job_id: JobName,
        request: controller_pb2.Controller.LaunchJobRequest,
        peer_id: str,
        owner_principal: str,
    ) -> None:
        """Persist a federated handle and synchronously hand the job to its peer.

        In one local transaction the store persists the
        ``jobs``/``job_config``/``federated_jobs`` handle (no task rows) in
        ``PENDING_HANDOFF``. When freshly admitted it then calls the peer's
        ``LaunchJob`` — the peer runs the job under the same, cluster-invariant
        ``local_job_id`` — and flips the handle to ``HANDED_OFF``; an idempotent
        resubmit skips delivery. A failed delivery is not fatal — the handle
        persists and the sync loop re-drives it.
        """
        if self._store is None:
            raise RuntimeError("federation handoff requires a store")
        self._require_peer(peer_id)

        spec = HandoffSpec(
            local_job_id=local_job_id,
            peer_id=peer_id,
            owner_principal=owner_principal,
            request=request,
        )
        if self._store.admit_and_persist_handoff(spec) is HandoffAdmission.ADMITTED:
            self._deliver_handoff(spec)

    # -- cancel (parent side) ------------------------------------------------

    def cancel_federated(self, local_job_id: JobName) -> None:
        """Route a versioned cancel for a federated job to its peer.

        Bumps ``cancel_intent_version`` (so a cancelled pending handoff is never
        delivered and a retried cancel is a no-op) and routes the idempotent
        ``TerminateJob(local_job_id)`` (the peer runs the same id). A transient
        failure is not fatal — the sync loop re-drives the cancel until the peer acks
        or sync observes the job terminal/pruned.
        """
        if self._store is None:
            raise RuntimeError("federation cancel requires a store")
        target = self._store.bump_cancel_intent(local_job_id)
        if target is not None:
            self._deliver_cancel(target)

    def _deliver_cancel(self, target: CancelTarget) -> None:
        """Route one ``TerminateJob`` to the peer.

        A peer ``NOT_FOUND`` means the job is already gone (terminal-and-pruned),
        which satisfies the cancel — terminalize the local mirror so the re-drive
        stops. Any other RPC error is left for the next sync to retry.
        """
        assert self._store is not None
        peer = self._peers.get(target.peer_id)
        if peer is None:
            logger.warning(
                "Cannot cancel federated job %s: peer %s is not configured", target.local_job_id, target.peer_id
            )
            return
        try:
            peer.terminate_job(target.local_job_id)
        except ConnectError as exc:
            if exc.code == Code.NOT_FOUND:
                self._store.mark_cancel_satisfied(target.local_job_id, now_ms=Timestamp.now().epoch_ms())
                return
            logger.warning(
                "Routed cancel of %s to peer %s failed (will retry): %s",
                target.local_job_id,
                target.peer_id,
                exc,
            )
        except (ConnectionError, OSError) as exc:
            logger.warning(
                "Routed cancel of %s to peer %s failed (will retry): %s",
                target.local_job_id,
                target.peer_id,
                exc,
            )

    # -- on-demand proxy (parent side) ---------------------------------------

    def proxy_profile(
        self,
        *,
        peer_id: str,
        request: job_pb2.ProfileTaskRequest,
    ) -> job_pb2.ProfileTaskResponse:
        """Forward a profile RPC for a federated task to its peer controller.

        Job ids are cluster-invariant, so the request's target names the same task
        on the peer — it is proxied verbatim to the peer's ``ProfileTask``. The peer
        is authoritative: its answer — including a ``NOT_FOUND`` for a task it has
        since moved or finished — propagates back.
        """
        peer = self._require_peer(peer_id)
        return peer.profile_task(request)

    def proxy_exec(
        self,
        *,
        peer_id: str,
        request: controller_pb2.Controller.ExecInContainerRequest,
    ) -> controller_pb2.Controller.ExecInContainerResponse:
        """Forward an exec RPC for a federated task to its peer controller.

        Job ids are cluster-invariant, so the request's task id names the same task
        on the peer — it is proxied verbatim to the peer's ``ExecInContainer``. The
        peer is authoritative: its answer — including a ``NOT_FOUND`` — propagates
        back.
        """
        peer = self._require_peer(peer_id)
        return peer.exec_in_container(request)

    # -- background loops ----------------------------------------------------

    def _require_peer(self, peer_id: str) -> FederationPeer:
        """The configured peer named ``peer_id``, or raise if it is unknown."""
        peer = self._peers.get(peer_id)
        if peer is None:
            raise ValueError(f"unknown federation peer {peer_id!r}")
        return peer

    def _run_loop(self, stop_event: threading.Event, step: Callable[[], None], interval: float) -> None:
        """Run ``step`` every ``interval`` seconds until ``stop_event`` is set."""
        while not stop_event.is_set():
            step()
            stop_event.wait(timeout=interval)

    def _run_heartbeat_loop(self, stop_event: threading.Event) -> None:
        self._run_loop(stop_event, self._probe_all_peers, self._heartbeat_interval.to_seconds())

    def _run_sync_loop(self, stop_event: threading.Event) -> None:
        self._run_loop(stop_event, self.sync_once, self._sync_interval.to_seconds())

    def _probe_all_peers(self) -> None:
        for peer in self._peers.values():
            peer.probe()

    def sync_once(self) -> None:
        """One sync pass: re-drive pending handoffs and cancels, then pull each peer.

        The unit the sync loop repeats; a no-op without a store.
        """
        if self._store is None:
            return
        self._redrive_pending_handoffs()
        self._redrive_pending_cancels()
        for peer in self._peers.values():
            self._sync_peer(peer)

    def _redrive_pending_handoffs(self) -> None:
        """Re-deliver every handle still awaiting its peer (boot recovery + retry)."""
        assert self._store is not None
        for spec in self._store.pending_handoffs():
            self._deliver_handoff(spec)

    def _redrive_pending_cancels(self) -> None:
        """Re-route ``TerminateJob`` for every cancel intent the peer has not yet
        acknowledged (a transient failure left it undelivered)."""
        assert self._store is not None
        for target in self._store.pending_cancels():
            self._deliver_cancel(target)

    def _deliver_handoff(self, spec: HandoffSpec) -> None:
        assert self._store is not None
        peer = self._peers.get(spec.peer_id)
        if peer is None:
            logger.warning("Cannot hand off %s: peer %s is not configured", spec.local_job_id, spec.peer_id)
            return
        handoff = self._build_handoff_request(spec.request, spec.local_job_id, spec.owner_principal)
        try:
            peer.launch_job(handoff)
        except ConnectError as exc:
            # A genuine name collision on the peer (a local job, or a job it received
            # from a different requester) is terminal, not transient — terminalize the
            # handle so the re-drive stops rather than re-delivering forever.
            if exc.code == Code.ALREADY_EXISTS:
                self._store.mark_handoff_rejected(
                    spec.local_job_id,
                    reason=f"Peer {spec.peer_id} rejected the handoff: job {spec.local_job_id} already exists there",
                )
                return
            logger.warning("Handoff of %s to peer %s failed (will retry): %s", spec.local_job_id, spec.peer_id, exc)
            return
        except (ConnectionError, OSError) as exc:
            logger.warning("Handoff of %s to peer %s failed (will retry): %s", spec.local_job_id, spec.peer_id, exc)
            return
        self._store.mark_handed_off(spec.local_job_id)

    def _sync_peer(self, peer: FederationPeer) -> None:
        assert self._store is not None
        cursor = self._store.read_cursor(peer.peer_id)
        request = controller_pb2.Controller.FederationSyncRequest(requester_id=self._cluster_id, cursor=cursor)
        try:
            response = peer.federation_sync(request)
        except _PEER_RPC_ERRORS as exc:
            logger.warning("Federation sync with peer %s failed: %s", peer.peer_id, exc)
            return
        self._store.apply_sync_batch(
            peer.peer_id,
            list(response.deltas),
            next_cursor=response.next_cursor,
            cursor_stale=response.cursor_stale,
        )

    # -- helpers -------------------------------------------------------------

    def _build_handoff_request(
        self,
        request: controller_pb2.Controller.LaunchJobRequest,
        local_job_id: JobName,
        owner_principal: str,
    ) -> controller_pb2.Controller.LaunchJobRequest:
        """The request delivered to the peer: the same cluster-invariant job name,
        federation attribution, and the routing directives stripped (the peer matches
        workers, not the parent's ``backend``/``cluster`` pins). Idempotency of a
        re-drive is owned by the peer's federation-aware admission, which returns the
        existing job for a re-drive from the same requester."""
        handoff = controller_pb2.Controller.LaunchJobRequest()
        handoff.CopyFrom(request)
        handoff.name = local_job_id.to_wire()
        kept = [c for c in request.constraints if c.key not in (BACKEND_CONSTRAINT_KEY, CLUSTER_CONSTRAINT_KEY)]
        del handoff.constraints[:]
        handoff.constraints.extend(kept)
        handoff.federation.CopyFrom(
            controller_pb2.Controller.FederationHandoff(
                requester_id=self._cluster_id,
                owner_principal=owner_principal,
            )
        )
        return handoff

    def _build_summary(self, peer: FederationPeer) -> controller_pb2.Controller.PeerSummary:
        heartbeat = peer.heartbeat()
        active = self._store.active_federated_job_count(peer.peer_id) if self._store is not None else 0
        return controller_pb2.Controller.PeerSummary(
            peer_id=peer.peer_id,
            controller_address=peer.controller_address,
            dashboard_url=peer.dashboard_url,
            reachable=heartbeat.reachable,
            last_contact_ms=heartbeat.last_contact_ms,
            active_federated_jobs=active,
            backends=heartbeat.backends,
        )
