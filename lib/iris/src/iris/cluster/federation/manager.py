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

from iris.cluster.bundle import BundleStore
from iris.cluster.constraints import BACKEND_CONSTRAINT_KEY, CLUSTER_CONSTRAINT_KEY
from iris.cluster.federation.availability import (
    BackendAvailability,
    PeerAvailability,
    Promotion,
    QueuedCandidate,
    ReservationLedger,
    assign_queued,
)
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.federation.router import PeerRouter, RoutingRequest, SubmitPlan
from iris.cluster.federation.store import (
    CancelTarget,
    FederationStore,
    HandoffSpec,
)
from iris.cluster.types import JobName
from iris.managed_thread import ManagedThread, ThreadContainer
from iris.rpc import controller_pb2, job_pb2

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_INTERVAL = Duration.from_seconds(30)
DEFAULT_SYNC_INTERVAL = Duration.from_seconds(3)
# Safety cap on federation queue promotions per peer per control tick, on top of
# the reservation ledger. Consume up to a peer's advertised free capacity per
# heartbeat, but never promote more than this many jobs to one peer in a single
# tick, so a burst can't over-commit a peer against one stale observation.
DEFAULT_MAX_HANDOFFS_PER_CYCLE = 8
_JOIN_TIMEOUT = Duration.from_seconds(5.0)

_PEER_RPC_ERRORS = (ConnectError, ConnectionError, OSError)

# A peer's verdict on the handoff itself, which it will repeat on every retry: the job
# id already exists there (ALREADY_EXISTS), its allowlist refuses the submitter
# (PERMISSION_DENIED), or the request is malformed (INVALID_ARGUMENT). Transport and
# auth failures are excluded — a federation bearer is minted per request, so
# UNAUTHENTICATED is a key/clock/rollout transient that a later attempt can clear.
_TERMINAL_HANDOFF_CODES = frozenset({Code.ALREADY_EXISTS, Code.PERMISSION_DENIED, Code.INVALID_ARGUMENT})


class FederationManager:
    """Owns the federation peer registry, handoff, delta-sync, and cancel."""

    def __init__(
        self,
        peers: Sequence[FederationPeer],
        *,
        threads: ThreadContainer,
        store: FederationStore | None = None,
        bundles: BundleStore | None = None,
        cluster_id: str = "",
        heartbeat_interval: Duration = DEFAULT_HEARTBEAT_INTERVAL,
        sync_interval: Duration = DEFAULT_SYNC_INTERVAL,
        max_handoffs_per_cycle: int = DEFAULT_MAX_HANDOFFS_PER_CYCLE,
    ):
        self._peers = {peer.peer_id: peer for peer in peers}
        self._threads = threads
        self._store = store
        self._bundles = bundles
        self._cluster_id = cluster_id
        self._heartbeat_interval = heartbeat_interval
        self._sync_interval = sync_interval
        self._max_handoffs_per_cycle = max_handoffs_per_cycle
        self._router = PeerRouter(peers)
        # In-memory reservation ledger for the control-tick federation pass: capacity
        # already promoted against each peer backend since its last heartbeat, so
        # successive ticks between heartbeats do not each re-spend the same number.
        self._ledger = ReservationLedger()
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

    def classify_submit(self, request: RoutingRequest) -> SubmitPlan:
        """Classify a submission as local, federation-queued, or unschedulable.

        Submit never selects a peer — that is a control-tick decision. See
        :class:`~iris.cluster.federation.router.PeerRouter`.
        """
        return self._router.classify(request)

    def has_peer(self, peer_id: str) -> bool:
        """Whether ``peer_id`` names a configured federation peer."""
        return peer_id in self._peers

    def peer_controller_address(self, peer_id: str) -> str | None:
        """The configured controller base URL of ``peer_id``, or None when it is not a peer."""
        peer = self._peers.get(peer_id)
        return peer.controller_address if peer is not None else None

    def peer_summaries(self) -> list[controller_pb2.Controller.PeerSummary]:
        """A ``PeerSummary`` for every configured peer, ordered by peer id."""
        return [self._build_summary(peer) for _, peer in sorted(self._peers.items())]

    # -- queue admission (parent side) ---------------------------------------

    def queue_federated(
        self,
        *,
        local_job_id: JobName,
        request: controller_pb2.Controller.LaunchJobRequest,
        pinned_peer_id: str,
        owner_principal: str,
        submitting_user: str,
    ) -> None:
        """Admit a job to the federation queue (``QUEUED_HANDOFF``), choosing no peer.

        In one local transaction the store persists the ``jobs``/``job_config``/
        ``federated_jobs`` handle (no task rows, no cluster) queued on the parent. The
        control tick's federation pass later picks a peer that has room and promotes
        the handle to ``PENDING_HANDOFF``, and the sync loop delivers it. ``pinned_peer_id``
        is "" for an unpinned candidate, or the peer a ``cluster=<peer>`` pin named (the
        tick then only ever assigns it there). An idempotent resubmit is a no-op.

        A peer's own rejection (e.g. its allowlist) is not visible at admission: it
        arrives later as a failed job once the tick promotes the handle and delivery is
        attempted.
        """
        if self._store is None:
            raise RuntimeError("federation queueing requires a store")
        if pinned_peer_id:
            self._require_peer(pinned_peer_id)
        self._store.admit_and_persist_queued(
            HandoffSpec(
                local_job_id=local_job_id,
                peer_id=pinned_peer_id,
                owner_principal=owner_principal,
                submitting_user=submitting_user,
                request=request,
            )
        )

    # -- control-tick federation pass (peer selection) -----------------------

    def peer_availability(self) -> list[PeerAvailability]:
        """Each configured peer's reachability + per-backend advertised availability.

        Read off the latest capability heartbeat. A backend that set the availability
        wrapper supplies a free-capacity metric (``supplies_metric``); a legacy backend
        that did not is matched on shape alone.
        """
        result: list[PeerAvailability] = []
        for peer_id, peer in sorted(self._peers.items()):
            heartbeat = peer.heartbeat()
            backends = [
                BackendAvailability(
                    backend_id=backend.backend_id,
                    supplies_metric=backend.HasField("availability"),
                    generation=backend.availability.observation_epoch_ms,
                    amounts=dict(backend.availability.amounts),
                    advertised_shape={key: list(values.values) for key, values in backend.advertised_attributes.items()},
                )
                for backend in heartbeat.backends
            ]
            result.append(PeerAvailability(peer_id=peer_id, reachable=heartbeat.reachable, backends=backends))
        return result

    def plan_federation(self, candidates: list[QueuedCandidate]) -> list[Promotion]:
        """Decide which queued candidates to promote to which peer this tick (pure).

        Snapshots peer availability, runs the availability-gated assignment against the
        reservation ledger, and returns the promotions. Does not mutate the ledger — the
        controller applies each promotion as a conditional CAS and calls
        :meth:`confirm_promotions` for the ones that actually committed.
        """
        if not self._peers or not candidates:
            return []
        self._ledger.drop_peers(set(self._peers))
        return assign_queued(
            candidates,
            self.peer_availability(),
            self._ledger,
            max_per_peer_per_cycle=self._max_handoffs_per_cycle,
        )

    def confirm_promotions(self, promotions: list[Promotion]) -> None:
        """Charge the reservation ledger for promotions whose CAS committed this tick."""
        for promotion in promotions:
            self._ledger.commit(promotion)

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
        """Deliver one handle to its peer, terminalizing a rejection the peer will repeat.

        A rejection marks the handle ``HANDOFF_REJECTED`` so the re-drive stops; the
        user learns the peer's verdict from the failed job. Never raises on a peer's
        answer: the re-drive loop calls this on the sync thread, which dies on an
        uncaught exception.
        """
        assert self._store is not None
        peer = self._peers.get(spec.peer_id)
        if peer is None:
            logger.warning("Cannot hand off %s: peer %s is not configured", spec.local_job_id, spec.peer_id)
            return
        try:
            handoff = self._build_handoff_request(
                spec.request, spec.local_job_id, spec.owner_principal, spec.submitting_user
            )
            peer.launch_job(handoff)
        except ConnectError as exc:
            # The peer answers a rejected handoff the same way every time — a name
            # collision there, a submitter its allowlist refuses, a malformed request.
            # Terminalize the handle so the re-drive stops rather than re-delivering
            # forever. Everything else (unreachable, unauthenticated, timed out) can
            # succeed on a later attempt and stays pending.
            if exc.code not in _TERMINAL_HANDOFF_CODES:
                logger.warning("Handoff of %s to peer %s failed (will retry): %s", spec.local_job_id, spec.peer_id, exc)
                return
            self._store.mark_handoff_rejected(
                spec.local_job_id,
                reason=f"Peer {spec.peer_id} rejected the handoff: {exc.message}",
            )
            return
        except (ConnectionError, OSError) as exc:
            # Also covers a blob this cluster's bundle store could not read back for
            # the handoff; a later attempt can still succeed.
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
            endpoints=list(response.endpoints),
        )

    # -- helpers -------------------------------------------------------------

    def _build_handoff_request(
        self,
        request: controller_pb2.Controller.LaunchJobRequest,
        local_job_id: JobName,
        owner_principal: str,
        submitting_user: str,
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
                submitting_user=submitting_user,
            )
        )
        self._inline_blobs(handoff)
        return handoff

    def _inline_blobs(self, handoff: controller_pb2.Controller.LaunchJobRequest) -> None:
        """Carry the bytes behind every content id, which only this cluster can resolve.

        ``launch_job`` replaces the submitted workspace bundle and any large workdir
        file with a content id in this cluster's bundle store, and a task fetches its
        bundle from the controller that runs it. A peer reads its own store, so the
        handoff carries the bytes; the peer re-externalizes them under the same
        content ids on the way in.
        """
        refs = dict(handoff.entrypoint.workdir_file_refs)
        if not handoff.bundle_id and not refs:
            return
        assert self._bundles is not None, "federating a job that references blobs needs a bundle store"
        if handoff.bundle_id:
            handoff.bundle_blob = self._bundles.get(handoff.bundle_id)
            handoff.ClearField("bundle_id")
        for name, blob_id in refs.items():
            handoff.entrypoint.workdir_files[name] = self._bundles.get(blob_id)
        handoff.entrypoint.ClearField("workdir_file_refs")

    def _build_summary(self, peer: FederationPeer) -> controller_pb2.Controller.PeerSummary:
        heartbeat = peer.heartbeat()
        active = self._store.active_federated_job_count(peer.peer_id) if self._store is not None else 0
        return controller_pb2.Controller.PeerSummary(
            peer_id=peer.peer_id,
            controller_address=peer.controller_address,
            reachable=heartbeat.reachable,
            last_contact_ms=heartbeat.last_contact_ms,
            active_federated_jobs=active,
            backends=heartbeat.backends,
        )
