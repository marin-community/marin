# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-controller federation journeys over the authenticated peer boundary."""

from pathlib import Path

import pytest
from iris.cluster.config import PeerConfig
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY
from iris.cluster.controller.controller import Controller
from iris.cluster.controller.resources.jobs import FederationSubmission
from iris.cluster.federation.legacy_rpc import federation_batch_from_legacy
from iris.cluster.federation.peer import FederationPeer, HandoffDelivery
from iris.cluster.federation.store import FederationSyncBatch
from iris.cluster.resources.endpoint import ExecRequest, ExecResult, ProfileRequest, ProfileResult
from iris.cluster.resources.identity import ResourceKey, ResourceKind
from iris.cluster.resources.job import JobDetail
from iris.cluster.resources.task import TaskSummary
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from rigging.server_auth import VerifiedIdentity, identity_scope
from tests.journeys.world import JobRef, JourneyWorld, TaskRef

PARENT_CLUSTER_ID = "journey-parent"
PEER_ID = "peer-b"
_PEER_CONTROLLER_ADDRESS = "http://peer-b.invalid"
_PEER_IDENTITY = VerifiedIdentity(user_id=PARENT_CLUSTER_ID, role="admin")
_READER_IDENTITY = VerifiedIdentity(user_id="journey-reader", role="admin")


class InProcessPeerConnection:
    """Authenticated parent-to-peer RPCs against a real in-process service."""

    def __init__(self, controller: Controller) -> None:
        self._controller = controller
        self._reachable = True

    def set_reachable(self, reachable: bool) -> None:
        self._reachable = reachable

    def _require_reachable(self) -> None:
        if not self._reachable:
            raise ConnectionError("peer-b is unreachable")

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]:
        self._require_reachable()
        return [controller_pb2.Controller.BackendSummary(backend_id="default")]

    def launch_job(self, delivery: HandoffDelivery) -> None:
        self._require_reachable()
        with identity_scope(_PEER_IDENTITY):
            self._controller.resources.submit_federated_job(
                delivery.spec,
                delivery.bundle_blob,
                FederationSubmission(
                    requester_id=delivery.requester_id,
                    owner_principal=delivery.owner_principal,
                    submitting_user=delivery.submitting_user,
                    handoff_nonce=delivery.handoff_nonce,
                ),
            )

    def federation_sync(self, requester_id: str, cursor: str) -> FederationSyncBatch:
        self._require_reachable()
        with identity_scope(_PEER_IDENTITY):
            response = self._controller.federation_sync(
                controller_pb2.Controller.FederationSyncRequest(requester_id=requester_id, cursor=cursor)
            )
        return federation_batch_from_legacy(response)

    def terminate_job(self, job_id: JobName) -> None:
        self._require_reachable()
        with identity_scope(_PEER_IDENTITY):
            identity = self._controller.resources.describe_job(
                ResourceKey(PARENT_CLUSTER_ID, ResourceKind.JOB, job_id.to_wire())
            ).summary.identity
            self._controller.resources.cancel_job(
                identity,
                idempotency_key=f"journey-federated-cancel:{identity.job_uid}",
                principal_id=job_id.user,
            )

    def profile_task(self, request: ProfileRequest) -> ProfileResult:
        raise NotImplementedError("federation journey does not provide a process runtime")

    def exec_in_container(self, request: ExecRequest) -> ExecResult:
        raise NotImplementedError("federation journey does not provide a container runtime")

    def get_process_status(self, request: job_pb2.GetProcessStatusRequest) -> job_pb2.GetProcessStatusResponse:
        raise NotImplementedError("federation journey does not provide a process runtime")

    def shutdown(self) -> None:
        pass


class FederationJourney:
    """Drive two real controllers while scripting only peer reachability."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.peer = JourneyWorld(root / "peer", monkeypatch, cluster_id=PEER_ID)
        self.connection = InProcessPeerConnection(self.peer.controller)
        self._federation_peer = FederationPeer(
            PEER_ID,
            PeerConfig(controller_address=_PEER_CONTROLLER_ADDRESS),
            self.connection,
        )
        self._federation_peer.probe()
        self.parent = JourneyWorld(
            root / "parent",
            monkeypatch,
            cluster_id=PARENT_CLUSTER_ID,
            peer_configs={PEER_ID: PeerConfig(controller_address=_PEER_CONTROLLER_ADDRESS)},
            federation_peers=[self._federation_peer],
        )
        self.manager = self.parent.controller.federation

    def close(self) -> None:
        self.parent.close()
        self.peer.close()

    def submit(self, name: str, *, tasks: int = 1) -> JobRef:
        return self.parent.submit(name, tasks=tasks, required_attributes={CLUSTER_CONSTRAINT_KEY: PEER_ID})

    def submit_child_on_peer(self, parent: JobRef, name: str, *, tasks: int = 1) -> JobRef:
        """Submit one descendant from the execution cluster."""
        return self.peer.submit_child(parent, name, tasks=tasks)

    def promote(self) -> None:
        """Run the parent tick that assigns a queued Job to peer B."""
        self.parent.step()

    def sync(self) -> None:
        """Run one parent-to-peer delivery and delta-sync pass."""
        self.manager.sync_once()

    def set_peer_reachable(self, reachable: bool) -> None:
        self.connection.set_reachable(reachable)
        self._federation_peer.probe()

    def run_peer(self) -> None:
        self.peer.step()

    def succeed_on_peer(self, task: TaskRef) -> None:
        self.peer.succeed(task)
        self.peer.settle()

    def cancel(self, job: JobRef) -> None:
        self.parent.cancel(job)
        self.parent.step()

    def parent_job(self, job: JobRef) -> JobDetail:
        return self.parent.job(job)

    def peer_job(self, job: JobRef) -> JobDetail:
        return self.peer.job(job)

    def parent_tasks(self, job: JobRef) -> list[TaskSummary]:
        return list(self.parent.tasks(job))

    def peer_tasks(self, job: JobRef) -> list[TaskSummary]:
        return list(self.peer.tasks(job))

    def peer_summary(self) -> controller_pb2.Controller.PeerSummary:
        with identity_scope(_READER_IDENTITY):
            response = self.parent.controller.list_peers()
        (summary,) = response.peers
        return summary
