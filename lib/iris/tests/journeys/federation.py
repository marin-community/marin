# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-controller federation journeys over the authenticated peer boundary."""

from pathlib import Path

import pytest
from iris.cluster.config import PeerConfig
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY
from iris.cluster.controller.controller import Controller
from iris.cluster.federation.peer import FederationPeer
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

    def launch_job(
        self, request: controller_pb2.Controller.LaunchJobRequest
    ) -> controller_pb2.Controller.LaunchJobResponse:
        self._require_reachable()
        with identity_scope(_PEER_IDENTITY):
            return self._controller.launch_job(request)

    def federation_sync(
        self, request: controller_pb2.Controller.FederationSyncRequest
    ) -> controller_pb2.Controller.FederationSyncResponse:
        self._require_reachable()
        with identity_scope(_PEER_IDENTITY):
            return self._controller.federation_sync(request)

    def terminate_job(self, job_id: JobName) -> None:
        self._require_reachable()
        with identity_scope(_PEER_IDENTITY):
            self._controller.terminate_job(job_id.to_wire())

    def profile_task(self, request: job_pb2.ProfileTaskRequest) -> job_pb2.ProfileTaskResponse:
        raise NotImplementedError("federation journey does not provide a process runtime")

    def exec_in_container(
        self, request: controller_pb2.Controller.ExecInContainerRequest
    ) -> controller_pb2.Controller.ExecInContainerResponse:
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

    def parent_job(self, job: JobRef) -> job_pb2.JobStatus:
        return self.parent.job(job)

    def peer_job(self, job: JobRef) -> job_pb2.JobStatus:
        return self.peer.controller.get_job_status(job.wire_id).job

    def parent_tasks(self, job: JobRef) -> list[job_pb2.TaskStatus]:
        return list(self.parent.tasks(job))

    def peer_tasks(self, job: JobRef) -> list[job_pb2.TaskStatus]:
        return list(self.peer.controller.list_tasks(job.wire_id).tasks)

    def peer_summary(self) -> controller_pb2.Controller.PeerSummary:
        with identity_scope(_READER_IDENTITY):
            response = self.parent.controller.list_peers()
        (summary,) = response.peers
        return summary
