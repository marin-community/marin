# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-controller federation journeys over the authenticated peer boundary."""

from pathlib import Path

import pytest
from google.protobuf import any_pb2
from iris.cluster.config import PeerConfig
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY
from iris.cluster.controller.composition import wire_resource_service
from iris.cluster.controller.job import FederationSubmission
from iris.cluster.controller.process import ControllerProcess
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.federation.protocol import (
    FederationBackendObservation,
    FederationPeerObservation,
    FederationSyncBatch,
    HandoffDelivery,
    PeerCallError,
    PeerErrorCode,
)
from iris.resources.action import ActionReceipt
from iris.resources.endpoint import ExecRequest, ExecResult, ProfileRequest, ProfileResult
from iris.resources.identity import AttemptIdentity, JobIdentity, ResourceKey, ResourceKind, TaskIdentity
from iris.resources.job import JobDetail
from iris.resources.names import JobName
from iris.resources.system import ProcessInfo
from iris.resources.task import TaskSummary
from iris.rpc import (
    controller_pb2,
    resource_action_pb2,
    resource_command_pb2,
    resource_job_pb2,
    resource_pb2,
    resource_task_pb2,
)
from iris.rpc.auth import FEDERATION_PEER_ROLE
from iris.rpc.federation_client import federation_batch_from_legacy
from iris.rpc.profile_codec import profile_configuration_to_proto
from iris.rpc.resource_client_codec import (
    exec_result_from_proto,
    profile_result_from_proto,
)
from iris.rpc.resource_codec import (
    action_receipt_from_proto,
    attempt_identity_to_proto,
)
from iris.rpc.resource_types import ATTEMPT, EXEC_SESSION, JOB, PROFILE_CAPTURE, TASK
from iris.time_proto import duration_to_proto
from rigging.server_auth import VerifiedIdentity, identity_scope
from tests.journeys.world import JobRef, JourneyWorld, TaskRef

PARENT_CLUSTER_ID = "journey-parent"
PEER_ID = "peer-b"
_PEER_CONTROLLER_ADDRESS = "http://peer-b.invalid"
_PEER_IDENTITY = VerifiedIdentity(user_id=PARENT_CLUSTER_ID, role=FEDERATION_PEER_ROLE)


def _pack(value) -> any_pb2.Any:
    result = any_pb2.Any()
    result.Pack(value)
    return result


def _ref(cluster_id: str, resource_type: str, resource_id: str, uid: str) -> resource_pb2.ResourceRef:
    return resource_pb2.ResourceRef(
        authority_cluster_id=cluster_id,
        type=resource_type,
        id=resource_id,
        uid=uid,
    )


def _attempt_ref(identity: AttemptIdentity) -> resource_pb2.ResourceRef:
    return _ref(
        identity.task.cluster_id,
        ATTEMPT,
        f"{identity.task.resource_id}:{identity.attempt_number}",
        identity.attempt_uid,
    )


class InProcessPeerConnection:
    """Authenticated parent-to-peer RPCs against a real in-process service."""

    def __init__(self, controller: ControllerProcess) -> None:
        self._controller = controller
        self._resources = wire_resource_service(controller.controller)
        self._reachable = True

    def set_reachable(self, reachable: bool) -> None:
        self._reachable = reachable

    def _require_reachable(self) -> None:
        if not self._reachable:
            raise PeerCallError(PeerErrorCode.UNAVAILABLE, "peer-b is unreachable")

    def list_backends(self) -> tuple[FederationBackendObservation, ...]:
        self._require_reachable()
        return (FederationBackendObservation(backend_id="default"),)

    def launch_job(self, delivery: HandoffDelivery) -> None:
        self._require_reachable()
        with identity_scope(_PEER_IDENTITY):
            self._controller.controller.submit_federated_job(
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
            response = self._controller.controller_service.federation_sync(
                controller_pb2.Controller.FederationSyncRequest(requester_id=requester_id, cursor=cursor),
                None,
            )
        return federation_batch_from_legacy(response)

    def terminate_job(self, job_id: JobName) -> None:
        self._require_reachable()
        with identity_scope(_PEER_IDENTITY):
            identity = self._controller.controller.describe_job(
                ResourceKey(PARENT_CLUSTER_ID, ResourceKind.JOB, job_id.to_wire())
            ).summary.identity
            self._controller.controller.cancel_job(
                identity,
                idempotency_key=f"journey-federated-cancel:{identity.job_uid}",
                principal_id=job_id.user,
            )

    def cancel_job(self, identity: JobIdentity, *, idempotency_key: str, reason: str) -> ActionReceipt:
        self._require_reachable()
        with identity_scope(_PEER_IDENTITY):
            operation = self._resources.update_resource(
                resource_pb2.UpdateResourceRequest(
                    mutation=resource_pb2.MutationMetadata(request_id=idempotency_key, reason=reason),
                    ref=_ref(identity.key.cluster_id, JOB, identity.key.resource_id, identity.job_uid),
                    update=_pack(resource_job_pb2.JobUpdate(cancel=resource_job_pb2.CancelJobUpdate())),
                ),
                None,
            )
        receipt = resource_action_pb2.ActionReceipt()
        assert operation.result.Unpack(receipt)
        return action_receipt_from_proto(receipt)

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
        reason: str,
    ) -> ActionReceipt:
        self._require_reachable()
        with identity_scope(_PEER_IDENTITY):
            operation = self._resources.update_resource(
                resource_pb2.UpdateResourceRequest(
                    mutation=resource_pb2.MutationMetadata(request_id=idempotency_key, reason=reason),
                    ref=_ref(identity.key.cluster_id, TASK, identity.key.resource_id, identity.task_uid),
                    update=_pack(
                        resource_task_pb2.TaskUpdate(
                            expected_attempt_uid=expected_attempt_uid,
                            preempt=resource_task_pb2.PreemptTaskUpdate(),
                        )
                    ),
                ),
                None,
            )
        receipt = resource_action_pb2.ActionReceipt()
        assert operation.result.Unpack(receipt)
        return action_receipt_from_proto(receipt)

    def terminate_attempt(
        self,
        identity: AttemptIdentity,
        *,
        idempotency_key: str,
        reason: str,
    ) -> ActionReceipt:
        self._require_reachable()
        with identity_scope(_PEER_IDENTITY):
            operation = self._resources.update_resource(
                resource_pb2.UpdateResourceRequest(
                    mutation=resource_pb2.MutationMetadata(request_id=idempotency_key, reason=reason),
                    ref=_attempt_ref(identity),
                    update=_pack(resource_task_pb2.AttemptUpdate(terminate=resource_task_pb2.TerminateAttemptUpdate())),
                ),
                None,
            )
        receipt = resource_action_pb2.ActionReceipt()
        assert operation.result.Unpack(receipt)
        return action_receipt_from_proto(receipt)

    def fail_attempt(
        self,
        identity: AttemptIdentity,
        *,
        idempotency_key: str,
        reason: str,
    ) -> ActionReceipt:
        self._require_reachable()
        with identity_scope(_PEER_IDENTITY):
            operation = self._resources.update_resource(
                resource_pb2.UpdateResourceRequest(
                    mutation=resource_pb2.MutationMetadata(request_id=idempotency_key, reason=reason),
                    ref=_attempt_ref(identity),
                    update=_pack(resource_task_pb2.AttemptUpdate(fail=resource_task_pb2.FailAttemptUpdate())),
                ),
                None,
            )
        receipt = resource_action_pb2.ActionReceipt()
        assert operation.result.Unpack(receipt)
        return action_receipt_from_proto(receipt)

    def profile_task(self, request: ProfileRequest) -> ProfileResult:
        assert request.attempt is not None
        self._require_reachable()
        wire = resource_command_pb2.CreateProfileCapture(
            attempt=attempt_identity_to_proto(request.attempt),
            profile=profile_configuration_to_proto(request.profile),
        )
        if request.duration is not None:
            wire.duration.CopyFrom(duration_to_proto(request.duration))
        with identity_scope(_PEER_IDENTITY):
            operation = self._resources.create_resource(
                resource_pb2.CreateResourceRequest(
                    mutation=resource_pb2.MutationMetadata(request_id="federation-profile"),
                    type=PROFILE_CAPTURE,
                    parent=_attempt_ref(request.attempt),
                    body=_pack(wire),
                ),
                None,
            )
        response = resource_command_pb2.ProfileCaptureResult()
        assert operation.result.Unpack(response)
        return profile_result_from_proto(response)

    def exec_in_container(self, request: ExecRequest) -> ExecResult:
        self._require_reachable()
        wire = resource_command_pb2.CreateExecSession(
            attempt=attempt_identity_to_proto(request.attempt),
            command=request.command,
        )
        if request.timeout is not None:
            wire.timeout.CopyFrom(duration_to_proto(request.timeout))
        with identity_scope(_PEER_IDENTITY):
            operation = self._resources.create_resource(
                resource_pb2.CreateResourceRequest(
                    mutation=resource_pb2.MutationMetadata(request_id="federation-exec"),
                    type=EXEC_SESSION,
                    parent=_attempt_ref(request.attempt),
                    body=_pack(wire),
                ),
                None,
            )
        response = resource_command_pb2.ExecSessionResult()
        assert operation.result.Unpack(response)
        return exec_result_from_proto(response)

    def get_process_status(self, target: str) -> ProcessInfo:
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
        self.manager = self.parent.controller.runtime.federation

    def close(self) -> None:
        self.parent.close()
        self.peer.close()

    def submit(
        self,
        name: str,
        *,
        user: str = "journey",
        tasks: int = 1,
        preemption_retries: int = 0,
    ) -> JobRef:
        return self.parent.submit(
            name,
            user=user,
            tasks=tasks,
            preemption_retries=preemption_retries,
            required_attributes={CLUSTER_CONSTRAINT_KEY: PEER_ID},
        )

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

    def restart_parent(self) -> None:
        self.parent.restart()
        self.manager = self.parent.controller.runtime.federation

    def parent_job(self, job: JobRef) -> JobDetail:
        return self.parent.job(job)

    def peer_job(self, job: JobRef) -> JobDetail:
        return self.peer.job(job)

    def parent_tasks(self, job: JobRef) -> list[TaskSummary]:
        return list(self.parent.tasks(job))

    def peer_tasks(self, job: JobRef) -> list[TaskSummary]:
        return list(self.peer.tasks(job))

    def peer_summary(self) -> FederationPeerObservation:
        (summary,) = self.parent.controller.runtime.federation.peer_observations()
        return summary
