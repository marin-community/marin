# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct endpoints cross real controller persistence and the native K8s adapter."""

from dataclasses import dataclass

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.backends.k8s.tasks import K8sTaskProvider
from iris.cluster.bundle import BundleStore
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.reconcile import dispatch
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.network_policies import (
    DRIVER_OWNER_LABEL,
    DRIVER_POD_UID_LABEL,
    DRIVER_PORT_LABEL,
    build_egress_network_policy,
)
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.runtime.sandbox import TaskNetwork
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from iris.testing.controller import MockController, make_controller_state, make_direct_job_request
from iris.testing.k8s import make_batch, make_kueue_provider
from iris.testing.transitions import commit_dispatch_updates
from rigging.server_auth import VerifiedIdentity, identity_scope
from rigging.timing import Timestamp


@dataclass
class ConnectionCase:
    service: ControllerServiceImpl
    backend: K8sTaskProvider
    db: ControllerDB
    k8s: InMemoryK8sService
    driver: job_pb2.RunTaskRequest
    child: job_pb2.RunTaskRequest
    driver_pod: dict
    child_pod: dict

    def request(self) -> controller_pb2.Controller.GetTaskConnectionRequest:
        return controller_pb2.Controller.GetTaskConnectionRequest(
            driver_task_id=self.driver.task_id,
            driver_attempt_uid=self.driver.attempt_uid,
            task_id=self.child.task_id,
            attempt_uid=self.child.attempt_uid,
        )

    def identity(self) -> VerifiedIdentity:
        return VerifiedIdentity(user_id="test-user", role="task", job_id="/test-user/driver")


@pytest.fixture
def connection(log_client, tmp_path):
    k8s = InMemoryK8sService()
    k8s.apply_json(build_egress_network_policy("iris", TaskNetwork.NONE))
    backend = make_kueue_provider(k8s)
    runtime = MockController()
    runtime.backend = backend
    with make_controller_state() as state:
        service = ControllerServiceImpl(
            controller=runtime,
            bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
            log_client=log_client,
            db=state._db,
            endpoint_service=EndpointServiceImpl(db=state._db),
            auth=ControllerAuth(provider="static"),
        )

        def launch(request, identity, uid, ip):
            with identity_scope(identity):
                service.launch_job(request, None)
            with state._db.transaction() as tx:
                batch = dispatch.drain_for_dispatch(tx)
            [run] = batch.tasks_to_run
            if request.environment.HasField("driver_connection"):
                # An endpoint must be selected by ingress isolation before Pod creation,
                # including when no egress policy would otherwise isolate its ingress.
                assert any(
                    policy["spec"]["podSelector"]["matchLabels"].get(DRIVER_OWNER_LABEL) == "driver-pod-uid"
                    for policy in k8s.list_json(K8sResource.NETWORK_POLICIES)
                )
            backend.sync(make_batch(tasks_to_run=[run], running_tasks=list(batch.running_tasks)))
            pod = next(
                pod
                for pod in k8s.list_json(K8sResource.PODS)
                if pod["metadata"]["labels"].get("iris.attempt_uid") == run.attempt_uid
            )
            pod["metadata"]["uid"] = uid
            pod["status"] = {
                "phase": "Running",
                "podIP": ip,
                "containerStatuses": [{"name": "task", "state": {"running": {}}}],
            }
            k8s.seed_resource(K8sResource.PODS, pod["metadata"]["name"], pod)
            with state._db.transaction() as tx:
                commit_dispatch_updates(
                    tx,
                    [
                        TaskUpdate(
                            task_id=JobName.from_wire(run.task_id),
                            attempt_id=run.attempt_id,
                            new_state=job_pb2.TASK_STATE_RUNNING,
                            pod_uid=uid,
                        )
                    ],
                    now=Timestamp.now(),
                )
            return run, pod

        driver_request = make_direct_job_request("driver")
        driver_request.environment.network_attachment = job_pb2.NETWORK_ATTACHMENT_POD
        driver, driver_pod = launch(
            driver_request, VerifiedIdentity(user_id="test-user", role="user"), "driver-pod-uid", "10.2.0.1"
        )
        request = make_direct_job_request("child")
        request.name = "/test-user/driver/child"
        request.container_profile = job_pb2.CONTAINER_PROFILE_SANDBOX
        request.egress_policy = job_pb2.EGRESS_POLICY_NONE
        request.environment.network_attachment = job_pb2.NETWORK_ATTACHMENT_POD
        request.environment.driver_connection.CopyFrom(
            job_pb2.DriverConnectionSpec(
                driver_task_id=driver.task_id,
                driver_attempt_uid=driver.attempt_uid,
                port=10003,
            )
        )
        child, child_pod = launch(
            request,
            VerifiedIdentity(user_id="test-user", role="task", job_id="/test-user/driver"),
            "child-pod-uid",
            "10.2.0.2",
        )
        try:
            yield ConnectionCase(service, backend, state._db, k8s, driver, child, driver_pod, child_pod)
        finally:
            backend.close()


def test_driver_connection_returns_actual_endpoint_and_authorizes_only_its_owner_pod(connection):
    with identity_scope(connection.identity()):
        endpoint = connection.service.get_task_connection(connection.request(), None)
    assert (endpoint.pod_ip, endpoint.port, endpoint.pod_uid, endpoint.attempt_uid) == (
        "10.2.0.2",
        10003,
        "child-pod-uid",
        connection.child.attempt_uid,
    )
    policies = connection.k8s.list_json(K8sResource.NETWORK_POLICIES)
    ingress = next(policy for policy in policies if policy["metadata"]["name"].startswith("iris-driver-"))
    assert ingress["spec"]["podSelector"]["matchLabels"] == {
        DRIVER_OWNER_LABEL: "driver-pod-uid",
        DRIVER_PORT_LABEL: "10003",
    }
    assert ingress["spec"]["ingress"] == [
        {
            "from": [
                {
                    "podSelector": {
                        "matchLabels": {
                            DRIVER_POD_UID_LABEL: "driver-pod-uid",
                            "iris.attempt_uid": connection.driver.attempt_uid,
                        }
                    }
                }
            ],
            "ports": [{"protocol": "TCP", "port": 10003}],
        }
    ]
    assert ingress["metadata"]["ownerReferences"][0]["uid"] == "driver-pod-uid"
    assert "egress" not in ingress["spec"]
    denied = next(policy for policy in policies if policy["metadata"]["name"] == "iris-egress-none")
    assert denied["spec"]["ingress"] == []
    assert [port["port"] for port in denied["spec"]["egress"][0]["ports"]] == [53, 53]


@pytest.mark.parametrize(
    "identity",
    [
        VerifiedIdentity(user_id="test-user", role="admin"),
        VerifiedIdentity(user_id="test-user", role="task", job_id="/test-user/other"),
        VerifiedIdentity(user_id="other-user", role="task", job_id="/test-user/driver"),
    ],
)
def test_driver_connection_rejects_external_admin_and_other_job_identities(connection, identity):
    with identity_scope(identity), pytest.raises(ConnectError) as rejected:
        connection.service.get_task_connection(connection.request(), None)
    assert rejected.value.code == Code.PERMISSION_DENIED
    assert len(connection.k8s.list_json(K8sResource.NETWORK_POLICIES)) == 2


@pytest.mark.parametrize("change", ["driver_attempt", "child_attempt", "driver_pod", "host_network", "terminating"])
def test_driver_connection_fails_closed_after_identity_or_liveness_changes(connection, change):
    request = connection.request()
    if change == "driver_attempt":
        request.driver_attempt_uid = "stale"
    elif change == "child_attempt":
        request.attempt_uid = "stale"
    else:
        pod = connection.driver_pod
        if change == "driver_pod":
            pod["metadata"]["uid"] = "replacement"
        elif change == "host_network":
            pod["spec"]["hostNetwork"] = True
        else:
            pod["metadata"]["deletionTimestamp"] = "2026-10-31T00:00:00Z"
        connection.k8s.seed_resource(K8sResource.PODS, pod["metadata"]["name"], pod)
    with identity_scope(connection.identity()), pytest.raises(ConnectError) as rejected:
        connection.service.get_task_connection(request, None)
    assert rejected.value.code in (Code.PERMISSION_DENIED, Code.FAILED_PRECONDITION)
    assert len(connection.k8s.list_json(K8sResource.NETWORK_POLICIES)) == 2


def test_driver_ingress_precedes_cluster_endpoint_pod_creation_and_is_shared_across_children(connection):
    request = make_direct_job_request("another-child")
    request.name = "/test-user/driver/another-child"
    request.egress_policy = job_pb2.EGRESS_POLICY_CLUSTER
    request.environment.network_attachment = job_pb2.NETWORK_ATTACHMENT_POD
    request.environment.driver_connection.CopyFrom(
        job_pb2.DriverConnectionSpec(
            driver_task_id=connection.driver.task_id,
            driver_attempt_uid=connection.driver.attempt_uid,
            port=10003,
        )
    )
    with identity_scope(connection.identity()):
        connection.service.launch_job(request, None)
    policies = connection.k8s.list_json(K8sResource.NETWORK_POLICIES)
    ingress = [policy for policy in policies if policy["metadata"]["name"].startswith("iris-driver-")]
    assert len(ingress) == 1
    assert len(connection.k8s.list_json(K8sResource.PODS)) == 2
    with connection.db.transaction() as tx:
        batch = dispatch.drain_for_dispatch(tx)
    [run] = batch.tasks_to_run
    connection.backend.sync(batch)
    pod = next(
        pod
        for pod in connection.k8s.list_json(K8sResource.PODS)
        if pod["metadata"]["labels"].get("iris.attempt_uid") == run.attempt_uid
    )
    assert pod["metadata"]["labels"][DRIVER_OWNER_LABEL] == "driver-pod-uid"
    assert pod["metadata"]["labels"][DRIVER_PORT_LABEL] == "10003"
    assert connection.k8s.list_json(K8sResource.NETWORK_POLICIES) == policies


@pytest.mark.parametrize("change", ["supplied_pod_uid", "stale_driver", "host_network", "isolated_driver"])
def test_child_launch_rejects_unbound_or_untrusted_driver(connection, change):
    request = make_direct_job_request("another-child")
    request.name = "/test-user/driver/another-child"
    request.environment.network_attachment = job_pb2.NETWORK_ATTACHMENT_POD
    request.environment.driver_connection.CopyFrom(
        job_pb2.DriverConnectionSpec(
            driver_task_id=connection.driver.task_id,
            driver_attempt_uid=connection.driver.attempt_uid,
            port=10003,
        )
    )
    if change == "supplied_pod_uid":
        request.environment.driver_connection.driver_pod_uid = "driver-pod-uid"
    elif change == "stale_driver":
        request.environment.driver_connection.driver_attempt_uid = "stale"
    elif change == "isolated_driver":
        request.name = "/test-user/driver/child/grandchild"
        request.environment.driver_connection.driver_task_id = connection.child.task_id
        request.environment.driver_connection.driver_attempt_uid = connection.child.attempt_uid
    else:
        connection.driver_pod["spec"]["hostNetwork"] = True
        connection.k8s.seed_resource(K8sResource.PODS, connection.driver_pod["metadata"]["name"], connection.driver_pod)
    identity = connection.identity()
    if change == "isolated_driver":
        identity = VerifiedIdentity(user_id="test-user", role="task", job_id="/test-user/driver/child")
    with identity_scope(identity), pytest.raises(ConnectError) as rejected:
        connection.service.launch_job(request, None)
    assert rejected.value.code in (Code.INVALID_ARGUMENT, Code.PERMISSION_DENIED, Code.FAILED_PRECONDITION)
    with identity_scope(connection.identity()):
        jobs = connection.service.list_jobs(controller_pb2.Controller.ListJobsRequest(), None)
    assert request.name not in {job.job_id for job in jobs.jobs}


def test_pod_attachment_rejects_distributed_gpu_infiniband(connection):
    request = make_direct_job_request("distributed-driver")
    request.environment.network_attachment = job_pb2.NETWORK_ATTACHMENT_POD
    request.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="H100", count=8))
    request.replicas = 2
    with identity_scope(VerifiedIdentity(user_id="test-user", role="user")), pytest.raises(ConnectError) as rejected:
        connection.service.launch_job(request, None)
    assert rejected.value.code == Code.INVALID_ARGUMENT
    assert "InfiniBand" in rejected.value.message
