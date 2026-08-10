# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import pytest
from iris.backends.protocol import BackendCapability
from iris.backends.status import (
    AutoscalerStatus,
    BackendStatus,
    GroupRoutingStatus,
    KubernetesStatus,
    NodeStatus,
    RoutingStatus,
    ScaleGroupStatus,
    SliceStatus,
    VmStatus,
    WorkerFleetStatus,
)
from iris.cluster.config import BackendConfig
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.controller import CapabilityUrlConfig, Controller
from iris.cluster.controller.endpoint_registry import EndpointRegistry
from iris.cluster.controller.persistence.database import ControllerDB
from iris.cluster.controller.persistence.projections.endpoints import EndpointsProjection
from iris.cluster.controller.persistence.schema import worker_attributes_table, workers_table
from iris.cluster.controller.worker_health import WorkerLiveness
from iris.cluster.types import (
    DEFAULT_BACKEND_ID,
    UserBudgetDefaults,
)
from iris.resources.endpoint import EndpointQuery
from iris.resources.errors import ResourceNotFound
from iris.resources.identity import NodeLocator, ResourceKind, SliceLocator
from iris.resources.node import NodeHealth, NodeQuery
from iris.resources.slice import SliceCapacityState, SliceLifecycle, SliceQuery
from iris.resources.source import SourceState
from iris.rpc import resource_pb2
from iris.rpc.resource_service import ResourceServiceImpl
from rigging.timing import Timestamp
from sqlalchemy import event

NOW = Timestamp.from_ms(1_000)


def _kubernetes_backend() -> Mock:
    backend = Mock()
    backend.name = "kubernetes"
    backend.autoscaler = None
    backend.capabilities = frozenset({BackendCapability.CLUSTER_VIEW})
    backend.advertised_attributes.return_value = {"region": {"us-central1"}}
    backend.resource_capacity.return_value = None
    backend.status.return_value = BackendStatus(
        kubernetes=KubernetesStatus(
            nodes=(
                NodeStatus(
                    name="node-alpha",
                    ready=True,
                    schedulable=True,
                    instance_type="h100-8",
                    region="us-central1",
                    gpu_count=8,
                    gpu_model="h100",
                    cpu_millicores=96_000,
                    memory_bytes=640_000,
                    disk_bytes=1_000_000,
                    running_pods=2,
                    created="2026-01-01T00:00:00Z",
                ),
                NodeStatus(
                    name="node-beta",
                    ready=False,
                    schedulable=False,
                    instance_type="h100-8",
                    region="us-central1",
                    created="2026-01-02T00:00:00Z",
                ),
            )
        )
    )
    return backend


def _autoscaling_backend() -> Mock:
    backend = Mock()
    backend.name = "worker fleet"
    backend.autoscaler = Mock()
    backend.capabilities = frozenset({BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER})
    backend.advertised_attributes.return_value = {"device_variant": {"h100"}, "region": {"us-central1"}}
    backend.resource_capacity.return_value = None
    autoscaler = AutoscalerStatus(
        last_evaluation=NOW,
        groups=(
            ScaleGroupStatus(
                name="pool-a",
                device_type="gpu",
                device_variant="h100",
                region="us-central1",
                current_demand=3,
                slices=(
                    SliceStatus(
                        slice_id="slice-a",
                        scale_group="pool-a",
                        state="ready",
                        created_at=Timestamp.from_ms(10),
                        last_active=Timestamp.from_ms(900),
                        capacity_status="in_use",
                        degraded_slot_count=1,
                        vms=(
                            VmStatus(
                                vm_id="vm-a",
                                worker_id="node-a",
                                worker_healthy=True,
                                usability="healthy",
                                running_task_count=2,
                                zone="us-central1-a",
                            ),
                        ),
                    ),
                    SliceStatus(
                        slice_id="slice-b",
                        scale_group="pool-a",
                        state="booting",
                        created_at=Timestamp.from_ms(20),
                    ),
                ),
            ),
            ScaleGroupStatus(
                name="pool-b",
                slices=(
                    SliceStatus(
                        slice_id="slice-c",
                        scale_group="pool-b",
                        state="failed",
                        error_message="quota denied",
                        created_at=Timestamp.from_ms(30),
                    ),
                ),
            ),
        ),
        last_routing_decision=RoutingStatus(
            group_statuses=(
                GroupRoutingStatus(
                    group="pool-a",
                    assigned=2,
                    launch=1,
                    decision="launch",
                    reason="one more slice required",
                ),
            )
        ),
    )
    backend.status.return_value = BackendStatus(
        worker=WorkerFleetStatus(autoscaler=autoscaler, healthy_worker_count=1, total_worker_count=1)
    )
    backend.autoscaler_status.return_value = autoscaler
    return backend


def _unavailable_backend() -> Mock:
    backend = Mock()
    backend.name = "unavailable"
    backend.autoscaler = Mock()
    backend.capabilities = frozenset({BackendCapability.CLUSTER_VIEW, BackendCapability.IRIS_AUTOSCALER})
    backend.advertised_attributes.return_value = {}
    backend.resource_capacity.return_value = None
    backend.status.side_effect = ConnectionError("provider offline")
    backend.autoscaler_status.side_effect = ConnectionError("provider offline")
    return backend


@pytest.fixture
def resources(tmp_path: Path):
    db = ControllerDB(tmp_path / "db")
    runtime = Mock()
    runtime.all_liveness.return_value = {}
    runtime.backend_id_for_scale_group.return_value = "rpc"
    runtime.federation.peer_observations.return_value = ()
    runtime.last_unroutable_jobs = {}
    facade = Controller(
        cluster_id="cluster-a",
        db=db,
        runtime=runtime,
        bundle_store=Mock(),
        endpoint_registry=Mock(),
        auth=ControllerAuth(),
        user_budget_defaults=UserBudgetDefaults(),
        capability_url_config=CapabilityUrlConfig(),
        backends={
            "down": _unavailable_backend(),
            "k8s": _kubernetes_backend(),
            "rpc": _autoscaling_backend(),
        },
        backend_configs={
            "down": BackendConfig(kind="k8s"),
            "k8s": BackendConfig(kind="k8s"),
            "rpc": BackendConfig(kind="worker_daemon"),
        },
    )
    yield facade
    db.close()


@pytest.fixture
def worker_resources(tmp_path: Path):
    db = ControllerDB(tmp_path / "worker-db")
    runtime = Mock()
    runtime.backend_id_for_scale_group.return_value = DEFAULT_BACKEND_ID
    runtime.liveness_for_worker.return_value = WorkerLiveness(
        healthy=True,
        active=True,
        last_heartbeat_ms=NOW.epoch_ms(),
    )
    backend = Mock()
    backend.capabilities = frozenset({BackendCapability.WORKER_DAEMON})
    backend.status.return_value = BackendStatus(worker=WorkerFleetStatus())
    facade = Controller(
        cluster_id="cluster-a",
        db=db,
        runtime=runtime,
        bundle_store=Mock(),
        endpoint_registry=Mock(),
        auth=ControllerAuth(),
        user_budget_defaults=UserBudgetDefaults(),
        capability_url_config=CapabilityUrlConfig(),
        backends={DEFAULT_BACKEND_ID: backend},
        backend_configs={DEFAULT_BACKEND_ID: BackendConfig(kind="worker_daemon")},
    )
    yield facade, db, backend
    db.close()


def test_nodes_filter_page_and_describe_an_exact_incarnation(resources: Controller) -> None:
    first = resources.list_nodes(NodeQuery(backend_id="k8s", page_size=1))
    assert [node.identity.key.resource_id for node in first.items] == ["node-alpha"]
    assert first.items[0].region == "us-central1"
    assert first.next_page_token is not None

    second = resources.list_nodes(NodeQuery(backend_id="k8s", page_size=1, page_token=first.next_page_token))
    assert [node.identity.key.resource_id for node in second.items] == ["node-beta"]
    assert second.next_page_token is None

    filtered = resources.list_nodes(NodeQuery(backend_id="k8s", contains="ALPHA", health=frozenset({NodeHealth.READY})))
    assert [node.identity for node in filtered.items] == [first.items[0].identity]

    identity = first.items[0].identity
    detail = resources.describe_node(NodeLocator(identity.key, identity.backend_id, identity.node_uid))
    assert detail.summary.identity == identity
    assert [(attribute.key, attribute.string_value) for attribute in detail.attributes] == [
        ("instance_type", "h100-8"),
        ("region", "us-central1"),
    ]

    with pytest.raises(ResourceNotFound):
        resources.describe_node(NodeLocator(identity.key, identity.backend_id, "replacement-node-uid"))

    source_states = {status.source_id: status.state for status in first.source_statuses}
    assert source_states == {
        "backend:down": SourceState.UNAVAILABLE,
        "backend:k8s": SourceState.AVAILABLE,
        "backend:rpc": SourceState.AVAILABLE,
    }


def test_system_endpoints_are_resource_visible_and_paginated(tmp_path: Path) -> None:
    db = ControllerDB(tmp_path / "system-endpoint-db")
    EndpointsProjection(db)
    endpoint_service = EndpointRegistry(db=db)
    endpoint_service.register_system_endpoint("/system/controller", "http://controller:8080")
    endpoint_service.register_system_endpoint("/system/log-server", "http://logs:9000")
    runtime = Mock()
    runtime.federation.peer_summaries.return_value = []
    resources = Controller(
        cluster_id="cluster-a",
        db=db,
        runtime=runtime,
        bundle_store=Mock(),
        endpoint_registry=endpoint_service,
        auth=ControllerAuth(),
        user_budget_defaults=UserBudgetDefaults(),
        capability_url_config=CapabilityUrlConfig(),
        backends={},
        backend_configs={},
    )

    first = resources.list_endpoints(EndpointQuery(name_prefix="/system/", page_size=1))
    second = resources.list_endpoints(
        EndpointQuery(name_prefix="/system/", page_size=1, page_token=first.next_page_token)
    )
    listed = first.items + second.items
    log_server = next(endpoint for endpoint in listed if endpoint.name == "/system/log-server")
    detail = resources.describe_endpoint(log_server.key)

    assert [endpoint.name for endpoint in listed] == ["/system/controller", "/system/log-server"]
    assert first.next_page_token is not None
    assert second.next_page_token is None
    assert log_server.task is None
    assert log_server.key.kind is ResourceKind.ENDPOINT
    assert detail.address == "http://logs:9000"
    assert detail.metadata == {}
    db.close()


def test_node_pages_are_bounded_at_the_sqlite_bind_ceiling(worker_resources) -> None:
    resources, db, backend = worker_resources
    worker_count = 32_767
    with db.transaction() as tx:
        tx.execute(
            workers_table.insert(),
            [
                {"worker_id": f"worker-{index:05d}", "address": f"worker-{index:05d}:8080"}
                for index in range(worker_count)
            ],
        )

    selects: list[str] = []

    def capture_select(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    event.listen(db.sa_read_engine, "before_cursor_execute", capture_select)
    try:
        page_token = None
        page_select_counts = []
        observed = []
        for _ in range(3):
            before = len(selects)
            page = resources.list_nodes(NodeQuery(page_size=1, page_token=page_token))
            page_select_counts.append(len(selects) - before)
            observed.append(page.items[0].identity.key.resource_id)
            page_token = page.next_page_token
    finally:
        event.remove(db.sa_read_engine, "before_cursor_execute", capture_select)

    assert observed == ["worker-00000", "worker-00001", "worker-00002"]
    assert page_token is not None
    assert page_select_counts == [3, 3, 3]
    assert backend.status.call_count == 3


def test_worker_node_uses_normalized_capacity_slice_and_typed_attributes(worker_resources) -> None:
    resources, db, _backend = worker_resources
    with db.transaction() as tx:
        tx.execute(
            workers_table.insert().values(
                worker_id="worker-a",
                address="worker-a:8080",
                total_cpu_millicores=8_000,
                total_memory_bytes=64_000,
                total_gpu_count=4,
                device_type="gpu",
                device_variant="h100",
                slice_id="slice-a",
                md_disk_bytes=1_000,
            )
        )
        tx.execute(
            worker_attributes_table.insert(),
            [
                {
                    "worker_id": "worker-a",
                    "key": "region",
                    "value_type": "str",
                    "str_value": "us-east1",
                    "int_value": None,
                    "float_value": None,
                },
                {
                    "worker_id": "worker-a",
                    "key": "rack",
                    "value_type": "int",
                    "str_value": None,
                    "int_value": 7,
                    "float_value": None,
                },
            ],
        )

    (node,) = resources.list_nodes(NodeQuery()).items
    detail = resources.describe_node(NodeLocator(node.identity.key, node.identity.backend_id, node.identity.node_uid))

    assert node.capacity.cpu_millicores == 8_000
    assert node.capacity.memory_bytes == 64_000
    assert node.capacity.disk_bytes == 1_000
    assert (node.capacity.accelerator_kind, node.capacity.accelerator_variant, node.capacity.accelerator_count) == (
        "gpu",
        "h100",
        4,
    )
    assert node.slice is not None and node.slice.key.resource_id == "slice-a"
    assert node.region == "us-east1"
    assert [(attribute.key, attribute.string_value, attribute.integer_value) for attribute in detail.attributes] == [
        ("rack", None, 7),
        ("region", "us-east1", None),
    ]


def test_slices_filter_page_and_describe_observed_membership(resources: Controller) -> None:
    first = resources.list_slices(SliceQuery(backend_id="rpc", scaling_group_id="pool-a", page_size=1))
    assert [item.identity.key.resource_id for item in first.items] == ["slice-a"]
    assert first.next_page_token is not None

    second = resources.list_slices(
        SliceQuery(
            backend_id="rpc",
            scaling_group_id="pool-a",
            page_size=1,
            page_token=first.next_page_token,
        )
    )
    assert [item.identity.key.resource_id for item in second.items] == ["slice-b"]
    assert second.items[0].lifecycle is SliceLifecycle.CREATING

    identity = first.items[0].identity
    detail = resources.describe_slice(SliceLocator(identity.key, identity.backend_id, identity.slice_uid))
    assert detail.summary.identity == identity
    assert detail.summary.capacity_state is SliceCapacityState.IN_USE
    assert detail.summary.last_active_at == Timestamp.from_ms(900)
    assert (detail.summary.healthy_member_count, detail.summary.degraded_member_count) == (1, 1)
    assert detail.summary.running_task_count == 2
    assert [
        (member.provider_node_id, member.node.key.resource_id if member.node else None) for member in detail.members
    ] == [("vm-a", "node-a")]
    assert (detail.members[0].worker_id, detail.members[0].zone) == ("node-a", "us-central1-a")

    with pytest.raises(ResourceNotFound):
        resources.describe_slice(SliceLocator(identity.key, identity.backend_id, "replacement-slice-uid"))

    source_states = {status.source_id: status.state for status in first.source_statuses}
    assert source_states == {
        "backend:down": SourceState.UNAVAILABLE,
        "backend:k8s": SourceState.UNSUPPORTED,
        "backend:rpc": SourceState.AVAILABLE,
    }


def test_capacity_resource_carries_backend_authored_fleet_and_routing_facts(resources: Controller) -> None:
    capacity = ResourceServiceImpl(resources).get_capacity_status(resource_pb2.GetCapacityStatusRequest(), None)
    backend = next(item for item in capacity.backends if item.backend_id == "rpc")
    group = next(item for item in backend.scaling_groups if item.name == "pool-a")
    capacity_slice = next(item for item in group.slices if item.summary.identity.key.resource_id == "slice-a")

    assert (backend.kind, backend.healthy_worker_count, group.device_variant, group.region) == (
        "worker-daemon",
        1,
        "h100",
        "us-central1",
    )
    assert (group.current_demand, backend.routing.groups[0].launch) == (3, 1)
    assert capacity_slice.summary.running_task_count == 2
    assert capacity_slice.summary.capacity_state == resource_pb2.SLICE_CAPACITY_STATE_IN_USE
    assert capacity_slice.members[0].node.key.resource_id == "node-a"
