# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import pytest
from iris.cluster.controller.backend import BackendCapability
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.resources.facade import ResourceController
from iris.cluster.resources.errors import ResourceNotFound
from iris.cluster.resources.identity import NodeLocator, SliceLocator
from iris.cluster.resources.node import NodeHealth, NodeQuery
from iris.cluster.resources.slice import SliceLifecycle, SliceQuery
from iris.cluster.resources.source import SourceState
from iris.rpc import controller_pb2, vm_pb2
from iris.time_proto import timestamp_to_proto
from rigging.timing import Timestamp

NOW = Timestamp.from_ms(1_000)


def _kubernetes_backend() -> Mock:
    backend = Mock()
    backend.capabilities = frozenset({BackendCapability.CLUSTER_VIEW})
    backend.status.return_value = controller_pb2.Controller.BackendStatus(
        kubernetes=controller_pb2.Controller.GetKubernetesClusterStatusResponse(
            nodes=[
                controller_pb2.Controller.NodeStatus(
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
                controller_pb2.Controller.NodeStatus(
                    name="node-beta",
                    ready=False,
                    schedulable=False,
                    instance_type="h100-8",
                    region="us-central1",
                    created="2026-01-02T00:00:00Z",
                ),
            ]
        )
    )
    return backend


def _autoscaling_backend() -> Mock:
    backend = Mock()
    backend.capabilities = frozenset({BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER})
    backend.status.return_value = controller_pb2.Controller.BackendStatus(
        worker=controller_pb2.Controller.WorkerFleetDetail()
    )
    backend.autoscaler_status.return_value = vm_pb2.AutoscalerStatus(
        last_evaluation=timestamp_to_proto(NOW),
        groups=[
            vm_pb2.ScaleGroupStatus(
                name="pool-a",
                slices=[
                    vm_pb2.SliceInfo(
                        slice_id="slice-a",
                        scale_group="pool-a",
                        state="ready",
                        created_at=timestamp_to_proto(Timestamp.from_ms(10)),
                        vms=[vm_pb2.VmInfo(vm_id="vm-a", worker_id="node-a")],
                    ),
                    vm_pb2.SliceInfo(
                        slice_id="slice-b",
                        scale_group="pool-a",
                        state="booting",
                        created_at=timestamp_to_proto(Timestamp.from_ms(20)),
                    ),
                ],
            ),
            vm_pb2.ScaleGroupStatus(
                name="pool-b",
                slices=[
                    vm_pb2.SliceInfo(
                        slice_id="slice-c",
                        scale_group="pool-b",
                        state="failed",
                        error_message="quota denied",
                        created_at=timestamp_to_proto(Timestamp.from_ms(30)),
                    )
                ],
            ),
        ],
    )
    return backend


def _unavailable_backend() -> Mock:
    backend = Mock()
    backend.capabilities = frozenset({BackendCapability.CLUSTER_VIEW, BackendCapability.IRIS_AUTOSCALER})
    backend.status.side_effect = ConnectionError("provider offline")
    backend.autoscaler_status.side_effect = ConnectionError("provider offline")
    return backend


@pytest.fixture
def resources(tmp_path: Path):
    db = ControllerDB(tmp_path / "db")
    legacy = Mock()
    legacy.list_workers.return_value = controller_pb2.Controller.ListWorkersResponse()
    facade = ResourceController(
        cluster_id="cluster-a",
        db=db,
        legacy=legacy,
        backends={
            "down": _unavailable_backend(),
            "k8s": _kubernetes_backend(),
            "rpc": _autoscaling_backend(),
        },
    )
    yield facade
    db.close()


def test_nodes_filter_page_and_describe_an_exact_incarnation(resources: ResourceController) -> None:
    first = resources.list_nodes(NodeQuery(backend_id="k8s", page_size=1))
    assert [node.identity.key.resource_id for node in first.items] == ["node-alpha"]
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


def test_slices_filter_page_and_describe_observed_membership(resources: ResourceController) -> None:
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
    assert [
        (member.provider_node_id, member.node.key.resource_id if member.node else None) for member in detail.members
    ] == [("vm-a", "node-a")]

    with pytest.raises(ResourceNotFound):
        resources.describe_slice(SliceLocator(identity.key, identity.backend_id, "replacement-slice-uid"))

    source_states = {status.source_id: status.state for status in first.source_statuses}
    assert source_states == {
        "backend:down": SourceState.UNAVAILABLE,
        "backend:k8s": SourceState.UNSUPPORTED,
        "backend:rpc": SourceState.AVAILABLE,
    }
