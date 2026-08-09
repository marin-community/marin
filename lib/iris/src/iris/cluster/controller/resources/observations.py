# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapters from execution and persistence boundaries to resource observations."""

from collections.abc import Mapping
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.controller.backend import TaskBackend
from iris.cluster.resources.node import (
    NodeAttribute,
    NodeAttributeKind,
    NodeCapacity,
)
from iris.time_proto import timestamp_from_proto


@dataclass(frozen=True, slots=True)
class WorkerNodeMetadata:
    capacity: NodeCapacity
    slice_id: str | None
    attributes: tuple[NodeAttribute, ...]
    region: str | None


@dataclass(frozen=True, slots=True)
class ProviderNodeObservation:
    provider_node_id: str
    incarnation: str
    ready: bool
    schedulable: bool
    capacity: NodeCapacity
    running_task_count: int
    region: str | None
    instance_type: str | None


@dataclass(frozen=True, slots=True)
class BackendResourceObservation:
    nodes: tuple[ProviderNodeObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class ProviderSliceObservation:
    slice_id: str
    scaling_group_id: str
    lifecycle_state: str
    created_at: Timestamp | None
    provider_node_ids: tuple[str, ...]
    error_message: str


@dataclass(frozen=True, slots=True)
class AutoscalerResourceObservation:
    observed_at: Timestamp
    slices: tuple[ProviderSliceObservation, ...]


def observe_backend_resources(backend: TaskBackend) -> BackendResourceObservation:
    """Read one backend status response into the native resource inventory."""
    status = backend.status()
    if not status.HasField("kubernetes"):
        return BackendResourceObservation()
    return BackendResourceObservation(
        nodes=tuple(
            ProviderNodeObservation(
                provider_node_id=node.name,
                incarnation=node.created,
                ready=node.ready,
                schedulable=node.schedulable,
                capacity=NodeCapacity(
                    cpu_millicores=node.cpu_millicores,
                    memory_bytes=node.memory_bytes,
                    disk_bytes=node.disk_bytes,
                    accelerator_kind="gpu" if node.gpu_count else "",
                    accelerator_variant=node.gpu_model,
                    accelerator_count=node.gpu_count,
                ),
                running_task_count=node.running_pods,
                region=node.region or None,
                instance_type=node.instance_type or None,
            )
            for node in status.kubernetes.nodes
        )
    )


def worker_node_metadata(
    worker,
    attributes: Mapping[str, str | int | float],
) -> WorkerNodeMetadata:
    """Project normalized worker columns and typed attributes to resource metadata."""
    accelerator_count = 0
    if worker.device_type == "gpu":
        accelerator_count = worker.total_gpu_count
    elif worker.device_type == "tpu":
        accelerator_count = worker.total_tpu_count
    region = attributes.get("region")
    return WorkerNodeMetadata(
        capacity=NodeCapacity(
            cpu_millicores=worker.total_cpu_millicores,
            memory_bytes=worker.total_memory_bytes,
            disk_bytes=worker.md_disk_bytes,
            accelerator_kind=worker.device_type,
            accelerator_variant=worker.device_variant,
            accelerator_count=accelerator_count,
        ),
        slice_id=worker.slice_id or None,
        attributes=tuple(_node_attribute(key, value) for key, value in sorted(attributes.items())),
        region=region if isinstance(region, str) and region else None,
    )


def observe_autoscaler_resources(backend: TaskBackend) -> AutoscalerResourceObservation:
    """Read one autoscaler response into native Slice observations."""
    status = backend.autoscaler_status()
    observed_at = timestamp_from_proto(status.last_evaluation) if status.HasField("last_evaluation") else Timestamp.now()
    return AutoscalerResourceObservation(
        observed_at=observed_at,
        slices=tuple(
            ProviderSliceObservation(
                slice_id=item.slice_id,
                scaling_group_id=group.name,
                lifecycle_state=item.state,
                created_at=timestamp_from_proto(item.created_at) if item.HasField("created_at") else None,
                provider_node_ids=tuple(vm.vm_id for vm in item.vms),
                error_message=item.error_message,
            )
            for group in status.groups
            for item in group.slices
        ),
    )


def _node_attribute(key: str, value: str | int | float) -> NodeAttribute:
    if isinstance(value, str):
        return NodeAttribute(key, NodeAttributeKind.STRING, string_value=value)
    if isinstance(value, int):
        return NodeAttribute(key, NodeAttributeKind.INTEGER, integer_value=value)
    return NodeAttribute(key, NodeAttributeKind.FLOAT, float_value=value)
