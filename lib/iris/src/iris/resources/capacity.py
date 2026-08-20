# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed capacity, routing, and execution-target observations."""

from collections.abc import Mapping
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.resources.slice import SliceDetail
from iris.resources.source import ResourceSourceStatus


@dataclass(frozen=True, slots=True)
class ResourceAvailability:
    """Free and total capacity reported by one backend."""

    version: int
    observed_at: Timestamp
    amounts: Mapping[str, int]
    total_amounts: Mapping[str, int]
    held_by_band: Mapping[int, Mapping[str, int]]


@dataclass(frozen=True, slots=True)
class RunningTaskPlacement:
    """One Job's running Task count on a worker."""

    backend_id: str
    worker_id: str
    job_id: str
    user_id: str
    task_count: int


@dataclass(frozen=True, slots=True)
class CapacityScalingGroup:
    name: str
    backend_id: str
    device_type: str
    device_variant: str
    quota_pool: str
    allocation_tier: int
    region: str
    current_demand: int
    peak_demand: int
    backoff_until: Timestamp | None
    consecutive_failures: int
    last_scale_up: Timestamp | None
    last_scale_down: Timestamp | None
    slices: tuple[SliceDetail, ...]
    slice_state_counts: Mapping[str, int]
    availability_status: str
    availability_reason: str
    blocked_until: Timestamp | None
    scale_up_cooldown_until: Timestamp | None
    idle_threshold_ms: int


@dataclass(frozen=True, slots=True)
class CapacityAction:
    timestamp: Timestamp | None
    action_type: str
    scaling_group_id: str
    slice_id: str
    reason: str
    status: str


@dataclass(frozen=True, slots=True)
class CapacityDemandEntry:
    task_ids: tuple[str, ...]
    coschedule_group_id: str
    device_type: str
    device_variant: str
    preemptible: bool


@dataclass(frozen=True, slots=True)
class CapacityUnmetDemand:
    entry: CapacityDemandEntry
    reason: str


@dataclass(frozen=True, slots=True)
class CapacityGroupRouting:
    scaling_group_id: str
    priority: int
    assigned: int
    launch: int
    decision: str
    reason: str


@dataclass(frozen=True, slots=True)
class CapacityRouting:
    unmet: tuple[CapacityUnmetDemand, ...]
    groups: tuple[CapacityGroupRouting, ...]


@dataclass(frozen=True, slots=True)
class CapacityKubernetesPod:
    pod_name: str
    task_id: str
    phase: str
    reason: str
    message: str
    last_transition: Timestamp | None
    node_name: str


@dataclass(frozen=True, slots=True)
class CapacityKubernetesPool:
    name: str
    instance_type: str
    scaling_group_id: str
    target_nodes: int
    current_nodes: int
    queued_nodes: int
    in_progress_nodes: int
    autoscaling: bool
    min_nodes: int
    max_nodes: int
    capacity: str
    quota: str


@dataclass(frozen=True, slots=True)
class CapacityKubernetesNode:
    name: str
    ready: bool
    schedulable: bool
    status_summary: str
    instance_type: str
    region: str
    accelerator_count: int
    accelerator_variant: str
    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int
    running_pods: int
    created: str


@dataclass(frozen=True, slots=True)
class CapacityKubernetesStatus:
    namespace: str
    total_nodes: int
    schedulable_nodes: int
    allocatable_cpu: str
    allocatable_memory: str
    pods: tuple[CapacityKubernetesPod, ...]
    provider_version: str
    pools: tuple[CapacityKubernetesPool, ...]
    nodes: tuple[CapacityKubernetesNode, ...]


@dataclass(frozen=True, slots=True)
class CapacityBackend:
    """A local execution backend and its latest authored capacity view."""

    backend_id: str
    name: str
    kind: str
    capabilities: tuple[str, ...]
    advertised_attributes: Mapping[str, tuple[str, ...]]
    worker_count: int
    pending_task_count: int
    running_task_count: int
    has_autoscaler: bool
    capacity_health: Mapping[str, int]
    availability: ResourceAvailability | None
    scaling_groups: tuple[CapacityScalingGroup, ...]
    recent_actions: tuple[CapacityAction, ...]
    routing: CapacityRouting | None
    last_evaluation: Timestamp | None
    healthy_worker_count: int
    kubernetes: CapacityKubernetesStatus | None


@dataclass(frozen=True, slots=True)
class CapacityPeerBackend:
    """Capacity facts last advertised by one federation peer backend."""

    backend_id: str
    name: str
    kind: str
    capabilities: tuple[str, ...]
    advertised_attributes: Mapping[str, tuple[str, ...]]
    scale_groups: tuple[str, ...]
    worker_count: int
    pending_task_count: int
    running_task_count: int
    has_autoscaler: bool
    capacity_health: Mapping[str, int]
    availability: ResourceAvailability | None


@dataclass(frozen=True, slots=True)
class CapacityPeer:
    """A federation execution target and its last capability heartbeat."""

    peer_id: str
    controller_address: str
    reachable: bool
    last_contact_ms: int
    active_federated_jobs: int
    backends: tuple[CapacityPeerBackend, ...]


@dataclass(frozen=True, slots=True)
class UnroutableJob:
    job_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class CapacityStatus:
    """One controller-owned snapshot for the Capacity dashboard."""

    backends: tuple[CapacityBackend, ...]
    peers: tuple[CapacityPeer, ...]
    running_placements: tuple[RunningTaskPlacement, ...]
    unroutable_job_count: int
    unroutable_jobs: tuple[UnroutableJob, ...]
    source_statuses: tuple[ResourceSourceStatus, ...]
