# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native status records authored by execution backends."""

from dataclasses import dataclass, field

from rigging.timing import Timestamp


@dataclass(frozen=True, slots=True)
class VmStatus:
    vm_id: str = ""
    slice_id: str = ""
    scale_group: str = ""
    state: int = 0
    address: str = ""
    zone: str = ""
    created_at: Timestamp | None = None
    state_changed_at: Timestamp | None = None
    worker_id: str = ""
    worker_healthy: bool = False
    usability: str = ""
    init_phase: str = ""
    init_log_tail: str = ""
    init_error: str = ""
    running_task_count: int = 0
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SliceStatus:
    slice_id: str = ""
    scale_group: str = ""
    created_at: Timestamp | None = None
    vms: tuple[VmStatus, ...] = ()
    error_message: str = ""
    last_active: Timestamp | None = None
    idle: bool = False
    state: str = ""
    degraded_slot_count: int = 0
    capacity_status: str = ""


@dataclass(frozen=True, slots=True)
class ScaleGroupStatus:
    name: str = ""
    backend_id: str = ""
    device_type: str = ""
    device_variant: str = ""
    quota_pool: str = ""
    allocation_tier: int = 0
    region: str = ""
    current_demand: int = 0
    peak_demand: int = 0
    backoff_until: Timestamp | None = None
    consecutive_failures: int = 0
    last_scale_up: Timestamp | None = None
    last_scale_down: Timestamp | None = None
    slices: tuple[SliceStatus, ...] = ()
    slice_state_counts: dict[str, int] = field(default_factory=dict)
    availability_status: str = ""
    availability_reason: str = ""
    blocked_until: Timestamp | None = None
    scale_up_cooldown_until: Timestamp | None = None
    idle_threshold_ms: int = 0


@dataclass(frozen=True, slots=True)
class AutoscalerActionStatus:
    timestamp: Timestamp | None = None
    action_type: str = ""
    scale_group: str = ""
    slice_id: str = ""
    reason: str = ""
    status: str = ""


@dataclass(frozen=True, slots=True)
class ResourceStatus:
    cpu_millicores: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0
    gpu_count: int = 0
    tpu_count: int = 0


@dataclass(frozen=True, slots=True)
class DemandEntryStatus:
    task_ids: tuple[str, ...] = ()
    coschedule_group_id: str = ""
    device_type: str = ""
    device_variant: str = ""
    preemptible: bool = False
    resources: ResourceStatus = field(default_factory=ResourceStatus)


@dataclass(frozen=True, slots=True)
class UnmetDemandStatus:
    entry: DemandEntryStatus = field(default_factory=DemandEntryStatus)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class GroupRoutingStatus:
    group: str = ""
    priority: int = 0
    assigned: int = 0
    launch: int = 0
    decision: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RoutingStatus:
    group_to_launch: dict[str, int] = field(default_factory=dict)
    group_reasons: dict[str, str] = field(default_factory=dict)
    routed_entries: dict[str, tuple[DemandEntryStatus, ...]] = field(default_factory=dict)
    unmet_entries: tuple[UnmetDemandStatus, ...] = ()
    group_statuses: tuple[GroupRoutingStatus, ...] = ()


@dataclass(frozen=True, slots=True)
class AutoscalerStatus:
    groups: tuple[ScaleGroupStatus, ...] = ()
    current_demand: dict[str, int] = field(default_factory=dict)
    last_evaluation: Timestamp | None = None
    recent_actions: tuple[AutoscalerActionStatus, ...] = ()
    last_routing_decision: RoutingStatus | None = None


@dataclass(frozen=True, slots=True)
class KubernetesPodStatus:
    pod_name: str = ""
    task_id: str = ""
    phase: str = ""
    reason: str = ""
    message: str = ""
    last_transition: Timestamp | None = None
    node_name: str = ""


@dataclass(frozen=True, slots=True)
class NodePoolStatus:
    name: str = ""
    instance_type: str = ""
    scale_group: str = ""
    target_nodes: int = 0
    current_nodes: int = 0
    queued_nodes: int = 0
    in_progress_nodes: int = 0
    autoscaling: bool = False
    min_nodes: int = 0
    max_nodes: int = 0
    capacity: str = ""
    quota: str = ""


@dataclass(frozen=True, slots=True)
class NodeStatus:
    name: str = ""
    ready: bool = False
    schedulable: bool = False
    status_summary: str = ""
    instance_type: str = ""
    region: str = ""
    gpu_count: int = 0
    gpu_model: str = ""
    cpu_millicores: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0
    running_pods: int = 0
    created: str = ""


@dataclass(frozen=True, slots=True)
class KubernetesStatus:
    namespace: str = ""
    total_nodes: int = 0
    schedulable_nodes: int = 0
    allocatable_cpu: str = ""
    allocatable_memory: str = ""
    pod_statuses: tuple[KubernetesPodStatus, ...] = ()
    provider_version: str = ""
    node_pools: tuple[NodePoolStatus, ...] = ()
    nodes: tuple[NodeStatus, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkerFleetStatus:
    autoscaler: AutoscalerStatus = field(default_factory=AutoscalerStatus)
    healthy_worker_count: int = 0
    total_worker_count: int = 0


@dataclass(frozen=True, slots=True)
class BackendStatus:
    kubernetes: KubernetesStatus | None = None
    worker: WorkerFleetStatus | None = None

    def __post_init__(self) -> None:
        if (self.kubernetes is None) == (self.worker is None):
            raise ValueError("BackendStatus requires exactly one detail variant")
