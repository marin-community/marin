from . import resource_pb2 as _resource_pb2
from . import resource_identity_pb2 as _resource_identity_pb2
from . import resource_task_pb2 as _resource_task_pb2
from . import time_pb2 as _time_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NodeHealth(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NODE_HEALTH_UNSPECIFIED: _ClassVar[NodeHealth]
    NODE_HEALTH_READY: _ClassVar[NodeHealth]
    NODE_HEALTH_DEGRADED: _ClassVar[NodeHealth]
    NODE_HEALTH_UNAVAILABLE: _ClassVar[NodeHealth]
    NODE_HEALTH_RETIRED: _ClassVar[NodeHealth]

class SliceLifecycle(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SLICE_LIFECYCLE_UNSPECIFIED: _ClassVar[SliceLifecycle]
    SLICE_LIFECYCLE_CREATING: _ClassVar[SliceLifecycle]
    SLICE_LIFECYCLE_READY: _ClassVar[SliceLifecycle]
    SLICE_LIFECYCLE_DELETING: _ClassVar[SliceLifecycle]
    SLICE_LIFECYCLE_FAILED: _ClassVar[SliceLifecycle]

class MembershipState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MEMBERSHIP_STATE_UNSPECIFIED: _ClassVar[MembershipState]
    MEMBERSHIP_STATE_UNKNOWN: _ClassVar[MembershipState]
    MEMBERSHIP_STATE_OBSERVED: _ClassVar[MembershipState]

class SliceCapacityState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SLICE_CAPACITY_STATE_UNSPECIFIED: _ClassVar[SliceCapacityState]
    SLICE_CAPACITY_STATE_UNKNOWN: _ClassVar[SliceCapacityState]
    SLICE_CAPACITY_STATE_AVAILABLE: _ClassVar[SliceCapacityState]
    SLICE_CAPACITY_STATE_IN_USE: _ClassVar[SliceCapacityState]
    SLICE_CAPACITY_STATE_IDLE: _ClassVar[SliceCapacityState]
    SLICE_CAPACITY_STATE_DEGRADED: _ClassVar[SliceCapacityState]
NODE_HEALTH_UNSPECIFIED: NodeHealth
NODE_HEALTH_READY: NodeHealth
NODE_HEALTH_DEGRADED: NodeHealth
NODE_HEALTH_UNAVAILABLE: NodeHealth
NODE_HEALTH_RETIRED: NodeHealth
SLICE_LIFECYCLE_UNSPECIFIED: SliceLifecycle
SLICE_LIFECYCLE_CREATING: SliceLifecycle
SLICE_LIFECYCLE_READY: SliceLifecycle
SLICE_LIFECYCLE_DELETING: SliceLifecycle
SLICE_LIFECYCLE_FAILED: SliceLifecycle
MEMBERSHIP_STATE_UNSPECIFIED: MembershipState
MEMBERSHIP_STATE_UNKNOWN: MembershipState
MEMBERSHIP_STATE_OBSERVED: MembershipState
SLICE_CAPACITY_STATE_UNSPECIFIED: SliceCapacityState
SLICE_CAPACITY_STATE_UNKNOWN: SliceCapacityState
SLICE_CAPACITY_STATE_AVAILABLE: SliceCapacityState
SLICE_CAPACITY_STATE_IN_USE: SliceCapacityState
SLICE_CAPACITY_STATE_IDLE: SliceCapacityState
SLICE_CAPACITY_STATE_DEGRADED: SliceCapacityState

class NodeQuery(_message.Message):
    __slots__ = ("backend_id", "contains", "health", "page")
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    CONTAINS_FIELD_NUMBER: _ClassVar[int]
    HEALTH_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    backend_id: str
    contains: str
    health: _containers.RepeatedScalarFieldContainer[NodeHealth]
    page: _resource_pb2.PageRequest
    def __init__(self, backend_id: _Optional[str] = ..., contains: _Optional[str] = ..., health: _Optional[_Iterable[_Union[NodeHealth, str]]] = ..., page: _Optional[_Union[_resource_pb2.PageRequest, _Mapping]] = ...) -> None: ...

class NodeCapacity(_message.Message):
    __slots__ = ("cpu_millicores", "memory_bytes", "disk_bytes", "accelerator_kind", "accelerator_variant", "accelerator_count")
    CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_KIND_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_VARIANT_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_COUNT_FIELD_NUMBER: _ClassVar[int]
    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int
    accelerator_kind: str
    accelerator_variant: str
    accelerator_count: int
    def __init__(self, cpu_millicores: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., accelerator_kind: _Optional[str] = ..., accelerator_variant: _Optional[str] = ..., accelerator_count: _Optional[int] = ...) -> None: ...

class NodeSummary(_message.Message):
    __slots__ = ("identity", "health", "schedulable", "capacity", "scaling_group_id", "slice", "running_task_count", "observed_at", "region")
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    HEALTH_FIELD_NUMBER: _ClassVar[int]
    SCHEDULABLE_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    SLICE_FIELD_NUMBER: _ClassVar[int]
    RUNNING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    identity: _resource_identity_pb2.NodeIdentity
    health: NodeHealth
    schedulable: bool
    capacity: NodeCapacity
    scaling_group_id: str
    slice: _resource_identity_pb2.SliceIdentity
    running_task_count: int
    observed_at: _time_pb2.Timestamp
    region: str
    def __init__(self, identity: _Optional[_Union[_resource_identity_pb2.NodeIdentity, _Mapping]] = ..., health: _Optional[_Union[NodeHealth, str]] = ..., schedulable: _Optional[bool] = ..., capacity: _Optional[_Union[NodeCapacity, _Mapping]] = ..., scaling_group_id: _Optional[str] = ..., slice: _Optional[_Union[_resource_identity_pb2.SliceIdentity, _Mapping]] = ..., running_task_count: _Optional[int] = ..., observed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., region: _Optional[str] = ...) -> None: ...

class NodeAttribute(_message.Message):
    __slots__ = ("key", "string_value", "integer_value", "float_value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    INTEGER_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    string_value: str
    integer_value: int
    float_value: float
    def __init__(self, key: _Optional[str] = ..., string_value: _Optional[str] = ..., integer_value: _Optional[int] = ..., float_value: _Optional[float] = ...) -> None: ...

class NodeDetail(_message.Message):
    __slots__ = ("summary", "address", "attributes", "recent_attempts", "bootstrap_logs", "source_statuses")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    RECENT_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    BOOTSTRAP_LOGS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    summary: NodeSummary
    address: str
    attributes: _containers.RepeatedCompositeFieldContainer[NodeAttribute]
    recent_attempts: _containers.RepeatedCompositeFieldContainer[_resource_task_pb2.AttemptSummary]
    bootstrap_logs: str
    source_statuses: _containers.RepeatedCompositeFieldContainer[_resource_pb2.ResourceSourceStatus]
    def __init__(self, summary: _Optional[_Union[NodeSummary, _Mapping]] = ..., address: _Optional[str] = ..., attributes: _Optional[_Iterable[_Union[NodeAttribute, _Mapping]]] = ..., recent_attempts: _Optional[_Iterable[_Union[_resource_task_pb2.AttemptSummary, _Mapping]]] = ..., bootstrap_logs: _Optional[str] = ..., source_statuses: _Optional[_Iterable[_Union[_resource_pb2.ResourceSourceStatus, _Mapping]]] = ...) -> None: ...

class SliceQuery(_message.Message):
    __slots__ = ("backend_id", "scaling_group_id", "page")
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    backend_id: str
    scaling_group_id: str
    page: _resource_pb2.PageRequest
    def __init__(self, backend_id: _Optional[str] = ..., scaling_group_id: _Optional[str] = ..., page: _Optional[_Union[_resource_pb2.PageRequest, _Mapping]] = ...) -> None: ...

class SliceSummary(_message.Message):
    __slots__ = ("identity", "scaling_group_id", "lifecycle", "membership_state", "observed_member_count", "observed_at", "error_message", "created_at", "last_active_at", "capacity_state", "healthy_member_count", "degraded_member_count", "running_task_count")
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    LIFECYCLE_FIELD_NUMBER: _ClassVar[int]
    MEMBERSHIP_STATE_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_MEMBER_COUNT_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_ACTIVE_AT_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_STATE_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_MEMBER_COUNT_FIELD_NUMBER: _ClassVar[int]
    DEGRADED_MEMBER_COUNT_FIELD_NUMBER: _ClassVar[int]
    RUNNING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    identity: _resource_identity_pb2.SliceIdentity
    scaling_group_id: str
    lifecycle: SliceLifecycle
    membership_state: MembershipState
    observed_member_count: int
    observed_at: _time_pb2.Timestamp
    error_message: str
    created_at: _time_pb2.Timestamp
    last_active_at: _time_pb2.Timestamp
    capacity_state: SliceCapacityState
    healthy_member_count: int
    degraded_member_count: int
    running_task_count: int
    def __init__(self, identity: _Optional[_Union[_resource_identity_pb2.SliceIdentity, _Mapping]] = ..., scaling_group_id: _Optional[str] = ..., lifecycle: _Optional[_Union[SliceLifecycle, str]] = ..., membership_state: _Optional[_Union[MembershipState, str]] = ..., observed_member_count: _Optional[int] = ..., observed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., error_message: _Optional[str] = ..., created_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., last_active_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., capacity_state: _Optional[_Union[SliceCapacityState, str]] = ..., healthy_member_count: _Optional[int] = ..., degraded_member_count: _Optional[int] = ..., running_task_count: _Optional[int] = ...) -> None: ...

class SliceMember(_message.Message):
    __slots__ = ("provider_node_id", "node", "observed_at", "worker_id", "healthy", "usability", "running_task_count", "zone")
    PROVIDER_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    USABILITY_FIELD_NUMBER: _ClassVar[int]
    RUNNING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    ZONE_FIELD_NUMBER: _ClassVar[int]
    provider_node_id: str
    node: _resource_identity_pb2.NodeIdentity
    observed_at: _time_pb2.Timestamp
    worker_id: str
    healthy: bool
    usability: str
    running_task_count: int
    zone: str
    def __init__(self, provider_node_id: _Optional[str] = ..., node: _Optional[_Union[_resource_identity_pb2.NodeIdentity, _Mapping]] = ..., observed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., worker_id: _Optional[str] = ..., healthy: _Optional[bool] = ..., usability: _Optional[str] = ..., running_task_count: _Optional[int] = ..., zone: _Optional[str] = ...) -> None: ...

class SliceDetail(_message.Message):
    __slots__ = ("summary", "members", "source_statuses")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    MEMBERS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    summary: SliceSummary
    members: _containers.RepeatedCompositeFieldContainer[SliceMember]
    source_statuses: _containers.RepeatedCompositeFieldContainer[_resource_pb2.ResourceSourceStatus]
    def __init__(self, summary: _Optional[_Union[SliceSummary, _Mapping]] = ..., members: _Optional[_Iterable[_Union[SliceMember, _Mapping]]] = ..., source_statuses: _Optional[_Iterable[_Union[_resource_pb2.ResourceSourceStatus, _Mapping]]] = ...) -> None: ...

class StringValues(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, values: _Optional[_Iterable[str]] = ...) -> None: ...

class CapacityResourceAvailability(_message.Message):
    __slots__ = ("version", "observed_at", "amounts", "total_amounts", "held_by_band")
    class AmountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class TotalAmountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    VERSION_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    AMOUNTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_AMOUNTS_FIELD_NUMBER: _ClassVar[int]
    HELD_BY_BAND_FIELD_NUMBER: _ClassVar[int]
    version: int
    observed_at: _time_pb2.Timestamp
    amounts: _containers.ScalarMap[str, int]
    total_amounts: _containers.ScalarMap[str, int]
    held_by_band: _containers.RepeatedCompositeFieldContainer[CapacityBandAvailability]
    def __init__(self, version: _Optional[int] = ..., observed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., amounts: _Optional[_Mapping[str, int]] = ..., total_amounts: _Optional[_Mapping[str, int]] = ..., held_by_band: _Optional[_Iterable[_Union[CapacityBandAvailability, _Mapping]]] = ...) -> None: ...

class CapacityBandAvailability(_message.Message):
    __slots__ = ("band", "amounts")
    class AmountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    BAND_FIELD_NUMBER: _ClassVar[int]
    AMOUNTS_FIELD_NUMBER: _ClassVar[int]
    band: int
    amounts: _containers.ScalarMap[str, int]
    def __init__(self, band: _Optional[int] = ..., amounts: _Optional[_Mapping[str, int]] = ...) -> None: ...

class CapacitySlice(_message.Message):
    __slots__ = ("summary", "members")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    MEMBERS_FIELD_NUMBER: _ClassVar[int]
    summary: SliceSummary
    members: _containers.RepeatedCompositeFieldContainer[SliceMember]
    def __init__(self, summary: _Optional[_Union[SliceSummary, _Mapping]] = ..., members: _Optional[_Iterable[_Union[SliceMember, _Mapping]]] = ...) -> None: ...

class CapacityScalingGroup(_message.Message):
    __slots__ = ("name", "backend_id", "device_type", "device_variant", "quota_pool", "allocation_tier", "region", "current_demand", "peak_demand", "backoff_until", "consecutive_failures", "last_scale_up", "last_scale_down", "slices", "slice_state_counts", "availability_status", "availability_reason", "blocked_until", "scale_up_cooldown_until", "idle_threshold_ms")
    class SliceStateCountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_VARIANT_FIELD_NUMBER: _ClassVar[int]
    QUOTA_POOL_FIELD_NUMBER: _ClassVar[int]
    ALLOCATION_TIER_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    CURRENT_DEMAND_FIELD_NUMBER: _ClassVar[int]
    PEAK_DEMAND_FIELD_NUMBER: _ClassVar[int]
    BACKOFF_UNTIL_FIELD_NUMBER: _ClassVar[int]
    CONSECUTIVE_FAILURES_FIELD_NUMBER: _ClassVar[int]
    LAST_SCALE_UP_FIELD_NUMBER: _ClassVar[int]
    LAST_SCALE_DOWN_FIELD_NUMBER: _ClassVar[int]
    SLICES_FIELD_NUMBER: _ClassVar[int]
    SLICE_STATE_COUNTS_FIELD_NUMBER: _ClassVar[int]
    AVAILABILITY_STATUS_FIELD_NUMBER: _ClassVar[int]
    AVAILABILITY_REASON_FIELD_NUMBER: _ClassVar[int]
    BLOCKED_UNTIL_FIELD_NUMBER: _ClassVar[int]
    SCALE_UP_COOLDOWN_UNTIL_FIELD_NUMBER: _ClassVar[int]
    IDLE_THRESHOLD_MS_FIELD_NUMBER: _ClassVar[int]
    name: str
    backend_id: str
    device_type: str
    device_variant: str
    quota_pool: str
    allocation_tier: int
    region: str
    current_demand: int
    peak_demand: int
    backoff_until: _time_pb2.Timestamp
    consecutive_failures: int
    last_scale_up: _time_pb2.Timestamp
    last_scale_down: _time_pb2.Timestamp
    slices: _containers.RepeatedCompositeFieldContainer[CapacitySlice]
    slice_state_counts: _containers.ScalarMap[str, int]
    availability_status: str
    availability_reason: str
    blocked_until: _time_pb2.Timestamp
    scale_up_cooldown_until: _time_pb2.Timestamp
    idle_threshold_ms: int
    def __init__(self, name: _Optional[str] = ..., backend_id: _Optional[str] = ..., device_type: _Optional[str] = ..., device_variant: _Optional[str] = ..., quota_pool: _Optional[str] = ..., allocation_tier: _Optional[int] = ..., region: _Optional[str] = ..., current_demand: _Optional[int] = ..., peak_demand: _Optional[int] = ..., backoff_until: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., consecutive_failures: _Optional[int] = ..., last_scale_up: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., last_scale_down: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., slices: _Optional[_Iterable[_Union[CapacitySlice, _Mapping]]] = ..., slice_state_counts: _Optional[_Mapping[str, int]] = ..., availability_status: _Optional[str] = ..., availability_reason: _Optional[str] = ..., blocked_until: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., scale_up_cooldown_until: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., idle_threshold_ms: _Optional[int] = ...) -> None: ...

class CapacityAction(_message.Message):
    __slots__ = ("timestamp", "action_type", "scaling_group_id", "slice_id", "reason", "status")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    ACTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    timestamp: _time_pb2.Timestamp
    action_type: str
    scaling_group_id: str
    slice_id: str
    reason: str
    status: str
    def __init__(self, timestamp: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., action_type: _Optional[str] = ..., scaling_group_id: _Optional[str] = ..., slice_id: _Optional[str] = ..., reason: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class CapacityDemandEntry(_message.Message):
    __slots__ = ("task_ids", "coschedule_group_id", "device_type", "device_variant", "preemptible")
    TASK_IDS_FIELD_NUMBER: _ClassVar[int]
    COSCHEDULE_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_VARIANT_FIELD_NUMBER: _ClassVar[int]
    PREEMPTIBLE_FIELD_NUMBER: _ClassVar[int]
    task_ids: _containers.RepeatedScalarFieldContainer[str]
    coschedule_group_id: str
    device_type: str
    device_variant: str
    preemptible: bool
    def __init__(self, task_ids: _Optional[_Iterable[str]] = ..., coschedule_group_id: _Optional[str] = ..., device_type: _Optional[str] = ..., device_variant: _Optional[str] = ..., preemptible: _Optional[bool] = ...) -> None: ...

class CapacityUnmetDemand(_message.Message):
    __slots__ = ("entry", "reason")
    ENTRY_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    entry: CapacityDemandEntry
    reason: str
    def __init__(self, entry: _Optional[_Union[CapacityDemandEntry, _Mapping]] = ..., reason: _Optional[str] = ...) -> None: ...

class CapacityGroupRouting(_message.Message):
    __slots__ = ("scaling_group_id", "priority", "assigned", "launch", "decision", "reason")
    SCALING_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    ASSIGNED_FIELD_NUMBER: _ClassVar[int]
    LAUNCH_FIELD_NUMBER: _ClassVar[int]
    DECISION_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    scaling_group_id: str
    priority: int
    assigned: int
    launch: int
    decision: str
    reason: str
    def __init__(self, scaling_group_id: _Optional[str] = ..., priority: _Optional[int] = ..., assigned: _Optional[int] = ..., launch: _Optional[int] = ..., decision: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class CapacityRouting(_message.Message):
    __slots__ = ("unmet", "groups")
    UNMET_FIELD_NUMBER: _ClassVar[int]
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    unmet: _containers.RepeatedCompositeFieldContainer[CapacityUnmetDemand]
    groups: _containers.RepeatedCompositeFieldContainer[CapacityGroupRouting]
    def __init__(self, unmet: _Optional[_Iterable[_Union[CapacityUnmetDemand, _Mapping]]] = ..., groups: _Optional[_Iterable[_Union[CapacityGroupRouting, _Mapping]]] = ...) -> None: ...

class CapacityKubernetesPod(_message.Message):
    __slots__ = ("pod_name", "task_id", "phase", "reason", "message", "last_transition", "node_name")
    POD_NAME_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    LAST_TRANSITION_FIELD_NUMBER: _ClassVar[int]
    NODE_NAME_FIELD_NUMBER: _ClassVar[int]
    pod_name: str
    task_id: str
    phase: str
    reason: str
    message: str
    last_transition: _time_pb2.Timestamp
    node_name: str
    def __init__(self, pod_name: _Optional[str] = ..., task_id: _Optional[str] = ..., phase: _Optional[str] = ..., reason: _Optional[str] = ..., message: _Optional[str] = ..., last_transition: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., node_name: _Optional[str] = ...) -> None: ...

class CapacityKubernetesPool(_message.Message):
    __slots__ = ("name", "instance_type", "scaling_group_id", "target_nodes", "current_nodes", "queued_nodes", "in_progress_nodes", "autoscaling", "min_nodes", "max_nodes", "capacity", "quota")
    NAME_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_NODES_FIELD_NUMBER: _ClassVar[int]
    CURRENT_NODES_FIELD_NUMBER: _ClassVar[int]
    QUEUED_NODES_FIELD_NUMBER: _ClassVar[int]
    IN_PROGRESS_NODES_FIELD_NUMBER: _ClassVar[int]
    AUTOSCALING_FIELD_NUMBER: _ClassVar[int]
    MIN_NODES_FIELD_NUMBER: _ClassVar[int]
    MAX_NODES_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_FIELD_NUMBER: _ClassVar[int]
    QUOTA_FIELD_NUMBER: _ClassVar[int]
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
    def __init__(self, name: _Optional[str] = ..., instance_type: _Optional[str] = ..., scaling_group_id: _Optional[str] = ..., target_nodes: _Optional[int] = ..., current_nodes: _Optional[int] = ..., queued_nodes: _Optional[int] = ..., in_progress_nodes: _Optional[int] = ..., autoscaling: _Optional[bool] = ..., min_nodes: _Optional[int] = ..., max_nodes: _Optional[int] = ..., capacity: _Optional[str] = ..., quota: _Optional[str] = ...) -> None: ...

class CapacityKubernetesNode(_message.Message):
    __slots__ = ("name", "ready", "schedulable", "status_summary", "instance_type", "region", "accelerator_count", "accelerator_variant", "cpu_millicores", "memory_bytes", "disk_bytes", "running_pods", "created")
    NAME_FIELD_NUMBER: _ClassVar[int]
    READY_FIELD_NUMBER: _ClassVar[int]
    SCHEDULABLE_FIELD_NUMBER: _ClassVar[int]
    STATUS_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_COUNT_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_VARIANT_FIELD_NUMBER: _ClassVar[int]
    CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    RUNNING_PODS_FIELD_NUMBER: _ClassVar[int]
    CREATED_FIELD_NUMBER: _ClassVar[int]
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
    def __init__(self, name: _Optional[str] = ..., ready: _Optional[bool] = ..., schedulable: _Optional[bool] = ..., status_summary: _Optional[str] = ..., instance_type: _Optional[str] = ..., region: _Optional[str] = ..., accelerator_count: _Optional[int] = ..., accelerator_variant: _Optional[str] = ..., cpu_millicores: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., running_pods: _Optional[int] = ..., created: _Optional[str] = ...) -> None: ...

class CapacityKubernetesStatus(_message.Message):
    __slots__ = ("namespace", "total_nodes", "schedulable_nodes", "allocatable_cpu", "allocatable_memory", "pods", "provider_version", "pools", "nodes")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_NODES_FIELD_NUMBER: _ClassVar[int]
    SCHEDULABLE_NODES_FIELD_NUMBER: _ClassVar[int]
    ALLOCATABLE_CPU_FIELD_NUMBER: _ClassVar[int]
    ALLOCATABLE_MEMORY_FIELD_NUMBER: _ClassVar[int]
    PODS_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_VERSION_FIELD_NUMBER: _ClassVar[int]
    POOLS_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    total_nodes: int
    schedulable_nodes: int
    allocatable_cpu: str
    allocatable_memory: str
    pods: _containers.RepeatedCompositeFieldContainer[CapacityKubernetesPod]
    provider_version: str
    pools: _containers.RepeatedCompositeFieldContainer[CapacityKubernetesPool]
    nodes: _containers.RepeatedCompositeFieldContainer[CapacityKubernetesNode]
    def __init__(self, namespace: _Optional[str] = ..., total_nodes: _Optional[int] = ..., schedulable_nodes: _Optional[int] = ..., allocatable_cpu: _Optional[str] = ..., allocatable_memory: _Optional[str] = ..., pods: _Optional[_Iterable[_Union[CapacityKubernetesPod, _Mapping]]] = ..., provider_version: _Optional[str] = ..., pools: _Optional[_Iterable[_Union[CapacityKubernetesPool, _Mapping]]] = ..., nodes: _Optional[_Iterable[_Union[CapacityKubernetesNode, _Mapping]]] = ...) -> None: ...

class CapacityBackend(_message.Message):
    __slots__ = ("backend_id", "name", "kind", "capabilities", "advertised_attributes", "worker_count", "pending_task_count", "running_task_count", "has_autoscaler", "capacity_health", "availability", "scaling_groups", "recent_actions", "routing", "last_evaluation", "healthy_worker_count", "kubernetes")
    class AdvertisedAttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: StringValues
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[StringValues, _Mapping]] = ...) -> None: ...
    class CapacityHealthEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    ADVERTISED_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    WORKER_COUNT_FIELD_NUMBER: _ClassVar[int]
    PENDING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    RUNNING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    HAS_AUTOSCALER_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_HEALTH_FIELD_NUMBER: _ClassVar[int]
    AVAILABILITY_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUPS_FIELD_NUMBER: _ClassVar[int]
    RECENT_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    ROUTING_FIELD_NUMBER: _ClassVar[int]
    LAST_EVALUATION_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_WORKER_COUNT_FIELD_NUMBER: _ClassVar[int]
    KUBERNETES_FIELD_NUMBER: _ClassVar[int]
    backend_id: str
    name: str
    kind: str
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    advertised_attributes: _containers.MessageMap[str, StringValues]
    worker_count: int
    pending_task_count: int
    running_task_count: int
    has_autoscaler: bool
    capacity_health: _containers.ScalarMap[str, int]
    availability: CapacityResourceAvailability
    scaling_groups: _containers.RepeatedCompositeFieldContainer[CapacityScalingGroup]
    recent_actions: _containers.RepeatedCompositeFieldContainer[CapacityAction]
    routing: CapacityRouting
    last_evaluation: _time_pb2.Timestamp
    healthy_worker_count: int
    kubernetes: CapacityKubernetesStatus
    def __init__(self, backend_id: _Optional[str] = ..., name: _Optional[str] = ..., kind: _Optional[str] = ..., capabilities: _Optional[_Iterable[str]] = ..., advertised_attributes: _Optional[_Mapping[str, StringValues]] = ..., worker_count: _Optional[int] = ..., pending_task_count: _Optional[int] = ..., running_task_count: _Optional[int] = ..., has_autoscaler: _Optional[bool] = ..., capacity_health: _Optional[_Mapping[str, int]] = ..., availability: _Optional[_Union[CapacityResourceAvailability, _Mapping]] = ..., scaling_groups: _Optional[_Iterable[_Union[CapacityScalingGroup, _Mapping]]] = ..., recent_actions: _Optional[_Iterable[_Union[CapacityAction, _Mapping]]] = ..., routing: _Optional[_Union[CapacityRouting, _Mapping]] = ..., last_evaluation: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., healthy_worker_count: _Optional[int] = ..., kubernetes: _Optional[_Union[CapacityKubernetesStatus, _Mapping]] = ...) -> None: ...

class CapacityPeerBackend(_message.Message):
    __slots__ = ("backend_id", "name", "kind", "capabilities", "advertised_attributes", "scaling_groups", "worker_count", "pending_task_count", "running_task_count", "has_autoscaler", "capacity_health", "availability")
    class AdvertisedAttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: StringValues
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[StringValues, _Mapping]] = ...) -> None: ...
    class CapacityHealthEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    ADVERTISED_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUPS_FIELD_NUMBER: _ClassVar[int]
    WORKER_COUNT_FIELD_NUMBER: _ClassVar[int]
    PENDING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    RUNNING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    HAS_AUTOSCALER_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_HEALTH_FIELD_NUMBER: _ClassVar[int]
    AVAILABILITY_FIELD_NUMBER: _ClassVar[int]
    backend_id: str
    name: str
    kind: str
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    advertised_attributes: _containers.MessageMap[str, StringValues]
    scaling_groups: _containers.RepeatedScalarFieldContainer[str]
    worker_count: int
    pending_task_count: int
    running_task_count: int
    has_autoscaler: bool
    capacity_health: _containers.ScalarMap[str, int]
    availability: CapacityResourceAvailability
    def __init__(self, backend_id: _Optional[str] = ..., name: _Optional[str] = ..., kind: _Optional[str] = ..., capabilities: _Optional[_Iterable[str]] = ..., advertised_attributes: _Optional[_Mapping[str, StringValues]] = ..., scaling_groups: _Optional[_Iterable[str]] = ..., worker_count: _Optional[int] = ..., pending_task_count: _Optional[int] = ..., running_task_count: _Optional[int] = ..., has_autoscaler: _Optional[bool] = ..., capacity_health: _Optional[_Mapping[str, int]] = ..., availability: _Optional[_Union[CapacityResourceAvailability, _Mapping]] = ...) -> None: ...

class CapacityPeer(_message.Message):
    __slots__ = ("peer_id", "controller_address", "reachable", "last_contact_ms", "active_federated_jobs", "backends")
    PEER_ID_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    REACHABLE_FIELD_NUMBER: _ClassVar[int]
    LAST_CONTACT_MS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_FEDERATED_JOBS_FIELD_NUMBER: _ClassVar[int]
    BACKENDS_FIELD_NUMBER: _ClassVar[int]
    peer_id: str
    controller_address: str
    reachable: bool
    last_contact_ms: int
    active_federated_jobs: int
    backends: _containers.RepeatedCompositeFieldContainer[CapacityPeerBackend]
    def __init__(self, peer_id: _Optional[str] = ..., controller_address: _Optional[str] = ..., reachable: _Optional[bool] = ..., last_contact_ms: _Optional[int] = ..., active_federated_jobs: _Optional[int] = ..., backends: _Optional[_Iterable[_Union[CapacityPeerBackend, _Mapping]]] = ...) -> None: ...

class CapacityRunningPlacement(_message.Message):
    __slots__ = ("backend_id", "worker_id", "job_id", "user_id", "task_count")
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    backend_id: str
    worker_id: str
    job_id: str
    user_id: str
    task_count: int
    def __init__(self, backend_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., job_id: _Optional[str] = ..., user_id: _Optional[str] = ..., task_count: _Optional[int] = ...) -> None: ...

class CapacityUnroutableJob(_message.Message):
    __slots__ = ("job_id", "reason")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    reason: str
    def __init__(self, job_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class CapacityStatus(_message.Message):
    __slots__ = ("backends", "peers", "running_placements", "unroutable_job_count", "unroutable_jobs", "source_statuses")
    BACKENDS_FIELD_NUMBER: _ClassVar[int]
    PEERS_FIELD_NUMBER: _ClassVar[int]
    RUNNING_PLACEMENTS_FIELD_NUMBER: _ClassVar[int]
    UNROUTABLE_JOB_COUNT_FIELD_NUMBER: _ClassVar[int]
    UNROUTABLE_JOBS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    backends: _containers.RepeatedCompositeFieldContainer[CapacityBackend]
    peers: _containers.RepeatedCompositeFieldContainer[CapacityPeer]
    running_placements: _containers.RepeatedCompositeFieldContainer[CapacityRunningPlacement]
    unroutable_job_count: int
    unroutable_jobs: _containers.RepeatedCompositeFieldContainer[CapacityUnroutableJob]
    source_statuses: _containers.RepeatedCompositeFieldContainer[_resource_pb2.ResourceSourceStatus]
    def __init__(self, backends: _Optional[_Iterable[_Union[CapacityBackend, _Mapping]]] = ..., peers: _Optional[_Iterable[_Union[CapacityPeer, _Mapping]]] = ..., running_placements: _Optional[_Iterable[_Union[CapacityRunningPlacement, _Mapping]]] = ..., unroutable_job_count: _Optional[int] = ..., unroutable_jobs: _Optional[_Iterable[_Union[CapacityUnroutableJob, _Mapping]]] = ..., source_statuses: _Optional[_Iterable[_Union[_resource_pb2.ResourceSourceStatus, _Mapping]]] = ...) -> None: ...
