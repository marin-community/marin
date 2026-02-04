from . import time_pb2 as _time_pb2
from . import config_pb2 as _config_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class VmState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VM_STATE_UNSPECIFIED: _ClassVar[VmState]
    VM_STATE_BOOTING: _ClassVar[VmState]
    VM_STATE_INITIALIZING: _ClassVar[VmState]
    VM_STATE_READY: _ClassVar[VmState]
    VM_STATE_UNHEALTHY: _ClassVar[VmState]
    VM_STATE_STOPPING: _ClassVar[VmState]
    VM_STATE_TERMINATED: _ClassVar[VmState]
    VM_STATE_FAILED: _ClassVar[VmState]
    VM_STATE_PREEMPTED: _ClassVar[VmState]
    VM_STATE_REQUESTING: _ClassVar[VmState]

class ScalingAction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SCALING_ACTION_UNSPECIFIED: _ClassVar[ScalingAction]
    SCALING_ACTION_SCALE_UP: _ClassVar[ScalingAction]
    SCALING_ACTION_SCALE_DOWN: _ClassVar[ScalingAction]
    SCALING_ACTION_NONE: _ClassVar[ScalingAction]
VM_STATE_UNSPECIFIED: VmState
VM_STATE_BOOTING: VmState
VM_STATE_INITIALIZING: VmState
VM_STATE_READY: VmState
VM_STATE_UNHEALTHY: VmState
VM_STATE_STOPPING: VmState
VM_STATE_TERMINATED: VmState
VM_STATE_FAILED: VmState
VM_STATE_PREEMPTED: VmState
VM_STATE_REQUESTING: VmState
SCALING_ACTION_UNSPECIFIED: ScalingAction
SCALING_ACTION_SCALE_UP: ScalingAction
SCALING_ACTION_SCALE_DOWN: ScalingAction
SCALING_ACTION_NONE: ScalingAction

class ResourceSpec(_message.Message):
    __slots__ = ("cpu", "memory_bytes", "disk_bytes", "gpu_count", "tpu_chips")
    CPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    TPU_CHIPS_FIELD_NUMBER: _ClassVar[int]
    cpu: int
    memory_bytes: int
    disk_bytes: int
    gpu_count: int
    tpu_chips: int
    def __init__(self, cpu: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., gpu_count: _Optional[int] = ..., tpu_chips: _Optional[int] = ...) -> None: ...

class VmInfo(_message.Message):
    __slots__ = ("vm_id", "slice_id", "scale_group", "state", "address", "zone", "created_at", "state_changed_at", "worker_id", "worker_healthy", "init_phase", "init_log_tail", "init_error", "labels")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VM_ID_FIELD_NUMBER: _ClassVar[int]
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    ZONE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    STATE_CHANGED_AT_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_HEALTHY_FIELD_NUMBER: _ClassVar[int]
    INIT_PHASE_FIELD_NUMBER: _ClassVar[int]
    INIT_LOG_TAIL_FIELD_NUMBER: _ClassVar[int]
    INIT_ERROR_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    vm_id: str
    slice_id: str
    scale_group: str
    state: VmState
    address: str
    zone: str
    created_at: _time_pb2.Timestamp
    state_changed_at: _time_pb2.Timestamp
    worker_id: str
    worker_healthy: bool
    init_phase: str
    init_log_tail: str
    init_error: str
    labels: _containers.ScalarMap[str, str]
    def __init__(self, vm_id: _Optional[str] = ..., slice_id: _Optional[str] = ..., scale_group: _Optional[str] = ..., state: _Optional[_Union[VmState, str]] = ..., address: _Optional[str] = ..., zone: _Optional[str] = ..., created_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., state_changed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., worker_id: _Optional[str] = ..., worker_healthy: _Optional[bool] = ..., init_phase: _Optional[str] = ..., init_log_tail: _Optional[str] = ..., init_error: _Optional[str] = ..., labels: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SliceInfo(_message.Message):
    __slots__ = ("slice_id", "scale_group", "created_at", "vms")
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    VMS_FIELD_NUMBER: _ClassVar[int]
    slice_id: str
    scale_group: str
    created_at: _time_pb2.Timestamp
    vms: _containers.RepeatedCompositeFieldContainer[VmInfo]
    def __init__(self, slice_id: _Optional[str] = ..., scale_group: _Optional[str] = ..., created_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., vms: _Optional[_Iterable[_Union[VmInfo, _Mapping]]] = ...) -> None: ...

class ScalingDecision(_message.Message):
    __slots__ = ("scale_group", "action", "slice_delta", "reason")
    SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    SLICE_DELTA_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    scale_group: str
    action: ScalingAction
    slice_delta: int
    reason: str
    def __init__(self, scale_group: _Optional[str] = ..., action: _Optional[_Union[ScalingAction, str]] = ..., slice_delta: _Optional[int] = ..., reason: _Optional[str] = ...) -> None: ...

class ScaleGroupStatus(_message.Message):
    __slots__ = ("name", "config", "current_demand", "peak_demand", "backoff_until", "consecutive_failures", "last_scale_up", "last_scale_down", "slices")
    NAME_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    CURRENT_DEMAND_FIELD_NUMBER: _ClassVar[int]
    PEAK_DEMAND_FIELD_NUMBER: _ClassVar[int]
    BACKOFF_UNTIL_FIELD_NUMBER: _ClassVar[int]
    CONSECUTIVE_FAILURES_FIELD_NUMBER: _ClassVar[int]
    LAST_SCALE_UP_FIELD_NUMBER: _ClassVar[int]
    LAST_SCALE_DOWN_FIELD_NUMBER: _ClassVar[int]
    SLICES_FIELD_NUMBER: _ClassVar[int]
    name: str
    config: _config_pb2.ScaleGroupConfig
    current_demand: int
    peak_demand: int
    backoff_until: _time_pb2.Timestamp
    consecutive_failures: int
    last_scale_up: _time_pb2.Timestamp
    last_scale_down: _time_pb2.Timestamp
    slices: _containers.RepeatedCompositeFieldContainer[SliceInfo]
    def __init__(self, name: _Optional[str] = ..., config: _Optional[_Union[_config_pb2.ScaleGroupConfig, _Mapping]] = ..., current_demand: _Optional[int] = ..., peak_demand: _Optional[int] = ..., backoff_until: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., consecutive_failures: _Optional[int] = ..., last_scale_up: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., last_scale_down: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., slices: _Optional[_Iterable[_Union[SliceInfo, _Mapping]]] = ...) -> None: ...

class AutoscalerAction(_message.Message):
    __slots__ = ("timestamp", "action_type", "scale_group", "slice_id", "reason", "status")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    ACTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    timestamp: _time_pb2.Timestamp
    action_type: str
    scale_group: str
    slice_id: str
    reason: str
    status: str
    def __init__(self, timestamp: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., action_type: _Optional[str] = ..., scale_group: _Optional[str] = ..., slice_id: _Optional[str] = ..., reason: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class DemandEntryStatus(_message.Message):
    __slots__ = ("task_ids", "coschedule_group_id", "accelerator_type", "accelerator_variant", "preemptible", "resources")
    TASK_IDS_FIELD_NUMBER: _ClassVar[int]
    COSCHEDULE_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_VARIANT_FIELD_NUMBER: _ClassVar[int]
    PREEMPTIBLE_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    task_ids: _containers.RepeatedScalarFieldContainer[str]
    coschedule_group_id: str
    accelerator_type: _config_pb2.AcceleratorType
    accelerator_variant: str
    preemptible: bool
    resources: ResourceSpec
    def __init__(self, task_ids: _Optional[_Iterable[str]] = ..., coschedule_group_id: _Optional[str] = ..., accelerator_type: _Optional[_Union[_config_pb2.AcceleratorType, str]] = ..., accelerator_variant: _Optional[str] = ..., preemptible: _Optional[bool] = ..., resources: _Optional[_Union[ResourceSpec, _Mapping]] = ...) -> None: ...

class UnmetDemand(_message.Message):
    __slots__ = ("entry", "reason")
    ENTRY_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    entry: DemandEntryStatus
    reason: str
    def __init__(self, entry: _Optional[_Union[DemandEntryStatus, _Mapping]] = ..., reason: _Optional[str] = ...) -> None: ...

class RoutingDecision(_message.Message):
    __slots__ = ("group_to_launch", "group_reasons", "routed_entries", "unmet_entries")
    class GroupToLaunchEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class GroupReasonsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class RoutedEntriesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: DemandEntryStatusList
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[DemandEntryStatusList, _Mapping]] = ...) -> None: ...
    GROUP_TO_LAUNCH_FIELD_NUMBER: _ClassVar[int]
    GROUP_REASONS_FIELD_NUMBER: _ClassVar[int]
    ROUTED_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    UNMET_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    group_to_launch: _containers.ScalarMap[str, int]
    group_reasons: _containers.ScalarMap[str, str]
    routed_entries: _containers.MessageMap[str, DemandEntryStatusList]
    unmet_entries: _containers.RepeatedCompositeFieldContainer[UnmetDemand]
    def __init__(self, group_to_launch: _Optional[_Mapping[str, int]] = ..., group_reasons: _Optional[_Mapping[str, str]] = ..., routed_entries: _Optional[_Mapping[str, DemandEntryStatusList]] = ..., unmet_entries: _Optional[_Iterable[_Union[UnmetDemand, _Mapping]]] = ...) -> None: ...

class DemandEntryStatusList(_message.Message):
    __slots__ = ("entries",)
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[DemandEntryStatus]
    def __init__(self, entries: _Optional[_Iterable[_Union[DemandEntryStatus, _Mapping]]] = ...) -> None: ...

class AutoscalerStatus(_message.Message):
    __slots__ = ("groups", "current_demand", "last_evaluation", "recent_actions", "last_routing_decision")
    class CurrentDemandEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_DEMAND_FIELD_NUMBER: _ClassVar[int]
    LAST_EVALUATION_FIELD_NUMBER: _ClassVar[int]
    RECENT_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    LAST_ROUTING_DECISION_FIELD_NUMBER: _ClassVar[int]
    groups: _containers.RepeatedCompositeFieldContainer[ScaleGroupStatus]
    current_demand: _containers.ScalarMap[str, int]
    last_evaluation: _time_pb2.Timestamp
    recent_actions: _containers.RepeatedCompositeFieldContainer[AutoscalerAction]
    last_routing_decision: RoutingDecision
    def __init__(self, groups: _Optional[_Iterable[_Union[ScaleGroupStatus, _Mapping]]] = ..., current_demand: _Optional[_Mapping[str, int]] = ..., last_evaluation: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., recent_actions: _Optional[_Iterable[_Union[AutoscalerAction, _Mapping]]] = ..., last_routing_decision: _Optional[_Union[RoutingDecision, _Mapping]] = ...) -> None: ...
