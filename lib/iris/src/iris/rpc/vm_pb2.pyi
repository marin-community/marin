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

class VmInfo(_message.Message):
    __slots__ = ("vm_id", "slice_id", "scale_group", "state", "address", "zone", "created_at_ms", "state_changed_at_ms", "worker_id", "worker_healthy", "init_phase", "init_log_tail", "init_error", "labels")
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
    CREATED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    STATE_CHANGED_AT_MS_FIELD_NUMBER: _ClassVar[int]
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
    created_at_ms: int
    state_changed_at_ms: int
    worker_id: str
    worker_healthy: bool
    init_phase: str
    init_log_tail: str
    init_error: str
    labels: _containers.ScalarMap[str, str]
    def __init__(self, vm_id: _Optional[str] = ..., slice_id: _Optional[str] = ..., scale_group: _Optional[str] = ..., state: _Optional[_Union[VmState, str]] = ..., address: _Optional[str] = ..., zone: _Optional[str] = ..., created_at_ms: _Optional[int] = ..., state_changed_at_ms: _Optional[int] = ..., worker_id: _Optional[str] = ..., worker_healthy: _Optional[bool] = ..., init_phase: _Optional[str] = ..., init_log_tail: _Optional[str] = ..., init_error: _Optional[str] = ..., labels: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SliceInfo(_message.Message):
    __slots__ = ("slice_id", "scale_group", "created_at_ms", "vms")
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    VMS_FIELD_NUMBER: _ClassVar[int]
    slice_id: str
    scale_group: str
    created_at_ms: int
    vms: _containers.RepeatedCompositeFieldContainer[VmInfo]
    def __init__(self, slice_id: _Optional[str] = ..., scale_group: _Optional[str] = ..., created_at_ms: _Optional[int] = ..., vms: _Optional[_Iterable[_Union[VmInfo, _Mapping]]] = ...) -> None: ...

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
    __slots__ = ("name", "config", "current_demand", "peak_demand", "backoff_until_ms", "consecutive_failures", "last_scale_up_ms", "last_scale_down_ms", "slices")
    NAME_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    CURRENT_DEMAND_FIELD_NUMBER: _ClassVar[int]
    PEAK_DEMAND_FIELD_NUMBER: _ClassVar[int]
    BACKOFF_UNTIL_MS_FIELD_NUMBER: _ClassVar[int]
    CONSECUTIVE_FAILURES_FIELD_NUMBER: _ClassVar[int]
    LAST_SCALE_UP_MS_FIELD_NUMBER: _ClassVar[int]
    LAST_SCALE_DOWN_MS_FIELD_NUMBER: _ClassVar[int]
    SLICES_FIELD_NUMBER: _ClassVar[int]
    name: str
    config: _config_pb2.ScaleGroupConfig
    current_demand: int
    peak_demand: int
    backoff_until_ms: int
    consecutive_failures: int
    last_scale_up_ms: int
    last_scale_down_ms: int
    slices: _containers.RepeatedCompositeFieldContainer[SliceInfo]
    def __init__(self, name: _Optional[str] = ..., config: _Optional[_Union[_config_pb2.ScaleGroupConfig, _Mapping]] = ..., current_demand: _Optional[int] = ..., peak_demand: _Optional[int] = ..., backoff_until_ms: _Optional[int] = ..., consecutive_failures: _Optional[int] = ..., last_scale_up_ms: _Optional[int] = ..., last_scale_down_ms: _Optional[int] = ..., slices: _Optional[_Iterable[_Union[SliceInfo, _Mapping]]] = ...) -> None: ...

class AutoscalerAction(_message.Message):
    __slots__ = ("timestamp_ms", "action_type", "scale_group", "slice_id", "reason", "status")
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    ACTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    timestamp_ms: int
    action_type: str
    scale_group: str
    slice_id: str
    reason: str
    status: str
    def __init__(self, timestamp_ms: _Optional[int] = ..., action_type: _Optional[str] = ..., scale_group: _Optional[str] = ..., slice_id: _Optional[str] = ..., reason: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class AutoscalerStatus(_message.Message):
    __slots__ = ("groups", "current_demand", "last_evaluation_ms", "recent_actions")
    class CurrentDemandEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_DEMAND_FIELD_NUMBER: _ClassVar[int]
    LAST_EVALUATION_MS_FIELD_NUMBER: _ClassVar[int]
    RECENT_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    groups: _containers.RepeatedCompositeFieldContainer[ScaleGroupStatus]
    current_demand: _containers.ScalarMap[str, int]
    last_evaluation_ms: int
    recent_actions: _containers.RepeatedCompositeFieldContainer[AutoscalerAction]
    def __init__(self, groups: _Optional[_Iterable[_Union[ScaleGroupStatus, _Mapping]]] = ..., current_demand: _Optional[_Mapping[str, int]] = ..., last_evaluation_ms: _Optional[int] = ..., recent_actions: _Optional[_Iterable[_Union[AutoscalerAction, _Mapping]]] = ...) -> None: ...
