from . import time_pb2 as _time_pb2
from . import config_pb2 as _config_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar

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
    __slots__ = (
        "address",
        "created_at",
        "init_error",
        "init_log_tail",
        "init_phase",
        "labels",
        "scale_group",
        "slice_id",
        "state",
        "state_changed_at",
        "vm_id",
        "worker_healthy",
        "worker_id",
        "zone",
    )
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...

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
    def __init__(
        self,
        vm_id: str | None = ...,
        slice_id: str | None = ...,
        scale_group: str | None = ...,
        state: VmState | str | None = ...,
        address: str | None = ...,
        zone: str | None = ...,
        created_at: _time_pb2.Timestamp | _Mapping | None = ...,
        state_changed_at: _time_pb2.Timestamp | _Mapping | None = ...,
        worker_id: str | None = ...,
        worker_healthy: bool | None = ...,
        init_phase: str | None = ...,
        init_log_tail: str | None = ...,
        init_error: str | None = ...,
        labels: _Mapping[str, str] | None = ...,
    ) -> None: ...

class SliceInfo(_message.Message):
    __slots__ = ("created_at", "scale_group", "slice_id", "vms")
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    VMS_FIELD_NUMBER: _ClassVar[int]
    slice_id: str
    scale_group: str
    created_at: _time_pb2.Timestamp
    vms: _containers.RepeatedCompositeFieldContainer[VmInfo]
    def __init__(
        self,
        slice_id: str | None = ...,
        scale_group: str | None = ...,
        created_at: _time_pb2.Timestamp | _Mapping | None = ...,
        vms: _Iterable[VmInfo | _Mapping] | None = ...,
    ) -> None: ...

class ScalingDecision(_message.Message):
    __slots__ = ("action", "reason", "scale_group", "slice_delta")
    SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    SLICE_DELTA_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    scale_group: str
    action: ScalingAction
    slice_delta: int
    reason: str
    def __init__(
        self,
        scale_group: str | None = ...,
        action: ScalingAction | str | None = ...,
        slice_delta: int | None = ...,
        reason: str | None = ...,
    ) -> None: ...

class ScaleGroupStatus(_message.Message):
    __slots__ = (
        "backoff_until",
        "config",
        "consecutive_failures",
        "current_demand",
        "last_scale_down",
        "last_scale_up",
        "name",
        "peak_demand",
        "slices",
    )
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
    def __init__(
        self,
        name: str | None = ...,
        config: _config_pb2.ScaleGroupConfig | _Mapping | None = ...,
        current_demand: int | None = ...,
        peak_demand: int | None = ...,
        backoff_until: _time_pb2.Timestamp | _Mapping | None = ...,
        consecutive_failures: int | None = ...,
        last_scale_up: _time_pb2.Timestamp | _Mapping | None = ...,
        last_scale_down: _time_pb2.Timestamp | _Mapping | None = ...,
        slices: _Iterable[SliceInfo | _Mapping] | None = ...,
    ) -> None: ...

class AutoscalerAction(_message.Message):
    __slots__ = ("action_type", "reason", "scale_group", "slice_id", "status", "timestamp")
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
    def __init__(
        self,
        timestamp: _time_pb2.Timestamp | _Mapping | None = ...,
        action_type: str | None = ...,
        scale_group: str | None = ...,
        slice_id: str | None = ...,
        reason: str | None = ...,
        status: str | None = ...,
    ) -> None: ...

class AutoscalerStatus(_message.Message):
    __slots__ = ("current_demand", "groups", "last_evaluation", "recent_actions")
    class CurrentDemandEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: str | None = ..., value: int | None = ...) -> None: ...

    GROUPS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_DEMAND_FIELD_NUMBER: _ClassVar[int]
    LAST_EVALUATION_FIELD_NUMBER: _ClassVar[int]
    RECENT_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    groups: _containers.RepeatedCompositeFieldContainer[ScaleGroupStatus]
    current_demand: _containers.ScalarMap[str, int]
    last_evaluation: _time_pb2.Timestamp
    recent_actions: _containers.RepeatedCompositeFieldContainer[AutoscalerAction]
    def __init__(
        self,
        groups: _Iterable[ScaleGroupStatus | _Mapping] | None = ...,
        current_demand: _Mapping[str, int] | None = ...,
        last_evaluation: _time_pb2.Timestamp | _Mapping | None = ...,
        recent_actions: _Iterable[AutoscalerAction | _Mapping] | None = ...,
    ) -> None: ...
