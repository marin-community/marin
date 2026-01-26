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

class ScaleGroupConfig(_message.Message):
    __slots__ = ("name", "min_slices", "max_slices", "accelerator_type", "runtime_version", "preemptible", "zones", "priority")
    NAME_FIELD_NUMBER: _ClassVar[int]
    MIN_SLICES_FIELD_NUMBER: _ClassVar[int]
    MAX_SLICES_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_VERSION_FIELD_NUMBER: _ClassVar[int]
    PREEMPTIBLE_FIELD_NUMBER: _ClassVar[int]
    ZONES_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    name: str
    min_slices: int
    max_slices: int
    accelerator_type: str
    runtime_version: str
    preemptible: bool
    zones: _containers.RepeatedScalarFieldContainer[str]
    priority: int
    def __init__(self, name: _Optional[str] = ..., min_slices: _Optional[int] = ..., max_slices: _Optional[int] = ..., accelerator_type: _Optional[str] = ..., runtime_version: _Optional[str] = ..., preemptible: _Optional[bool] = ..., zones: _Optional[_Iterable[str]] = ..., priority: _Optional[int] = ...) -> None: ...

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

class BootstrapConfig(_message.Message):
    __slots__ = ("controller_address", "worker_id", "worker_port", "docker_image", "cache_dir", "env_vars")
    class EnvVarsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CONTROLLER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_PORT_FIELD_NUMBER: _ClassVar[int]
    DOCKER_IMAGE_FIELD_NUMBER: _ClassVar[int]
    CACHE_DIR_FIELD_NUMBER: _ClassVar[int]
    ENV_VARS_FIELD_NUMBER: _ClassVar[int]
    controller_address: str
    worker_id: str
    worker_port: int
    docker_image: str
    cache_dir: str
    env_vars: _containers.ScalarMap[str, str]
    def __init__(self, controller_address: _Optional[str] = ..., worker_id: _Optional[str] = ..., worker_port: _Optional[int] = ..., docker_image: _Optional[str] = ..., cache_dir: _Optional[str] = ..., env_vars: _Optional[_Mapping[str, str]] = ...) -> None: ...

class TimeoutConfig(_message.Message):
    __slots__ = ("boot_timeout_seconds", "init_timeout_seconds", "ssh_connect_timeout_seconds", "ssh_poll_interval_seconds")
    BOOT_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    INIT_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SSH_CONNECT_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SSH_POLL_INTERVAL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    boot_timeout_seconds: int
    init_timeout_seconds: int
    ssh_connect_timeout_seconds: int
    ssh_poll_interval_seconds: int
    def __init__(self, boot_timeout_seconds: _Optional[int] = ..., init_timeout_seconds: _Optional[int] = ..., ssh_connect_timeout_seconds: _Optional[int] = ..., ssh_poll_interval_seconds: _Optional[int] = ...) -> None: ...

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
    config: ScaleGroupConfig
    current_demand: int
    peak_demand: int
    backoff_until_ms: int
    consecutive_failures: int
    last_scale_up_ms: int
    last_scale_down_ms: int
    slices: _containers.RepeatedCompositeFieldContainer[SliceInfo]
    def __init__(self, name: _Optional[str] = ..., config: _Optional[_Union[ScaleGroupConfig, _Mapping]] = ..., current_demand: _Optional[int] = ..., peak_demand: _Optional[int] = ..., backoff_until_ms: _Optional[int] = ..., consecutive_failures: _Optional[int] = ..., last_scale_up_ms: _Optional[int] = ..., last_scale_down_ms: _Optional[int] = ..., slices: _Optional[_Iterable[_Union[SliceInfo, _Mapping]]] = ...) -> None: ...

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

class ControllerVmConfig(_message.Message):
    __slots__ = ("enabled", "image", "machine_type", "boot_disk_size_gb", "port", "host")
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    MACHINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    BOOT_DISK_SIZE_GB_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    image: str
    machine_type: str
    boot_disk_size_gb: int
    port: int
    host: str
    def __init__(self, enabled: _Optional[bool] = ..., image: _Optional[str] = ..., machine_type: _Optional[str] = ..., boot_disk_size_gb: _Optional[int] = ..., port: _Optional[int] = ..., host: _Optional[str] = ...) -> None: ...

class IrisClusterConfig(_message.Message):
    __slots__ = ("provider_type", "project_id", "region", "zone", "ssh_user", "ssh_private_key", "docker_image", "worker_port", "controller_address", "controller_vm", "manual_hosts", "scale_groups", "boot_timeout_seconds", "init_timeout_seconds", "ssh_connect_timeout_seconds", "ssh_poll_interval_seconds", "label_prefix")
    class ScaleGroupsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ScaleGroupConfig
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ScaleGroupConfig, _Mapping]] = ...) -> None: ...
    PROVIDER_TYPE_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    ZONE_FIELD_NUMBER: _ClassVar[int]
    SSH_USER_FIELD_NUMBER: _ClassVar[int]
    SSH_PRIVATE_KEY_FIELD_NUMBER: _ClassVar[int]
    DOCKER_IMAGE_FIELD_NUMBER: _ClassVar[int]
    WORKER_PORT_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_VM_FIELD_NUMBER: _ClassVar[int]
    MANUAL_HOSTS_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUPS_FIELD_NUMBER: _ClassVar[int]
    BOOT_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    INIT_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SSH_CONNECT_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SSH_POLL_INTERVAL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    LABEL_PREFIX_FIELD_NUMBER: _ClassVar[int]
    provider_type: str
    project_id: str
    region: str
    zone: str
    ssh_user: str
    ssh_private_key: str
    docker_image: str
    worker_port: int
    controller_address: str
    controller_vm: ControllerVmConfig
    manual_hosts: _containers.RepeatedScalarFieldContainer[str]
    scale_groups: _containers.MessageMap[str, ScaleGroupConfig]
    boot_timeout_seconds: int
    init_timeout_seconds: int
    ssh_connect_timeout_seconds: int
    ssh_poll_interval_seconds: int
    label_prefix: str
    def __init__(self, provider_type: _Optional[str] = ..., project_id: _Optional[str] = ..., region: _Optional[str] = ..., zone: _Optional[str] = ..., ssh_user: _Optional[str] = ..., ssh_private_key: _Optional[str] = ..., docker_image: _Optional[str] = ..., worker_port: _Optional[int] = ..., controller_address: _Optional[str] = ..., controller_vm: _Optional[_Union[ControllerVmConfig, _Mapping]] = ..., manual_hosts: _Optional[_Iterable[str]] = ..., scale_groups: _Optional[_Mapping[str, ScaleGroupConfig]] = ..., boot_timeout_seconds: _Optional[int] = ..., init_timeout_seconds: _Optional[int] = ..., ssh_connect_timeout_seconds: _Optional[int] = ..., ssh_poll_interval_seconds: _Optional[int] = ..., label_prefix: _Optional[str] = ...) -> None: ...
