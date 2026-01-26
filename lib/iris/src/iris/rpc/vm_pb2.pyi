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
    __slots__ = (
        "address",
        "created_at_ms",
        "init_error",
        "init_log_tail",
        "init_phase",
        "labels",
        "scale_group",
        "slice_id",
        "state",
        "state_changed_at_ms",
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
    def __init__(
        self,
        vm_id: str | None = ...,
        slice_id: str | None = ...,
        scale_group: str | None = ...,
        state: VmState | str | None = ...,
        address: str | None = ...,
        zone: str | None = ...,
        created_at_ms: int | None = ...,
        state_changed_at_ms: int | None = ...,
        worker_id: str | None = ...,
        worker_healthy: bool | None = ...,
        init_phase: str | None = ...,
        init_log_tail: str | None = ...,
        init_error: str | None = ...,
        labels: _Mapping[str, str] | None = ...,
    ) -> None: ...

class SliceInfo(_message.Message):
    __slots__ = ("created_at_ms", "scale_group", "slice_id", "vms")
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    VMS_FIELD_NUMBER: _ClassVar[int]
    slice_id: str
    scale_group: str
    created_at_ms: int
    vms: _containers.RepeatedCompositeFieldContainer[VmInfo]
    def __init__(
        self,
        slice_id: str | None = ...,
        scale_group: str | None = ...,
        created_at_ms: int | None = ...,
        vms: _Iterable[VmInfo | _Mapping] | None = ...,
    ) -> None: ...

class TpuProvider(_message.Message):
    __slots__ = ("project_id",)
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    def __init__(self, project_id: str | None = ...) -> None: ...

class ManualProvider(_message.Message):
    __slots__ = ("hosts", "ssh_key_file", "ssh_port", "ssh_user")
    HOSTS_FIELD_NUMBER: _ClassVar[int]
    SSH_USER_FIELD_NUMBER: _ClassVar[int]
    SSH_KEY_FILE_FIELD_NUMBER: _ClassVar[int]
    SSH_PORT_FIELD_NUMBER: _ClassVar[int]
    hosts: _containers.RepeatedScalarFieldContainer[str]
    ssh_user: str
    ssh_key_file: str
    ssh_port: int
    def __init__(
        self,
        hosts: _Iterable[str] | None = ...,
        ssh_user: str | None = ...,
        ssh_key_file: str | None = ...,
        ssh_port: int | None = ...,
    ) -> None: ...

class ProviderConfig(_message.Message):
    __slots__ = ("manual", "tpu")
    TPU_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    tpu: TpuProvider
    manual: ManualProvider
    def __init__(
        self, tpu: TpuProvider | _Mapping | None = ..., manual: ManualProvider | _Mapping | None = ...
    ) -> None: ...

class ScaleGroupConfig(_message.Message):
    __slots__ = (
        "accelerator_type",
        "max_slices",
        "min_slices",
        "name",
        "preemptible",
        "priority",
        "provider",
        "runtime_version",
        "zones",
    )
    NAME_FIELD_NUMBER: _ClassVar[int]
    MIN_SLICES_FIELD_NUMBER: _ClassVar[int]
    MAX_SLICES_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_VERSION_FIELD_NUMBER: _ClassVar[int]
    PREEMPTIBLE_FIELD_NUMBER: _ClassVar[int]
    ZONES_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    name: str
    min_slices: int
    max_slices: int
    accelerator_type: str
    runtime_version: str
    preemptible: bool
    zones: _containers.RepeatedScalarFieldContainer[str]
    priority: int
    provider: ProviderConfig
    def __init__(
        self,
        name: str | None = ...,
        min_slices: int | None = ...,
        max_slices: int | None = ...,
        accelerator_type: str | None = ...,
        runtime_version: str | None = ...,
        preemptible: bool | None = ...,
        zones: _Iterable[str] | None = ...,
        priority: int | None = ...,
        provider: ProviderConfig | _Mapping | None = ...,
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

class BootstrapConfig(_message.Message):
    __slots__ = ("cache_dir", "controller_address", "docker_image", "env_vars", "worker_id", "worker_port")
    class EnvVarsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...

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
    def __init__(
        self,
        controller_address: str | None = ...,
        worker_id: str | None = ...,
        worker_port: int | None = ...,
        docker_image: str | None = ...,
        cache_dir: str | None = ...,
        env_vars: _Mapping[str, str] | None = ...,
    ) -> None: ...

class TimeoutConfig(_message.Message):
    __slots__ = (
        "boot_timeout_seconds",
        "init_timeout_seconds",
        "ssh_connect_timeout_seconds",
        "ssh_poll_interval_seconds",
    )
    BOOT_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    INIT_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SSH_CONNECT_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SSH_POLL_INTERVAL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    boot_timeout_seconds: int
    init_timeout_seconds: int
    ssh_connect_timeout_seconds: int
    ssh_poll_interval_seconds: int
    def __init__(
        self,
        boot_timeout_seconds: int | None = ...,
        init_timeout_seconds: int | None = ...,
        ssh_connect_timeout_seconds: int | None = ...,
        ssh_poll_interval_seconds: int | None = ...,
    ) -> None: ...

class ScaleGroupStatus(_message.Message):
    __slots__ = (
        "backoff_until_ms",
        "config",
        "consecutive_failures",
        "current_demand",
        "last_scale_down_ms",
        "last_scale_up_ms",
        "name",
        "peak_demand",
        "slices",
    )
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
    def __init__(
        self,
        name: str | None = ...,
        config: ScaleGroupConfig | _Mapping | None = ...,
        current_demand: int | None = ...,
        peak_demand: int | None = ...,
        backoff_until_ms: int | None = ...,
        consecutive_failures: int | None = ...,
        last_scale_up_ms: int | None = ...,
        last_scale_down_ms: int | None = ...,
        slices: _Iterable[SliceInfo | _Mapping] | None = ...,
    ) -> None: ...

class AutoscalerAction(_message.Message):
    __slots__ = ("action_type", "reason", "scale_group", "slice_id", "status", "timestamp_ms")
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
    def __init__(
        self,
        timestamp_ms: int | None = ...,
        action_type: str | None = ...,
        scale_group: str | None = ...,
        slice_id: str | None = ...,
        reason: str | None = ...,
        status: str | None = ...,
    ) -> None: ...

class AutoscalerStatus(_message.Message):
    __slots__ = ("current_demand", "groups", "last_evaluation_ms", "recent_actions")
    class CurrentDemandEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: str | None = ..., value: int | None = ...) -> None: ...

    GROUPS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_DEMAND_FIELD_NUMBER: _ClassVar[int]
    LAST_EVALUATION_MS_FIELD_NUMBER: _ClassVar[int]
    RECENT_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    groups: _containers.RepeatedCompositeFieldContainer[ScaleGroupStatus]
    current_demand: _containers.ScalarMap[str, int]
    last_evaluation_ms: int
    recent_actions: _containers.RepeatedCompositeFieldContainer[AutoscalerAction]
    def __init__(
        self,
        groups: _Iterable[ScaleGroupStatus | _Mapping] | None = ...,
        current_demand: _Mapping[str, int] | None = ...,
        last_evaluation_ms: int | None = ...,
        recent_actions: _Iterable[AutoscalerAction | _Mapping] | None = ...,
    ) -> None: ...

class GcpControllerConfig(_message.Message):
    __slots__ = ("boot_disk_size_gb", "image", "machine_type", "port")
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    MACHINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    BOOT_DISK_SIZE_GB_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    image: str
    machine_type: str
    boot_disk_size_gb: int
    port: int
    def __init__(
        self,
        image: str | None = ...,
        machine_type: str | None = ...,
        boot_disk_size_gb: int | None = ...,
        port: int | None = ...,
    ) -> None: ...

class ManualControllerConfig(_message.Message):
    __slots__ = ("host", "image", "port")
    HOST_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    host: str
    image: str
    port: int
    def __init__(self, host: str | None = ..., image: str | None = ..., port: int | None = ...) -> None: ...

class ControllerVmConfig(_message.Message):
    __slots__ = ("gcp", "manual")
    GCP_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    gcp: GcpControllerConfig
    manual: ManualControllerConfig
    def __init__(
        self, gcp: GcpControllerConfig | _Mapping | None = ..., manual: ManualControllerConfig | _Mapping | None = ...
    ) -> None: ...

class IrisClusterConfig(_message.Message):
    __slots__ = (
        "boot_timeout_seconds",
        "controller_address",
        "controller_vm",
        "docker_image",
        "init_timeout_seconds",
        "label_prefix",
        "manual_hosts",
        "project_id",
        "provider_type",
        "region",
        "scale_groups",
        "ssh_connect_timeout_seconds",
        "ssh_poll_interval_seconds",
        "ssh_private_key",
        "ssh_user",
        "worker_port",
        "zone",
    )
    class ScaleGroupsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ScaleGroupConfig
        def __init__(self, key: str | None = ..., value: ScaleGroupConfig | _Mapping | None = ...) -> None: ...

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
    def __init__(
        self,
        provider_type: str | None = ...,
        project_id: str | None = ...,
        region: str | None = ...,
        zone: str | None = ...,
        ssh_user: str | None = ...,
        ssh_private_key: str | None = ...,
        docker_image: str | None = ...,
        worker_port: int | None = ...,
        controller_address: str | None = ...,
        controller_vm: ControllerVmConfig | _Mapping | None = ...,
        manual_hosts: _Iterable[str] | None = ...,
        scale_groups: _Mapping[str, ScaleGroupConfig] | None = ...,
        boot_timeout_seconds: int | None = ...,
        init_timeout_seconds: int | None = ...,
        ssh_connect_timeout_seconds: int | None = ...,
        ssh_poll_interval_seconds: int | None = ...,
        label_prefix: str | None = ...,
    ) -> None: ...
