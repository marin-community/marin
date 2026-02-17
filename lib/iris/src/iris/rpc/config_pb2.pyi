from . import time_pb2 as _time_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class AcceleratorType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACCELERATOR_TYPE_UNSPECIFIED: _ClassVar[AcceleratorType]
    ACCELERATOR_TYPE_CPU: _ClassVar[AcceleratorType]
    ACCELERATOR_TYPE_GPU: _ClassVar[AcceleratorType]
    ACCELERATOR_TYPE_TPU: _ClassVar[AcceleratorType]
ACCELERATOR_TYPE_UNSPECIFIED: AcceleratorType
ACCELERATOR_TYPE_CPU: AcceleratorType
ACCELERATOR_TYPE_GPU: AcceleratorType
ACCELERATOR_TYPE_TPU: AcceleratorType

class GcpPlatformConfig(_message.Message):
    __slots__ = ("project_id", "zones")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    ZONES_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    zones: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, project_id: str | None = ..., zones: _Iterable[str] | None = ...) -> None: ...

class ManualPlatformConfig(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class LocalPlatformConfig(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CoreweavePlatformConfig(_message.Message):
    __slots__ = ("region",)
    REGION_FIELD_NUMBER: _ClassVar[int]
    region: str
    def __init__(self, region: str | None = ...) -> None: ...

class PlatformConfig(_message.Message):
    __slots__ = ("coreweave", "gcp", "label_prefix", "local", "manual")
    LABEL_PREFIX_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    LOCAL_FIELD_NUMBER: _ClassVar[int]
    COREWEAVE_FIELD_NUMBER: _ClassVar[int]
    label_prefix: str
    gcp: GcpPlatformConfig
    manual: ManualPlatformConfig
    local: LocalPlatformConfig
    coreweave: CoreweavePlatformConfig
    def __init__(self, label_prefix: str | None = ..., gcp: GcpPlatformConfig | _Mapping | None = ..., manual: ManualPlatformConfig | _Mapping | None = ..., local: LocalPlatformConfig | _Mapping | None = ..., coreweave: CoreweavePlatformConfig | _Mapping | None = ...) -> None: ...

class ManualProvider(_message.Message):
    __slots__ = ("hosts", "ssh_key_file", "ssh_user")
    HOSTS_FIELD_NUMBER: _ClassVar[int]
    SSH_USER_FIELD_NUMBER: _ClassVar[int]
    SSH_KEY_FILE_FIELD_NUMBER: _ClassVar[int]
    hosts: _containers.RepeatedScalarFieldContainer[str]
    ssh_user: str
    ssh_key_file: str
    def __init__(self, hosts: _Iterable[str] | None = ..., ssh_user: str | None = ..., ssh_key_file: str | None = ...) -> None: ...

class GcpVmConfig(_message.Message):
    __slots__ = ("boot_disk_size_gb", "machine_type", "zone")
    ZONE_FIELD_NUMBER: _ClassVar[int]
    MACHINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    BOOT_DISK_SIZE_GB_FIELD_NUMBER: _ClassVar[int]
    zone: str
    machine_type: str
    boot_disk_size_gb: int
    def __init__(self, zone: str | None = ..., machine_type: str | None = ..., boot_disk_size_gb: int | None = ...) -> None: ...

class ManualVmConfig(_message.Message):
    __slots__ = ("host", "ssh_key_file", "ssh_user")
    HOST_FIELD_NUMBER: _ClassVar[int]
    SSH_USER_FIELD_NUMBER: _ClassVar[int]
    SSH_KEY_FILE_FIELD_NUMBER: _ClassVar[int]
    host: str
    ssh_user: str
    ssh_key_file: str
    def __init__(self, host: str | None = ..., ssh_user: str | None = ..., ssh_key_file: str | None = ...) -> None: ...

class VmConfig(_message.Message):
    __slots__ = ("gcp", "labels", "manual", "metadata", "name")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    name: str
    labels: _containers.ScalarMap[str, str]
    metadata: _containers.ScalarMap[str, str]
    gcp: GcpVmConfig
    manual: ManualVmConfig
    def __init__(self, name: str | None = ..., labels: _Mapping[str, str] | None = ..., metadata: _Mapping[str, str] | None = ..., gcp: GcpVmConfig | _Mapping | None = ..., manual: ManualVmConfig | _Mapping | None = ...) -> None: ...

class GcpSliceConfig(_message.Message):
    __slots__ = ("runtime_version", "topology", "zone")
    ZONE_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_VERSION_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_FIELD_NUMBER: _ClassVar[int]
    zone: str
    runtime_version: str
    topology: str
    def __init__(self, zone: str | None = ..., runtime_version: str | None = ..., topology: str | None = ...) -> None: ...

class CoreweaveSliceConfig(_message.Message):
    __slots__ = ("instance_type", "region")
    REGION_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    region: str
    instance_type: str
    def __init__(self, region: str | None = ..., instance_type: str | None = ...) -> None: ...

class ManualSliceConfig(_message.Message):
    __slots__ = ("hosts", "ssh_key_file", "ssh_user")
    HOSTS_FIELD_NUMBER: _ClassVar[int]
    SSH_USER_FIELD_NUMBER: _ClassVar[int]
    SSH_KEY_FILE_FIELD_NUMBER: _ClassVar[int]
    hosts: _containers.RepeatedScalarFieldContainer[str]
    ssh_user: str
    ssh_key_file: str
    def __init__(self, hosts: _Iterable[str] | None = ..., ssh_user: str | None = ..., ssh_key_file: str | None = ...) -> None: ...

class LocalSliceConfig(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SliceConfig(_message.Message):
    __slots__ = ("accelerator_type", "accelerator_variant", "coreweave", "gcp", "labels", "local", "manual", "name_prefix", "num_vms", "preemptible")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...
    NAME_PREFIX_FIELD_NUMBER: _ClassVar[int]
    NUM_VMS_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_VARIANT_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    PREEMPTIBLE_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    COREWEAVE_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    LOCAL_FIELD_NUMBER: _ClassVar[int]
    name_prefix: str
    num_vms: int
    accelerator_type: AcceleratorType
    accelerator_variant: str
    labels: _containers.ScalarMap[str, str]
    preemptible: bool
    gcp: GcpSliceConfig
    coreweave: CoreweaveSliceConfig
    manual: ManualSliceConfig
    local: LocalSliceConfig
    def __init__(self, name_prefix: str | None = ..., num_vms: int | None = ..., accelerator_type: AcceleratorType | str | None = ..., accelerator_variant: str | None = ..., labels: _Mapping[str, str] | None = ..., preemptible: bool | None = ..., gcp: GcpSliceConfig | _Mapping | None = ..., coreweave: CoreweaveSliceConfig | _Mapping | None = ..., manual: ManualSliceConfig | _Mapping | None = ..., local: LocalSliceConfig | _Mapping | None = ...) -> None: ...

class ScaleGroupResources(_message.Message):
    __slots__ = ("cpu", "disk_bytes", "gpu_count", "memory_bytes", "tpu_count")
    CPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    TPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    cpu: int
    memory_bytes: int
    disk_bytes: int
    gpu_count: int
    tpu_count: int
    def __init__(self, cpu: int | None = ..., memory_bytes: int | None = ..., disk_bytes: int | None = ..., gpu_count: int | None = ..., tpu_count: int | None = ...) -> None: ...

class ScaleGroupConfig(_message.Message):
    __slots__ = ("accelerator_type", "accelerator_variant", "max_slices", "min_slices", "name", "num_vms", "priority", "resources", "slice_template")
    NAME_FIELD_NUMBER: _ClassVar[int]
    MIN_SLICES_FIELD_NUMBER: _ClassVar[int]
    MAX_SLICES_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_VARIANT_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    NUM_VMS_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    SLICE_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    name: str
    min_slices: int
    max_slices: int
    accelerator_type: AcceleratorType
    accelerator_variant: str
    resources: ScaleGroupResources
    num_vms: int
    priority: int
    slice_template: SliceConfig
    def __init__(self, name: str | None = ..., min_slices: int | None = ..., max_slices: int | None = ..., accelerator_type: AcceleratorType | str | None = ..., accelerator_variant: str | None = ..., resources: ScaleGroupResources | _Mapping | None = ..., num_vms: int | None = ..., priority: int | None = ..., slice_template: SliceConfig | _Mapping | None = ...) -> None: ...

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
    def __init__(self, controller_address: str | None = ..., worker_id: str | None = ..., worker_port: int | None = ..., docker_image: str | None = ..., cache_dir: str | None = ..., env_vars: _Mapping[str, str] | None = ...) -> None: ...

class TimeoutConfig(_message.Message):
    __slots__ = ("boot_timeout", "init_timeout", "ssh_poll_interval")
    BOOT_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    INIT_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    SSH_POLL_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    boot_timeout: _time_pb2.Duration
    init_timeout: _time_pb2.Duration
    ssh_poll_interval: _time_pb2.Duration
    def __init__(self, boot_timeout: _time_pb2.Duration | _Mapping | None = ..., init_timeout: _time_pb2.Duration | _Mapping | None = ..., ssh_poll_interval: _time_pb2.Duration | _Mapping | None = ...) -> None: ...

class SshConfig(_message.Message):
    __slots__ = ("connect_timeout", "key_file", "port", "user")
    USER_FIELD_NUMBER: _ClassVar[int]
    KEY_FILE_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    CONNECT_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    user: str
    key_file: str
    port: int
    connect_timeout: _time_pb2.Duration
    def __init__(self, user: str | None = ..., key_file: str | None = ..., port: int | None = ..., connect_timeout: _time_pb2.Duration | _Mapping | None = ...) -> None: ...

class GcpControllerConfig(_message.Message):
    __slots__ = ("boot_disk_size_gb", "machine_type", "port", "zone")
    ZONE_FIELD_NUMBER: _ClassVar[int]
    MACHINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    BOOT_DISK_SIZE_GB_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    zone: str
    machine_type: str
    boot_disk_size_gb: int
    port: int
    def __init__(self, zone: str | None = ..., machine_type: str | None = ..., boot_disk_size_gb: int | None = ..., port: int | None = ...) -> None: ...

class ManualControllerConfig(_message.Message):
    __slots__ = ("host", "port")
    HOST_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    host: str
    port: int
    def __init__(self, host: str | None = ..., port: int | None = ...) -> None: ...

class LocalControllerConfig(_message.Message):
    __slots__ = ("port",)
    PORT_FIELD_NUMBER: _ClassVar[int]
    port: int
    def __init__(self, port: int | None = ...) -> None: ...

class ControllerVmConfig(_message.Message):
    __slots__ = ("bundle_prefix", "gcp", "heartbeat_failure_threshold", "image", "local", "manual", "worker_timeout")
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_PREFIX_FIELD_NUMBER: _ClassVar[int]
    WORKER_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_FAILURE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    LOCAL_FIELD_NUMBER: _ClassVar[int]
    image: str
    bundle_prefix: str
    worker_timeout: _time_pb2.Duration
    heartbeat_failure_threshold: int
    gcp: GcpControllerConfig
    manual: ManualControllerConfig
    local: LocalControllerConfig
    def __init__(self, image: str | None = ..., bundle_prefix: str | None = ..., worker_timeout: _time_pb2.Duration | _Mapping | None = ..., heartbeat_failure_threshold: int | None = ..., gcp: GcpControllerConfig | _Mapping | None = ..., manual: ManualControllerConfig | _Mapping | None = ..., local: LocalControllerConfig | _Mapping | None = ...) -> None: ...

class AutoscalerConfig(_message.Message):
    __slots__ = ("evaluation_interval", "heartbeat_grace_period", "requesting_timeout", "scale_down_delay", "scale_up_delay", "startup_grace_period")
    EVALUATION_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    REQUESTING_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    SCALE_UP_DELAY_FIELD_NUMBER: _ClassVar[int]
    SCALE_DOWN_DELAY_FIELD_NUMBER: _ClassVar[int]
    STARTUP_GRACE_PERIOD_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_GRACE_PERIOD_FIELD_NUMBER: _ClassVar[int]
    evaluation_interval: _time_pb2.Duration
    requesting_timeout: _time_pb2.Duration
    scale_up_delay: _time_pb2.Duration
    scale_down_delay: _time_pb2.Duration
    startup_grace_period: _time_pb2.Duration
    heartbeat_grace_period: _time_pb2.Duration
    def __init__(self, evaluation_interval: _time_pb2.Duration | _Mapping | None = ..., requesting_timeout: _time_pb2.Duration | _Mapping | None = ..., scale_up_delay: _time_pb2.Duration | _Mapping | None = ..., scale_down_delay: _time_pb2.Duration | _Mapping | None = ..., startup_grace_period: _time_pb2.Duration | _Mapping | None = ..., heartbeat_grace_period: _time_pb2.Duration | _Mapping | None = ...) -> None: ...

class DefaultsConfig(_message.Message):
    __slots__ = ("autoscaler", "bootstrap", "ssh", "timeouts")
    TIMEOUTS_FIELD_NUMBER: _ClassVar[int]
    SSH_FIELD_NUMBER: _ClassVar[int]
    AUTOSCALER_FIELD_NUMBER: _ClassVar[int]
    BOOTSTRAP_FIELD_NUMBER: _ClassVar[int]
    timeouts: TimeoutConfig
    ssh: SshConfig
    autoscaler: AutoscalerConfig
    bootstrap: BootstrapConfig
    def __init__(self, timeouts: TimeoutConfig | _Mapping | None = ..., ssh: SshConfig | _Mapping | None = ..., autoscaler: AutoscalerConfig | _Mapping | None = ..., bootstrap: BootstrapConfig | _Mapping | None = ...) -> None: ...

class IrisClusterConfig(_message.Message):
    __slots__ = ("controller", "defaults", "platform", "scale_groups")
    class ScaleGroupsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ScaleGroupConfig
        def __init__(self, key: str | None = ..., value: ScaleGroupConfig | _Mapping | None = ...) -> None: ...
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    DEFAULTS_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUPS_FIELD_NUMBER: _ClassVar[int]
    platform: PlatformConfig
    defaults: DefaultsConfig
    controller: ControllerVmConfig
    scale_groups: _containers.MessageMap[str, ScaleGroupConfig]
    def __init__(self, platform: PlatformConfig | _Mapping | None = ..., defaults: DefaultsConfig | _Mapping | None = ..., controller: ControllerVmConfig | _Mapping | None = ..., scale_groups: _Mapping[str, ScaleGroupConfig] | None = ...) -> None: ...
