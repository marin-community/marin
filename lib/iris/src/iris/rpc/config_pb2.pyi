from . import time_pb2 as _time_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

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
    __slots__ = ("project_id",)
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    def __init__(self, project_id: _Optional[str] = ...) -> None: ...

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
    def __init__(self, region: _Optional[str] = ...) -> None: ...

class PlatformConfig(_message.Message):
    __slots__ = ("label_prefix", "gcp", "manual", "local", "coreweave")
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
    def __init__(self, label_prefix: _Optional[str] = ..., gcp: _Optional[_Union[GcpPlatformConfig, _Mapping]] = ..., manual: _Optional[_Union[ManualPlatformConfig, _Mapping]] = ..., local: _Optional[_Union[LocalPlatformConfig, _Mapping]] = ..., coreweave: _Optional[_Union[CoreweavePlatformConfig, _Mapping]] = ...) -> None: ...

class ManualProvider(_message.Message):
    __slots__ = ("hosts", "ssh_user", "ssh_key_file")
    HOSTS_FIELD_NUMBER: _ClassVar[int]
    SSH_USER_FIELD_NUMBER: _ClassVar[int]
    SSH_KEY_FILE_FIELD_NUMBER: _ClassVar[int]
    hosts: _containers.RepeatedScalarFieldContainer[str]
    ssh_user: str
    ssh_key_file: str
    def __init__(self, hosts: _Optional[_Iterable[str]] = ..., ssh_user: _Optional[str] = ..., ssh_key_file: _Optional[str] = ...) -> None: ...

class GcpVmConfig(_message.Message):
    __slots__ = ("zone", "machine_type", "boot_disk_size_gb")
    ZONE_FIELD_NUMBER: _ClassVar[int]
    MACHINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    BOOT_DISK_SIZE_GB_FIELD_NUMBER: _ClassVar[int]
    zone: str
    machine_type: str
    boot_disk_size_gb: int
    def __init__(self, zone: _Optional[str] = ..., machine_type: _Optional[str] = ..., boot_disk_size_gb: _Optional[int] = ...) -> None: ...

class ManualVmConfig(_message.Message):
    __slots__ = ("host", "ssh_user", "ssh_key_file")
    HOST_FIELD_NUMBER: _ClassVar[int]
    SSH_USER_FIELD_NUMBER: _ClassVar[int]
    SSH_KEY_FILE_FIELD_NUMBER: _ClassVar[int]
    host: str
    ssh_user: str
    ssh_key_file: str
    def __init__(self, host: _Optional[str] = ..., ssh_user: _Optional[str] = ..., ssh_key_file: _Optional[str] = ...) -> None: ...

class VmConfig(_message.Message):
    __slots__ = ("name", "labels", "metadata", "gcp", "manual")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
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
    def __init__(self, name: _Optional[str] = ..., labels: _Optional[_Mapping[str, str]] = ..., metadata: _Optional[_Mapping[str, str]] = ..., gcp: _Optional[_Union[GcpVmConfig, _Mapping]] = ..., manual: _Optional[_Union[ManualVmConfig, _Mapping]] = ...) -> None: ...

class GcpSliceConfig(_message.Message):
    __slots__ = ("zone", "runtime_version", "topology", "zones")
    ZONE_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_VERSION_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_FIELD_NUMBER: _ClassVar[int]
    ZONES_FIELD_NUMBER: _ClassVar[int]
    zone: str
    runtime_version: str
    topology: str
    zones: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, zone: _Optional[str] = ..., runtime_version: _Optional[str] = ..., topology: _Optional[str] = ..., zones: _Optional[_Iterable[str]] = ...) -> None: ...

class CoreweaveSliceConfig(_message.Message):
    __slots__ = ("region", "instance_type")
    REGION_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    region: str
    instance_type: str
    def __init__(self, region: _Optional[str] = ..., instance_type: _Optional[str] = ...) -> None: ...

class ManualSliceConfig(_message.Message):
    __slots__ = ("hosts", "ssh_user", "ssh_key_file")
    HOSTS_FIELD_NUMBER: _ClassVar[int]
    SSH_USER_FIELD_NUMBER: _ClassVar[int]
    SSH_KEY_FILE_FIELD_NUMBER: _ClassVar[int]
    hosts: _containers.RepeatedScalarFieldContainer[str]
    ssh_user: str
    ssh_key_file: str
    def __init__(self, hosts: _Optional[_Iterable[str]] = ..., ssh_user: _Optional[str] = ..., ssh_key_file: _Optional[str] = ...) -> None: ...

class SliceConfig(_message.Message):
    __slots__ = ("name_prefix", "slice_size", "accelerator_type", "accelerator_variant", "labels", "preemptible", "gcp", "coreweave", "manual")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_PREFIX_FIELD_NUMBER: _ClassVar[int]
    SLICE_SIZE_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_VARIANT_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    PREEMPTIBLE_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    COREWEAVE_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    name_prefix: str
    slice_size: int
    accelerator_type: AcceleratorType
    accelerator_variant: str
    labels: _containers.ScalarMap[str, str]
    preemptible: bool
    gcp: GcpSliceConfig
    coreweave: CoreweaveSliceConfig
    manual: ManualSliceConfig
    def __init__(self, name_prefix: _Optional[str] = ..., slice_size: _Optional[int] = ..., accelerator_type: _Optional[_Union[AcceleratorType, str]] = ..., accelerator_variant: _Optional[str] = ..., labels: _Optional[_Mapping[str, str]] = ..., preemptible: _Optional[bool] = ..., gcp: _Optional[_Union[GcpSliceConfig, _Mapping]] = ..., coreweave: _Optional[_Union[CoreweaveSliceConfig, _Mapping]] = ..., manual: _Optional[_Union[ManualSliceConfig, _Mapping]] = ...) -> None: ...

class ScaleGroupResources(_message.Message):
    __slots__ = ("cpu", "memory_bytes", "disk_bytes", "gpu_count", "tpu_count")
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
    def __init__(self, cpu: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., gpu_count: _Optional[int] = ..., tpu_count: _Optional[int] = ...) -> None: ...

class ScaleGroupConfig(_message.Message):
    __slots__ = ("name", "min_slices", "max_slices", "accelerator_type", "accelerator_variant", "resources", "slice_size", "priority", "slice_template")
    NAME_FIELD_NUMBER: _ClassVar[int]
    MIN_SLICES_FIELD_NUMBER: _ClassVar[int]
    MAX_SLICES_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_VARIANT_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    SLICE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    SLICE_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    name: str
    min_slices: int
    max_slices: int
    accelerator_type: AcceleratorType
    accelerator_variant: str
    resources: ScaleGroupResources
    slice_size: int
    priority: int
    slice_template: SliceConfig
    def __init__(self, name: _Optional[str] = ..., min_slices: _Optional[int] = ..., max_slices: _Optional[int] = ..., accelerator_type: _Optional[_Union[AcceleratorType, str]] = ..., accelerator_variant: _Optional[str] = ..., resources: _Optional[_Union[ScaleGroupResources, _Mapping]] = ..., slice_size: _Optional[int] = ..., priority: _Optional[int] = ..., slice_template: _Optional[_Union[SliceConfig, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("boot_timeout", "init_timeout", "ssh_poll_interval")
    BOOT_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    INIT_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    SSH_POLL_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    boot_timeout: _time_pb2.Duration
    init_timeout: _time_pb2.Duration
    ssh_poll_interval: _time_pb2.Duration
    def __init__(self, boot_timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., init_timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., ssh_poll_interval: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ...) -> None: ...

class SshConfig(_message.Message):
    __slots__ = ("user", "key_file", "connect_timeout")
    USER_FIELD_NUMBER: _ClassVar[int]
    KEY_FILE_FIELD_NUMBER: _ClassVar[int]
    CONNECT_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    user: str
    key_file: str
    connect_timeout: _time_pb2.Duration
    def __init__(self, user: _Optional[str] = ..., key_file: _Optional[str] = ..., connect_timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ...) -> None: ...

class GcpControllerConfig(_message.Message):
    __slots__ = ("zone", "machine_type", "boot_disk_size_gb", "port")
    ZONE_FIELD_NUMBER: _ClassVar[int]
    MACHINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    BOOT_DISK_SIZE_GB_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    zone: str
    machine_type: str
    boot_disk_size_gb: int
    port: int
    def __init__(self, zone: _Optional[str] = ..., machine_type: _Optional[str] = ..., boot_disk_size_gb: _Optional[int] = ..., port: _Optional[int] = ...) -> None: ...

class ManualControllerConfig(_message.Message):
    __slots__ = ("host", "port")
    HOST_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    host: str
    port: int
    def __init__(self, host: _Optional[str] = ..., port: _Optional[int] = ...) -> None: ...

class LocalControllerConfig(_message.Message):
    __slots__ = ("port",)
    PORT_FIELD_NUMBER: _ClassVar[int]
    port: int
    def __init__(self, port: _Optional[int] = ...) -> None: ...

class ControllerVmConfig(_message.Message):
    __slots__ = ("image", "bundle_prefix", "worker_timeout", "gcp", "manual", "local")
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_PREFIX_FIELD_NUMBER: _ClassVar[int]
    WORKER_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    LOCAL_FIELD_NUMBER: _ClassVar[int]
    image: str
    bundle_prefix: str
    worker_timeout: _time_pb2.Duration
    gcp: GcpControllerConfig
    manual: ManualControllerConfig
    local: LocalControllerConfig
    def __init__(self, image: _Optional[str] = ..., bundle_prefix: _Optional[str] = ..., worker_timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., gcp: _Optional[_Union[GcpControllerConfig, _Mapping]] = ..., manual: _Optional[_Union[ManualControllerConfig, _Mapping]] = ..., local: _Optional[_Union[LocalControllerConfig, _Mapping]] = ...) -> None: ...

class AutoscalerConfig(_message.Message):
    __slots__ = ("evaluation_interval", "requesting_timeout", "scale_up_delay", "scale_down_delay", "startup_grace_period", "heartbeat_grace_period")
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
    def __init__(self, evaluation_interval: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., requesting_timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., scale_up_delay: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., scale_down_delay: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., startup_grace_period: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., heartbeat_grace_period: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ...) -> None: ...

class DefaultsConfig(_message.Message):
    __slots__ = ("timeouts", "ssh", "autoscaler", "bootstrap")
    TIMEOUTS_FIELD_NUMBER: _ClassVar[int]
    SSH_FIELD_NUMBER: _ClassVar[int]
    AUTOSCALER_FIELD_NUMBER: _ClassVar[int]
    BOOTSTRAP_FIELD_NUMBER: _ClassVar[int]
    timeouts: TimeoutConfig
    ssh: SshConfig
    autoscaler: AutoscalerConfig
    bootstrap: BootstrapConfig
    def __init__(self, timeouts: _Optional[_Union[TimeoutConfig, _Mapping]] = ..., ssh: _Optional[_Union[SshConfig, _Mapping]] = ..., autoscaler: _Optional[_Union[AutoscalerConfig, _Mapping]] = ..., bootstrap: _Optional[_Union[BootstrapConfig, _Mapping]] = ...) -> None: ...

class IrisClusterConfig(_message.Message):
    __slots__ = ("platform", "defaults", "controller", "scale_groups")
    class ScaleGroupsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ScaleGroupConfig
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ScaleGroupConfig, _Mapping]] = ...) -> None: ...
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    DEFAULTS_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUPS_FIELD_NUMBER: _ClassVar[int]
    platform: PlatformConfig
    defaults: DefaultsConfig
    controller: ControllerVmConfig
    scale_groups: _containers.MessageMap[str, ScaleGroupConfig]
    def __init__(self, platform: _Optional[_Union[PlatformConfig, _Mapping]] = ..., defaults: _Optional[_Union[DefaultsConfig, _Mapping]] = ..., controller: _Optional[_Union[ControllerVmConfig, _Mapping]] = ..., scale_groups: _Optional[_Mapping[str, ScaleGroupConfig]] = ...) -> None: ...
