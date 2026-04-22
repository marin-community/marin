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

class CapacityType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CAPACITY_TYPE_UNSPECIFIED: _ClassVar[CapacityType]
    CAPACITY_TYPE_PREEMPTIBLE: _ClassVar[CapacityType]
    CAPACITY_TYPE_ON_DEMAND: _ClassVar[CapacityType]
    CAPACITY_TYPE_RESERVED: _ClassVar[CapacityType]
ACCELERATOR_TYPE_UNSPECIFIED: AcceleratorType
ACCELERATOR_TYPE_CPU: AcceleratorType
ACCELERATOR_TYPE_GPU: AcceleratorType
ACCELERATOR_TYPE_TPU: AcceleratorType
CAPACITY_TYPE_UNSPECIFIED: CapacityType
CAPACITY_TYPE_PREEMPTIBLE: CapacityType
CAPACITY_TYPE_ON_DEMAND: CapacityType
CAPACITY_TYPE_RESERVED: CapacityType

class GcpPlatformConfig(_message.Message):
    __slots__ = ("project_id", "zones")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    ZONES_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    zones: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, project_id: _Optional[str] = ..., zones: _Optional[_Iterable[str]] = ...) -> None: ...

class ManualPlatformConfig(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class LocalPlatformConfig(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CoreweavePlatformConfig(_message.Message):
    __slots__ = ("region", "namespace", "kubeconfig_path", "object_storage_endpoint")
    REGION_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    KUBECONFIG_PATH_FIELD_NUMBER: _ClassVar[int]
    OBJECT_STORAGE_ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    region: str
    namespace: str
    kubeconfig_path: str
    object_storage_endpoint: str
    def __init__(self, region: _Optional[str] = ..., namespace: _Optional[str] = ..., kubeconfig_path: _Optional[str] = ..., object_storage_endpoint: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("zone", "machine_type", "boot_disk_size_gb", "service_account")
    ZONE_FIELD_NUMBER: _ClassVar[int]
    MACHINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    BOOT_DISK_SIZE_GB_FIELD_NUMBER: _ClassVar[int]
    SERVICE_ACCOUNT_FIELD_NUMBER: _ClassVar[int]
    zone: str
    machine_type: str
    boot_disk_size_gb: int
    service_account: str
    def __init__(self, zone: _Optional[str] = ..., machine_type: _Optional[str] = ..., boot_disk_size_gb: _Optional[int] = ..., service_account: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("mode", "zone", "runtime_version", "topology", "machine_type", "service_account")
    class GcpSliceMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        GCP_SLICE_MODE_TPU: _ClassVar[GcpSliceConfig.GcpSliceMode]
        GCP_SLICE_MODE_VM: _ClassVar[GcpSliceConfig.GcpSliceMode]
    GCP_SLICE_MODE_TPU: GcpSliceConfig.GcpSliceMode
    GCP_SLICE_MODE_VM: GcpSliceConfig.GcpSliceMode
    MODE_FIELD_NUMBER: _ClassVar[int]
    ZONE_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_VERSION_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_FIELD_NUMBER: _ClassVar[int]
    MACHINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SERVICE_ACCOUNT_FIELD_NUMBER: _ClassVar[int]
    mode: GcpSliceConfig.GcpSliceMode
    zone: str
    runtime_version: str
    topology: str
    machine_type: str
    service_account: str
    def __init__(self, mode: _Optional[_Union[GcpSliceConfig.GcpSliceMode, str]] = ..., zone: _Optional[str] = ..., runtime_version: _Optional[str] = ..., topology: _Optional[str] = ..., machine_type: _Optional[str] = ..., service_account: _Optional[str] = ...) -> None: ...

class CoreweaveSliceConfig(_message.Message):
    __slots__ = ("region", "instance_type", "gpu_class", "infiniband")
    REGION_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    GPU_CLASS_FIELD_NUMBER: _ClassVar[int]
    INFINIBAND_FIELD_NUMBER: _ClassVar[int]
    region: str
    instance_type: str
    gpu_class: str
    infiniband: bool
    def __init__(self, region: _Optional[str] = ..., instance_type: _Optional[str] = ..., gpu_class: _Optional[str] = ..., infiniband: _Optional[bool] = ...) -> None: ...

class ManualSliceConfig(_message.Message):
    __slots__ = ("hosts", "ssh_user", "ssh_key_file")
    HOSTS_FIELD_NUMBER: _ClassVar[int]
    SSH_USER_FIELD_NUMBER: _ClassVar[int]
    SSH_KEY_FILE_FIELD_NUMBER: _ClassVar[int]
    hosts: _containers.RepeatedScalarFieldContainer[str]
    ssh_user: str
    ssh_key_file: str
    def __init__(self, hosts: _Optional[_Iterable[str]] = ..., ssh_user: _Optional[str] = ..., ssh_key_file: _Optional[str] = ...) -> None: ...

class LocalSliceConfig(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SliceConfig(_message.Message):
    __slots__ = ("name_prefix", "num_vms", "accelerator_type", "accelerator_variant", "labels", "gpu_count", "disk_size_gb", "capacity_type", "gcp", "coreweave", "manual", "local")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_PREFIX_FIELD_NUMBER: _ClassVar[int]
    NUM_VMS_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_VARIANT_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    DISK_SIZE_GB_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    COREWEAVE_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    LOCAL_FIELD_NUMBER: _ClassVar[int]
    name_prefix: str
    num_vms: int
    accelerator_type: AcceleratorType
    accelerator_variant: str
    labels: _containers.ScalarMap[str, str]
    gpu_count: int
    disk_size_gb: int
    capacity_type: CapacityType
    gcp: GcpSliceConfig
    coreweave: CoreweaveSliceConfig
    manual: ManualSliceConfig
    local: LocalSliceConfig
    def __init__(self, name_prefix: _Optional[str] = ..., num_vms: _Optional[int] = ..., accelerator_type: _Optional[_Union[AcceleratorType, str]] = ..., accelerator_variant: _Optional[str] = ..., labels: _Optional[_Mapping[str, str]] = ..., gpu_count: _Optional[int] = ..., disk_size_gb: _Optional[int] = ..., capacity_type: _Optional[_Union[CapacityType, str]] = ..., gcp: _Optional[_Union[GcpSliceConfig, _Mapping]] = ..., coreweave: _Optional[_Union[CoreweaveSliceConfig, _Mapping]] = ..., manual: _Optional[_Union[ManualSliceConfig, _Mapping]] = ..., local: _Optional[_Union[LocalSliceConfig, _Mapping]] = ...) -> None: ...

class ScaleGroupResources(_message.Message):
    __slots__ = ("cpu_millicores", "memory_bytes", "disk_bytes", "device_type", "device_variant", "device_count", "capacity_type")
    CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_VARIANT_FIELD_NUMBER: _ClassVar[int]
    DEVICE_COUNT_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int
    device_type: AcceleratorType
    device_variant: str
    device_count: int
    capacity_type: CapacityType
    def __init__(self, cpu_millicores: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., device_type: _Optional[_Union[AcceleratorType, str]] = ..., device_variant: _Optional[str] = ..., device_count: _Optional[int] = ..., capacity_type: _Optional[_Union[CapacityType, str]] = ...) -> None: ...

class WorkerSettings(_message.Message):
    __slots__ = ("attributes",)
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    attributes: _containers.ScalarMap[str, str]
    def __init__(self, attributes: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ScaleGroupConfig(_message.Message):
    __slots__ = ("name", "buffer_slices", "max_slices", "resources", "num_vms", "priority", "scale_up_rate_limit", "scale_down_rate_limit", "slice_template", "worker", "quota_pool", "allocation_tier")
    NAME_FIELD_NUMBER: _ClassVar[int]
    BUFFER_SLICES_FIELD_NUMBER: _ClassVar[int]
    MAX_SLICES_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    NUM_VMS_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    SCALE_UP_RATE_LIMIT_FIELD_NUMBER: _ClassVar[int]
    SCALE_DOWN_RATE_LIMIT_FIELD_NUMBER: _ClassVar[int]
    SLICE_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    WORKER_FIELD_NUMBER: _ClassVar[int]
    QUOTA_POOL_FIELD_NUMBER: _ClassVar[int]
    ALLOCATION_TIER_FIELD_NUMBER: _ClassVar[int]
    name: str
    buffer_slices: int
    max_slices: int
    resources: ScaleGroupResources
    num_vms: int
    priority: int
    scale_up_rate_limit: int
    scale_down_rate_limit: int
    slice_template: SliceConfig
    worker: WorkerSettings
    quota_pool: str
    allocation_tier: int
    def __init__(self, name: _Optional[str] = ..., buffer_slices: _Optional[int] = ..., max_slices: _Optional[int] = ..., resources: _Optional[_Union[ScaleGroupResources, _Mapping]] = ..., num_vms: _Optional[int] = ..., priority: _Optional[int] = ..., scale_up_rate_limit: _Optional[int] = ..., scale_down_rate_limit: _Optional[int] = ..., slice_template: _Optional[_Union[SliceConfig, _Mapping]] = ..., worker: _Optional[_Union[WorkerSettings, _Mapping]] = ..., quota_pool: _Optional[str] = ..., allocation_tier: _Optional[int] = ...) -> None: ...

class WorkerConfig(_message.Message):
    __slots__ = ("docker_image", "host", "port", "port_range", "worker_id", "controller_address", "cache_dir", "default_task_image", "task_env", "runtime", "accelerator_type", "accelerator_variant", "gpu_count", "capacity_type", "worker_attributes", "poll_interval", "heartbeat_timeout", "slice_id", "platform", "storage_prefix", "auth_token")
    class TaskEnvEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class WorkerAttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DOCKER_IMAGE_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    PORT_RANGE_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    CACHE_DIR_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_TASK_IMAGE_FIELD_NUMBER: _ClassVar[int]
    TASK_ENV_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_VARIANT_FIELD_NUMBER: _ClassVar[int]
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    WORKER_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    POLL_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    STORAGE_PREFIX_FIELD_NUMBER: _ClassVar[int]
    AUTH_TOKEN_FIELD_NUMBER: _ClassVar[int]
    docker_image: str
    host: str
    port: int
    port_range: str
    worker_id: str
    controller_address: str
    cache_dir: str
    default_task_image: str
    task_env: _containers.ScalarMap[str, str]
    runtime: str
    accelerator_type: AcceleratorType
    accelerator_variant: str
    gpu_count: int
    capacity_type: CapacityType
    worker_attributes: _containers.ScalarMap[str, str]
    poll_interval: _time_pb2.Duration
    heartbeat_timeout: _time_pb2.Duration
    slice_id: str
    platform: PlatformConfig
    storage_prefix: str
    auth_token: str
    def __init__(self, docker_image: _Optional[str] = ..., host: _Optional[str] = ..., port: _Optional[int] = ..., port_range: _Optional[str] = ..., worker_id: _Optional[str] = ..., controller_address: _Optional[str] = ..., cache_dir: _Optional[str] = ..., default_task_image: _Optional[str] = ..., task_env: _Optional[_Mapping[str, str]] = ..., runtime: _Optional[str] = ..., accelerator_type: _Optional[_Union[AcceleratorType, str]] = ..., accelerator_variant: _Optional[str] = ..., gpu_count: _Optional[int] = ..., capacity_type: _Optional[_Union[CapacityType, str]] = ..., worker_attributes: _Optional[_Mapping[str, str]] = ..., poll_interval: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., heartbeat_timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., slice_id: _Optional[str] = ..., platform: _Optional[_Union[PlatformConfig, _Mapping]] = ..., storage_prefix: _Optional[str] = ..., auth_token: _Optional[str] = ...) -> None: ...

class SshConfig(_message.Message):
    __slots__ = ("user", "key_file", "port", "connect_timeout", "auth_mode", "os_login_user", "impersonate_service_account")
    class AuthMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SSH_AUTH_MODE_METADATA: _ClassVar[SshConfig.AuthMode]
        SSH_AUTH_MODE_OS_LOGIN: _ClassVar[SshConfig.AuthMode]
    SSH_AUTH_MODE_METADATA: SshConfig.AuthMode
    SSH_AUTH_MODE_OS_LOGIN: SshConfig.AuthMode
    USER_FIELD_NUMBER: _ClassVar[int]
    KEY_FILE_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    CONNECT_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    AUTH_MODE_FIELD_NUMBER: _ClassVar[int]
    OS_LOGIN_USER_FIELD_NUMBER: _ClassVar[int]
    IMPERSONATE_SERVICE_ACCOUNT_FIELD_NUMBER: _ClassVar[int]
    user: str
    key_file: str
    port: int
    connect_timeout: _time_pb2.Duration
    auth_mode: SshConfig.AuthMode
    os_login_user: str
    impersonate_service_account: str
    def __init__(self, user: _Optional[str] = ..., key_file: _Optional[str] = ..., port: _Optional[int] = ..., connect_timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., auth_mode: _Optional[_Union[SshConfig.AuthMode, str]] = ..., os_login_user: _Optional[str] = ..., impersonate_service_account: _Optional[str] = ...) -> None: ...

class StorageConfig(_message.Message):
    __slots__ = ("local_state_dir", "remote_state_dir")
    LOCAL_STATE_DIR_FIELD_NUMBER: _ClassVar[int]
    REMOTE_STATE_DIR_FIELD_NUMBER: _ClassVar[int]
    local_state_dir: str
    remote_state_dir: str
    def __init__(self, local_state_dir: _Optional[str] = ..., remote_state_dir: _Optional[str] = ...) -> None: ...

class GcpControllerConfig(_message.Message):
    __slots__ = ("zone", "machine_type", "boot_disk_size_gb", "port", "service_account")
    ZONE_FIELD_NUMBER: _ClassVar[int]
    MACHINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    BOOT_DISK_SIZE_GB_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    SERVICE_ACCOUNT_FIELD_NUMBER: _ClassVar[int]
    zone: str
    machine_type: str
    boot_disk_size_gb: int
    port: int
    service_account: str
    def __init__(self, zone: _Optional[str] = ..., machine_type: _Optional[str] = ..., boot_disk_size_gb: _Optional[int] = ..., port: _Optional[int] = ..., service_account: _Optional[str] = ...) -> None: ...

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

class CoreweaveControllerConfig(_message.Message):
    __slots__ = ("port", "service_name", "scale_group")
    PORT_FIELD_NUMBER: _ClassVar[int]
    SERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
    port: int
    service_name: str
    scale_group: str
    def __init__(self, port: _Optional[int] = ..., service_name: _Optional[str] = ..., scale_group: _Optional[str] = ...) -> None: ...

class ControllerVmConfig(_message.Message):
    __slots__ = ("image", "worker_timeout", "use_split_heartbeat", "gcp", "manual", "local", "coreweave")
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    WORKER_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    USE_SPLIT_HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    LOCAL_FIELD_NUMBER: _ClassVar[int]
    COREWEAVE_FIELD_NUMBER: _ClassVar[int]
    image: str
    worker_timeout: _time_pb2.Duration
    use_split_heartbeat: bool
    gcp: GcpControllerConfig
    manual: ManualControllerConfig
    local: LocalControllerConfig
    coreweave: CoreweaveControllerConfig
    def __init__(self, image: _Optional[str] = ..., worker_timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., use_split_heartbeat: _Optional[bool] = ..., gcp: _Optional[_Union[GcpControllerConfig, _Mapping]] = ..., manual: _Optional[_Union[ManualControllerConfig, _Mapping]] = ..., local: _Optional[_Union[LocalControllerConfig, _Mapping]] = ..., coreweave: _Optional[_Union[CoreweaveControllerConfig, _Mapping]] = ...) -> None: ...

class AutoscalerConfig(_message.Message):
    __slots__ = ("evaluation_interval", "scale_up_delay", "scale_down_delay", "startup_grace_period", "heartbeat_grace_period")
    EVALUATION_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    SCALE_UP_DELAY_FIELD_NUMBER: _ClassVar[int]
    SCALE_DOWN_DELAY_FIELD_NUMBER: _ClassVar[int]
    STARTUP_GRACE_PERIOD_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_GRACE_PERIOD_FIELD_NUMBER: _ClassVar[int]
    evaluation_interval: _time_pb2.Duration
    scale_up_delay: _time_pb2.Duration
    scale_down_delay: _time_pb2.Duration
    startup_grace_period: _time_pb2.Duration
    heartbeat_grace_period: _time_pb2.Duration
    def __init__(self, evaluation_interval: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., scale_up_delay: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., scale_down_delay: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., startup_grace_period: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., heartbeat_grace_period: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ...) -> None: ...

class DefaultsConfig(_message.Message):
    __slots__ = ("ssh", "autoscaler", "worker", "task_env")
    class TaskEnvEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SSH_FIELD_NUMBER: _ClassVar[int]
    AUTOSCALER_FIELD_NUMBER: _ClassVar[int]
    WORKER_FIELD_NUMBER: _ClassVar[int]
    TASK_ENV_FIELD_NUMBER: _ClassVar[int]
    ssh: SshConfig
    autoscaler: AutoscalerConfig
    worker: WorkerConfig
    task_env: _containers.ScalarMap[str, str]
    def __init__(self, ssh: _Optional[_Union[SshConfig, _Mapping]] = ..., autoscaler: _Optional[_Union[AutoscalerConfig, _Mapping]] = ..., worker: _Optional[_Union[WorkerConfig, _Mapping]] = ..., task_env: _Optional[_Mapping[str, str]] = ...) -> None: ...

class GcpAuthConfig(_message.Message):
    __slots__ = ("project_id",)
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    def __init__(self, project_id: _Optional[str] = ...) -> None: ...

class StaticAuthConfig(_message.Message):
    __slots__ = ("tokens",)
    class TokensEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    tokens: _containers.ScalarMap[str, str]
    def __init__(self, tokens: _Optional[_Mapping[str, str]] = ...) -> None: ...

class AuthConfig(_message.Message):
    __slots__ = ("gcp", "static", "admin_users", "optional")
    GCP_FIELD_NUMBER: _ClassVar[int]
    STATIC_FIELD_NUMBER: _ClassVar[int]
    ADMIN_USERS_FIELD_NUMBER: _ClassVar[int]
    OPTIONAL_FIELD_NUMBER: _ClassVar[int]
    gcp: GcpAuthConfig
    static: StaticAuthConfig
    admin_users: _containers.RepeatedScalarFieldContainer[str]
    optional: bool
    def __init__(self, gcp: _Optional[_Union[GcpAuthConfig, _Mapping]] = ..., static: _Optional[_Union[StaticAuthConfig, _Mapping]] = ..., admin_users: _Optional[_Iterable[str]] = ..., optional: _Optional[bool] = ...) -> None: ...

class WorkerProviderConfig(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class KubernetesProviderConfig(_message.Message):
    __slots__ = ("namespace", "kubeconfig", "default_image", "colocation_topology_key", "service_account", "host_network", "cache_dir", "controller_address")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    KUBECONFIG_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_IMAGE_FIELD_NUMBER: _ClassVar[int]
    COLOCATION_TOPOLOGY_KEY_FIELD_NUMBER: _ClassVar[int]
    SERVICE_ACCOUNT_FIELD_NUMBER: _ClassVar[int]
    HOST_NETWORK_FIELD_NUMBER: _ClassVar[int]
    CACHE_DIR_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    kubeconfig: str
    default_image: str
    colocation_topology_key: str
    service_account: str
    host_network: bool
    cache_dir: str
    controller_address: str
    def __init__(self, namespace: _Optional[str] = ..., kubeconfig: _Optional[str] = ..., default_image: _Optional[str] = ..., colocation_topology_key: _Optional[str] = ..., service_account: _Optional[str] = ..., host_network: _Optional[bool] = ..., cache_dir: _Optional[str] = ..., controller_address: _Optional[str] = ...) -> None: ...

class IrisClusterConfig(_message.Message):
    __slots__ = ("name", "platform", "defaults", "storage", "controller", "scale_groups", "auth", "kubernetes_provider", "worker_provider")
    class ScaleGroupsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ScaleGroupConfig
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ScaleGroupConfig, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    DEFAULTS_FIELD_NUMBER: _ClassVar[int]
    STORAGE_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUPS_FIELD_NUMBER: _ClassVar[int]
    AUTH_FIELD_NUMBER: _ClassVar[int]
    KUBERNETES_PROVIDER_FIELD_NUMBER: _ClassVar[int]
    WORKER_PROVIDER_FIELD_NUMBER: _ClassVar[int]
    name: str
    platform: PlatformConfig
    defaults: DefaultsConfig
    storage: StorageConfig
    controller: ControllerVmConfig
    scale_groups: _containers.MessageMap[str, ScaleGroupConfig]
    auth: AuthConfig
    kubernetes_provider: KubernetesProviderConfig
    worker_provider: WorkerProviderConfig
    def __init__(self, name: _Optional[str] = ..., platform: _Optional[_Union[PlatformConfig, _Mapping]] = ..., defaults: _Optional[_Union[DefaultsConfig, _Mapping]] = ..., storage: _Optional[_Union[StorageConfig, _Mapping]] = ..., controller: _Optional[_Union[ControllerVmConfig, _Mapping]] = ..., scale_groups: _Optional[_Mapping[str, ScaleGroupConfig]] = ..., auth: _Optional[_Union[AuthConfig, _Mapping]] = ..., kubernetes_provider: _Optional[_Union[KubernetesProviderConfig, _Mapping]] = ..., worker_provider: _Optional[_Union[WorkerProviderConfig, _Mapping]] = ...) -> None: ...
