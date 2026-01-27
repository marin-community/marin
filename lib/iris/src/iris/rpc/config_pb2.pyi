from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TpuProvider(_message.Message):
    __slots__ = ("project_id",)
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    def __init__(self, project_id: _Optional[str] = ...) -> None: ...

class ManualProvider(_message.Message):
    __slots__ = ("hosts", "ssh_user", "ssh_key_file", "ssh_port")
    HOSTS_FIELD_NUMBER: _ClassVar[int]
    SSH_USER_FIELD_NUMBER: _ClassVar[int]
    SSH_KEY_FILE_FIELD_NUMBER: _ClassVar[int]
    SSH_PORT_FIELD_NUMBER: _ClassVar[int]
    hosts: _containers.RepeatedScalarFieldContainer[str]
    ssh_user: str
    ssh_key_file: str
    ssh_port: int
    def __init__(self, hosts: _Optional[_Iterable[str]] = ..., ssh_user: _Optional[str] = ..., ssh_key_file: _Optional[str] = ..., ssh_port: _Optional[int] = ...) -> None: ...

class ProviderConfig(_message.Message):
    __slots__ = ("tpu", "manual")
    TPU_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    tpu: TpuProvider
    manual: ManualProvider
    def __init__(self, tpu: _Optional[_Union[TpuProvider, _Mapping]] = ..., manual: _Optional[_Union[ManualProvider, _Mapping]] = ...) -> None: ...

class ScaleGroupConfig(_message.Message):
    __slots__ = ("name", "min_slices", "max_slices", "accelerator_type", "runtime_version", "preemptible", "zones", "priority", "provider")
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
    def __init__(self, name: _Optional[str] = ..., min_slices: _Optional[int] = ..., max_slices: _Optional[int] = ..., accelerator_type: _Optional[str] = ..., runtime_version: _Optional[str] = ..., preemptible: _Optional[bool] = ..., zones: _Optional[_Iterable[str]] = ..., priority: _Optional[int] = ..., provider: _Optional[_Union[ProviderConfig, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("boot_timeout_seconds", "init_timeout_seconds", "ssh_poll_interval_seconds")
    BOOT_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    INIT_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SSH_POLL_INTERVAL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    boot_timeout_seconds: int
    init_timeout_seconds: int
    ssh_poll_interval_seconds: int
    def __init__(self, boot_timeout_seconds: _Optional[int] = ..., init_timeout_seconds: _Optional[int] = ..., ssh_poll_interval_seconds: _Optional[int] = ...) -> None: ...

class SshConfig(_message.Message):
    __slots__ = ("user", "key_file", "port", "connect_timeout")
    USER_FIELD_NUMBER: _ClassVar[int]
    KEY_FILE_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    CONNECT_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    user: str
    key_file: str
    port: int
    connect_timeout: int
    def __init__(self, user: _Optional[str] = ..., key_file: _Optional[str] = ..., port: _Optional[int] = ..., connect_timeout: _Optional[int] = ...) -> None: ...

class GcpControllerConfig(_message.Message):
    __slots__ = ("image", "machine_type", "boot_disk_size_gb", "port")
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    MACHINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    BOOT_DISK_SIZE_GB_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    image: str
    machine_type: str
    boot_disk_size_gb: int
    port: int
    def __init__(self, image: _Optional[str] = ..., machine_type: _Optional[str] = ..., boot_disk_size_gb: _Optional[int] = ..., port: _Optional[int] = ...) -> None: ...

class ManualControllerConfig(_message.Message):
    __slots__ = ("host", "image", "port")
    HOST_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    host: str
    image: str
    port: int
    def __init__(self, host: _Optional[str] = ..., image: _Optional[str] = ..., port: _Optional[int] = ...) -> None: ...

class ControllerVmConfig(_message.Message):
    __slots__ = ("gcp", "manual")
    GCP_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    gcp: GcpControllerConfig
    manual: ManualControllerConfig
    def __init__(self, gcp: _Optional[_Union[GcpControllerConfig, _Mapping]] = ..., manual: _Optional[_Union[ManualControllerConfig, _Mapping]] = ...) -> None: ...

class IrisClusterConfig(_message.Message):
    __slots__ = ("provider_type", "project_id", "region", "zone", "controller_vm", "scale_groups", "label_prefix", "bootstrap", "timeouts", "ssh")
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
    CONTROLLER_VM_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUPS_FIELD_NUMBER: _ClassVar[int]
    LABEL_PREFIX_FIELD_NUMBER: _ClassVar[int]
    BOOTSTRAP_FIELD_NUMBER: _ClassVar[int]
    TIMEOUTS_FIELD_NUMBER: _ClassVar[int]
    SSH_FIELD_NUMBER: _ClassVar[int]
    provider_type: str
    project_id: str
    region: str
    zone: str
    controller_vm: ControllerVmConfig
    scale_groups: _containers.MessageMap[str, ScaleGroupConfig]
    label_prefix: str
    bootstrap: BootstrapConfig
    timeouts: TimeoutConfig
    ssh: SshConfig
    def __init__(self, provider_type: _Optional[str] = ..., project_id: _Optional[str] = ..., region: _Optional[str] = ..., zone: _Optional[str] = ..., controller_vm: _Optional[_Union[ControllerVmConfig, _Mapping]] = ..., scale_groups: _Optional[_Mapping[str, ScaleGroupConfig]] = ..., label_prefix: _Optional[str] = ..., bootstrap: _Optional[_Union[BootstrapConfig, _Mapping]] = ..., timeouts: _Optional[_Union[TimeoutConfig, _Mapping]] = ..., ssh: _Optional[_Union[SshConfig, _Mapping]] = ...) -> None: ...
