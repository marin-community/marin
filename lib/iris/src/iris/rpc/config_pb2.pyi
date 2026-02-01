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

class LocalProvider(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ProviderConfig(_message.Message):
    __slots__ = ("local", "manual", "tpu")
    TPU_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    LOCAL_FIELD_NUMBER: _ClassVar[int]
    tpu: TpuProvider
    manual: ManualProvider
    local: LocalProvider
    def __init__(
        self,
        tpu: TpuProvider | _Mapping | None = ...,
        manual: ManualProvider | _Mapping | None = ...,
        local: LocalProvider | _Mapping | None = ...,
    ) -> None: ...

class ScaleGroupConfig(_message.Message):
    __slots__ = (
        "accelerator_type",
        "accelerator_variant",
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
    ACCELERATOR_VARIANT_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_VERSION_FIELD_NUMBER: _ClassVar[int]
    PREEMPTIBLE_FIELD_NUMBER: _ClassVar[int]
    ZONES_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    name: str
    min_slices: int
    max_slices: int
    accelerator_type: AcceleratorType
    accelerator_variant: str
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
        accelerator_type: AcceleratorType | str | None = ...,
        accelerator_variant: str | None = ...,
        runtime_version: str | None = ...,
        preemptible: bool | None = ...,
        zones: _Iterable[str] | None = ...,
        priority: int | None = ...,
        provider: ProviderConfig | _Mapping | None = ...,
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
    __slots__ = ("boot_timeout", "init_timeout", "ssh_poll_interval")
    BOOT_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    INIT_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    SSH_POLL_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    boot_timeout: _time_pb2.Duration
    init_timeout: _time_pb2.Duration
    ssh_poll_interval: _time_pb2.Duration
    def __init__(
        self,
        boot_timeout: _time_pb2.Duration | _Mapping | None = ...,
        init_timeout: _time_pb2.Duration | _Mapping | None = ...,
        ssh_poll_interval: _time_pb2.Duration | _Mapping | None = ...,
    ) -> None: ...

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
    def __init__(
        self,
        user: str | None = ...,
        key_file: str | None = ...,
        port: int | None = ...,
        connect_timeout: _time_pb2.Duration | _Mapping | None = ...,
    ) -> None: ...

class GcpControllerConfig(_message.Message):
    __slots__ = ("boot_disk_size_gb", "machine_type", "port")
    MACHINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    BOOT_DISK_SIZE_GB_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    machine_type: str
    boot_disk_size_gb: int
    port: int
    def __init__(
        self, machine_type: str | None = ..., boot_disk_size_gb: int | None = ..., port: int | None = ...
    ) -> None: ...

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
    __slots__ = ("bundle_prefix", "gcp", "image", "local", "manual")
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_PREFIX_FIELD_NUMBER: _ClassVar[int]
    GCP_FIELD_NUMBER: _ClassVar[int]
    MANUAL_FIELD_NUMBER: _ClassVar[int]
    LOCAL_FIELD_NUMBER: _ClassVar[int]
    image: str
    bundle_prefix: str
    gcp: GcpControllerConfig
    manual: ManualControllerConfig
    local: LocalControllerConfig
    def __init__(
        self,
        image: str | None = ...,
        bundle_prefix: str | None = ...,
        gcp: GcpControllerConfig | _Mapping | None = ...,
        manual: ManualControllerConfig | _Mapping | None = ...,
        local: LocalControllerConfig | _Mapping | None = ...,
    ) -> None: ...

class AutoscalerConfig(_message.Message):
    __slots__ = ("evaluation_interval", "requesting_timeout")
    EVALUATION_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    REQUESTING_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    evaluation_interval: _time_pb2.Duration
    requesting_timeout: _time_pb2.Duration
    def __init__(
        self,
        evaluation_interval: _time_pb2.Duration | _Mapping | None = ...,
        requesting_timeout: _time_pb2.Duration | _Mapping | None = ...,
    ) -> None: ...

class IrisClusterConfig(_message.Message):
    __slots__ = (
        "autoscaler",
        "bootstrap",
        "controller_vm",
        "label_prefix",
        "project_id",
        "provider_type",
        "region",
        "scale_groups",
        "ssh",
        "timeouts",
        "zone",
    )
    class ScaleGroupsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ScaleGroupConfig
        def __init__(
            self, key: str | None = ..., value: ScaleGroupConfig | _Mapping | None = ...
        ) -> None: ...

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
    AUTOSCALER_FIELD_NUMBER: _ClassVar[int]
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
    autoscaler: AutoscalerConfig
    def __init__(
        self,
        provider_type: str | None = ...,
        project_id: str | None = ...,
        region: str | None = ...,
        zone: str | None = ...,
        controller_vm: ControllerVmConfig | _Mapping | None = ...,
        scale_groups: _Mapping[str, ScaleGroupConfig] | None = ...,
        label_prefix: str | None = ...,
        bootstrap: BootstrapConfig | _Mapping | None = ...,
        timeouts: TimeoutConfig | _Mapping | None = ...,
        ssh: SshConfig | _Mapping | None = ...,
        autoscaler: AutoscalerConfig | _Mapping | None = ...,
    ) -> None: ...
