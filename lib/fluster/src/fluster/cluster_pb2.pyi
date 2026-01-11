from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class JobState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_STATE_UNSPECIFIED: _ClassVar[JobState]
    JOB_STATE_PENDING: _ClassVar[JobState]
    JOB_STATE_BUILDING: _ClassVar[JobState]
    JOB_STATE_RUNNING: _ClassVar[JobState]
    JOB_STATE_SUCCEEDED: _ClassVar[JobState]
    JOB_STATE_FAILED: _ClassVar[JobState]
    JOB_STATE_KILLED: _ClassVar[JobState]
JOB_STATE_UNSPECIFIED: JobState
JOB_STATE_PENDING: JobState
JOB_STATE_BUILDING: JobState
JOB_STATE_RUNNING: JobState
JOB_STATE_SUCCEEDED: JobState
JOB_STATE_FAILED: JobState
JOB_STATE_KILLED: JobState

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DeviceConfig(_message.Message):
    __slots__ = ()
    CPU_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    TPU_FIELD_NUMBER: _ClassVar[int]
    cpu: CpuDevice
    gpu: GpuDevice
    tpu: TpuDevice
    def __init__(self, cpu: _Optional[_Union[CpuDevice, _Mapping]] = ..., gpu: _Optional[_Union[GpuDevice, _Mapping]] = ..., tpu: _Optional[_Union[TpuDevice, _Mapping]] = ...) -> None: ...

class CpuDevice(_message.Message):
    __slots__ = ()
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    variant: str
    def __init__(self, variant: _Optional[str] = ...) -> None: ...

class GpuDevice(_message.Message):
    __slots__ = ()
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    variant: str
    count: int
    def __init__(self, variant: _Optional[str] = ..., count: _Optional[int] = ...) -> None: ...

class TpuDevice(_message.Message):
    __slots__ = ()
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_FIELD_NUMBER: _ClassVar[int]
    variant: str
    topology: str
    def __init__(self, variant: _Optional[str] = ..., topology: _Optional[str] = ...) -> None: ...

class ResourceSpec(_message.Message):
    __slots__ = ()
    CPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    DISK_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    REPLICAS_FIELD_NUMBER: _ClassVar[int]
    PREEMPTIBLE_FIELD_NUMBER: _ClassVar[int]
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    cpu: int
    memory: str
    disk: str
    device: DeviceConfig
    replicas: int
    preemptible: bool
    regions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, cpu: _Optional[int] = ..., memory: _Optional[str] = ..., disk: _Optional[str] = ..., device: _Optional[_Union[DeviceConfig, _Mapping]] = ..., replicas: _Optional[int] = ..., preemptible: _Optional[bool] = ..., regions: _Optional[_Iterable[str]] = ...) -> None: ...

class EnvironmentConfig(_message.Message):
    __slots__ = ()
    class EnvVarsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    WORKSPACE_FIELD_NUMBER: _ClassVar[int]
    DOCKER_IMAGE_FIELD_NUMBER: _ClassVar[int]
    PIP_PACKAGES_FIELD_NUMBER: _ClassVar[int]
    ENV_VARS_FIELD_NUMBER: _ClassVar[int]
    EXTRAS_FIELD_NUMBER: _ClassVar[int]
    workspace: str
    docker_image: str
    pip_packages: _containers.RepeatedScalarFieldContainer[str]
    env_vars: _containers.ScalarMap[str, str]
    extras: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, workspace: _Optional[str] = ..., docker_image: _Optional[str] = ..., pip_packages: _Optional[_Iterable[str]] = ..., env_vars: _Optional[_Mapping[str, str]] = ..., extras: _Optional[_Iterable[str]] = ...) -> None: ...

class JobSpec(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_GCS_PATH_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_HASH_FIELD_NUMBER: _ClassVar[int]
    name: str
    serialized_entrypoint: bytes
    resources: ResourceSpec
    environment: EnvironmentConfig
    bundle_gcs_path: str
    bundle_hash: str
    def __init__(self, name: _Optional[str] = ..., serialized_entrypoint: _Optional[bytes] = ..., resources: _Optional[_Union[ResourceSpec, _Mapping]] = ..., environment: _Optional[_Union[EnvironmentConfig, _Mapping]] = ..., bundle_gcs_path: _Optional[str] = ..., bundle_hash: _Optional[str] = ...) -> None: ...

class JobHandle(_message.Message):
    __slots__ = ()
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: JobState
    worker_id: str
    error: str
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[_Union[JobState, str]] = ..., worker_id: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class ListJobsRequest(_message.Message):
    __slots__ = ()
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    def __init__(self, namespace: _Optional[str] = ...) -> None: ...

class ListJobsResponse(_message.Message):
    __slots__ = ()
    JOBS_FIELD_NUMBER: _ClassVar[int]
    jobs: _containers.RepeatedCompositeFieldContainer[JobStatus]
    def __init__(self, jobs: _Optional[_Iterable[_Union[JobStatus, _Mapping]]] = ...) -> None: ...

class Endpoint(_message.Message):
    __slots__ = ()
    class MetadataEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    endpoint_id: str
    name: str
    address: str
    job_id: str
    namespace: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, endpoint_id: _Optional[str] = ..., name: _Optional[str] = ..., address: _Optional[str] = ..., job_id: _Optional[str] = ..., namespace: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class RegisterEndpointRequest(_message.Message):
    __slots__ = ()
    class MetadataEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    name: str
    address: str
    job_id: str
    namespace: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, name: _Optional[str] = ..., address: _Optional[str] = ..., job_id: _Optional[str] = ..., namespace: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class LookupRequest(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    name: str
    namespace: str
    def __init__(self, name: _Optional[str] = ..., namespace: _Optional[str] = ...) -> None: ...

class LookupResponse(_message.Message):
    __slots__ = ()
    ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    endpoints: _containers.RepeatedCompositeFieldContainer[Endpoint]
    def __init__(self, endpoints: _Optional[_Iterable[_Union[Endpoint, _Mapping]]] = ...) -> None: ...

class ListEndpointsRequest(_message.Message):
    __slots__ = ()
    PREFIX_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    prefix: str
    namespace: str
    def __init__(self, prefix: _Optional[str] = ..., namespace: _Optional[str] = ...) -> None: ...

class RunJobRequest(_message.Message):
    __slots__ = ()
    class EnvVarsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_GCS_PATH_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    ENV_VARS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    PORTS_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    serialized_entrypoint: bytes
    environment: EnvironmentConfig
    bundle_gcs_path: str
    resources: ResourceSpec
    env_vars: _containers.ScalarMap[str, str]
    timeout_seconds: int
    ports: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, job_id: _Optional[str] = ..., serialized_entrypoint: _Optional[bytes] = ..., environment: _Optional[_Union[EnvironmentConfig, _Mapping]] = ..., bundle_gcs_path: _Optional[str] = ..., resources: _Optional[_Union[ResourceSpec, _Mapping]] = ..., env_vars: _Optional[_Mapping[str, str]] = ..., timeout_seconds: _Optional[int] = ..., ports: _Optional[_Iterable[str]] = ...) -> None: ...

class RunJobResponse(_message.Message):
    __slots__ = ()
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    state: JobState
    def __init__(self, job_id: _Optional[str] = ..., state: _Optional[_Union[JobState, str]] = ...) -> None: ...

class GetStatusRequest(_message.Message):
    __slots__ = ()
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class ResourceUsage(_message.Message):
    __slots__ = ()
    MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    DISK_MB_FIELD_NUMBER: _ClassVar[int]
    CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    memory_mb: int
    disk_mb: int
    cpu_millicores: int
    def __init__(self, memory_mb: _Optional[int] = ..., disk_mb: _Optional[int] = ..., cpu_millicores: _Optional[int] = ...) -> None: ...

class JobStatus(_message.Message):
    __slots__ = ()
    class PortsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    PORTS_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_USAGE_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    state: JobState
    exit_code: int
    error: str
    started_at_ms: int
    finished_at_ms: int
    ports: _containers.ScalarMap[str, int]
    resource_usage: ResourceUsage
    def __init__(self, job_id: _Optional[str] = ..., state: _Optional[_Union[JobState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at_ms: _Optional[int] = ..., finished_at_ms: _Optional[int] = ..., ports: _Optional[_Mapping[str, int]] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ...) -> None: ...

class LogEntry(_message.Message):
    __slots__ = ()
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    timestamp_ms: int
    source: str
    data: str
    def __init__(self, timestamp_ms: _Optional[int] = ..., source: _Optional[str] = ..., data: _Optional[str] = ...) -> None: ...

class FetchLogsFilter(_message.Message):
    __slots__ = ()
    REGEX_FIELD_NUMBER: _ClassVar[int]
    START_LINE_FIELD_NUMBER: _ClassVar[int]
    START_MS_FIELD_NUMBER: _ClassVar[int]
    END_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_LINES_FIELD_NUMBER: _ClassVar[int]
    regex: str
    start_line: int
    start_ms: int
    end_ms: int
    max_lines: int
    def __init__(self, regex: _Optional[str] = ..., start_line: _Optional[int] = ..., start_ms: _Optional[int] = ..., end_ms: _Optional[int] = ..., max_lines: _Optional[int] = ...) -> None: ...

class FetchLogsRequest(_message.Message):
    __slots__ = ()
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    filter: FetchLogsFilter
    def __init__(self, job_id: _Optional[str] = ..., filter: _Optional[_Union[FetchLogsFilter, _Mapping]] = ...) -> None: ...

class FetchLogsResponse(_message.Message):
    __slots__ = ()
    LOGS_FIELD_NUMBER: _ClassVar[int]
    logs: _containers.RepeatedCompositeFieldContainer[LogEntry]
    def __init__(self, logs: _Optional[_Iterable[_Union[LogEntry, _Mapping]]] = ...) -> None: ...

class KillJobRequest(_message.Message):
    __slots__ = ()
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    TERM_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    term_timeout_ms: int
    def __init__(self, job_id: _Optional[str] = ..., term_timeout_ms: _Optional[int] = ...) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ()
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    UPTIME_MS_FIELD_NUMBER: _ClassVar[int]
    RUNNING_JOBS_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    uptime_ms: int
    running_jobs: int
    def __init__(self, healthy: _Optional[bool] = ..., uptime_ms: _Optional[int] = ..., running_jobs: _Optional[int] = ...) -> None: ...

class ControllerService(_service.service): ...

class ControllerService_Stub(ControllerService): ...

class WorkerService(_service.service): ...

class WorkerService_Stub(WorkerService): ...
