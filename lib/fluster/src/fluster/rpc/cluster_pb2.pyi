from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
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
    JOB_STATE_WORKER_FAILED: _ClassVar[JobState]
    JOB_STATE_UNSCHEDULABLE: _ClassVar[JobState]
JOB_STATE_UNSPECIFIED: JobState
JOB_STATE_PENDING: JobState
JOB_STATE_BUILDING: JobState
JOB_STATE_RUNNING: JobState
JOB_STATE_SUCCEEDED: JobState
JOB_STATE_FAILED: JobState
JOB_STATE_KILLED: JobState
JOB_STATE_WORKER_FAILED: JobState
JOB_STATE_UNSCHEDULABLE: JobState

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResourceUsage(_message.Message):
    __slots__ = ("memory_mb", "disk_mb", "cpu_millicores", "memory_peak_mb", "cpu_percent", "process_count")
    MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    DISK_MB_FIELD_NUMBER: _ClassVar[int]
    CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_PEAK_MB_FIELD_NUMBER: _ClassVar[int]
    CPU_PERCENT_FIELD_NUMBER: _ClassVar[int]
    PROCESS_COUNT_FIELD_NUMBER: _ClassVar[int]
    memory_mb: int
    disk_mb: int
    cpu_millicores: int
    memory_peak_mb: int
    cpu_percent: int
    process_count: int
    def __init__(self, memory_mb: _Optional[int] = ..., disk_mb: _Optional[int] = ..., cpu_millicores: _Optional[int] = ..., memory_peak_mb: _Optional[int] = ..., cpu_percent: _Optional[int] = ..., process_count: _Optional[int] = ...) -> None: ...

class BuildMetrics(_message.Message):
    __slots__ = ("build_started_ms", "build_finished_ms", "from_cache", "image_tag")
    BUILD_STARTED_MS_FIELD_NUMBER: _ClassVar[int]
    BUILD_FINISHED_MS_FIELD_NUMBER: _ClassVar[int]
    FROM_CACHE_FIELD_NUMBER: _ClassVar[int]
    IMAGE_TAG_FIELD_NUMBER: _ClassVar[int]
    build_started_ms: int
    build_finished_ms: int
    from_cache: bool
    image_tag: str
    def __init__(self, build_started_ms: _Optional[int] = ..., build_finished_ms: _Optional[int] = ..., from_cache: _Optional[bool] = ..., image_tag: _Optional[str] = ...) -> None: ...

class JobStatus(_message.Message):
    __slots__ = ("job_id", "state", "exit_code", "error", "started_at_ms", "finished_at_ms", "ports", "resource_usage", "status_message", "build_metrics", "worker_id", "worker_address", "serialized_result", "parent_job_id")
    class PortsEntry(_message.Message):
        __slots__ = ("key", "value")
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
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    BUILD_METRICS_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_RESULT_FIELD_NUMBER: _ClassVar[int]
    PARENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    state: JobState
    exit_code: int
    error: str
    started_at_ms: int
    finished_at_ms: int
    ports: _containers.ScalarMap[str, int]
    resource_usage: ResourceUsage
    status_message: str
    build_metrics: BuildMetrics
    worker_id: str
    worker_address: str
    serialized_result: bytes
    parent_job_id: str
    def __init__(self, job_id: _Optional[str] = ..., state: _Optional[_Union[JobState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at_ms: _Optional[int] = ..., finished_at_ms: _Optional[int] = ..., ports: _Optional[_Mapping[str, int]] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., status_message: _Optional[str] = ..., build_metrics: _Optional[_Union[BuildMetrics, _Mapping]] = ..., worker_id: _Optional[str] = ..., worker_address: _Optional[str] = ..., serialized_result: _Optional[bytes] = ..., parent_job_id: _Optional[str] = ...) -> None: ...

class DeviceConfig(_message.Message):
    __slots__ = ("cpu", "gpu", "tpu")
    CPU_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    TPU_FIELD_NUMBER: _ClassVar[int]
    cpu: CpuDevice
    gpu: GpuDevice
    tpu: TpuDevice
    def __init__(self, cpu: _Optional[_Union[CpuDevice, _Mapping]] = ..., gpu: _Optional[_Union[GpuDevice, _Mapping]] = ..., tpu: _Optional[_Union[TpuDevice, _Mapping]] = ...) -> None: ...

class CpuDevice(_message.Message):
    __slots__ = ("variant",)
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    variant: str
    def __init__(self, variant: _Optional[str] = ...) -> None: ...

class GpuDevice(_message.Message):
    __slots__ = ("variant", "count")
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    variant: str
    count: int
    def __init__(self, variant: _Optional[str] = ..., count: _Optional[int] = ...) -> None: ...

class TpuDevice(_message.Message):
    __slots__ = ("variant", "topology")
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_FIELD_NUMBER: _ClassVar[int]
    variant: str
    topology: str
    def __init__(self, variant: _Optional[str] = ..., topology: _Optional[str] = ...) -> None: ...

class ResourceSpec(_message.Message):
    __slots__ = ("cpu", "memory", "disk", "device", "replicas", "preemptible", "regions")
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
    __slots__ = ("workspace", "pip_packages", "env_vars", "extras")
    class EnvVarsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    WORKSPACE_FIELD_NUMBER: _ClassVar[int]
    PIP_PACKAGES_FIELD_NUMBER: _ClassVar[int]
    ENV_VARS_FIELD_NUMBER: _ClassVar[int]
    EXTRAS_FIELD_NUMBER: _ClassVar[int]
    workspace: str
    pip_packages: _containers.RepeatedScalarFieldContainer[str]
    env_vars: _containers.ScalarMap[str, str]
    extras: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, workspace: _Optional[str] = ..., pip_packages: _Optional[_Iterable[str]] = ..., env_vars: _Optional[_Mapping[str, str]] = ..., extras: _Optional[_Iterable[str]] = ...) -> None: ...

class WorkerMetadata(_message.Message):
    __slots__ = ("hostname", "ip_address", "cpu_count", "memory_bytes", "tpu_name", "tpu_worker_hostnames", "tpu_worker_id", "tpu_chips_per_host_bounds", "gpu_count", "gpu_name", "gpu_memory_mb", "gce_instance_name", "gce_zone")
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    IP_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    CPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    TPU_NAME_FIELD_NUMBER: _ClassVar[int]
    TPU_WORKER_HOSTNAMES_FIELD_NUMBER: _ClassVar[int]
    TPU_WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    TPU_CHIPS_PER_HOST_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    GPU_NAME_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    GCE_INSTANCE_NAME_FIELD_NUMBER: _ClassVar[int]
    GCE_ZONE_FIELD_NUMBER: _ClassVar[int]
    hostname: str
    ip_address: str
    cpu_count: int
    memory_bytes: int
    tpu_name: str
    tpu_worker_hostnames: str
    tpu_worker_id: str
    tpu_chips_per_host_bounds: str
    gpu_count: int
    gpu_name: str
    gpu_memory_mb: int
    gce_instance_name: str
    gce_zone: str
    def __init__(self, hostname: _Optional[str] = ..., ip_address: _Optional[str] = ..., cpu_count: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., tpu_name: _Optional[str] = ..., tpu_worker_hostnames: _Optional[str] = ..., tpu_worker_id: _Optional[str] = ..., tpu_chips_per_host_bounds: _Optional[str] = ..., gpu_count: _Optional[int] = ..., gpu_name: _Optional[str] = ..., gpu_memory_mb: _Optional[int] = ..., gce_instance_name: _Optional[str] = ..., gce_zone: _Optional[str] = ...) -> None: ...

class Controller(_message.Message):
    __slots__ = ()
    class LaunchJobRequest(_message.Message):
        __slots__ = ("name", "serialized_entrypoint", "resources", "environment", "bundle_gcs_path", "bundle_hash", "bundle_blob", "scheduling_timeout_seconds", "ports", "parent_job_id")
        NAME_FIELD_NUMBER: _ClassVar[int]
        SERIALIZED_ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
        RESOURCES_FIELD_NUMBER: _ClassVar[int]
        ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
        BUNDLE_GCS_PATH_FIELD_NUMBER: _ClassVar[int]
        BUNDLE_HASH_FIELD_NUMBER: _ClassVar[int]
        BUNDLE_BLOB_FIELD_NUMBER: _ClassVar[int]
        SCHEDULING_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
        PORTS_FIELD_NUMBER: _ClassVar[int]
        PARENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
        name: str
        serialized_entrypoint: bytes
        resources: ResourceSpec
        environment: EnvironmentConfig
        bundle_gcs_path: str
        bundle_hash: str
        bundle_blob: bytes
        scheduling_timeout_seconds: int
        ports: _containers.RepeatedScalarFieldContainer[str]
        parent_job_id: str
        def __init__(self, name: _Optional[str] = ..., serialized_entrypoint: _Optional[bytes] = ..., resources: _Optional[_Union[ResourceSpec, _Mapping]] = ..., environment: _Optional[_Union[EnvironmentConfig, _Mapping]] = ..., bundle_gcs_path: _Optional[str] = ..., bundle_hash: _Optional[str] = ..., bundle_blob: _Optional[bytes] = ..., scheduling_timeout_seconds: _Optional[int] = ..., ports: _Optional[_Iterable[str]] = ..., parent_job_id: _Optional[str] = ...) -> None: ...
    class LaunchJobResponse(_message.Message):
        __slots__ = ("job_id",)
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        def __init__(self, job_id: _Optional[str] = ...) -> None: ...
    class GetJobStatusRequest(_message.Message):
        __slots__ = ("job_id", "include_result")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        INCLUDE_RESULT_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        include_result: bool
        def __init__(self, job_id: _Optional[str] = ..., include_result: _Optional[bool] = ...) -> None: ...
    class GetJobStatusResponse(_message.Message):
        __slots__ = ("job",)
        JOB_FIELD_NUMBER: _ClassVar[int]
        job: JobStatus
        def __init__(self, job: _Optional[_Union[JobStatus, _Mapping]] = ...) -> None: ...
    class TerminateJobRequest(_message.Message):
        __slots__ = ("job_id",)
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        def __init__(self, job_id: _Optional[str] = ...) -> None: ...
    class ListJobsRequest(_message.Message):
        __slots__ = ("namespace",)
        NAMESPACE_FIELD_NUMBER: _ClassVar[int]
        namespace: str
        def __init__(self, namespace: _Optional[str] = ...) -> None: ...
    class ListJobsResponse(_message.Message):
        __slots__ = ("jobs",)
        JOBS_FIELD_NUMBER: _ClassVar[int]
        jobs: _containers.RepeatedCompositeFieldContainer[JobStatus]
        def __init__(self, jobs: _Optional[_Iterable[_Union[JobStatus, _Mapping]]] = ...) -> None: ...
    class WorkerInfo(_message.Message):
        __slots__ = ("worker_id", "address", "resources", "registered_at_ms", "metadata")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        RESOURCES_FIELD_NUMBER: _ClassVar[int]
        REGISTERED_AT_MS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        address: str
        resources: ResourceSpec
        registered_at_ms: int
        metadata: WorkerMetadata
        def __init__(self, worker_id: _Optional[str] = ..., address: _Optional[str] = ..., resources: _Optional[_Union[ResourceSpec, _Mapping]] = ..., registered_at_ms: _Optional[int] = ..., metadata: _Optional[_Union[WorkerMetadata, _Mapping]] = ...) -> None: ...
    class RegisterWorkerRequest(_message.Message):
        __slots__ = ("worker_id", "address", "resources", "metadata")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        RESOURCES_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        address: str
        resources: ResourceSpec
        metadata: WorkerMetadata
        def __init__(self, worker_id: _Optional[str] = ..., address: _Optional[str] = ..., resources: _Optional[_Union[ResourceSpec, _Mapping]] = ..., metadata: _Optional[_Union[WorkerMetadata, _Mapping]] = ...) -> None: ...
    class RegisterWorkerResponse(_message.Message):
        __slots__ = ("accepted", "controller_address")
        ACCEPTED_FIELD_NUMBER: _ClassVar[int]
        CONTROLLER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
        accepted: bool
        controller_address: str
        def __init__(self, accepted: _Optional[bool] = ..., controller_address: _Optional[str] = ...) -> None: ...
    class WorkerHealthStatus(_message.Message):
        __slots__ = ("worker_id", "healthy", "consecutive_failures", "last_heartbeat_ms", "running_job_ids")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        HEALTHY_FIELD_NUMBER: _ClassVar[int]
        CONSECUTIVE_FAILURES_FIELD_NUMBER: _ClassVar[int]
        LAST_HEARTBEAT_MS_FIELD_NUMBER: _ClassVar[int]
        RUNNING_JOB_IDS_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        healthy: bool
        consecutive_failures: int
        last_heartbeat_ms: int
        running_job_ids: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, worker_id: _Optional[str] = ..., healthy: _Optional[bool] = ..., consecutive_failures: _Optional[int] = ..., last_heartbeat_ms: _Optional[int] = ..., running_job_ids: _Optional[_Iterable[str]] = ...) -> None: ...
    class ListWorkersRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class ListWorkersResponse(_message.Message):
        __slots__ = ("workers",)
        WORKERS_FIELD_NUMBER: _ClassVar[int]
        workers: _containers.RepeatedCompositeFieldContainer[Controller.WorkerHealthStatus]
        def __init__(self, workers: _Optional[_Iterable[_Union[Controller.WorkerHealthStatus, _Mapping]]] = ...) -> None: ...
    class ReportJobStateRequest(_message.Message):
        __slots__ = ("worker_id", "job_id", "state", "exit_code", "error", "finished_at_ms")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        STATE_FIELD_NUMBER: _ClassVar[int]
        EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
        ERROR_FIELD_NUMBER: _ClassVar[int]
        FINISHED_AT_MS_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        job_id: str
        state: JobState
        exit_code: int
        error: str
        finished_at_ms: int
        def __init__(self, worker_id: _Optional[str] = ..., job_id: _Optional[str] = ..., state: _Optional[_Union[JobState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., finished_at_ms: _Optional[int] = ...) -> None: ...
    class ReportJobStateResponse(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class Endpoint(_message.Message):
        __slots__ = ("endpoint_id", "name", "address", "job_id", "metadata")
        class MetadataEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
        ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
        NAME_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        endpoint_id: str
        name: str
        address: str
        job_id: str
        metadata: _containers.ScalarMap[str, str]
        def __init__(self, endpoint_id: _Optional[str] = ..., name: _Optional[str] = ..., address: _Optional[str] = ..., job_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...
    class RegisterEndpointRequest(_message.Message):
        __slots__ = ("name", "address", "job_id", "metadata")
        class MetadataEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
        NAME_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        name: str
        address: str
        job_id: str
        metadata: _containers.ScalarMap[str, str]
        def __init__(self, name: _Optional[str] = ..., address: _Optional[str] = ..., job_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...
    class RegisterEndpointResponse(_message.Message):
        __slots__ = ("endpoint_id",)
        ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
        endpoint_id: str
        def __init__(self, endpoint_id: _Optional[str] = ...) -> None: ...
    class UnregisterEndpointRequest(_message.Message):
        __slots__ = ("endpoint_id",)
        ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
        endpoint_id: str
        def __init__(self, endpoint_id: _Optional[str] = ...) -> None: ...
    class LookupEndpointRequest(_message.Message):
        __slots__ = ("name",)
        NAME_FIELD_NUMBER: _ClassVar[int]
        name: str
        def __init__(self, name: _Optional[str] = ...) -> None: ...
    class LookupEndpointResponse(_message.Message):
        __slots__ = ("endpoint",)
        ENDPOINT_FIELD_NUMBER: _ClassVar[int]
        endpoint: Controller.Endpoint
        def __init__(self, endpoint: _Optional[_Union[Controller.Endpoint, _Mapping]] = ...) -> None: ...
    class ListEndpointsRequest(_message.Message):
        __slots__ = ("prefix",)
        PREFIX_FIELD_NUMBER: _ClassVar[int]
        prefix: str
        def __init__(self, prefix: _Optional[str] = ...) -> None: ...
    class ListEndpointsResponse(_message.Message):
        __slots__ = ("endpoints",)
        ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
        endpoints: _containers.RepeatedCompositeFieldContainer[Controller.Endpoint]
        def __init__(self, endpoints: _Optional[_Iterable[_Union[Controller.Endpoint, _Mapping]]] = ...) -> None: ...
    def __init__(self) -> None: ...

class Worker(_message.Message):
    __slots__ = ()
    class RunJobRequest(_message.Message):
        __slots__ = ("job_id", "serialized_entrypoint", "environment", "bundle_gcs_path", "resources", "timeout_seconds", "ports")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        SERIALIZED_ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
        ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
        BUNDLE_GCS_PATH_FIELD_NUMBER: _ClassVar[int]
        RESOURCES_FIELD_NUMBER: _ClassVar[int]
        TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
        PORTS_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        serialized_entrypoint: bytes
        environment: EnvironmentConfig
        bundle_gcs_path: str
        resources: ResourceSpec
        timeout_seconds: int
        ports: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, job_id: _Optional[str] = ..., serialized_entrypoint: _Optional[bytes] = ..., environment: _Optional[_Union[EnvironmentConfig, _Mapping]] = ..., bundle_gcs_path: _Optional[str] = ..., resources: _Optional[_Union[ResourceSpec, _Mapping]] = ..., timeout_seconds: _Optional[int] = ..., ports: _Optional[_Iterable[str]] = ...) -> None: ...
    class RunJobResponse(_message.Message):
        __slots__ = ("job_id", "state")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        STATE_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        state: JobState
        def __init__(self, job_id: _Optional[str] = ..., state: _Optional[_Union[JobState, str]] = ...) -> None: ...
    class GetJobStatusRequest(_message.Message):
        __slots__ = ("job_id", "include_result")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        INCLUDE_RESULT_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        include_result: bool
        def __init__(self, job_id: _Optional[str] = ..., include_result: _Optional[bool] = ...) -> None: ...
    class ListJobsRequest(_message.Message):
        __slots__ = ("namespace",)
        NAMESPACE_FIELD_NUMBER: _ClassVar[int]
        namespace: str
        def __init__(self, namespace: _Optional[str] = ...) -> None: ...
    class ListJobsResponse(_message.Message):
        __slots__ = ("jobs",)
        JOBS_FIELD_NUMBER: _ClassVar[int]
        jobs: _containers.RepeatedCompositeFieldContainer[JobStatus]
        def __init__(self, jobs: _Optional[_Iterable[_Union[JobStatus, _Mapping]]] = ...) -> None: ...
    class LogEntry(_message.Message):
        __slots__ = ("timestamp_ms", "source", "data")
        TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
        SOURCE_FIELD_NUMBER: _ClassVar[int]
        DATA_FIELD_NUMBER: _ClassVar[int]
        timestamp_ms: int
        source: str
        data: str
        def __init__(self, timestamp_ms: _Optional[int] = ..., source: _Optional[str] = ..., data: _Optional[str] = ...) -> None: ...
    class FetchLogsFilter(_message.Message):
        __slots__ = ("regex", "start_line", "start_ms", "end_ms", "max_lines")
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
        __slots__ = ("job_id", "filter")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        FILTER_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        filter: Worker.FetchLogsFilter
        def __init__(self, job_id: _Optional[str] = ..., filter: _Optional[_Union[Worker.FetchLogsFilter, _Mapping]] = ...) -> None: ...
    class FetchLogsResponse(_message.Message):
        __slots__ = ("logs",)
        LOGS_FIELD_NUMBER: _ClassVar[int]
        logs: _containers.RepeatedCompositeFieldContainer[Worker.LogEntry]
        def __init__(self, logs: _Optional[_Iterable[_Union[Worker.LogEntry, _Mapping]]] = ...) -> None: ...
    class KillJobRequest(_message.Message):
        __slots__ = ("job_id", "term_timeout_ms")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        TERM_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        term_timeout_ms: int
        def __init__(self, job_id: _Optional[str] = ..., term_timeout_ms: _Optional[int] = ...) -> None: ...
    class HealthResponse(_message.Message):
        __slots__ = ("healthy", "uptime_ms", "running_jobs")
        HEALTHY_FIELD_NUMBER: _ClassVar[int]
        UPTIME_MS_FIELD_NUMBER: _ClassVar[int]
        RUNNING_JOBS_FIELD_NUMBER: _ClassVar[int]
        healthy: bool
        uptime_ms: int
        running_jobs: int
        def __init__(self, healthy: _Optional[bool] = ..., uptime_ms: _Optional[int] = ..., running_jobs: _Optional[int] = ...) -> None: ...
    def __init__(self) -> None: ...
