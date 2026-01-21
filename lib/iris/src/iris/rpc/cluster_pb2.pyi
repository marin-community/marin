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

class TaskState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TASK_STATE_UNSPECIFIED: _ClassVar[TaskState]
    TASK_STATE_PENDING: _ClassVar[TaskState]
    TASK_STATE_BUILDING: _ClassVar[TaskState]
    TASK_STATE_RUNNING: _ClassVar[TaskState]
    TASK_STATE_SUCCEEDED: _ClassVar[TaskState]
    TASK_STATE_FAILED: _ClassVar[TaskState]
    TASK_STATE_KILLED: _ClassVar[TaskState]
    TASK_STATE_WORKER_FAILED: _ClassVar[TaskState]
    TASK_STATE_UNSCHEDULABLE: _ClassVar[TaskState]
JOB_STATE_UNSPECIFIED: JobState
JOB_STATE_PENDING: JobState
JOB_STATE_BUILDING: JobState
JOB_STATE_RUNNING: JobState
JOB_STATE_SUCCEEDED: JobState
JOB_STATE_FAILED: JobState
JOB_STATE_KILLED: JobState
JOB_STATE_WORKER_FAILED: JobState
JOB_STATE_UNSCHEDULABLE: JobState
TASK_STATE_UNSPECIFIED: TaskState
TASK_STATE_PENDING: TaskState
TASK_STATE_BUILDING: TaskState
TASK_STATE_RUNNING: TaskState
TASK_STATE_SUCCEEDED: TaskState
TASK_STATE_FAILED: TaskState
TASK_STATE_KILLED: TaskState
TASK_STATE_WORKER_FAILED: TaskState
TASK_STATE_UNSCHEDULABLE: TaskState

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TaskStatus(_message.Message):
    __slots__ = ("task_id", "job_id", "task_index", "state", "worker_id", "worker_address", "exit_code", "error", "started_at_ms", "finished_at_ms", "ports", "resource_usage", "build_metrics", "current_attempt_id", "attempts")
    class PortsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_INDEX_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    PORTS_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_USAGE_FIELD_NUMBER: _ClassVar[int]
    BUILD_METRICS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    job_id: str
    task_index: int
    state: TaskState
    worker_id: str
    worker_address: str
    exit_code: int
    error: str
    started_at_ms: int
    finished_at_ms: int
    ports: _containers.ScalarMap[str, int]
    resource_usage: ResourceUsage
    build_metrics: BuildMetrics
    current_attempt_id: int
    attempts: _containers.RepeatedCompositeFieldContainer[TaskAttempt]
    def __init__(self, task_id: _Optional[str] = ..., job_id: _Optional[str] = ..., task_index: _Optional[int] = ..., state: _Optional[_Union[TaskState, str]] = ..., worker_id: _Optional[str] = ..., worker_address: _Optional[str] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at_ms: _Optional[int] = ..., finished_at_ms: _Optional[int] = ..., ports: _Optional[_Mapping[str, int]] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., build_metrics: _Optional[_Union[BuildMetrics, _Mapping]] = ..., current_attempt_id: _Optional[int] = ..., attempts: _Optional[_Iterable[_Union[TaskAttempt, _Mapping]]] = ...) -> None: ...

class TaskAttempt(_message.Message):
    __slots__ = ("attempt_id", "worker_id", "state", "exit_code", "error", "started_at_ms", "finished_at_ms", "is_worker_failure")
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    IS_WORKER_FAILURE_FIELD_NUMBER: _ClassVar[int]
    attempt_id: int
    worker_id: str
    state: TaskState
    exit_code: int
    error: str
    started_at_ms: int
    finished_at_ms: int
    is_worker_failure: bool
    def __init__(self, attempt_id: _Optional[int] = ..., worker_id: _Optional[str] = ..., state: _Optional[_Union[TaskState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at_ms: _Optional[int] = ..., finished_at_ms: _Optional[int] = ..., is_worker_failure: _Optional[bool] = ...) -> None: ...

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
    __slots__ = ("job_id", "state", "exit_code", "error", "started_at_ms", "finished_at_ms", "ports", "resource_usage", "status_message", "build_metrics", "serialized_result", "parent_job_id", "failure_count", "preemption_count", "num_tasks", "tasks_pending", "tasks_running", "tasks_succeeded", "tasks_failed", "tasks")
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
    SERIALIZED_RESULT_FIELD_NUMBER: _ClassVar[int]
    PARENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    FAILURE_COUNT_FIELD_NUMBER: _ClassVar[int]
    PREEMPTION_COUNT_FIELD_NUMBER: _ClassVar[int]
    NUM_TASKS_FIELD_NUMBER: _ClassVar[int]
    TASKS_PENDING_FIELD_NUMBER: _ClassVar[int]
    TASKS_RUNNING_FIELD_NUMBER: _ClassVar[int]
    TASKS_SUCCEEDED_FIELD_NUMBER: _ClassVar[int]
    TASKS_FAILED_FIELD_NUMBER: _ClassVar[int]
    TASKS_FIELD_NUMBER: _ClassVar[int]
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
    serialized_result: bytes
    parent_job_id: str
    failure_count: int
    preemption_count: int
    num_tasks: int
    tasks_pending: int
    tasks_running: int
    tasks_succeeded: int
    tasks_failed: int
    tasks: _containers.RepeatedCompositeFieldContainer[TaskStatus]
    def __init__(self, job_id: _Optional[str] = ..., state: _Optional[_Union[JobState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at_ms: _Optional[int] = ..., finished_at_ms: _Optional[int] = ..., ports: _Optional[_Mapping[str, int]] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., status_message: _Optional[str] = ..., build_metrics: _Optional[_Union[BuildMetrics, _Mapping]] = ..., serialized_result: _Optional[bytes] = ..., parent_job_id: _Optional[str] = ..., failure_count: _Optional[int] = ..., preemption_count: _Optional[int] = ..., num_tasks: _Optional[int] = ..., tasks_pending: _Optional[int] = ..., tasks_running: _Optional[int] = ..., tasks_succeeded: _Optional[int] = ..., tasks_failed: _Optional[int] = ..., tasks: _Optional[_Iterable[_Union[TaskStatus, _Mapping]]] = ...) -> None: ...

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

class ResourceSpecProto(_message.Message):
    __slots__ = ("cpu", "memory_bytes", "disk_bytes", "device", "replicas", "preemptible", "regions")
    CPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    REPLICAS_FIELD_NUMBER: _ClassVar[int]
    PREEMPTIBLE_FIELD_NUMBER: _ClassVar[int]
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    cpu: int
    memory_bytes: int
    disk_bytes: int
    device: DeviceConfig
    replicas: int
    preemptible: bool
    regions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, cpu: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., device: _Optional[_Union[DeviceConfig, _Mapping]] = ..., replicas: _Optional[int] = ..., preemptible: _Optional[bool] = ..., regions: _Optional[_Iterable[str]] = ...) -> None: ...

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

class AttributeValue(_message.Message):
    __slots__ = ("string_value", "int_value", "float_value")
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    string_value: str
    int_value: int
    float_value: float
    def __init__(self, string_value: _Optional[str] = ..., int_value: _Optional[int] = ..., float_value: _Optional[float] = ...) -> None: ...

class WorkerMetadata(_message.Message):
    __slots__ = ("hostname", "ip_address", "cpu_count", "memory_bytes", "disk_bytes", "device", "tpu_name", "tpu_worker_hostnames", "tpu_worker_id", "tpu_chips_per_host_bounds", "gpu_count", "gpu_name", "gpu_memory_mb", "gce_instance_name", "gce_zone", "attributes")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: AttributeValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[AttributeValue, _Mapping]] = ...) -> None: ...
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    IP_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    CPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    TPU_NAME_FIELD_NUMBER: _ClassVar[int]
    TPU_WORKER_HOSTNAMES_FIELD_NUMBER: _ClassVar[int]
    TPU_WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    TPU_CHIPS_PER_HOST_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    GPU_NAME_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    GCE_INSTANCE_NAME_FIELD_NUMBER: _ClassVar[int]
    GCE_ZONE_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    hostname: str
    ip_address: str
    cpu_count: int
    memory_bytes: int
    disk_bytes: int
    device: DeviceConfig
    tpu_name: str
    tpu_worker_hostnames: str
    tpu_worker_id: str
    tpu_chips_per_host_bounds: str
    gpu_count: int
    gpu_name: str
    gpu_memory_mb: int
    gce_instance_name: str
    gce_zone: str
    attributes: _containers.MessageMap[str, AttributeValue]
    def __init__(self, hostname: _Optional[str] = ..., ip_address: _Optional[str] = ..., cpu_count: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., device: _Optional[_Union[DeviceConfig, _Mapping]] = ..., tpu_name: _Optional[str] = ..., tpu_worker_hostnames: _Optional[str] = ..., tpu_worker_id: _Optional[str] = ..., tpu_chips_per_host_bounds: _Optional[str] = ..., gpu_count: _Optional[int] = ..., gpu_name: _Optional[str] = ..., gpu_memory_mb: _Optional[int] = ..., gce_instance_name: _Optional[str] = ..., gce_zone: _Optional[str] = ..., attributes: _Optional[_Mapping[str, AttributeValue]] = ...) -> None: ...

class Controller(_message.Message):
    __slots__ = ()
    class LaunchJobRequest(_message.Message):
        __slots__ = ("name", "serialized_entrypoint", "resources", "environment", "bundle_gcs_path", "bundle_hash", "bundle_blob", "scheduling_timeout_seconds", "ports", "parent_job_id", "max_task_failures", "max_retries_failure", "max_retries_preemption")
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
        MAX_TASK_FAILURES_FIELD_NUMBER: _ClassVar[int]
        MAX_RETRIES_FAILURE_FIELD_NUMBER: _ClassVar[int]
        MAX_RETRIES_PREEMPTION_FIELD_NUMBER: _ClassVar[int]
        name: str
        serialized_entrypoint: bytes
        resources: ResourceSpecProto
        environment: EnvironmentConfig
        bundle_gcs_path: str
        bundle_hash: str
        bundle_blob: bytes
        scheduling_timeout_seconds: int
        ports: _containers.RepeatedScalarFieldContainer[str]
        parent_job_id: str
        max_task_failures: int
        max_retries_failure: int
        max_retries_preemption: int
        def __init__(self, name: _Optional[str] = ..., serialized_entrypoint: _Optional[bytes] = ..., resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ..., environment: _Optional[_Union[EnvironmentConfig, _Mapping]] = ..., bundle_gcs_path: _Optional[str] = ..., bundle_hash: _Optional[str] = ..., bundle_blob: _Optional[bytes] = ..., scheduling_timeout_seconds: _Optional[int] = ..., ports: _Optional[_Iterable[str]] = ..., parent_job_id: _Optional[str] = ..., max_task_failures: _Optional[int] = ..., max_retries_failure: _Optional[int] = ..., max_retries_preemption: _Optional[int] = ...) -> None: ...
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
        __slots__ = ()
        def __init__(self) -> None: ...
    class ListJobsResponse(_message.Message):
        __slots__ = ("jobs",)
        JOBS_FIELD_NUMBER: _ClassVar[int]
        jobs: _containers.RepeatedCompositeFieldContainer[JobStatus]
        def __init__(self, jobs: _Optional[_Iterable[_Union[JobStatus, _Mapping]]] = ...) -> None: ...
    class GetTaskStatusRequest(_message.Message):
        __slots__ = ("job_id", "task_index")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        TASK_INDEX_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        task_index: int
        def __init__(self, job_id: _Optional[str] = ..., task_index: _Optional[int] = ...) -> None: ...
    class GetTaskStatusResponse(_message.Message):
        __slots__ = ("task",)
        TASK_FIELD_NUMBER: _ClassVar[int]
        task: TaskStatus
        def __init__(self, task: _Optional[_Union[TaskStatus, _Mapping]] = ...) -> None: ...
    class ListTasksRequest(_message.Message):
        __slots__ = ("job_id",)
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        def __init__(self, job_id: _Optional[str] = ...) -> None: ...
    class ListTasksResponse(_message.Message):
        __slots__ = ("tasks",)
        TASKS_FIELD_NUMBER: _ClassVar[int]
        tasks: _containers.RepeatedCompositeFieldContainer[TaskStatus]
        def __init__(self, tasks: _Optional[_Iterable[_Union[TaskStatus, _Mapping]]] = ...) -> None: ...
    class WorkerInfo(_message.Message):
        __slots__ = ("worker_id", "address", "metadata", "registered_at_ms")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        REGISTERED_AT_MS_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        address: str
        metadata: WorkerMetadata
        registered_at_ms: int
        def __init__(self, worker_id: _Optional[str] = ..., address: _Optional[str] = ..., metadata: _Optional[_Union[WorkerMetadata, _Mapping]] = ..., registered_at_ms: _Optional[int] = ...) -> None: ...
    class RegisterWorkerRequest(_message.Message):
        __slots__ = ("worker_id", "address", "metadata")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        address: str
        metadata: WorkerMetadata
        def __init__(self, worker_id: _Optional[str] = ..., address: _Optional[str] = ..., metadata: _Optional[_Union[WorkerMetadata, _Mapping]] = ...) -> None: ...
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
    class ReportTaskStateRequest(_message.Message):
        __slots__ = ("worker_id", "task_id", "job_id", "task_index", "state", "exit_code", "error", "finished_at_ms", "attempt_id")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        TASK_INDEX_FIELD_NUMBER: _ClassVar[int]
        STATE_FIELD_NUMBER: _ClassVar[int]
        EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
        ERROR_FIELD_NUMBER: _ClassVar[int]
        FINISHED_AT_MS_FIELD_NUMBER: _ClassVar[int]
        ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        task_id: str
        job_id: str
        task_index: int
        state: TaskState
        exit_code: int
        error: str
        finished_at_ms: int
        attempt_id: int
        def __init__(self, worker_id: _Optional[str] = ..., task_id: _Optional[str] = ..., job_id: _Optional[str] = ..., task_index: _Optional[int] = ..., state: _Optional[_Union[TaskState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., finished_at_ms: _Optional[int] = ..., attempt_id: _Optional[int] = ...) -> None: ...
    class ReportTaskStateResponse(_message.Message):
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
    class RunTaskRequest(_message.Message):
        __slots__ = ("job_id", "task_id", "task_index", "num_tasks", "serialized_entrypoint", "environment", "bundle_gcs_path", "resources", "timeout_seconds", "ports", "attempt_id")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        TASK_INDEX_FIELD_NUMBER: _ClassVar[int]
        NUM_TASKS_FIELD_NUMBER: _ClassVar[int]
        SERIALIZED_ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
        ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
        BUNDLE_GCS_PATH_FIELD_NUMBER: _ClassVar[int]
        RESOURCES_FIELD_NUMBER: _ClassVar[int]
        TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
        PORTS_FIELD_NUMBER: _ClassVar[int]
        ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        task_id: str
        task_index: int
        num_tasks: int
        serialized_entrypoint: bytes
        environment: EnvironmentConfig
        bundle_gcs_path: str
        resources: ResourceSpecProto
        timeout_seconds: int
        ports: _containers.RepeatedScalarFieldContainer[str]
        attempt_id: int
        def __init__(self, job_id: _Optional[str] = ..., task_id: _Optional[str] = ..., task_index: _Optional[int] = ..., num_tasks: _Optional[int] = ..., serialized_entrypoint: _Optional[bytes] = ..., environment: _Optional[_Union[EnvironmentConfig, _Mapping]] = ..., bundle_gcs_path: _Optional[str] = ..., resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ..., timeout_seconds: _Optional[int] = ..., ports: _Optional[_Iterable[str]] = ..., attempt_id: _Optional[int] = ...) -> None: ...
    class RunTaskResponse(_message.Message):
        __slots__ = ("task_id", "state")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        STATE_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        state: TaskState
        def __init__(self, task_id: _Optional[str] = ..., state: _Optional[_Union[TaskState, str]] = ...) -> None: ...
    class GetTaskStatusRequest(_message.Message):
        __slots__ = ("task_id", "include_result")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        INCLUDE_RESULT_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        include_result: bool
        def __init__(self, task_id: _Optional[str] = ..., include_result: _Optional[bool] = ...) -> None: ...
    class ListTasksRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class ListTasksResponse(_message.Message):
        __slots__ = ("tasks",)
        TASKS_FIELD_NUMBER: _ClassVar[int]
        tasks: _containers.RepeatedCompositeFieldContainer[TaskStatus]
        def __init__(self, tasks: _Optional[_Iterable[_Union[TaskStatus, _Mapping]]] = ...) -> None: ...
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
    class FetchTaskLogsRequest(_message.Message):
        __slots__ = ("task_id", "filter")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        FILTER_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        filter: Worker.FetchLogsFilter
        def __init__(self, task_id: _Optional[str] = ..., filter: _Optional[_Union[Worker.FetchLogsFilter, _Mapping]] = ...) -> None: ...
    class FetchTaskLogsResponse(_message.Message):
        __slots__ = ("logs",)
        LOGS_FIELD_NUMBER: _ClassVar[int]
        logs: _containers.RepeatedCompositeFieldContainer[Worker.LogEntry]
        def __init__(self, logs: _Optional[_Iterable[_Union[Worker.LogEntry, _Mapping]]] = ...) -> None: ...
    class KillTaskRequest(_message.Message):
        __slots__ = ("task_id", "term_timeout_ms")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        TERM_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        term_timeout_ms: int
        def __init__(self, task_id: _Optional[str] = ..., term_timeout_ms: _Optional[int] = ...) -> None: ...
    class HealthResponse(_message.Message):
        __slots__ = ("healthy", "uptime_ms", "running_tasks")
        HEALTHY_FIELD_NUMBER: _ClassVar[int]
        UPTIME_MS_FIELD_NUMBER: _ClassVar[int]
        RUNNING_TASKS_FIELD_NUMBER: _ClassVar[int]
        healthy: bool
        uptime_ms: int
        running_tasks: int
        def __init__(self, healthy: _Optional[bool] = ..., uptime_ms: _Optional[int] = ..., running_tasks: _Optional[int] = ...) -> None: ...
    def __init__(self) -> None: ...
