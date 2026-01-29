from . import vm_pb2 as _vm_pb2
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

class ConstraintOp(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONSTRAINT_OP_EQ: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_NE: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_EXISTS: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_NOT_EXISTS: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_GT: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_GE: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_LT: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_LE: _ClassVar[ConstraintOp]
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
CONSTRAINT_OP_EQ: ConstraintOp
CONSTRAINT_OP_NE: ConstraintOp
CONSTRAINT_OP_EXISTS: ConstraintOp
CONSTRAINT_OP_NOT_EXISTS: ConstraintOp
CONSTRAINT_OP_GT: ConstraintOp
CONSTRAINT_OP_GE: ConstraintOp
CONSTRAINT_OP_LT: ConstraintOp
CONSTRAINT_OP_LE: ConstraintOp

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TaskStatus(_message.Message):
    __slots__ = ("task_id", "job_id", "task_index", "state", "worker_id", "worker_address", "exit_code", "error", "started_at_ms", "finished_at_ms", "ports", "resource_usage", "build_metrics", "current_attempt_id", "attempts", "pending_reason", "can_be_scheduled")
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
    PENDING_REASON_FIELD_NUMBER: _ClassVar[int]
    CAN_BE_SCHEDULED_FIELD_NUMBER: _ClassVar[int]
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
    pending_reason: str
    can_be_scheduled: bool
    def __init__(self, task_id: _Optional[str] = ..., job_id: _Optional[str] = ..., task_index: _Optional[int] = ..., state: _Optional[_Union[TaskState, str]] = ..., worker_id: _Optional[str] = ..., worker_address: _Optional[str] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at_ms: _Optional[int] = ..., finished_at_ms: _Optional[int] = ..., ports: _Optional[_Mapping[str, int]] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., build_metrics: _Optional[_Union[BuildMetrics, _Mapping]] = ..., current_attempt_id: _Optional[int] = ..., attempts: _Optional[_Iterable[_Union[TaskAttempt, _Mapping]]] = ..., pending_reason: _Optional[str] = ..., can_be_scheduled: _Optional[bool] = ...) -> None: ...

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
    __slots__ = ("job_id", "state", "exit_code", "error", "started_at_ms", "finished_at_ms", "ports", "resource_usage", "status_message", "build_metrics", "serialized_result", "parent_job_id", "failure_count", "preemption_count", "tasks", "name", "submitted_at_ms", "resources", "task_state_counts", "task_count", "completed_count", "pending_reason")
    class PortsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class TaskStateCountsEntry(_message.Message):
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
    TASKS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SUBMITTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    TASK_STATE_COUNTS_FIELD_NUMBER: _ClassVar[int]
    TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_COUNT_FIELD_NUMBER: _ClassVar[int]
    PENDING_REASON_FIELD_NUMBER: _ClassVar[int]
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
    tasks: _containers.RepeatedCompositeFieldContainer[TaskStatus]
    name: str
    submitted_at_ms: int
    resources: ResourceSpecProto
    task_state_counts: _containers.ScalarMap[str, int]
    task_count: int
    completed_count: int
    pending_reason: str
    def __init__(self, job_id: _Optional[str] = ..., state: _Optional[_Union[JobState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at_ms: _Optional[int] = ..., finished_at_ms: _Optional[int] = ..., ports: _Optional[_Mapping[str, int]] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., status_message: _Optional[str] = ..., build_metrics: _Optional[_Union[BuildMetrics, _Mapping]] = ..., serialized_result: _Optional[bytes] = ..., parent_job_id: _Optional[str] = ..., failure_count: _Optional[int] = ..., preemption_count: _Optional[int] = ..., tasks: _Optional[_Iterable[_Union[TaskStatus, _Mapping]]] = ..., name: _Optional[str] = ..., submitted_at_ms: _Optional[int] = ..., resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ..., task_state_counts: _Optional[_Mapping[str, int]] = ..., task_count: _Optional[int] = ..., completed_count: _Optional[int] = ..., pending_reason: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("variant", "topology", "count")
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    variant: str
    topology: str
    count: int
    def __init__(self, variant: _Optional[str] = ..., topology: _Optional[str] = ..., count: _Optional[int] = ...) -> None: ...

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
    __slots__ = ("pip_packages", "env_vars", "extras", "python_version")
    class EnvVarsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PIP_PACKAGES_FIELD_NUMBER: _ClassVar[int]
    ENV_VARS_FIELD_NUMBER: _ClassVar[int]
    EXTRAS_FIELD_NUMBER: _ClassVar[int]
    PYTHON_VERSION_FIELD_NUMBER: _ClassVar[int]
    pip_packages: _containers.RepeatedScalarFieldContainer[str]
    env_vars: _containers.ScalarMap[str, str]
    extras: _containers.RepeatedScalarFieldContainer[str]
    python_version: str
    def __init__(self, pip_packages: _Optional[_Iterable[str]] = ..., env_vars: _Optional[_Mapping[str, str]] = ..., extras: _Optional[_Iterable[str]] = ..., python_version: _Optional[str] = ...) -> None: ...

class Entrypoint(_message.Message):
    __slots__ = ("callable", "command")
    CALLABLE_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    callable: bytes
    command: CommandEntrypoint
    def __init__(self, callable: _Optional[bytes] = ..., command: _Optional[_Union[CommandEntrypoint, _Mapping]] = ...) -> None: ...

class CommandEntrypoint(_message.Message):
    __slots__ = ("argv",)
    ARGV_FIELD_NUMBER: _ClassVar[int]
    argv: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, argv: _Optional[_Iterable[str]] = ...) -> None: ...

class AttributeValue(_message.Message):
    __slots__ = ("string_value", "int_value", "float_value")
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    string_value: str
    int_value: int
    float_value: float
    def __init__(self, string_value: _Optional[str] = ..., int_value: _Optional[int] = ..., float_value: _Optional[float] = ...) -> None: ...

class Constraint(_message.Message):
    __slots__ = ("key", "op", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    op: ConstraintOp
    value: AttributeValue
    def __init__(self, key: _Optional[str] = ..., op: _Optional[_Union[ConstraintOp, str]] = ..., value: _Optional[_Union[AttributeValue, _Mapping]] = ...) -> None: ...

class CoschedulingConfig(_message.Message):
    __slots__ = ("group_by",)
    GROUP_BY_FIELD_NUMBER: _ClassVar[int]
    group_by: str
    def __init__(self, group_by: _Optional[str] = ...) -> None: ...

class WorkerMetadata(_message.Message):
    __slots__ = ("hostname", "ip_address", "cpu_count", "memory_bytes", "disk_bytes", "device", "tpu_name", "tpu_worker_hostnames", "tpu_worker_id", "tpu_chips_per_host_bounds", "gpu_count", "gpu_name", "gpu_memory_mb", "gce_instance_name", "gce_zone", "attributes", "vm_address")
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
    VM_ADDRESS_FIELD_NUMBER: _ClassVar[int]
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
    vm_address: str
    def __init__(self, hostname: _Optional[str] = ..., ip_address: _Optional[str] = ..., cpu_count: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., device: _Optional[_Union[DeviceConfig, _Mapping]] = ..., tpu_name: _Optional[str] = ..., tpu_worker_hostnames: _Optional[str] = ..., tpu_worker_id: _Optional[str] = ..., tpu_chips_per_host_bounds: _Optional[str] = ..., gpu_count: _Optional[int] = ..., gpu_name: _Optional[str] = ..., gpu_memory_mb: _Optional[int] = ..., gce_instance_name: _Optional[str] = ..., gce_zone: _Optional[str] = ..., attributes: _Optional[_Mapping[str, AttributeValue]] = ..., vm_address: _Optional[str] = ...) -> None: ...

class Controller(_message.Message):
    __slots__ = ()
    class LaunchJobRequest(_message.Message):
        __slots__ = ("name", "entrypoint", "resources", "environment", "bundle_gcs_path", "bundle_hash", "bundle_blob", "scheduling_timeout_seconds", "ports", "parent_job_id", "max_task_failures", "max_retries_failure", "max_retries_preemption", "constraints", "coscheduling")
        NAME_FIELD_NUMBER: _ClassVar[int]
        ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
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
        CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
        COSCHEDULING_FIELD_NUMBER: _ClassVar[int]
        name: str
        entrypoint: Entrypoint
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
        constraints: _containers.RepeatedCompositeFieldContainer[Constraint]
        coscheduling: CoschedulingConfig
        def __init__(self, name: _Optional[str] = ..., entrypoint: _Optional[_Union[Entrypoint, _Mapping]] = ..., resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ..., environment: _Optional[_Union[EnvironmentConfig, _Mapping]] = ..., bundle_gcs_path: _Optional[str] = ..., bundle_hash: _Optional[str] = ..., bundle_blob: _Optional[bytes] = ..., scheduling_timeout_seconds: _Optional[int] = ..., ports: _Optional[_Iterable[str]] = ..., parent_job_id: _Optional[str] = ..., max_task_failures: _Optional[int] = ..., max_retries_failure: _Optional[int] = ..., max_retries_preemption: _Optional[int] = ..., constraints: _Optional[_Iterable[_Union[Constraint, _Mapping]]] = ..., coscheduling: _Optional[_Union[CoschedulingConfig, _Mapping]] = ...) -> None: ...
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
        __slots__ = ("worker_id", "address", "metadata", "running_task_ids")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        RUNNING_TASK_IDS_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        address: str
        metadata: WorkerMetadata
        running_task_ids: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, worker_id: _Optional[str] = ..., address: _Optional[str] = ..., metadata: _Optional[_Union[WorkerMetadata, _Mapping]] = ..., running_task_ids: _Optional[_Iterable[str]] = ...) -> None: ...
    class RegisterWorkerResponse(_message.Message):
        __slots__ = ("accepted", "controller_address", "should_reset")
        ACCEPTED_FIELD_NUMBER: _ClassVar[int]
        CONTROLLER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
        SHOULD_RESET_FIELD_NUMBER: _ClassVar[int]
        accepted: bool
        controller_address: str
        should_reset: bool
        def __init__(self, accepted: _Optional[bool] = ..., controller_address: _Optional[str] = ..., should_reset: _Optional[bool] = ...) -> None: ...
    class WorkerHealthStatus(_message.Message):
        __slots__ = ("worker_id", "healthy", "consecutive_failures", "last_heartbeat_ms", "running_job_ids", "address", "metadata", "status_message")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        HEALTHY_FIELD_NUMBER: _ClassVar[int]
        CONSECUTIVE_FAILURES_FIELD_NUMBER: _ClassVar[int]
        LAST_HEARTBEAT_MS_FIELD_NUMBER: _ClassVar[int]
        RUNNING_JOB_IDS_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        healthy: bool
        consecutive_failures: int
        last_heartbeat_ms: int
        running_job_ids: _containers.RepeatedScalarFieldContainer[str]
        address: str
        metadata: WorkerMetadata
        status_message: str
        def __init__(self, worker_id: _Optional[str] = ..., healthy: _Optional[bool] = ..., consecutive_failures: _Optional[int] = ..., last_heartbeat_ms: _Optional[int] = ..., running_job_ids: _Optional[_Iterable[str]] = ..., address: _Optional[str] = ..., metadata: _Optional[_Union[WorkerMetadata, _Mapping]] = ..., status_message: _Optional[str] = ...) -> None: ...
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
    class GetAutoscalerStatusRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class GetAutoscalerStatusResponse(_message.Message):
        __slots__ = ("status",)
        STATUS_FIELD_NUMBER: _ClassVar[int]
        status: _vm_pb2.AutoscalerStatus
        def __init__(self, status: _Optional[_Union[_vm_pb2.AutoscalerStatus, _Mapping]] = ...) -> None: ...
    class GetVmLogsRequest(_message.Message):
        __slots__ = ("vm_id", "tail")
        VM_ID_FIELD_NUMBER: _ClassVar[int]
        TAIL_FIELD_NUMBER: _ClassVar[int]
        vm_id: str
        tail: int
        def __init__(self, vm_id: _Optional[str] = ..., tail: _Optional[int] = ...) -> None: ...
    class GetVmLogsResponse(_message.Message):
        __slots__ = ("logs", "vm_id", "state")
        LOGS_FIELD_NUMBER: _ClassVar[int]
        VM_ID_FIELD_NUMBER: _ClassVar[int]
        STATE_FIELD_NUMBER: _ClassVar[int]
        logs: str
        vm_id: str
        state: _vm_pb2.VmState
        def __init__(self, logs: _Optional[str] = ..., vm_id: _Optional[str] = ..., state: _Optional[_Union[_vm_pb2.VmState, str]] = ...) -> None: ...
    class TransactionAction(_message.Message):
        __slots__ = ("timestamp_ms", "action", "entity_id", "details")
        TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
        ACTION_FIELD_NUMBER: _ClassVar[int]
        ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
        DETAILS_FIELD_NUMBER: _ClassVar[int]
        timestamp_ms: int
        action: str
        entity_id: str
        details: str
        def __init__(self, timestamp_ms: _Optional[int] = ..., action: _Optional[str] = ..., entity_id: _Optional[str] = ..., details: _Optional[str] = ...) -> None: ...
    class GetTransactionsRequest(_message.Message):
        __slots__ = ("limit",)
        LIMIT_FIELD_NUMBER: _ClassVar[int]
        limit: int
        def __init__(self, limit: _Optional[int] = ...) -> None: ...
    class GetTransactionsResponse(_message.Message):
        __slots__ = ("actions",)
        ACTIONS_FIELD_NUMBER: _ClassVar[int]
        actions: _containers.RepeatedCompositeFieldContainer[Controller.TransactionAction]
        def __init__(self, actions: _Optional[_Iterable[_Union[Controller.TransactionAction, _Mapping]]] = ...) -> None: ...
    class GetTaskLogsRequest(_message.Message):
        __slots__ = ("job_id", "task_index", "start_ms", "limit")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        TASK_INDEX_FIELD_NUMBER: _ClassVar[int]
        START_MS_FIELD_NUMBER: _ClassVar[int]
        LIMIT_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        task_index: int
        start_ms: int
        limit: int
        def __init__(self, job_id: _Optional[str] = ..., task_index: _Optional[int] = ..., start_ms: _Optional[int] = ..., limit: _Optional[int] = ...) -> None: ...
    class GetTaskLogsResponse(_message.Message):
        __slots__ = ("logs", "worker_address")
        LOGS_FIELD_NUMBER: _ClassVar[int]
        WORKER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
        logs: _containers.RepeatedCompositeFieldContainer[Worker.LogEntry]
        worker_address: str
        def __init__(self, logs: _Optional[_Iterable[_Union[Worker.LogEntry, _Mapping]]] = ..., worker_address: _Optional[str] = ...) -> None: ...
    class GetControllerLogsRequest(_message.Message):
        __slots__ = ("prefix", "limit")
        PREFIX_FIELD_NUMBER: _ClassVar[int]
        LIMIT_FIELD_NUMBER: _ClassVar[int]
        prefix: str
        limit: int
        def __init__(self, prefix: _Optional[str] = ..., limit: _Optional[int] = ...) -> None: ...
    class ControllerLogRecord(_message.Message):
        __slots__ = ("timestamp", "level", "logger_name", "message")
        TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
        LEVEL_FIELD_NUMBER: _ClassVar[int]
        LOGGER_NAME_FIELD_NUMBER: _ClassVar[int]
        MESSAGE_FIELD_NUMBER: _ClassVar[int]
        timestamp: float
        level: str
        logger_name: str
        message: str
        def __init__(self, timestamp: _Optional[float] = ..., level: _Optional[str] = ..., logger_name: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...
    class GetControllerLogsResponse(_message.Message):
        __slots__ = ("records",)
        RECORDS_FIELD_NUMBER: _ClassVar[int]
        records: _containers.RepeatedCompositeFieldContainer[Controller.ControllerLogRecord]
        def __init__(self, records: _Optional[_Iterable[_Union[Controller.ControllerLogRecord, _Mapping]]] = ...) -> None: ...
    def __init__(self) -> None: ...

class Worker(_message.Message):
    __slots__ = ()
    class RunTaskRequest(_message.Message):
        __slots__ = ("job_id", "task_id", "task_index", "num_tasks", "entrypoint", "environment", "bundle_gcs_path", "resources", "timeout_seconds", "ports", "attempt_id")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        TASK_INDEX_FIELD_NUMBER: _ClassVar[int]
        NUM_TASKS_FIELD_NUMBER: _ClassVar[int]
        ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
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
        entrypoint: Entrypoint
        environment: EnvironmentConfig
        bundle_gcs_path: str
        resources: ResourceSpecProto
        timeout_seconds: int
        ports: _containers.RepeatedScalarFieldContainer[str]
        attempt_id: int
        def __init__(self, job_id: _Optional[str] = ..., task_id: _Optional[str] = ..., task_index: _Optional[int] = ..., num_tasks: _Optional[int] = ..., entrypoint: _Optional[_Union[Entrypoint, _Mapping]] = ..., environment: _Optional[_Union[EnvironmentConfig, _Mapping]] = ..., bundle_gcs_path: _Optional[str] = ..., resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ..., timeout_seconds: _Optional[int] = ..., ports: _Optional[_Iterable[str]] = ..., attempt_id: _Optional[int] = ...) -> None: ...
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
