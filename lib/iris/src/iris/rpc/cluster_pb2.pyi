import vm_pb2 as _vm_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar

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
    __slots__ = (
        "attempts",
        "build_metrics",
        "can_be_scheduled",
        "current_attempt_id",
        "error",
        "exit_code",
        "finished_at_ms",
        "job_id",
        "pending_reason",
        "ports",
        "resource_usage",
        "started_at_ms",
        "state",
        "task_id",
        "task_index",
        "worker_address",
        "worker_id",
    )
    class PortsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: str | None = ..., value: int | None = ...) -> None: ...

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
    def __init__(
        self,
        task_id: str | None = ...,
        job_id: str | None = ...,
        task_index: int | None = ...,
        state: TaskState | str | None = ...,
        worker_id: str | None = ...,
        worker_address: str | None = ...,
        exit_code: int | None = ...,
        error: str | None = ...,
        started_at_ms: int | None = ...,
        finished_at_ms: int | None = ...,
        ports: _Mapping[str, int] | None = ...,
        resource_usage: ResourceUsage | _Mapping | None = ...,
        build_metrics: BuildMetrics | _Mapping | None = ...,
        current_attempt_id: int | None = ...,
        attempts: _Iterable[TaskAttempt | _Mapping] | None = ...,
        pending_reason: str | None = ...,
        can_be_scheduled: bool | None = ...,
    ) -> None: ...

class TaskAttempt(_message.Message):
    __slots__ = (
        "attempt_id",
        "error",
        "exit_code",
        "finished_at_ms",
        "is_worker_failure",
        "started_at_ms",
        "state",
        "worker_id",
    )
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
    def __init__(
        self,
        attempt_id: int | None = ...,
        worker_id: str | None = ...,
        state: TaskState | str | None = ...,
        exit_code: int | None = ...,
        error: str | None = ...,
        started_at_ms: int | None = ...,
        finished_at_ms: int | None = ...,
        is_worker_failure: bool | None = ...,
    ) -> None: ...

class ResourceUsage(_message.Message):
    __slots__ = ("cpu_millicores", "cpu_percent", "disk_mb", "memory_mb", "memory_peak_mb", "process_count")
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
    def __init__(
        self,
        memory_mb: int | None = ...,
        disk_mb: int | None = ...,
        cpu_millicores: int | None = ...,
        memory_peak_mb: int | None = ...,
        cpu_percent: int | None = ...,
        process_count: int | None = ...,
    ) -> None: ...

class BuildMetrics(_message.Message):
    __slots__ = ("build_finished_ms", "build_started_ms", "from_cache", "image_tag")
    BUILD_STARTED_MS_FIELD_NUMBER: _ClassVar[int]
    BUILD_FINISHED_MS_FIELD_NUMBER: _ClassVar[int]
    FROM_CACHE_FIELD_NUMBER: _ClassVar[int]
    IMAGE_TAG_FIELD_NUMBER: _ClassVar[int]
    build_started_ms: int
    build_finished_ms: int
    from_cache: bool
    image_tag: str
    def __init__(
        self,
        build_started_ms: int | None = ...,
        build_finished_ms: int | None = ...,
        from_cache: bool | None = ...,
        image_tag: str | None = ...,
    ) -> None: ...

class JobStatus(_message.Message):
    __slots__ = (
        "build_metrics",
        "completed_count",
        "error",
        "exit_code",
        "failure_count",
        "finished_at_ms",
        "job_id",
        "name",
        "parent_job_id",
        "ports",
        "preemption_count",
        "resource_usage",
        "resources",
        "serialized_result",
        "started_at_ms",
        "state",
        "status_message",
        "submitted_at_ms",
        "task_count",
        "task_state_counts",
        "tasks",
    )
    class PortsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: str | None = ..., value: int | None = ...) -> None: ...

    class TaskStateCountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: str | None = ..., value: int | None = ...) -> None: ...

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
    def __init__(
        self,
        job_id: str | None = ...,
        state: JobState | str | None = ...,
        exit_code: int | None = ...,
        error: str | None = ...,
        started_at_ms: int | None = ...,
        finished_at_ms: int | None = ...,
        ports: _Mapping[str, int] | None = ...,
        resource_usage: ResourceUsage | _Mapping | None = ...,
        status_message: str | None = ...,
        build_metrics: BuildMetrics | _Mapping | None = ...,
        serialized_result: bytes | None = ...,
        parent_job_id: str | None = ...,
        failure_count: int | None = ...,
        preemption_count: int | None = ...,
        tasks: _Iterable[TaskStatus | _Mapping] | None = ...,
        name: str | None = ...,
        submitted_at_ms: int | None = ...,
        resources: ResourceSpecProto | _Mapping | None = ...,
        task_state_counts: _Mapping[str, int] | None = ...,
        task_count: int | None = ...,
        completed_count: int | None = ...,
    ) -> None: ...

class DeviceConfig(_message.Message):
    __slots__ = ("cpu", "gpu", "tpu")
    CPU_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    TPU_FIELD_NUMBER: _ClassVar[int]
    cpu: CpuDevice
    gpu: GpuDevice
    tpu: TpuDevice
    def __init__(
        self,
        cpu: CpuDevice | _Mapping | None = ...,
        gpu: GpuDevice | _Mapping | None = ...,
        tpu: TpuDevice | _Mapping | None = ...,
    ) -> None: ...

class CpuDevice(_message.Message):
    __slots__ = ("variant",)
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    variant: str
    def __init__(self, variant: str | None = ...) -> None: ...

class GpuDevice(_message.Message):
    __slots__ = ("count", "variant")
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    variant: str
    count: int
    def __init__(self, variant: str | None = ..., count: int | None = ...) -> None: ...

class TpuDevice(_message.Message):
    __slots__ = ("count", "topology", "variant")
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    variant: str
    topology: str
    count: int
    def __init__(self, variant: str | None = ..., topology: str | None = ..., count: int | None = ...) -> None: ...

class ResourceSpecProto(_message.Message):
    __slots__ = ("cpu", "device", "disk_bytes", "memory_bytes", "preemptible", "regions", "replicas")
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
    def __init__(
        self,
        cpu: int | None = ...,
        memory_bytes: int | None = ...,
        disk_bytes: int | None = ...,
        device: DeviceConfig | _Mapping | None = ...,
        replicas: int | None = ...,
        preemptible: bool | None = ...,
        regions: _Iterable[str] | None = ...,
    ) -> None: ...

class EnvironmentConfig(_message.Message):
    __slots__ = ("env_vars", "extras", "pip_packages", "workspace")
    class EnvVarsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...

    WORKSPACE_FIELD_NUMBER: _ClassVar[int]
    PIP_PACKAGES_FIELD_NUMBER: _ClassVar[int]
    ENV_VARS_FIELD_NUMBER: _ClassVar[int]
    EXTRAS_FIELD_NUMBER: _ClassVar[int]
    workspace: str
    pip_packages: _containers.RepeatedScalarFieldContainer[str]
    env_vars: _containers.ScalarMap[str, str]
    extras: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        workspace: str | None = ...,
        pip_packages: _Iterable[str] | None = ...,
        env_vars: _Mapping[str, str] | None = ...,
        extras: _Iterable[str] | None = ...,
    ) -> None: ...

class AttributeValue(_message.Message):
    __slots__ = ("float_value", "int_value", "string_value")
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    string_value: str
    int_value: int
    float_value: float
    def __init__(
        self, string_value: str | None = ..., int_value: int | None = ..., float_value: float | None = ...
    ) -> None: ...

class Constraint(_message.Message):
    __slots__ = ("key", "op", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    op: ConstraintOp
    value: AttributeValue
    def __init__(
        self, key: str | None = ..., op: ConstraintOp | str | None = ..., value: AttributeValue | _Mapping | None = ...
    ) -> None: ...

class CoschedulingConfig(_message.Message):
    __slots__ = ("group_by",)
    GROUP_BY_FIELD_NUMBER: _ClassVar[int]
    group_by: str
    def __init__(self, group_by: str | None = ...) -> None: ...

class WorkerMetadata(_message.Message):
    __slots__ = (
        "attributes",
        "cpu_count",
        "device",
        "disk_bytes",
        "gce_instance_name",
        "gce_zone",
        "gpu_count",
        "gpu_memory_mb",
        "gpu_name",
        "hostname",
        "ip_address",
        "memory_bytes",
        "tpu_chips_per_host_bounds",
        "tpu_name",
        "tpu_worker_hostnames",
        "tpu_worker_id",
    )
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: AttributeValue
        def __init__(self, key: str | None = ..., value: AttributeValue | _Mapping | None = ...) -> None: ...

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
    def __init__(
        self,
        hostname: str | None = ...,
        ip_address: str | None = ...,
        cpu_count: int | None = ...,
        memory_bytes: int | None = ...,
        disk_bytes: int | None = ...,
        device: DeviceConfig | _Mapping | None = ...,
        tpu_name: str | None = ...,
        tpu_worker_hostnames: str | None = ...,
        tpu_worker_id: str | None = ...,
        tpu_chips_per_host_bounds: str | None = ...,
        gpu_count: int | None = ...,
        gpu_name: str | None = ...,
        gpu_memory_mb: int | None = ...,
        gce_instance_name: str | None = ...,
        gce_zone: str | None = ...,
        attributes: _Mapping[str, AttributeValue] | None = ...,
    ) -> None: ...

class Controller(_message.Message):
    __slots__ = ()
    class LaunchJobRequest(_message.Message):
        __slots__ = (
            "bundle_blob",
            "bundle_gcs_path",
            "bundle_hash",
            "constraints",
            "coscheduling",
            "environment",
            "max_retries_failure",
            "max_retries_preemption",
            "max_task_failures",
            "name",
            "parent_job_id",
            "ports",
            "resources",
            "scheduling_timeout_seconds",
            "serialized_entrypoint",
        )
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
        CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
        COSCHEDULING_FIELD_NUMBER: _ClassVar[int]
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
        constraints: _containers.RepeatedCompositeFieldContainer[Constraint]
        coscheduling: CoschedulingConfig
        def __init__(
            self,
            name: str | None = ...,
            serialized_entrypoint: bytes | None = ...,
            resources: ResourceSpecProto | _Mapping | None = ...,
            environment: EnvironmentConfig | _Mapping | None = ...,
            bundle_gcs_path: str | None = ...,
            bundle_hash: str | None = ...,
            bundle_blob: bytes | None = ...,
            scheduling_timeout_seconds: int | None = ...,
            ports: _Iterable[str] | None = ...,
            parent_job_id: str | None = ...,
            max_task_failures: int | None = ...,
            max_retries_failure: int | None = ...,
            max_retries_preemption: int | None = ...,
            constraints: _Iterable[Constraint | _Mapping] | None = ...,
            coscheduling: CoschedulingConfig | _Mapping | None = ...,
        ) -> None: ...

    class LaunchJobResponse(_message.Message):
        __slots__ = ("job_id",)
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        def __init__(self, job_id: str | None = ...) -> None: ...

    class GetJobStatusRequest(_message.Message):
        __slots__ = ("include_result", "job_id")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        INCLUDE_RESULT_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        include_result: bool
        def __init__(self, job_id: str | None = ..., include_result: bool | None = ...) -> None: ...

    class GetJobStatusResponse(_message.Message):
        __slots__ = ("job",)
        JOB_FIELD_NUMBER: _ClassVar[int]
        job: JobStatus
        def __init__(self, job: JobStatus | _Mapping | None = ...) -> None: ...

    class TerminateJobRequest(_message.Message):
        __slots__ = ("job_id",)
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        def __init__(self, job_id: str | None = ...) -> None: ...

    class ListJobsRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...

    class ListJobsResponse(_message.Message):
        __slots__ = ("jobs",)
        JOBS_FIELD_NUMBER: _ClassVar[int]
        jobs: _containers.RepeatedCompositeFieldContainer[JobStatus]
        def __init__(self, jobs: _Iterable[JobStatus | _Mapping] | None = ...) -> None: ...

    class GetTaskStatusRequest(_message.Message):
        __slots__ = ("job_id", "task_index")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        TASK_INDEX_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        task_index: int
        def __init__(self, job_id: str | None = ..., task_index: int | None = ...) -> None: ...

    class GetTaskStatusResponse(_message.Message):
        __slots__ = ("task",)
        TASK_FIELD_NUMBER: _ClassVar[int]
        task: TaskStatus
        def __init__(self, task: TaskStatus | _Mapping | None = ...) -> None: ...

    class ListTasksRequest(_message.Message):
        __slots__ = ("job_id",)
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        def __init__(self, job_id: str | None = ...) -> None: ...

    class ListTasksResponse(_message.Message):
        __slots__ = ("tasks",)
        TASKS_FIELD_NUMBER: _ClassVar[int]
        tasks: _containers.RepeatedCompositeFieldContainer[TaskStatus]
        def __init__(self, tasks: _Iterable[TaskStatus | _Mapping] | None = ...) -> None: ...

    class WorkerInfo(_message.Message):
        __slots__ = ("address", "metadata", "registered_at_ms", "worker_id")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        REGISTERED_AT_MS_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        address: str
        metadata: WorkerMetadata
        registered_at_ms: int
        def __init__(
            self,
            worker_id: str | None = ...,
            address: str | None = ...,
            metadata: WorkerMetadata | _Mapping | None = ...,
            registered_at_ms: int | None = ...,
        ) -> None: ...

    class RegisterWorkerRequest(_message.Message):
        __slots__ = ("address", "metadata", "worker_id")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        address: str
        metadata: WorkerMetadata
        def __init__(
            self,
            worker_id: str | None = ...,
            address: str | None = ...,
            metadata: WorkerMetadata | _Mapping | None = ...,
        ) -> None: ...

    class RegisterWorkerResponse(_message.Message):
        __slots__ = ("accepted", "controller_address")
        ACCEPTED_FIELD_NUMBER: _ClassVar[int]
        CONTROLLER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
        accepted: bool
        controller_address: str
        def __init__(self, accepted: bool | None = ..., controller_address: str | None = ...) -> None: ...

    class WorkerHealthStatus(_message.Message):
        __slots__ = (
            "address",
            "consecutive_failures",
            "healthy",
            "last_heartbeat_ms",
            "metadata",
            "running_job_ids",
            "worker_id",
        )
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        HEALTHY_FIELD_NUMBER: _ClassVar[int]
        CONSECUTIVE_FAILURES_FIELD_NUMBER: _ClassVar[int]
        LAST_HEARTBEAT_MS_FIELD_NUMBER: _ClassVar[int]
        RUNNING_JOB_IDS_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        healthy: bool
        consecutive_failures: int
        last_heartbeat_ms: int
        running_job_ids: _containers.RepeatedScalarFieldContainer[str]
        address: str
        metadata: WorkerMetadata
        def __init__(
            self,
            worker_id: str | None = ...,
            healthy: bool | None = ...,
            consecutive_failures: int | None = ...,
            last_heartbeat_ms: int | None = ...,
            running_job_ids: _Iterable[str] | None = ...,
            address: str | None = ...,
            metadata: WorkerMetadata | _Mapping | None = ...,
        ) -> None: ...

    class ListWorkersRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...

    class ListWorkersResponse(_message.Message):
        __slots__ = ("workers",)
        WORKERS_FIELD_NUMBER: _ClassVar[int]
        workers: _containers.RepeatedCompositeFieldContainer[Controller.WorkerHealthStatus]
        def __init__(self, workers: _Iterable[Controller.WorkerHealthStatus | _Mapping] | None = ...) -> None: ...

    class ReportTaskStateRequest(_message.Message):
        __slots__ = (
            "attempt_id",
            "error",
            "exit_code",
            "finished_at_ms",
            "job_id",
            "state",
            "task_id",
            "task_index",
            "worker_id",
        )
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
        def __init__(
            self,
            worker_id: str | None = ...,
            task_id: str | None = ...,
            job_id: str | None = ...,
            task_index: int | None = ...,
            state: TaskState | str | None = ...,
            exit_code: int | None = ...,
            error: str | None = ...,
            finished_at_ms: int | None = ...,
            attempt_id: int | None = ...,
        ) -> None: ...

    class ReportTaskStateResponse(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...

    class Endpoint(_message.Message):
        __slots__ = ("address", "endpoint_id", "job_id", "metadata", "name")
        class MetadataEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...

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
        def __init__(
            self,
            endpoint_id: str | None = ...,
            name: str | None = ...,
            address: str | None = ...,
            job_id: str | None = ...,
            metadata: _Mapping[str, str] | None = ...,
        ) -> None: ...

    class RegisterEndpointRequest(_message.Message):
        __slots__ = ("address", "job_id", "metadata", "name")
        class MetadataEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...

        NAME_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        name: str
        address: str
        job_id: str
        metadata: _containers.ScalarMap[str, str]
        def __init__(
            self,
            name: str | None = ...,
            address: str | None = ...,
            job_id: str | None = ...,
            metadata: _Mapping[str, str] | None = ...,
        ) -> None: ...

    class RegisterEndpointResponse(_message.Message):
        __slots__ = ("endpoint_id",)
        ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
        endpoint_id: str
        def __init__(self, endpoint_id: str | None = ...) -> None: ...

    class UnregisterEndpointRequest(_message.Message):
        __slots__ = ("endpoint_id",)
        ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
        endpoint_id: str
        def __init__(self, endpoint_id: str | None = ...) -> None: ...

    class LookupEndpointRequest(_message.Message):
        __slots__ = ("name",)
        NAME_FIELD_NUMBER: _ClassVar[int]
        name: str
        def __init__(self, name: str | None = ...) -> None: ...

    class LookupEndpointResponse(_message.Message):
        __slots__ = ("endpoint",)
        ENDPOINT_FIELD_NUMBER: _ClassVar[int]
        endpoint: Controller.Endpoint
        def __init__(self, endpoint: Controller.Endpoint | _Mapping | None = ...) -> None: ...

    class ListEndpointsRequest(_message.Message):
        __slots__ = ("prefix",)
        PREFIX_FIELD_NUMBER: _ClassVar[int]
        prefix: str
        def __init__(self, prefix: str | None = ...) -> None: ...

    class ListEndpointsResponse(_message.Message):
        __slots__ = ("endpoints",)
        ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
        endpoints: _containers.RepeatedCompositeFieldContainer[Controller.Endpoint]
        def __init__(self, endpoints: _Iterable[Controller.Endpoint | _Mapping] | None = ...) -> None: ...

    class GetAutoscalerStatusRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...

    class GetAutoscalerStatusResponse(_message.Message):
        __slots__ = ("status",)
        STATUS_FIELD_NUMBER: _ClassVar[int]
        status: _vm_pb2.AutoscalerStatus
        def __init__(self, status: _vm_pb2.AutoscalerStatus | _Mapping | None = ...) -> None: ...

    class GetVmLogsRequest(_message.Message):
        __slots__ = ("tail", "vm_id")
        VM_ID_FIELD_NUMBER: _ClassVar[int]
        TAIL_FIELD_NUMBER: _ClassVar[int]
        vm_id: str
        tail: int
        def __init__(self, vm_id: str | None = ..., tail: int | None = ...) -> None: ...

    class GetVmLogsResponse(_message.Message):
        __slots__ = ("logs", "state", "vm_id")
        LOGS_FIELD_NUMBER: _ClassVar[int]
        VM_ID_FIELD_NUMBER: _ClassVar[int]
        STATE_FIELD_NUMBER: _ClassVar[int]
        logs: str
        vm_id: str
        state: _vm_pb2.VmState
        def __init__(
            self, logs: str | None = ..., vm_id: str | None = ..., state: _vm_pb2.VmState | str | None = ...
        ) -> None: ...

    class TransactionAction(_message.Message):
        __slots__ = ("action", "details", "entity_id", "timestamp_ms")
        TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
        ACTION_FIELD_NUMBER: _ClassVar[int]
        ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
        DETAILS_FIELD_NUMBER: _ClassVar[int]
        timestamp_ms: int
        action: str
        entity_id: str
        details: str
        def __init__(
            self,
            timestamp_ms: int | None = ...,
            action: str | None = ...,
            entity_id: str | None = ...,
            details: str | None = ...,
        ) -> None: ...

    class GetTransactionsRequest(_message.Message):
        __slots__ = ("limit",)
        LIMIT_FIELD_NUMBER: _ClassVar[int]
        limit: int
        def __init__(self, limit: int | None = ...) -> None: ...

    class GetTransactionsResponse(_message.Message):
        __slots__ = ("actions",)
        ACTIONS_FIELD_NUMBER: _ClassVar[int]
        actions: _containers.RepeatedCompositeFieldContainer[Controller.TransactionAction]
        def __init__(self, actions: _Iterable[Controller.TransactionAction | _Mapping] | None = ...) -> None: ...

    def __init__(self) -> None: ...

class Worker(_message.Message):
    __slots__ = ()
    class RunTaskRequest(_message.Message):
        __slots__ = (
            "attempt_id",
            "bundle_gcs_path",
            "environment",
            "job_id",
            "num_tasks",
            "ports",
            "resources",
            "serialized_entrypoint",
            "task_id",
            "task_index",
            "timeout_seconds",
        )
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
        def __init__(
            self,
            job_id: str | None = ...,
            task_id: str | None = ...,
            task_index: int | None = ...,
            num_tasks: int | None = ...,
            serialized_entrypoint: bytes | None = ...,
            environment: EnvironmentConfig | _Mapping | None = ...,
            bundle_gcs_path: str | None = ...,
            resources: ResourceSpecProto | _Mapping | None = ...,
            timeout_seconds: int | None = ...,
            ports: _Iterable[str] | None = ...,
            attempt_id: int | None = ...,
        ) -> None: ...

    class RunTaskResponse(_message.Message):
        __slots__ = ("state", "task_id")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        STATE_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        state: TaskState
        def __init__(self, task_id: str | None = ..., state: TaskState | str | None = ...) -> None: ...

    class GetTaskStatusRequest(_message.Message):
        __slots__ = ("include_result", "task_id")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        INCLUDE_RESULT_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        include_result: bool
        def __init__(self, task_id: str | None = ..., include_result: bool | None = ...) -> None: ...

    class ListTasksRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...

    class ListTasksResponse(_message.Message):
        __slots__ = ("tasks",)
        TASKS_FIELD_NUMBER: _ClassVar[int]
        tasks: _containers.RepeatedCompositeFieldContainer[TaskStatus]
        def __init__(self, tasks: _Iterable[TaskStatus | _Mapping] | None = ...) -> None: ...

    class LogEntry(_message.Message):
        __slots__ = ("data", "source", "timestamp_ms")
        TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
        SOURCE_FIELD_NUMBER: _ClassVar[int]
        DATA_FIELD_NUMBER: _ClassVar[int]
        timestamp_ms: int
        source: str
        data: str
        def __init__(self, timestamp_ms: int | None = ..., source: str | None = ..., data: str | None = ...) -> None: ...

    class FetchLogsFilter(_message.Message):
        __slots__ = ("end_ms", "max_lines", "regex", "start_line", "start_ms")
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
        def __init__(
            self,
            regex: str | None = ...,
            start_line: int | None = ...,
            start_ms: int | None = ...,
            end_ms: int | None = ...,
            max_lines: int | None = ...,
        ) -> None: ...

    class FetchTaskLogsRequest(_message.Message):
        __slots__ = ("filter", "task_id")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        FILTER_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        filter: Worker.FetchLogsFilter
        def __init__(
            self, task_id: str | None = ..., filter: Worker.FetchLogsFilter | _Mapping | None = ...
        ) -> None: ...

    class FetchTaskLogsResponse(_message.Message):
        __slots__ = ("logs",)
        LOGS_FIELD_NUMBER: _ClassVar[int]
        logs: _containers.RepeatedCompositeFieldContainer[Worker.LogEntry]
        def __init__(self, logs: _Iterable[Worker.LogEntry | _Mapping] | None = ...) -> None: ...

    class KillTaskRequest(_message.Message):
        __slots__ = ("task_id", "term_timeout_ms")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        TERM_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        term_timeout_ms: int
        def __init__(self, task_id: str | None = ..., term_timeout_ms: int | None = ...) -> None: ...

    class HealthResponse(_message.Message):
        __slots__ = ("healthy", "running_tasks", "uptime_ms")
        HEALTHY_FIELD_NUMBER: _ClassVar[int]
        UPTIME_MS_FIELD_NUMBER: _ClassVar[int]
        RUNNING_TASKS_FIELD_NUMBER: _ClassVar[int]
        healthy: bool
        uptime_ms: int
        running_tasks: int
        def __init__(
            self, healthy: bool | None = ..., uptime_ms: int | None = ..., running_tasks: int | None = ...
        ) -> None: ...

    def __init__(self) -> None: ...
