from . import iris_logging_pb2 as _iris_logging_pb2
from . import iris_time_pb2 as _iris_time_pb2
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
    TASK_STATE_ASSIGNED: _ClassVar[TaskState]
    TASK_STATE_PREEMPTED: _ClassVar[TaskState]

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
    CONSTRAINT_OP_IN: _ClassVar[ConstraintOp]

class ConstraintMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONSTRAINT_MODE_REQUIRED: _ClassVar[ConstraintMode]
    CONSTRAINT_MODE_PREFERRED: _ClassVar[ConstraintMode]

class JobPreemptionPolicy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_PREEMPTION_POLICY_UNSPECIFIED: _ClassVar[JobPreemptionPolicy]
    JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN: _ClassVar[JobPreemptionPolicy]
    JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN: _ClassVar[JobPreemptionPolicy]

class ExistingJobPolicy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXISTING_JOB_POLICY_UNSPECIFIED: _ClassVar[ExistingJobPolicy]
    EXISTING_JOB_POLICY_ERROR: _ClassVar[ExistingJobPolicy]
    EXISTING_JOB_POLICY_KEEP: _ClassVar[ExistingJobPolicy]
    EXISTING_JOB_POLICY_RECREATE: _ClassVar[ExistingJobPolicy]

class PriorityBand(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PRIORITY_BAND_UNSPECIFIED: _ClassVar[PriorityBand]
    PRIORITY_BAND_PRODUCTION: _ClassVar[PriorityBand]
    PRIORITY_BAND_INTERACTIVE: _ClassVar[PriorityBand]
    PRIORITY_BAND_BATCH: _ClassVar[PriorityBand]
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
TASK_STATE_ASSIGNED: TaskState
TASK_STATE_PREEMPTED: TaskState
CONSTRAINT_OP_EQ: ConstraintOp
CONSTRAINT_OP_NE: ConstraintOp
CONSTRAINT_OP_EXISTS: ConstraintOp
CONSTRAINT_OP_NOT_EXISTS: ConstraintOp
CONSTRAINT_OP_GT: ConstraintOp
CONSTRAINT_OP_GE: ConstraintOp
CONSTRAINT_OP_LT: ConstraintOp
CONSTRAINT_OP_LE: ConstraintOp
CONSTRAINT_OP_IN: ConstraintOp
CONSTRAINT_MODE_REQUIRED: ConstraintMode
CONSTRAINT_MODE_PREFERRED: ConstraintMode
JOB_PREEMPTION_POLICY_UNSPECIFIED: JobPreemptionPolicy
JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN: JobPreemptionPolicy
JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN: JobPreemptionPolicy
EXISTING_JOB_POLICY_UNSPECIFIED: ExistingJobPolicy
EXISTING_JOB_POLICY_ERROR: ExistingJobPolicy
EXISTING_JOB_POLICY_KEEP: ExistingJobPolicy
EXISTING_JOB_POLICY_RECREATE: ExistingJobPolicy
PRIORITY_BAND_UNSPECIFIED: PriorityBand
PRIORITY_BAND_PRODUCTION: PriorityBand
PRIORITY_BAND_INTERACTIVE: PriorityBand
PRIORITY_BAND_BATCH: PriorityBand

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class LoginRequest(_message.Message):
    __slots__ = ("identity_token",)
    IDENTITY_TOKEN_FIELD_NUMBER: _ClassVar[int]
    identity_token: str
    def __init__(self, identity_token: _Optional[str] = ...) -> None: ...

class LoginResponse(_message.Message):
    __slots__ = ("token", "key_id", "user_id")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    KEY_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    token: str
    key_id: str
    user_id: str
    def __init__(self, token: _Optional[str] = ..., key_id: _Optional[str] = ..., user_id: _Optional[str] = ...) -> None: ...

class GetAuthInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetAuthInfoResponse(_message.Message):
    __slots__ = ("provider", "gcp_project_id")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    GCP_PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    provider: str
    gcp_project_id: str
    def __init__(self, provider: _Optional[str] = ..., gcp_project_id: _Optional[str] = ...) -> None: ...

class CreateApiKeyRequest(_message.Message):
    __slots__ = ("user_id", "name", "ttl_ms")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TTL_MS_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    name: str
    ttl_ms: int
    def __init__(self, user_id: _Optional[str] = ..., name: _Optional[str] = ..., ttl_ms: _Optional[int] = ...) -> None: ...

class CreateApiKeyResponse(_message.Message):
    __slots__ = ("key_id", "token", "key_prefix")
    KEY_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    KEY_PREFIX_FIELD_NUMBER: _ClassVar[int]
    key_id: str
    token: str
    key_prefix: str
    def __init__(self, key_id: _Optional[str] = ..., token: _Optional[str] = ..., key_prefix: _Optional[str] = ...) -> None: ...

class RevokeApiKeyRequest(_message.Message):
    __slots__ = ("key_id",)
    KEY_ID_FIELD_NUMBER: _ClassVar[int]
    key_id: str
    def __init__(self, key_id: _Optional[str] = ...) -> None: ...

class ListApiKeysRequest(_message.Message):
    __slots__ = ("user_id",)
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    def __init__(self, user_id: _Optional[str] = ...) -> None: ...

class ApiKeyInfo(_message.Message):
    __slots__ = ("key_id", "key_prefix", "user_id", "name", "created_at_ms", "last_used_at_ms", "expires_at_ms", "revoked")
    KEY_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_PREFIX_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    LAST_USED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_MS_FIELD_NUMBER: _ClassVar[int]
    REVOKED_FIELD_NUMBER: _ClassVar[int]
    key_id: str
    key_prefix: str
    user_id: str
    name: str
    created_at_ms: int
    last_used_at_ms: int
    expires_at_ms: int
    revoked: bool
    def __init__(self, key_id: _Optional[str] = ..., key_prefix: _Optional[str] = ..., user_id: _Optional[str] = ..., name: _Optional[str] = ..., created_at_ms: _Optional[int] = ..., last_used_at_ms: _Optional[int] = ..., expires_at_ms: _Optional[int] = ..., revoked: _Optional[bool] = ...) -> None: ...

class ListApiKeysResponse(_message.Message):
    __slots__ = ("keys",)
    KEYS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedCompositeFieldContainer[ApiKeyInfo]
    def __init__(self, keys: _Optional[_Iterable[_Union[ApiKeyInfo, _Mapping]]] = ...) -> None: ...

class GetCurrentUserRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetCurrentUserResponse(_message.Message):
    __slots__ = ("user_id", "role", "display_name")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    role: str
    display_name: str
    def __init__(self, user_id: _Optional[str] = ..., role: _Optional[str] = ..., display_name: _Optional[str] = ...) -> None: ...

class CpuProfile(_message.Message):
    __slots__ = ("format", "rate_hz", "native")
    class Format(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        FORMAT_UNSPECIFIED: _ClassVar[CpuProfile.Format]
        FLAMEGRAPH: _ClassVar[CpuProfile.Format]
        SPEEDSCOPE: _ClassVar[CpuProfile.Format]
        RAW: _ClassVar[CpuProfile.Format]
    FORMAT_UNSPECIFIED: CpuProfile.Format
    FLAMEGRAPH: CpuProfile.Format
    SPEEDSCOPE: CpuProfile.Format
    RAW: CpuProfile.Format
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    RATE_HZ_FIELD_NUMBER: _ClassVar[int]
    NATIVE_FIELD_NUMBER: _ClassVar[int]
    format: CpuProfile.Format
    rate_hz: int
    native: bool
    def __init__(self, format: _Optional[_Union[CpuProfile.Format, str]] = ..., rate_hz: _Optional[int] = ..., native: _Optional[bool] = ...) -> None: ...

class MemoryProfile(_message.Message):
    __slots__ = ("format", "leaks")
    class Format(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        FORMAT_UNSPECIFIED: _ClassVar[MemoryProfile.Format]
        FLAMEGRAPH: _ClassVar[MemoryProfile.Format]
        TABLE: _ClassVar[MemoryProfile.Format]
        STATS: _ClassVar[MemoryProfile.Format]
        RAW: _ClassVar[MemoryProfile.Format]
    FORMAT_UNSPECIFIED: MemoryProfile.Format
    FLAMEGRAPH: MemoryProfile.Format
    TABLE: MemoryProfile.Format
    STATS: MemoryProfile.Format
    RAW: MemoryProfile.Format
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    LEAKS_FIELD_NUMBER: _ClassVar[int]
    format: MemoryProfile.Format
    leaks: bool
    def __init__(self, format: _Optional[_Union[MemoryProfile.Format, str]] = ..., leaks: _Optional[bool] = ...) -> None: ...

class ThreadsProfile(_message.Message):
    __slots__ = ("locals",)
    LOCALS_FIELD_NUMBER: _ClassVar[int]
    locals: bool
    def __init__(self, locals: _Optional[bool] = ...) -> None: ...

class ProfileType(_message.Message):
    __slots__ = ("cpu", "memory", "threads")
    CPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    THREADS_FIELD_NUMBER: _ClassVar[int]
    cpu: CpuProfile
    memory: MemoryProfile
    threads: ThreadsProfile
    def __init__(self, cpu: _Optional[_Union[CpuProfile, _Mapping]] = ..., memory: _Optional[_Union[MemoryProfile, _Mapping]] = ..., threads: _Optional[_Union[ThreadsProfile, _Mapping]] = ...) -> None: ...

class ProfileTaskRequest(_message.Message):
    __slots__ = ("target", "duration_seconds", "profile_type")
    TARGET_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    PROFILE_TYPE_FIELD_NUMBER: _ClassVar[int]
    target: str
    duration_seconds: int
    profile_type: ProfileType
    def __init__(self, target: _Optional[str] = ..., duration_seconds: _Optional[int] = ..., profile_type: _Optional[_Union[ProfileType, _Mapping]] = ...) -> None: ...

class ProfileTaskResponse(_message.Message):
    __slots__ = ("profile_data", "error")
    PROFILE_DATA_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    profile_data: bytes
    error: str
    def __init__(self, profile_data: _Optional[bytes] = ..., error: _Optional[str] = ...) -> None: ...

class ProcessInfo(_message.Message):
    __slots__ = ("hostname", "pid", "python_version", "uptime_ms", "memory_rss_bytes", "memory_vms_bytes", "thread_count", "open_fd_count", "memory_total_bytes", "cpu_count", "git_hash", "cpu_millicores")
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    PYTHON_VERSION_FIELD_NUMBER: _ClassVar[int]
    UPTIME_MS_FIELD_NUMBER: _ClassVar[int]
    MEMORY_RSS_BYTES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_VMS_BYTES_FIELD_NUMBER: _ClassVar[int]
    THREAD_COUNT_FIELD_NUMBER: _ClassVar[int]
    OPEN_FD_COUNT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    CPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    GIT_HASH_FIELD_NUMBER: _ClassVar[int]
    CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    hostname: str
    pid: int
    python_version: str
    uptime_ms: int
    memory_rss_bytes: int
    memory_vms_bytes: int
    thread_count: int
    open_fd_count: int
    memory_total_bytes: int
    cpu_count: int
    git_hash: str
    cpu_millicores: int
    def __init__(self, hostname: _Optional[str] = ..., pid: _Optional[int] = ..., python_version: _Optional[str] = ..., uptime_ms: _Optional[int] = ..., memory_rss_bytes: _Optional[int] = ..., memory_vms_bytes: _Optional[int] = ..., thread_count: _Optional[int] = ..., open_fd_count: _Optional[int] = ..., memory_total_bytes: _Optional[int] = ..., cpu_count: _Optional[int] = ..., git_hash: _Optional[str] = ..., cpu_millicores: _Optional[int] = ...) -> None: ...

class GetProcessStatusRequest(_message.Message):
    __slots__ = ("max_log_lines", "log_substring", "min_log_level", "target")
    MAX_LOG_LINES_FIELD_NUMBER: _ClassVar[int]
    LOG_SUBSTRING_FIELD_NUMBER: _ClassVar[int]
    MIN_LOG_LEVEL_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    max_log_lines: int
    log_substring: str
    min_log_level: str
    target: str
    def __init__(self, max_log_lines: _Optional[int] = ..., log_substring: _Optional[str] = ..., min_log_level: _Optional[str] = ..., target: _Optional[str] = ...) -> None: ...

class GetProcessStatusResponse(_message.Message):
    __slots__ = ("process_info", "log_entries")
    PROCESS_INFO_FIELD_NUMBER: _ClassVar[int]
    LOG_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    process_info: ProcessInfo
    log_entries: _containers.RepeatedCompositeFieldContainer[_iris_logging_pb2.LogEntry]
    def __init__(self, process_info: _Optional[_Union[ProcessInfo, _Mapping]] = ..., log_entries: _Optional[_Iterable[_Union[_iris_logging_pb2.LogEntry, _Mapping]]] = ...) -> None: ...

class TaskStatus(_message.Message):
    __slots__ = ("task_id", "state", "worker_id", "worker_address", "exit_code", "error", "started_at", "finished_at", "ports", "resource_usage", "build_metrics", "current_attempt_id", "attempts", "pending_reason", "can_be_scheduled", "container_id", "resource_history")
    class PortsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    PORTS_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_USAGE_FIELD_NUMBER: _ClassVar[int]
    BUILD_METRICS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    PENDING_REASON_FIELD_NUMBER: _ClassVar[int]
    CAN_BE_SCHEDULED_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_HISTORY_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    state: TaskState
    worker_id: str
    worker_address: str
    exit_code: int
    error: str
    started_at: _iris_time_pb2.Timestamp
    finished_at: _iris_time_pb2.Timestamp
    ports: _containers.ScalarMap[str, int]
    resource_usage: ResourceUsage
    build_metrics: BuildMetrics
    current_attempt_id: int
    attempts: _containers.RepeatedCompositeFieldContainer[TaskAttempt]
    pending_reason: str
    can_be_scheduled: bool
    container_id: str
    resource_history: _containers.RepeatedCompositeFieldContainer[ResourceUsage]
    def __init__(self, task_id: _Optional[str] = ..., state: _Optional[_Union[TaskState, str]] = ..., worker_id: _Optional[str] = ..., worker_address: _Optional[str] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at: _Optional[_Union[_iris_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_iris_time_pb2.Timestamp, _Mapping]] = ..., ports: _Optional[_Mapping[str, int]] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., build_metrics: _Optional[_Union[BuildMetrics, _Mapping]] = ..., current_attempt_id: _Optional[int] = ..., attempts: _Optional[_Iterable[_Union[TaskAttempt, _Mapping]]] = ..., pending_reason: _Optional[str] = ..., can_be_scheduled: _Optional[bool] = ..., container_id: _Optional[str] = ..., resource_history: _Optional[_Iterable[_Union[ResourceUsage, _Mapping]]] = ...) -> None: ...

class TaskAttempt(_message.Message):
    __slots__ = ("attempt_id", "worker_id", "state", "exit_code", "error", "started_at", "finished_at", "is_worker_failure")
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    IS_WORKER_FAILURE_FIELD_NUMBER: _ClassVar[int]
    attempt_id: int
    worker_id: str
    state: TaskState
    exit_code: int
    error: str
    started_at: _iris_time_pb2.Timestamp
    finished_at: _iris_time_pb2.Timestamp
    is_worker_failure: bool
    def __init__(self, attempt_id: _Optional[int] = ..., worker_id: _Optional[str] = ..., state: _Optional[_Union[TaskState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at: _Optional[_Union[_iris_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_iris_time_pb2.Timestamp, _Mapping]] = ..., is_worker_failure: _Optional[bool] = ...) -> None: ...

class ResourceUsage(_message.Message):
    __slots__ = ("memory_mb", "disk_mb", "cpu_millicores", "memory_peak_mb", "process_count")
    MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    DISK_MB_FIELD_NUMBER: _ClassVar[int]
    CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_PEAK_MB_FIELD_NUMBER: _ClassVar[int]
    PROCESS_COUNT_FIELD_NUMBER: _ClassVar[int]
    memory_mb: int
    disk_mb: int
    cpu_millicores: int
    memory_peak_mb: int
    process_count: int
    def __init__(self, memory_mb: _Optional[int] = ..., disk_mb: _Optional[int] = ..., cpu_millicores: _Optional[int] = ..., memory_peak_mb: _Optional[int] = ..., process_count: _Optional[int] = ...) -> None: ...

class WorkerResourceSnapshot(_message.Message):
    __slots__ = ("timestamp", "host_cpu_percent", "memory_used_bytes", "memory_total_bytes", "disk_used_bytes", "disk_total_bytes", "running_task_count", "total_process_count", "net_recv_bps", "net_sent_bps")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    HOST_CPU_PERCENT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USED_BYTES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_USED_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    RUNNING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_PROCESS_COUNT_FIELD_NUMBER: _ClassVar[int]
    NET_RECV_BPS_FIELD_NUMBER: _ClassVar[int]
    NET_SENT_BPS_FIELD_NUMBER: _ClassVar[int]
    timestamp: _iris_time_pb2.Timestamp
    host_cpu_percent: int
    memory_used_bytes: int
    memory_total_bytes: int
    disk_used_bytes: int
    disk_total_bytes: int
    running_task_count: int
    total_process_count: int
    net_recv_bps: int
    net_sent_bps: int
    def __init__(self, timestamp: _Optional[_Union[_iris_time_pb2.Timestamp, _Mapping]] = ..., host_cpu_percent: _Optional[int] = ..., memory_used_bytes: _Optional[int] = ..., memory_total_bytes: _Optional[int] = ..., disk_used_bytes: _Optional[int] = ..., disk_total_bytes: _Optional[int] = ..., running_task_count: _Optional[int] = ..., total_process_count: _Optional[int] = ..., net_recv_bps: _Optional[int] = ..., net_sent_bps: _Optional[int] = ...) -> None: ...

class BuildMetrics(_message.Message):
    __slots__ = ("build_started", "build_finished", "from_cache", "image_tag")
    BUILD_STARTED_FIELD_NUMBER: _ClassVar[int]
    BUILD_FINISHED_FIELD_NUMBER: _ClassVar[int]
    FROM_CACHE_FIELD_NUMBER: _ClassVar[int]
    IMAGE_TAG_FIELD_NUMBER: _ClassVar[int]
    build_started: _iris_time_pb2.Timestamp
    build_finished: _iris_time_pb2.Timestamp
    from_cache: bool
    image_tag: str
    def __init__(self, build_started: _Optional[_Union[_iris_time_pb2.Timestamp, _Mapping]] = ..., build_finished: _Optional[_Union[_iris_time_pb2.Timestamp, _Mapping]] = ..., from_cache: _Optional[bool] = ..., image_tag: _Optional[str] = ...) -> None: ...

class JobStatus(_message.Message):
    __slots__ = ("job_id", "state", "exit_code", "error", "started_at", "finished_at", "ports", "resource_usage", "status_message", "build_metrics", "failure_count", "preemption_count", "tasks", "name", "submitted_at", "resources", "task_state_counts", "task_count", "completed_count", "pending_reason", "has_children")
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
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    PORTS_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_USAGE_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    BUILD_METRICS_FIELD_NUMBER: _ClassVar[int]
    FAILURE_COUNT_FIELD_NUMBER: _ClassVar[int]
    PREEMPTION_COUNT_FIELD_NUMBER: _ClassVar[int]
    TASKS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SUBMITTED_AT_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    TASK_STATE_COUNTS_FIELD_NUMBER: _ClassVar[int]
    TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_COUNT_FIELD_NUMBER: _ClassVar[int]
    PENDING_REASON_FIELD_NUMBER: _ClassVar[int]
    HAS_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    state: JobState
    exit_code: int
    error: str
    started_at: _iris_time_pb2.Timestamp
    finished_at: _iris_time_pb2.Timestamp
    ports: _containers.ScalarMap[str, int]
    resource_usage: ResourceUsage
    status_message: str
    build_metrics: BuildMetrics
    failure_count: int
    preemption_count: int
    tasks: _containers.RepeatedCompositeFieldContainer[TaskStatus]
    name: str
    submitted_at: _iris_time_pb2.Timestamp
    resources: ResourceSpecProto
    task_state_counts: _containers.ScalarMap[str, int]
    task_count: int
    completed_count: int
    pending_reason: str
    has_children: bool
    def __init__(self, job_id: _Optional[str] = ..., state: _Optional[_Union[JobState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at: _Optional[_Union[_iris_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_iris_time_pb2.Timestamp, _Mapping]] = ..., ports: _Optional[_Mapping[str, int]] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., status_message: _Optional[str] = ..., build_metrics: _Optional[_Union[BuildMetrics, _Mapping]] = ..., failure_count: _Optional[int] = ..., preemption_count: _Optional[int] = ..., tasks: _Optional[_Iterable[_Union[TaskStatus, _Mapping]]] = ..., name: _Optional[str] = ..., submitted_at: _Optional[_Union[_iris_time_pb2.Timestamp, _Mapping]] = ..., resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ..., task_state_counts: _Optional[_Mapping[str, int]] = ..., task_count: _Optional[int] = ..., completed_count: _Optional[int] = ..., pending_reason: _Optional[str] = ..., has_children: _Optional[bool] = ...) -> None: ...

class ReservationEntry(_message.Message):
    __slots__ = ("resources", "constraints")
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    resources: ResourceSpecProto
    constraints: _containers.RepeatedCompositeFieldContainer[Constraint]
    def __init__(self, resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ..., constraints: _Optional[_Iterable[_Union[Constraint, _Mapping]]] = ...) -> None: ...

class ReservationConfig(_message.Message):
    __slots__ = ("entries",)
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[ReservationEntry]
    def __init__(self, entries: _Optional[_Iterable[_Union[ReservationEntry, _Mapping]]] = ...) -> None: ...

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
    __slots__ = ("cpu_millicores", "memory_bytes", "disk_bytes", "device")
    CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int
    device: DeviceConfig
    def __init__(self, cpu_millicores: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., device: _Optional[_Union[DeviceConfig, _Mapping]] = ...) -> None: ...

class EnvironmentConfig(_message.Message):
    __slots__ = ("pip_packages", "env_vars", "extras", "python_version", "dockerfile")
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
    DOCKERFILE_FIELD_NUMBER: _ClassVar[int]
    pip_packages: _containers.RepeatedScalarFieldContainer[str]
    env_vars: _containers.ScalarMap[str, str]
    extras: _containers.RepeatedScalarFieldContainer[str]
    python_version: str
    dockerfile: str
    def __init__(self, pip_packages: _Optional[_Iterable[str]] = ..., env_vars: _Optional[_Mapping[str, str]] = ..., extras: _Optional[_Iterable[str]] = ..., python_version: _Optional[str] = ..., dockerfile: _Optional[str] = ...) -> None: ...

class CommandEntrypoint(_message.Message):
    __slots__ = ("argv",)
    ARGV_FIELD_NUMBER: _ClassVar[int]
    argv: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, argv: _Optional[_Iterable[str]] = ...) -> None: ...

class RuntimeEntrypoint(_message.Message):
    __slots__ = ("setup_commands", "run_command", "workdir_files", "workdir_file_refs")
    class WorkdirFilesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bytes
        def __init__(self, key: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...
    class WorkdirFileRefsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SETUP_COMMANDS_FIELD_NUMBER: _ClassVar[int]
    RUN_COMMAND_FIELD_NUMBER: _ClassVar[int]
    WORKDIR_FILES_FIELD_NUMBER: _ClassVar[int]
    WORKDIR_FILE_REFS_FIELD_NUMBER: _ClassVar[int]
    setup_commands: _containers.RepeatedScalarFieldContainer[str]
    run_command: CommandEntrypoint
    workdir_files: _containers.ScalarMap[str, bytes]
    workdir_file_refs: _containers.ScalarMap[str, str]
    def __init__(self, setup_commands: _Optional[_Iterable[str]] = ..., run_command: _Optional[_Union[CommandEntrypoint, _Mapping]] = ..., workdir_files: _Optional[_Mapping[str, bytes]] = ..., workdir_file_refs: _Optional[_Mapping[str, str]] = ...) -> None: ...

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
    __slots__ = ("key", "op", "value", "values", "mode")
    KEY_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    key: str
    op: ConstraintOp
    value: AttributeValue
    values: _containers.RepeatedCompositeFieldContainer[AttributeValue]
    mode: ConstraintMode
    def __init__(self, key: _Optional[str] = ..., op: _Optional[_Union[ConstraintOp, str]] = ..., value: _Optional[_Union[AttributeValue, _Mapping]] = ..., values: _Optional[_Iterable[_Union[AttributeValue, _Mapping]]] = ..., mode: _Optional[_Union[ConstraintMode, str]] = ...) -> None: ...

class ConstraintList(_message.Message):
    __slots__ = ("constraints",)
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    constraints: _containers.RepeatedCompositeFieldContainer[Constraint]
    def __init__(self, constraints: _Optional[_Iterable[_Union[Constraint, _Mapping]]] = ...) -> None: ...

class CoschedulingConfig(_message.Message):
    __slots__ = ("group_by",)
    GROUP_BY_FIELD_NUMBER: _ClassVar[int]
    group_by: str
    def __init__(self, group_by: _Optional[str] = ...) -> None: ...

class WorkerMetadata(_message.Message):
    __slots__ = ("hostname", "ip_address", "cpu_count", "memory_bytes", "disk_bytes", "device", "tpu_name", "tpu_worker_hostnames", "tpu_worker_id", "tpu_chips_per_host_bounds", "gpu_count", "gpu_name", "gpu_memory_mb", "gce_instance_name", "gce_zone", "attributes", "git_hash")
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
    GIT_HASH_FIELD_NUMBER: _ClassVar[int]
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
    git_hash: str
    def __init__(self, hostname: _Optional[str] = ..., ip_address: _Optional[str] = ..., cpu_count: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., device: _Optional[_Union[DeviceConfig, _Mapping]] = ..., tpu_name: _Optional[str] = ..., tpu_worker_hostnames: _Optional[str] = ..., tpu_worker_id: _Optional[str] = ..., tpu_chips_per_host_bounds: _Optional[str] = ..., gpu_count: _Optional[int] = ..., gpu_name: _Optional[str] = ..., gpu_memory_mb: _Optional[int] = ..., gce_instance_name: _Optional[str] = ..., gce_zone: _Optional[str] = ..., attributes: _Optional[_Mapping[str, AttributeValue]] = ..., git_hash: _Optional[str] = ...) -> None: ...

class RunTaskRequest(_message.Message):
    __slots__ = ("task_id", "num_tasks", "entrypoint", "environment", "bundle_id", "resources", "timeout", "ports", "attempt_id", "constraints", "task_image")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_TASKS_FIELD_NUMBER: _ClassVar[int]
    ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_ID_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    PORTS_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    TASK_IMAGE_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    num_tasks: int
    entrypoint: RuntimeEntrypoint
    environment: EnvironmentConfig
    bundle_id: str
    resources: ResourceSpecProto
    timeout: _iris_time_pb2.Duration
    ports: _containers.RepeatedScalarFieldContainer[str]
    attempt_id: int
    constraints: _containers.RepeatedCompositeFieldContainer[Constraint]
    task_image: str
    def __init__(self, task_id: _Optional[str] = ..., num_tasks: _Optional[int] = ..., entrypoint: _Optional[_Union[RuntimeEntrypoint, _Mapping]] = ..., environment: _Optional[_Union[EnvironmentConfig, _Mapping]] = ..., bundle_id: _Optional[str] = ..., resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ..., timeout: _Optional[_Union[_iris_time_pb2.Duration, _Mapping]] = ..., ports: _Optional[_Iterable[str]] = ..., attempt_id: _Optional[int] = ..., constraints: _Optional[_Iterable[_Union[Constraint, _Mapping]]] = ..., task_image: _Optional[str] = ...) -> None: ...

class WorkerTaskStatus(_message.Message):
    __slots__ = ("task_id", "attempt_id", "state", "exit_code", "error", "finished_at", "resource_usage", "container_id")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_USAGE_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    attempt_id: int
    state: TaskState
    exit_code: int
    error: str
    finished_at: _iris_time_pb2.Timestamp
    resource_usage: ResourceUsage
    container_id: str
    def __init__(self, task_id: _Optional[str] = ..., attempt_id: _Optional[int] = ..., state: _Optional[_Union[TaskState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., finished_at: _Optional[_Union[_iris_time_pb2.Timestamp, _Mapping]] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., container_id: _Optional[str] = ...) -> None: ...
