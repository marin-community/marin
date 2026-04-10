from . import time_pb2 as _time_pb2
from . import vm_pb2 as _vm_pb2
from . import logging_pb2 as _logging_pb2
from . import query_pb2 as _query_pb2
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
    __slots__ = ("hostname", "pid", "python_version", "uptime_ms", "memory_rss_bytes", "memory_vms_bytes", "cpu_percent", "thread_count", "open_fd_count", "memory_total_bytes", "cpu_count", "git_hash")
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    PYTHON_VERSION_FIELD_NUMBER: _ClassVar[int]
    UPTIME_MS_FIELD_NUMBER: _ClassVar[int]
    MEMORY_RSS_BYTES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_VMS_BYTES_FIELD_NUMBER: _ClassVar[int]
    CPU_PERCENT_FIELD_NUMBER: _ClassVar[int]
    THREAD_COUNT_FIELD_NUMBER: _ClassVar[int]
    OPEN_FD_COUNT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    CPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    GIT_HASH_FIELD_NUMBER: _ClassVar[int]
    hostname: str
    pid: int
    python_version: str
    uptime_ms: int
    memory_rss_bytes: int
    memory_vms_bytes: int
    cpu_percent: float
    thread_count: int
    open_fd_count: int
    memory_total_bytes: int
    cpu_count: int
    git_hash: str
    def __init__(self, hostname: _Optional[str] = ..., pid: _Optional[int] = ..., python_version: _Optional[str] = ..., uptime_ms: _Optional[int] = ..., memory_rss_bytes: _Optional[int] = ..., memory_vms_bytes: _Optional[int] = ..., cpu_percent: _Optional[float] = ..., thread_count: _Optional[int] = ..., open_fd_count: _Optional[int] = ..., memory_total_bytes: _Optional[int] = ..., cpu_count: _Optional[int] = ..., git_hash: _Optional[str] = ...) -> None: ...

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
    log_entries: _containers.RepeatedCompositeFieldContainer[_logging_pb2.LogEntry]
    def __init__(self, process_info: _Optional[_Union[ProcessInfo, _Mapping]] = ..., log_entries: _Optional[_Iterable[_Union[_logging_pb2.LogEntry, _Mapping]]] = ...) -> None: ...

class TaskStatus(_message.Message):
    __slots__ = ("task_id", "state", "worker_id", "worker_address", "exit_code", "error", "started_at", "finished_at", "ports", "resource_usage", "build_metrics", "current_attempt_id", "attempts", "pending_reason", "can_be_scheduled", "container_id")
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
    task_id: str
    state: TaskState
    worker_id: str
    worker_address: str
    exit_code: int
    error: str
    started_at: _time_pb2.Timestamp
    finished_at: _time_pb2.Timestamp
    ports: _containers.ScalarMap[str, int]
    resource_usage: ResourceUsage
    build_metrics: BuildMetrics
    current_attempt_id: int
    attempts: _containers.RepeatedCompositeFieldContainer[TaskAttempt]
    pending_reason: str
    can_be_scheduled: bool
    container_id: str
    def __init__(self, task_id: _Optional[str] = ..., state: _Optional[_Union[TaskState, str]] = ..., worker_id: _Optional[str] = ..., worker_address: _Optional[str] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., ports: _Optional[_Mapping[str, int]] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., build_metrics: _Optional[_Union[BuildMetrics, _Mapping]] = ..., current_attempt_id: _Optional[int] = ..., attempts: _Optional[_Iterable[_Union[TaskAttempt, _Mapping]]] = ..., pending_reason: _Optional[str] = ..., can_be_scheduled: _Optional[bool] = ..., container_id: _Optional[str] = ...) -> None: ...

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
    started_at: _time_pb2.Timestamp
    finished_at: _time_pb2.Timestamp
    is_worker_failure: bool
    def __init__(self, attempt_id: _Optional[int] = ..., worker_id: _Optional[str] = ..., state: _Optional[_Union[TaskState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., is_worker_failure: _Optional[bool] = ...) -> None: ...

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

class WorkerResourceSnapshot(_message.Message):
    __slots__ = ("timestamp", "cpu_percent", "memory_used_bytes", "memory_total_bytes", "disk_used_bytes", "disk_total_bytes", "running_task_count", "total_process_count", "net_recv_bps", "net_sent_bps")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    CPU_PERCENT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USED_BYTES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_USED_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    RUNNING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_PROCESS_COUNT_FIELD_NUMBER: _ClassVar[int]
    NET_RECV_BPS_FIELD_NUMBER: _ClassVar[int]
    NET_SENT_BPS_FIELD_NUMBER: _ClassVar[int]
    timestamp: _time_pb2.Timestamp
    cpu_percent: int
    memory_used_bytes: int
    memory_total_bytes: int
    disk_used_bytes: int
    disk_total_bytes: int
    running_task_count: int
    total_process_count: int
    net_recv_bps: int
    net_sent_bps: int
    def __init__(self, timestamp: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., cpu_percent: _Optional[int] = ..., memory_used_bytes: _Optional[int] = ..., memory_total_bytes: _Optional[int] = ..., disk_used_bytes: _Optional[int] = ..., disk_total_bytes: _Optional[int] = ..., running_task_count: _Optional[int] = ..., total_process_count: _Optional[int] = ..., net_recv_bps: _Optional[int] = ..., net_sent_bps: _Optional[int] = ...) -> None: ...

class BuildMetrics(_message.Message):
    __slots__ = ("build_started", "build_finished", "from_cache", "image_tag")
    BUILD_STARTED_FIELD_NUMBER: _ClassVar[int]
    BUILD_FINISHED_FIELD_NUMBER: _ClassVar[int]
    FROM_CACHE_FIELD_NUMBER: _ClassVar[int]
    IMAGE_TAG_FIELD_NUMBER: _ClassVar[int]
    build_started: _time_pb2.Timestamp
    build_finished: _time_pb2.Timestamp
    from_cache: bool
    image_tag: str
    def __init__(self, build_started: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., build_finished: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., from_cache: _Optional[bool] = ..., image_tag: _Optional[str] = ...) -> None: ...

class JobStatus(_message.Message):
    __slots__ = ("job_id", "state", "exit_code", "error", "started_at", "finished_at", "ports", "resource_usage", "status_message", "build_metrics", "failure_count", "preemption_count", "tasks", "name", "submitted_at", "resources", "task_state_counts", "task_count", "completed_count", "pending_reason")
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
    job_id: str
    state: JobState
    exit_code: int
    error: str
    started_at: _time_pb2.Timestamp
    finished_at: _time_pb2.Timestamp
    ports: _containers.ScalarMap[str, int]
    resource_usage: ResourceUsage
    status_message: str
    build_metrics: BuildMetrics
    failure_count: int
    preemption_count: int
    tasks: _containers.RepeatedCompositeFieldContainer[TaskStatus]
    name: str
    submitted_at: _time_pb2.Timestamp
    resources: ResourceSpecProto
    task_state_counts: _containers.ScalarMap[str, int]
    task_count: int
    completed_count: int
    pending_reason: str
    def __init__(self, job_id: _Optional[str] = ..., state: _Optional[_Union[JobState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., ports: _Optional[_Mapping[str, int]] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., status_message: _Optional[str] = ..., build_metrics: _Optional[_Union[BuildMetrics, _Mapping]] = ..., failure_count: _Optional[int] = ..., preemption_count: _Optional[int] = ..., tasks: _Optional[_Iterable[_Union[TaskStatus, _Mapping]]] = ..., name: _Optional[str] = ..., submitted_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ..., task_state_counts: _Optional[_Mapping[str, int]] = ..., task_count: _Optional[int] = ..., completed_count: _Optional[int] = ..., pending_reason: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("hostname", "ip_address", "cpu_count", "memory_bytes", "disk_bytes", "device", "tpu_name", "tpu_worker_hostnames", "tpu_worker_id", "tpu_chips_per_host_bounds", "gpu_count", "gpu_name", "gpu_memory_mb", "gce_instance_name", "gce_zone", "attributes", "vm_address", "git_hash")
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
    vm_address: str
    git_hash: str
    def __init__(self, hostname: _Optional[str] = ..., ip_address: _Optional[str] = ..., cpu_count: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., device: _Optional[_Union[DeviceConfig, _Mapping]] = ..., tpu_name: _Optional[str] = ..., tpu_worker_hostnames: _Optional[str] = ..., tpu_worker_id: _Optional[str] = ..., tpu_chips_per_host_bounds: _Optional[str] = ..., gpu_count: _Optional[int] = ..., gpu_name: _Optional[str] = ..., gpu_memory_mb: _Optional[int] = ..., gce_instance_name: _Optional[str] = ..., gce_zone: _Optional[str] = ..., attributes: _Optional[_Mapping[str, AttributeValue]] = ..., vm_address: _Optional[str] = ..., git_hash: _Optional[str] = ...) -> None: ...

class Controller(_message.Message):
    __slots__ = ()
    class JobSortField(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        JOB_SORT_FIELD_UNSPECIFIED: _ClassVar[Controller.JobSortField]
        JOB_SORT_FIELD_DATE: _ClassVar[Controller.JobSortField]
        JOB_SORT_FIELD_NAME: _ClassVar[Controller.JobSortField]
        JOB_SORT_FIELD_STATE: _ClassVar[Controller.JobSortField]
        JOB_SORT_FIELD_FAILURES: _ClassVar[Controller.JobSortField]
        JOB_SORT_FIELD_PREEMPTIONS: _ClassVar[Controller.JobSortField]
    JOB_SORT_FIELD_UNSPECIFIED: Controller.JobSortField
    JOB_SORT_FIELD_DATE: Controller.JobSortField
    JOB_SORT_FIELD_NAME: Controller.JobSortField
    JOB_SORT_FIELD_STATE: Controller.JobSortField
    JOB_SORT_FIELD_FAILURES: Controller.JobSortField
    JOB_SORT_FIELD_PREEMPTIONS: Controller.JobSortField
    class SortDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SORT_DIRECTION_UNSPECIFIED: _ClassVar[Controller.SortDirection]
        SORT_DIRECTION_ASC: _ClassVar[Controller.SortDirection]
        SORT_DIRECTION_DESC: _ClassVar[Controller.SortDirection]
    SORT_DIRECTION_UNSPECIFIED: Controller.SortDirection
    SORT_DIRECTION_ASC: Controller.SortDirection
    SORT_DIRECTION_DESC: Controller.SortDirection
    class LaunchJobRequest(_message.Message):
        __slots__ = ("name", "entrypoint", "resources", "environment", "bundle_id", "bundle_blob", "scheduling_timeout", "ports", "max_task_failures", "max_retries_failure", "max_retries_preemption", "constraints", "coscheduling", "replicas", "timeout", "fail_if_exists", "reservation", "preemption_policy", "existing_job_policy", "priority_band")
        NAME_FIELD_NUMBER: _ClassVar[int]
        ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
        RESOURCES_FIELD_NUMBER: _ClassVar[int]
        ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
        BUNDLE_ID_FIELD_NUMBER: _ClassVar[int]
        BUNDLE_BLOB_FIELD_NUMBER: _ClassVar[int]
        SCHEDULING_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
        PORTS_FIELD_NUMBER: _ClassVar[int]
        MAX_TASK_FAILURES_FIELD_NUMBER: _ClassVar[int]
        MAX_RETRIES_FAILURE_FIELD_NUMBER: _ClassVar[int]
        MAX_RETRIES_PREEMPTION_FIELD_NUMBER: _ClassVar[int]
        CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
        COSCHEDULING_FIELD_NUMBER: _ClassVar[int]
        REPLICAS_FIELD_NUMBER: _ClassVar[int]
        TIMEOUT_FIELD_NUMBER: _ClassVar[int]
        FAIL_IF_EXISTS_FIELD_NUMBER: _ClassVar[int]
        RESERVATION_FIELD_NUMBER: _ClassVar[int]
        PREEMPTION_POLICY_FIELD_NUMBER: _ClassVar[int]
        EXISTING_JOB_POLICY_FIELD_NUMBER: _ClassVar[int]
        PRIORITY_BAND_FIELD_NUMBER: _ClassVar[int]
        name: str
        entrypoint: RuntimeEntrypoint
        resources: ResourceSpecProto
        environment: EnvironmentConfig
        bundle_id: str
        bundle_blob: bytes
        scheduling_timeout: _time_pb2.Duration
        ports: _containers.RepeatedScalarFieldContainer[str]
        max_task_failures: int
        max_retries_failure: int
        max_retries_preemption: int
        constraints: _containers.RepeatedCompositeFieldContainer[Constraint]
        coscheduling: CoschedulingConfig
        replicas: int
        timeout: _time_pb2.Duration
        fail_if_exists: bool
        reservation: ReservationConfig
        preemption_policy: JobPreemptionPolicy
        existing_job_policy: ExistingJobPolicy
        priority_band: PriorityBand
        def __init__(self, name: _Optional[str] = ..., entrypoint: _Optional[_Union[RuntimeEntrypoint, _Mapping]] = ..., resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ..., environment: _Optional[_Union[EnvironmentConfig, _Mapping]] = ..., bundle_id: _Optional[str] = ..., bundle_blob: _Optional[bytes] = ..., scheduling_timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., ports: _Optional[_Iterable[str]] = ..., max_task_failures: _Optional[int] = ..., max_retries_failure: _Optional[int] = ..., max_retries_preemption: _Optional[int] = ..., constraints: _Optional[_Iterable[_Union[Constraint, _Mapping]]] = ..., coscheduling: _Optional[_Union[CoschedulingConfig, _Mapping]] = ..., replicas: _Optional[int] = ..., timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., fail_if_exists: _Optional[bool] = ..., reservation: _Optional[_Union[ReservationConfig, _Mapping]] = ..., preemption_policy: _Optional[_Union[JobPreemptionPolicy, str]] = ..., existing_job_policy: _Optional[_Union[ExistingJobPolicy, str]] = ..., priority_band: _Optional[_Union[PriorityBand, str]] = ...) -> None: ...
    class LaunchJobResponse(_message.Message):
        __slots__ = ("job_id",)
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        def __init__(self, job_id: _Optional[str] = ...) -> None: ...
    class GetJobStatusRequest(_message.Message):
        __slots__ = ("job_id",)
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        def __init__(self, job_id: _Optional[str] = ...) -> None: ...
    class GetJobStatusResponse(_message.Message):
        __slots__ = ("job", "request")
        JOB_FIELD_NUMBER: _ClassVar[int]
        REQUEST_FIELD_NUMBER: _ClassVar[int]
        job: JobStatus
        request: Controller.LaunchJobRequest
        def __init__(self, job: _Optional[_Union[JobStatus, _Mapping]] = ..., request: _Optional[_Union[Controller.LaunchJobRequest, _Mapping]] = ...) -> None: ...
    class GetJobStateRequest(_message.Message):
        __slots__ = ("job_ids",)
        JOB_IDS_FIELD_NUMBER: _ClassVar[int]
        job_ids: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, job_ids: _Optional[_Iterable[str]] = ...) -> None: ...
    class GetJobStateResponse(_message.Message):
        __slots__ = ("states",)
        class StatesEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: JobState
            def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[JobState, str]] = ...) -> None: ...
        STATES_FIELD_NUMBER: _ClassVar[int]
        states: _containers.ScalarMap[str, JobState]
        def __init__(self, states: _Optional[_Mapping[str, JobState]] = ...) -> None: ...
    class TerminateJobRequest(_message.Message):
        __slots__ = ("job_id",)
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        def __init__(self, job_id: _Optional[str] = ...) -> None: ...
    class ListJobsRequest(_message.Message):
        __slots__ = ("offset", "limit", "sort_field", "sort_direction", "name_filter", "state_filter")
        OFFSET_FIELD_NUMBER: _ClassVar[int]
        LIMIT_FIELD_NUMBER: _ClassVar[int]
        SORT_FIELD_FIELD_NUMBER: _ClassVar[int]
        SORT_DIRECTION_FIELD_NUMBER: _ClassVar[int]
        NAME_FILTER_FIELD_NUMBER: _ClassVar[int]
        STATE_FILTER_FIELD_NUMBER: _ClassVar[int]
        offset: int
        limit: int
        sort_field: Controller.JobSortField
        sort_direction: Controller.SortDirection
        name_filter: str
        state_filter: str
        def __init__(self, offset: _Optional[int] = ..., limit: _Optional[int] = ..., sort_field: _Optional[_Union[Controller.JobSortField, str]] = ..., sort_direction: _Optional[_Union[Controller.SortDirection, str]] = ..., name_filter: _Optional[str] = ..., state_filter: _Optional[str] = ...) -> None: ...
    class ListJobsResponse(_message.Message):
        __slots__ = ("jobs", "total_count", "has_more")
        JOBS_FIELD_NUMBER: _ClassVar[int]
        TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
        HAS_MORE_FIELD_NUMBER: _ClassVar[int]
        jobs: _containers.RepeatedCompositeFieldContainer[JobStatus]
        total_count: int
        has_more: bool
        def __init__(self, jobs: _Optional[_Iterable[_Union[JobStatus, _Mapping]]] = ..., total_count: _Optional[int] = ..., has_more: _Optional[bool] = ...) -> None: ...
    class GetTaskStatusRequest(_message.Message):
        __slots__ = ("task_id",)
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        def __init__(self, task_id: _Optional[str] = ...) -> None: ...
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
    class ExecInContainerRequest(_message.Message):
        __slots__ = ("task_id", "command", "timeout_seconds")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        COMMAND_FIELD_NUMBER: _ClassVar[int]
        TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        command: _containers.RepeatedScalarFieldContainer[str]
        timeout_seconds: int
        def __init__(self, task_id: _Optional[str] = ..., command: _Optional[_Iterable[str]] = ..., timeout_seconds: _Optional[int] = ...) -> None: ...
    class ExecInContainerResponse(_message.Message):
        __slots__ = ("exit_code", "stdout", "stderr", "error")
        EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
        STDOUT_FIELD_NUMBER: _ClassVar[int]
        STDERR_FIELD_NUMBER: _ClassVar[int]
        ERROR_FIELD_NUMBER: _ClassVar[int]
        exit_code: int
        stdout: str
        stderr: str
        error: str
        def __init__(self, exit_code: _Optional[int] = ..., stdout: _Optional[str] = ..., stderr: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...
    class WorkerInfo(_message.Message):
        __slots__ = ("worker_id", "address", "metadata", "registered_at")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        REGISTERED_AT_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        address: str
        metadata: WorkerMetadata
        registered_at: _time_pb2.Timestamp
        def __init__(self, worker_id: _Optional[str] = ..., address: _Optional[str] = ..., metadata: _Optional[_Union[WorkerMetadata, _Mapping]] = ..., registered_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...
    class WorkerTaskStatus(_message.Message):
        __slots__ = ("task_id", "attempt_id", "state", "exit_code", "error", "finished_at", "resource_usage", "log_entries", "container_id")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
        STATE_FIELD_NUMBER: _ClassVar[int]
        EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
        ERROR_FIELD_NUMBER: _ClassVar[int]
        FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
        RESOURCE_USAGE_FIELD_NUMBER: _ClassVar[int]
        LOG_ENTRIES_FIELD_NUMBER: _ClassVar[int]
        CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        attempt_id: int
        state: TaskState
        exit_code: int
        error: str
        finished_at: _time_pb2.Timestamp
        resource_usage: ResourceUsage
        log_entries: _containers.RepeatedCompositeFieldContainer[_logging_pb2.LogEntry]
        container_id: str
        def __init__(self, task_id: _Optional[str] = ..., attempt_id: _Optional[int] = ..., state: _Optional[_Union[TaskState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., log_entries: _Optional[_Iterable[_Union[_logging_pb2.LogEntry, _Mapping]]] = ..., container_id: _Optional[str] = ...) -> None: ...
    class WorkerHealthStatus(_message.Message):
        __slots__ = ("worker_id", "healthy", "consecutive_failures", "last_heartbeat", "running_job_ids", "address", "metadata", "status_message")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        HEALTHY_FIELD_NUMBER: _ClassVar[int]
        CONSECUTIVE_FAILURES_FIELD_NUMBER: _ClassVar[int]
        LAST_HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
        RUNNING_JOB_IDS_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        healthy: bool
        consecutive_failures: int
        last_heartbeat: _time_pb2.Timestamp
        running_job_ids: _containers.RepeatedScalarFieldContainer[str]
        address: str
        metadata: WorkerMetadata
        status_message: str
        def __init__(self, worker_id: _Optional[str] = ..., healthy: _Optional[bool] = ..., consecutive_failures: _Optional[int] = ..., last_heartbeat: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., running_job_ids: _Optional[_Iterable[str]] = ..., address: _Optional[str] = ..., metadata: _Optional[_Union[WorkerMetadata, _Mapping]] = ..., status_message: _Optional[str] = ...) -> None: ...
    class ListWorkersRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class ListWorkersResponse(_message.Message):
        __slots__ = ("workers",)
        WORKERS_FIELD_NUMBER: _ClassVar[int]
        workers: _containers.RepeatedCompositeFieldContainer[Controller.WorkerHealthStatus]
        def __init__(self, workers: _Optional[_Iterable[_Union[Controller.WorkerHealthStatus, _Mapping]]] = ...) -> None: ...
    class RegisterRequest(_message.Message):
        __slots__ = ("address", "metadata", "worker_id", "slice_id", "scale_group")
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        SLICE_ID_FIELD_NUMBER: _ClassVar[int]
        SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
        address: str
        metadata: WorkerMetadata
        worker_id: str
        slice_id: str
        scale_group: str
        def __init__(self, address: _Optional[str] = ..., metadata: _Optional[_Union[WorkerMetadata, _Mapping]] = ..., worker_id: _Optional[str] = ..., slice_id: _Optional[str] = ..., scale_group: _Optional[str] = ...) -> None: ...
    class RegisterResponse(_message.Message):
        __slots__ = ("worker_id", "accepted")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        ACCEPTED_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        accepted: bool
        def __init__(self, worker_id: _Optional[str] = ..., accepted: _Optional[bool] = ...) -> None: ...
    class Endpoint(_message.Message):
        __slots__ = ("endpoint_id", "name", "address", "task_id", "metadata")
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
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        endpoint_id: str
        name: str
        address: str
        task_id: str
        metadata: _containers.ScalarMap[str, str]
        def __init__(self, endpoint_id: _Optional[str] = ..., name: _Optional[str] = ..., address: _Optional[str] = ..., task_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...
    class RegisterEndpointRequest(_message.Message):
        __slots__ = ("name", "address", "task_id", "metadata", "attempt_id", "endpoint_id")
        class MetadataEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
        NAME_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
        ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
        name: str
        address: str
        task_id: str
        metadata: _containers.ScalarMap[str, str]
        attempt_id: int
        endpoint_id: str
        def __init__(self, name: _Optional[str] = ..., address: _Optional[str] = ..., task_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., attempt_id: _Optional[int] = ..., endpoint_id: _Optional[str] = ...) -> None: ...
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
    class ListEndpointsRequest(_message.Message):
        __slots__ = ("prefix", "exact")
        PREFIX_FIELD_NUMBER: _ClassVar[int]
        EXACT_FIELD_NUMBER: _ClassVar[int]
        prefix: str
        exact: bool
        def __init__(self, prefix: _Optional[str] = ..., exact: _Optional[bool] = ...) -> None: ...
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
    class TransactionAction(_message.Message):
        __slots__ = ("timestamp", "action", "entity_id", "details")
        TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
        ACTION_FIELD_NUMBER: _ClassVar[int]
        ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
        DETAILS_FIELD_NUMBER: _ClassVar[int]
        timestamp: _time_pb2.Timestamp
        action: str
        entity_id: str
        details: str
        def __init__(self, timestamp: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., action: _Optional[str] = ..., entity_id: _Optional[str] = ..., details: _Optional[str] = ...) -> None: ...
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
    class BeginCheckpointRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class BeginCheckpointResponse(_message.Message):
        __slots__ = ("checkpoint_path", "created_at", "job_count", "task_count", "worker_count")
        CHECKPOINT_PATH_FIELD_NUMBER: _ClassVar[int]
        CREATED_AT_FIELD_NUMBER: _ClassVar[int]
        JOB_COUNT_FIELD_NUMBER: _ClassVar[int]
        TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
        WORKER_COUNT_FIELD_NUMBER: _ClassVar[int]
        checkpoint_path: str
        created_at: _time_pb2.Timestamp
        job_count: int
        task_count: int
        worker_count: int
        def __init__(self, checkpoint_path: _Optional[str] = ..., created_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., job_count: _Optional[int] = ..., task_count: _Optional[int] = ..., worker_count: _Optional[int] = ...) -> None: ...
    class UserSummary(_message.Message):
        __slots__ = ("user", "task_state_counts", "job_state_counts")
        class TaskStateCountsEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: int
            def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
        class JobStateCountsEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: int
            def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
        USER_FIELD_NUMBER: _ClassVar[int]
        TASK_STATE_COUNTS_FIELD_NUMBER: _ClassVar[int]
        JOB_STATE_COUNTS_FIELD_NUMBER: _ClassVar[int]
        user: str
        task_state_counts: _containers.ScalarMap[str, int]
        job_state_counts: _containers.ScalarMap[str, int]
        def __init__(self, user: _Optional[str] = ..., task_state_counts: _Optional[_Mapping[str, int]] = ..., job_state_counts: _Optional[_Mapping[str, int]] = ...) -> None: ...
    class ListUsersRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class ListUsersResponse(_message.Message):
        __slots__ = ("users",)
        USERS_FIELD_NUMBER: _ClassVar[int]
        users: _containers.RepeatedCompositeFieldContainer[Controller.UserSummary]
        def __init__(self, users: _Optional[_Iterable[_Union[Controller.UserSummary, _Mapping]]] = ...) -> None: ...
    class GetTaskLogsRequest(_message.Message):
        __slots__ = ("id", "include_children", "since_ms", "max_total_lines", "substring", "attempt_id", "min_level", "cursor", "tail")
        ID_FIELD_NUMBER: _ClassVar[int]
        INCLUDE_CHILDREN_FIELD_NUMBER: _ClassVar[int]
        SINCE_MS_FIELD_NUMBER: _ClassVar[int]
        MAX_TOTAL_LINES_FIELD_NUMBER: _ClassVar[int]
        SUBSTRING_FIELD_NUMBER: _ClassVar[int]
        ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
        MIN_LEVEL_FIELD_NUMBER: _ClassVar[int]
        CURSOR_FIELD_NUMBER: _ClassVar[int]
        TAIL_FIELD_NUMBER: _ClassVar[int]
        id: str
        include_children: bool
        since_ms: int
        max_total_lines: int
        substring: str
        attempt_id: int
        min_level: str
        cursor: int
        tail: bool
        def __init__(self, id: _Optional[str] = ..., include_children: _Optional[bool] = ..., since_ms: _Optional[int] = ..., max_total_lines: _Optional[int] = ..., substring: _Optional[str] = ..., attempt_id: _Optional[int] = ..., min_level: _Optional[str] = ..., cursor: _Optional[int] = ..., tail: _Optional[bool] = ...) -> None: ...
    class TaskLogBatch(_message.Message):
        __slots__ = ("task_id", "logs", "error", "worker_id")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        LOGS_FIELD_NUMBER: _ClassVar[int]
        ERROR_FIELD_NUMBER: _ClassVar[int]
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        logs: _containers.RepeatedCompositeFieldContainer[_logging_pb2.LogEntry]
        error: str
        worker_id: str
        def __init__(self, task_id: _Optional[str] = ..., logs: _Optional[_Iterable[_Union[_logging_pb2.LogEntry, _Mapping]]] = ..., error: _Optional[str] = ..., worker_id: _Optional[str] = ...) -> None: ...
    class GetTaskLogsResponse(_message.Message):
        __slots__ = ("task_logs", "truncated", "child_job_statuses", "cursor")
        TASK_LOGS_FIELD_NUMBER: _ClassVar[int]
        TRUNCATED_FIELD_NUMBER: _ClassVar[int]
        CHILD_JOB_STATUSES_FIELD_NUMBER: _ClassVar[int]
        CURSOR_FIELD_NUMBER: _ClassVar[int]
        task_logs: _containers.RepeatedCompositeFieldContainer[Controller.TaskLogBatch]
        truncated: bool
        child_job_statuses: _containers.RepeatedCompositeFieldContainer[JobStatus]
        cursor: int
        def __init__(self, task_logs: _Optional[_Iterable[_Union[Controller.TaskLogBatch, _Mapping]]] = ..., truncated: _Optional[bool] = ..., child_job_statuses: _Optional[_Iterable[_Union[JobStatus, _Mapping]]] = ..., cursor: _Optional[int] = ...) -> None: ...
    class GetWorkerStatusRequest(_message.Message):
        __slots__ = ("id",)
        ID_FIELD_NUMBER: _ClassVar[int]
        id: str
        def __init__(self, id: _Optional[str] = ...) -> None: ...
    class GetWorkerStatusResponse(_message.Message):
        __slots__ = ("vm", "scale_group", "worker", "bootstrap_logs", "worker_log_entries", "recent_tasks", "current_resources", "resource_history")
        VM_FIELD_NUMBER: _ClassVar[int]
        SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
        WORKER_FIELD_NUMBER: _ClassVar[int]
        BOOTSTRAP_LOGS_FIELD_NUMBER: _ClassVar[int]
        WORKER_LOG_ENTRIES_FIELD_NUMBER: _ClassVar[int]
        RECENT_TASKS_FIELD_NUMBER: _ClassVar[int]
        CURRENT_RESOURCES_FIELD_NUMBER: _ClassVar[int]
        RESOURCE_HISTORY_FIELD_NUMBER: _ClassVar[int]
        vm: _vm_pb2.VmInfo
        scale_group: str
        worker: Controller.WorkerHealthStatus
        bootstrap_logs: str
        worker_log_entries: _containers.RepeatedCompositeFieldContainer[_logging_pb2.LogEntry]
        recent_tasks: _containers.RepeatedCompositeFieldContainer[TaskStatus]
        current_resources: WorkerResourceSnapshot
        resource_history: _containers.RepeatedCompositeFieldContainer[WorkerResourceSnapshot]
        def __init__(self, vm: _Optional[_Union[_vm_pb2.VmInfo, _Mapping]] = ..., scale_group: _Optional[str] = ..., worker: _Optional[_Union[Controller.WorkerHealthStatus, _Mapping]] = ..., bootstrap_logs: _Optional[str] = ..., worker_log_entries: _Optional[_Iterable[_Union[_logging_pb2.LogEntry, _Mapping]]] = ..., recent_tasks: _Optional[_Iterable[_Union[TaskStatus, _Mapping]]] = ..., current_resources: _Optional[_Union[WorkerResourceSnapshot, _Mapping]] = ..., resource_history: _Optional[_Iterable[_Union[WorkerResourceSnapshot, _Mapping]]] = ...) -> None: ...
    class SchedulingEvent(_message.Message):
        __slots__ = ("task_id", "attempt_id", "event_type", "reason", "message", "timestamp")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
        EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        MESSAGE_FIELD_NUMBER: _ClassVar[int]
        TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        attempt_id: int
        event_type: str
        reason: str
        message: str
        timestamp: _time_pb2.Timestamp
        def __init__(self, task_id: _Optional[str] = ..., attempt_id: _Optional[int] = ..., event_type: _Optional[str] = ..., reason: _Optional[str] = ..., message: _Optional[str] = ..., timestamp: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...
    class ClusterCapacity(_message.Message):
        __slots__ = ("schedulable_nodes", "total_cpu_millicores", "available_cpu_millicores", "total_memory_bytes", "available_memory_bytes")
        SCHEDULABLE_NODES_FIELD_NUMBER: _ClassVar[int]
        TOTAL_CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
        AVAILABLE_CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
        TOTAL_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
        AVAILABLE_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
        schedulable_nodes: int
        total_cpu_millicores: int
        available_cpu_millicores: int
        total_memory_bytes: int
        available_memory_bytes: int
        def __init__(self, schedulable_nodes: _Optional[int] = ..., total_cpu_millicores: _Optional[int] = ..., available_cpu_millicores: _Optional[int] = ..., total_memory_bytes: _Optional[int] = ..., available_memory_bytes: _Optional[int] = ...) -> None: ...
    class GetProviderStatusRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class GetProviderStatusResponse(_message.Message):
        __slots__ = ("has_direct_provider", "scheduling_events", "capacity")
        HAS_DIRECT_PROVIDER_FIELD_NUMBER: _ClassVar[int]
        SCHEDULING_EVENTS_FIELD_NUMBER: _ClassVar[int]
        CAPACITY_FIELD_NUMBER: _ClassVar[int]
        has_direct_provider: bool
        scheduling_events: _containers.RepeatedCompositeFieldContainer[Controller.SchedulingEvent]
        capacity: Controller.ClusterCapacity
        def __init__(self, has_direct_provider: _Optional[bool] = ..., scheduling_events: _Optional[_Iterable[_Union[Controller.SchedulingEvent, _Mapping]]] = ..., capacity: _Optional[_Union[Controller.ClusterCapacity, _Mapping]] = ...) -> None: ...
    class GetKubernetesClusterStatusRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class KubernetesPodStatus(_message.Message):
        __slots__ = ("pod_name", "task_id", "phase", "reason", "message", "last_transition", "node_name")
        POD_NAME_FIELD_NUMBER: _ClassVar[int]
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        PHASE_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        MESSAGE_FIELD_NUMBER: _ClassVar[int]
        LAST_TRANSITION_FIELD_NUMBER: _ClassVar[int]
        NODE_NAME_FIELD_NUMBER: _ClassVar[int]
        pod_name: str
        task_id: str
        phase: str
        reason: str
        message: str
        last_transition: _time_pb2.Timestamp
        node_name: str
        def __init__(self, pod_name: _Optional[str] = ..., task_id: _Optional[str] = ..., phase: _Optional[str] = ..., reason: _Optional[str] = ..., message: _Optional[str] = ..., last_transition: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., node_name: _Optional[str] = ...) -> None: ...
    class NodePoolStatus(_message.Message):
        __slots__ = ("name", "instance_type", "scale_group", "target_nodes", "current_nodes", "queued_nodes", "in_progress_nodes", "autoscaling", "min_nodes", "max_nodes", "capacity", "quota")
        NAME_FIELD_NUMBER: _ClassVar[int]
        INSTANCE_TYPE_FIELD_NUMBER: _ClassVar[int]
        SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
        TARGET_NODES_FIELD_NUMBER: _ClassVar[int]
        CURRENT_NODES_FIELD_NUMBER: _ClassVar[int]
        QUEUED_NODES_FIELD_NUMBER: _ClassVar[int]
        IN_PROGRESS_NODES_FIELD_NUMBER: _ClassVar[int]
        AUTOSCALING_FIELD_NUMBER: _ClassVar[int]
        MIN_NODES_FIELD_NUMBER: _ClassVar[int]
        MAX_NODES_FIELD_NUMBER: _ClassVar[int]
        CAPACITY_FIELD_NUMBER: _ClassVar[int]
        QUOTA_FIELD_NUMBER: _ClassVar[int]
        name: str
        instance_type: str
        scale_group: str
        target_nodes: int
        current_nodes: int
        queued_nodes: int
        in_progress_nodes: int
        autoscaling: bool
        min_nodes: int
        max_nodes: int
        capacity: str
        quota: str
        def __init__(self, name: _Optional[str] = ..., instance_type: _Optional[str] = ..., scale_group: _Optional[str] = ..., target_nodes: _Optional[int] = ..., current_nodes: _Optional[int] = ..., queued_nodes: _Optional[int] = ..., in_progress_nodes: _Optional[int] = ..., autoscaling: _Optional[bool] = ..., min_nodes: _Optional[int] = ..., max_nodes: _Optional[int] = ..., capacity: _Optional[str] = ..., quota: _Optional[str] = ...) -> None: ...
    class GetKubernetesClusterStatusResponse(_message.Message):
        __slots__ = ("namespace", "total_nodes", "schedulable_nodes", "allocatable_cpu", "allocatable_memory", "pod_statuses", "provider_version", "node_pools")
        NAMESPACE_FIELD_NUMBER: _ClassVar[int]
        TOTAL_NODES_FIELD_NUMBER: _ClassVar[int]
        SCHEDULABLE_NODES_FIELD_NUMBER: _ClassVar[int]
        ALLOCATABLE_CPU_FIELD_NUMBER: _ClassVar[int]
        ALLOCATABLE_MEMORY_FIELD_NUMBER: _ClassVar[int]
        POD_STATUSES_FIELD_NUMBER: _ClassVar[int]
        PROVIDER_VERSION_FIELD_NUMBER: _ClassVar[int]
        NODE_POOLS_FIELD_NUMBER: _ClassVar[int]
        namespace: str
        total_nodes: int
        schedulable_nodes: int
        allocatable_cpu: str
        allocatable_memory: str
        pod_statuses: _containers.RepeatedCompositeFieldContainer[Controller.KubernetesPodStatus]
        provider_version: str
        node_pools: _containers.RepeatedCompositeFieldContainer[Controller.NodePoolStatus]
        def __init__(self, namespace: _Optional[str] = ..., total_nodes: _Optional[int] = ..., schedulable_nodes: _Optional[int] = ..., allocatable_cpu: _Optional[str] = ..., allocatable_memory: _Optional[str] = ..., pod_statuses: _Optional[_Iterable[_Union[Controller.KubernetesPodStatus, _Mapping]]] = ..., provider_version: _Optional[str] = ..., node_pools: _Optional[_Iterable[_Union[Controller.NodePoolStatus, _Mapping]]] = ...) -> None: ...
    class RestartWorkerRequest(_message.Message):
        __slots__ = ("worker_id",)
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        def __init__(self, worker_id: _Optional[str] = ...) -> None: ...
    class RestartWorkerResponse(_message.Message):
        __slots__ = ("accepted", "error")
        ACCEPTED_FIELD_NUMBER: _ClassVar[int]
        ERROR_FIELD_NUMBER: _ClassVar[int]
        accepted: bool
        error: str
        def __init__(self, accepted: _Optional[bool] = ..., error: _Optional[str] = ...) -> None: ...
    class SetUserBudgetRequest(_message.Message):
        __slots__ = ("user_id", "budget_limit", "max_band")
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        BUDGET_LIMIT_FIELD_NUMBER: _ClassVar[int]
        MAX_BAND_FIELD_NUMBER: _ClassVar[int]
        user_id: str
        budget_limit: int
        max_band: PriorityBand
        def __init__(self, user_id: _Optional[str] = ..., budget_limit: _Optional[int] = ..., max_band: _Optional[_Union[PriorityBand, str]] = ...) -> None: ...
    class SetUserBudgetResponse(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class GetUserBudgetRequest(_message.Message):
        __slots__ = ("user_id",)
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        user_id: str
        def __init__(self, user_id: _Optional[str] = ...) -> None: ...
    class GetUserBudgetResponse(_message.Message):
        __slots__ = ("user_id", "budget_limit", "budget_spent", "max_band")
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        BUDGET_LIMIT_FIELD_NUMBER: _ClassVar[int]
        BUDGET_SPENT_FIELD_NUMBER: _ClassVar[int]
        MAX_BAND_FIELD_NUMBER: _ClassVar[int]
        user_id: str
        budget_limit: int
        budget_spent: int
        max_band: PriorityBand
        def __init__(self, user_id: _Optional[str] = ..., budget_limit: _Optional[int] = ..., budget_spent: _Optional[int] = ..., max_band: _Optional[_Union[PriorityBand, str]] = ...) -> None: ...
    class ListUserBudgetsRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class ListUserBudgetsResponse(_message.Message):
        __slots__ = ("users",)
        USERS_FIELD_NUMBER: _ClassVar[int]
        users: _containers.RepeatedCompositeFieldContainer[Controller.GetUserBudgetResponse]
        def __init__(self, users: _Optional[_Iterable[_Union[Controller.GetUserBudgetResponse, _Mapping]]] = ...) -> None: ...
    class GetSchedulerStateRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class SchedulerTaskEntry(_message.Message):
        __slots__ = ("task_id", "job_id", "user_id", "original_band", "effective_band", "queue_position", "resource_value")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        ORIGINAL_BAND_FIELD_NUMBER: _ClassVar[int]
        EFFECTIVE_BAND_FIELD_NUMBER: _ClassVar[int]
        QUEUE_POSITION_FIELD_NUMBER: _ClassVar[int]
        RESOURCE_VALUE_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        job_id: str
        user_id: str
        original_band: PriorityBand
        effective_band: PriorityBand
        queue_position: int
        resource_value: int
        def __init__(self, task_id: _Optional[str] = ..., job_id: _Optional[str] = ..., user_id: _Optional[str] = ..., original_band: _Optional[_Union[PriorityBand, str]] = ..., effective_band: _Optional[_Union[PriorityBand, str]] = ..., queue_position: _Optional[int] = ..., resource_value: _Optional[int] = ...) -> None: ...
    class SchedulerBandGroup(_message.Message):
        __slots__ = ("band", "tasks", "total_in_band")
        BAND_FIELD_NUMBER: _ClassVar[int]
        TASKS_FIELD_NUMBER: _ClassVar[int]
        TOTAL_IN_BAND_FIELD_NUMBER: _ClassVar[int]
        band: PriorityBand
        tasks: _containers.RepeatedCompositeFieldContainer[Controller.SchedulerTaskEntry]
        total_in_band: int
        def __init__(self, band: _Optional[_Union[PriorityBand, str]] = ..., tasks: _Optional[_Iterable[_Union[Controller.SchedulerTaskEntry, _Mapping]]] = ..., total_in_band: _Optional[int] = ...) -> None: ...
    class SchedulerUserBudget(_message.Message):
        __slots__ = ("user_id", "budget_limit", "budget_spent", "max_band", "effective_band", "utilization_percent")
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        BUDGET_LIMIT_FIELD_NUMBER: _ClassVar[int]
        BUDGET_SPENT_FIELD_NUMBER: _ClassVar[int]
        MAX_BAND_FIELD_NUMBER: _ClassVar[int]
        EFFECTIVE_BAND_FIELD_NUMBER: _ClassVar[int]
        UTILIZATION_PERCENT_FIELD_NUMBER: _ClassVar[int]
        user_id: str
        budget_limit: int
        budget_spent: int
        max_band: PriorityBand
        effective_band: PriorityBand
        utilization_percent: float
        def __init__(self, user_id: _Optional[str] = ..., budget_limit: _Optional[int] = ..., budget_spent: _Optional[int] = ..., max_band: _Optional[_Union[PriorityBand, str]] = ..., effective_band: _Optional[_Union[PriorityBand, str]] = ..., utilization_percent: _Optional[float] = ...) -> None: ...
    class SchedulerRunningTask(_message.Message):
        __slots__ = ("task_id", "job_id", "user_id", "worker_id", "effective_band", "resource_value", "preemptible", "preemptible_by", "is_coscheduled")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        EFFECTIVE_BAND_FIELD_NUMBER: _ClassVar[int]
        RESOURCE_VALUE_FIELD_NUMBER: _ClassVar[int]
        PREEMPTIBLE_FIELD_NUMBER: _ClassVar[int]
        PREEMPTIBLE_BY_FIELD_NUMBER: _ClassVar[int]
        IS_COSCHEDULED_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        job_id: str
        user_id: str
        worker_id: str
        effective_band: PriorityBand
        resource_value: int
        preemptible: bool
        preemptible_by: _containers.RepeatedScalarFieldContainer[PriorityBand]
        is_coscheduled: bool
        def __init__(self, task_id: _Optional[str] = ..., job_id: _Optional[str] = ..., user_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., effective_band: _Optional[_Union[PriorityBand, str]] = ..., resource_value: _Optional[int] = ..., preemptible: _Optional[bool] = ..., preemptible_by: _Optional[_Iterable[_Union[PriorityBand, str]]] = ..., is_coscheduled: _Optional[bool] = ...) -> None: ...
    class GetSchedulerStateResponse(_message.Message):
        __slots__ = ("pending_queue", "user_budgets", "running_tasks", "total_pending", "total_running")
        PENDING_QUEUE_FIELD_NUMBER: _ClassVar[int]
        USER_BUDGETS_FIELD_NUMBER: _ClassVar[int]
        RUNNING_TASKS_FIELD_NUMBER: _ClassVar[int]
        TOTAL_PENDING_FIELD_NUMBER: _ClassVar[int]
        TOTAL_RUNNING_FIELD_NUMBER: _ClassVar[int]
        pending_queue: _containers.RepeatedCompositeFieldContainer[Controller.SchedulerBandGroup]
        user_budgets: _containers.RepeatedCompositeFieldContainer[Controller.SchedulerUserBudget]
        running_tasks: _containers.RepeatedCompositeFieldContainer[Controller.SchedulerRunningTask]
        total_pending: int
        total_running: int
        def __init__(self, pending_queue: _Optional[_Iterable[_Union[Controller.SchedulerBandGroup, _Mapping]]] = ..., user_budgets: _Optional[_Iterable[_Union[Controller.SchedulerUserBudget, _Mapping]]] = ..., running_tasks: _Optional[_Iterable[_Union[Controller.SchedulerRunningTask, _Mapping]]] = ..., total_pending: _Optional[int] = ..., total_running: _Optional[int] = ...) -> None: ...
    def __init__(self) -> None: ...

class Worker(_message.Message):
    __slots__ = ()
    class RunTaskRequest(_message.Message):
        __slots__ = ("task_id", "num_tasks", "entrypoint", "environment", "bundle_id", "resources", "timeout", "ports", "attempt_id", "constraints")
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
        task_id: str
        num_tasks: int
        entrypoint: RuntimeEntrypoint
        environment: EnvironmentConfig
        bundle_id: str
        resources: ResourceSpecProto
        timeout: _time_pb2.Duration
        ports: _containers.RepeatedScalarFieldContainer[str]
        attempt_id: int
        constraints: _containers.RepeatedCompositeFieldContainer[Constraint]
        def __init__(self, task_id: _Optional[str] = ..., num_tasks: _Optional[int] = ..., entrypoint: _Optional[_Union[RuntimeEntrypoint, _Mapping]] = ..., environment: _Optional[_Union[EnvironmentConfig, _Mapping]] = ..., bundle_id: _Optional[str] = ..., resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ..., timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., ports: _Optional[_Iterable[str]] = ..., attempt_id: _Optional[int] = ..., constraints: _Optional[_Iterable[_Union[Constraint, _Mapping]]] = ...) -> None: ...
    class GetTaskStatusRequest(_message.Message):
        __slots__ = ("task_id",)
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        def __init__(self, task_id: _Optional[str] = ...) -> None: ...
    class ListTasksRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class ListTasksResponse(_message.Message):
        __slots__ = ("tasks",)
        TASKS_FIELD_NUMBER: _ClassVar[int]
        tasks: _containers.RepeatedCompositeFieldContainer[TaskStatus]
        def __init__(self, tasks: _Optional[_Iterable[_Union[TaskStatus, _Mapping]]] = ...) -> None: ...
    class KillTaskRequest(_message.Message):
        __slots__ = ("task_id", "term_timeout")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        TERM_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        term_timeout: _time_pb2.Duration
        def __init__(self, task_id: _Optional[str] = ..., term_timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ...) -> None: ...
    class HealthResponse(_message.Message):
        __slots__ = ("healthy", "uptime", "running_tasks")
        HEALTHY_FIELD_NUMBER: _ClassVar[int]
        UPTIME_FIELD_NUMBER: _ClassVar[int]
        RUNNING_TASKS_FIELD_NUMBER: _ClassVar[int]
        healthy: bool
        uptime: _time_pb2.Duration
        running_tasks: int
        def __init__(self, healthy: _Optional[bool] = ..., uptime: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., running_tasks: _Optional[int] = ...) -> None: ...
    class ExecInContainerRequest(_message.Message):
        __slots__ = ("task_id", "command", "timeout_seconds")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        COMMAND_FIELD_NUMBER: _ClassVar[int]
        TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        command: _containers.RepeatedScalarFieldContainer[str]
        timeout_seconds: int
        def __init__(self, task_id: _Optional[str] = ..., command: _Optional[_Iterable[str]] = ..., timeout_seconds: _Optional[int] = ...) -> None: ...
    class ExecInContainerResponse(_message.Message):
        __slots__ = ("exit_code", "stdout", "stderr", "error")
        EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
        STDOUT_FIELD_NUMBER: _ClassVar[int]
        STDERR_FIELD_NUMBER: _ClassVar[int]
        ERROR_FIELD_NUMBER: _ClassVar[int]
        exit_code: int
        stdout: str
        stderr: str
        error: str
        def __init__(self, exit_code: _Optional[int] = ..., stdout: _Optional[str] = ..., stderr: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...
    def __init__(self) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("tasks_to_run", "tasks_to_kill", "expected_tasks")
    TASKS_TO_RUN_FIELD_NUMBER: _ClassVar[int]
    TASKS_TO_KILL_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_TASKS_FIELD_NUMBER: _ClassVar[int]
    tasks_to_run: _containers.RepeatedCompositeFieldContainer[Worker.RunTaskRequest]
    tasks_to_kill: _containers.RepeatedScalarFieldContainer[str]
    expected_tasks: _containers.RepeatedCompositeFieldContainer[Controller.WorkerTaskStatus]
    def __init__(self, tasks_to_run: _Optional[_Iterable[_Union[Worker.RunTaskRequest, _Mapping]]] = ..., tasks_to_kill: _Optional[_Iterable[str]] = ..., expected_tasks: _Optional[_Iterable[_Union[Controller.WorkerTaskStatus, _Mapping]]] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("tasks", "resource_snapshot", "worker_healthy", "health_error")
    TASKS_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    WORKER_HEALTHY_FIELD_NUMBER: _ClassVar[int]
    HEALTH_ERROR_FIELD_NUMBER: _ClassVar[int]
    tasks: _containers.RepeatedCompositeFieldContainer[Controller.WorkerTaskStatus]
    resource_snapshot: WorkerResourceSnapshot
    worker_healthy: bool
    health_error: str
    def __init__(self, tasks: _Optional[_Iterable[_Union[Controller.WorkerTaskStatus, _Mapping]]] = ..., resource_snapshot: _Optional[_Union[WorkerResourceSnapshot, _Mapping]] = ..., worker_healthy: _Optional[bool] = ..., health_error: _Optional[str] = ...) -> None: ...
