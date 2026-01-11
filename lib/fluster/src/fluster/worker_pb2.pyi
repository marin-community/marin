from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class JobStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_STATUS_UNSPECIFIED: _ClassVar[JobStatus]
    JOB_STATUS_PENDING: _ClassVar[JobStatus]
    JOB_STATUS_BUILDING: _ClassVar[JobStatus]
    JOB_STATUS_RUNNING: _ClassVar[JobStatus]
    JOB_STATUS_SUCCEEDED: _ClassVar[JobStatus]
    JOB_STATUS_FAILED: _ClassVar[JobStatus]
    JOB_STATUS_KILLED: _ClassVar[JobStatus]
JOB_STATUS_UNSPECIFIED: JobStatus
JOB_STATUS_PENDING: JobStatus
JOB_STATUS_BUILDING: JobStatus
JOB_STATUS_RUNNING: JobStatus
JOB_STATUS_SUCCEEDED: JobStatus
JOB_STATUS_FAILED: JobStatus
JOB_STATUS_KILLED: JobStatus

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
    SERIALIZED_ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_GCS_PATH_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_HASH_FIELD_NUMBER: _ClassVar[int]
    LIMITS_FIELD_NUMBER: _ClassVar[int]
    ENV_VARS_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    serialized_entrypoint: bytes
    serialized_environment: bytes
    bundle_gcs_path: str
    bundle_hash: str
    limits: ResourceLimits
    env_vars: _containers.ScalarMap[str, str]
    def __init__(self, job_id: _Optional[str] = ..., serialized_entrypoint: _Optional[bytes] = ..., serialized_environment: _Optional[bytes] = ..., bundle_gcs_path: _Optional[str] = ..., bundle_hash: _Optional[str] = ..., limits: _Optional[_Union[ResourceLimits, _Mapping]] = ..., env_vars: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ResourceLimits(_message.Message):
    __slots__ = ()
    CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    cpu_millicores: int
    memory_mb: int
    timeout_seconds: int
    def __init__(self, cpu_millicores: _Optional[int] = ..., memory_mb: _Optional[int] = ..., timeout_seconds: _Optional[int] = ...) -> None: ...

class RunJobResponse(_message.Message):
    __slots__ = ()
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: JobStatus
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[_Union[JobStatus, str]] = ...) -> None: ...

class GetStatusRequest(_message.Message):
    __slots__ = ()
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class GetStatusResponse(_message.Message):
    __slots__ = ()
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: JobStatus
    exit_code: int
    error: str
    started_at_ms: int
    finished_at_ms: int
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[_Union[JobStatus, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., started_at_ms: _Optional[int] = ..., finished_at_ms: _Optional[int] = ...) -> None: ...

class LogEntry(_message.Message):
    __slots__ = ()
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    timestamp_ms: int
    stream: str
    data: str
    def __init__(self, timestamp_ms: _Optional[int] = ..., stream: _Optional[str] = ..., data: _Optional[str] = ...) -> None: ...

class StreamLogsRequest(_message.Message):
    __slots__ = ()
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class KillJobRequest(_message.Message):
    __slots__ = ()
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    force: bool
    def __init__(self, job_id: _Optional[str] = ..., force: _Optional[bool] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ()
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    UPTIME_MS_FIELD_NUMBER: _ClassVar[int]
    RUNNING_JOBS_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    uptime_ms: int
    running_jobs: int
    def __init__(self, healthy: _Optional[bool] = ..., uptime_ms: _Optional[int] = ..., running_jobs: _Optional[int] = ...) -> None: ...

class WorkerService(_service.service): ...

class WorkerService_Stub(WorkerService): ...
