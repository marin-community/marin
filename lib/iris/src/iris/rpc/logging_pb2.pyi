from . import time_pb2 as _time_pb2
from . import cluster_pb2 as _cluster_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LogEntry(_message.Message):
    __slots__ = ("timestamp", "source", "data", "attempt_id")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    timestamp: _time_pb2.Timestamp
    source: str
    data: str
    attempt_id: int
    def __init__(self, timestamp: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., source: _Optional[str] = ..., data: _Optional[str] = ..., attempt_id: _Optional[int] = ...) -> None: ...

class LogBatch(_message.Message):
    __slots__ = ("entries",)
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[LogEntry]
    def __init__(self, entries: _Optional[_Iterable[_Union[LogEntry, _Mapping]]] = ...) -> None: ...

class TaskAttemptMetadata(_message.Message):
    __slots__ = ("task_id", "attempt_id", "worker_id", "start_time", "end_time", "exit_code", "oom_killed", "status", "error_message", "resource_usage")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    OOM_KILLED_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_USAGE_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    attempt_id: int
    worker_id: str
    start_time: _time_pb2.Timestamp
    end_time: _time_pb2.Timestamp
    exit_code: int
    oom_killed: bool
    status: _cluster_pb2.TaskState
    error_message: str
    resource_usage: ResourceUsage
    def __init__(self, task_id: _Optional[str] = ..., attempt_id: _Optional[int] = ..., worker_id: _Optional[str] = ..., start_time: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., end_time: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., exit_code: _Optional[int] = ..., oom_killed: _Optional[bool] = ..., status: _Optional[_Union[_cluster_pb2.TaskState, str]] = ..., error_message: _Optional[str] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ...) -> None: ...

class ResourceUsage(_message.Message):
    __slots__ = ("peak_memory_bytes", "cpu_seconds", "gpu_memory_bytes")
    PEAK_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    CPU_SECONDS_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    peak_memory_bytes: int
    cpu_seconds: float
    gpu_memory_bytes: int
    def __init__(self, peak_memory_bytes: _Optional[int] = ..., cpu_seconds: _Optional[float] = ..., gpu_memory_bytes: _Optional[int] = ...) -> None: ...
