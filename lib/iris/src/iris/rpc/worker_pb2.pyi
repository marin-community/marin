from . import job_pb2 as _job_pb2
from . import time_pb2 as _time_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Worker(_message.Message):
    __slots__ = ()
    class StopReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STOP_REASON_UNSPECIFIED: _ClassVar[Worker.StopReason]
        STOP_REASON_CANCELLED: _ClassVar[Worker.StopReason]
        STOP_REASON_PREEMPTED: _ClassVar[Worker.StopReason]
        STOP_REASON_SUPERSEDED: _ClassVar[Worker.StopReason]
        STOP_REASON_JOB_TERMINATED: _ClassVar[Worker.StopReason]
        STOP_REASON_TASK_TIMEOUT: _ClassVar[Worker.StopReason]
        STOP_REASON_WORKER_DRAIN: _ClassVar[Worker.StopReason]
    STOP_REASON_UNSPECIFIED: Worker.StopReason
    STOP_REASON_CANCELLED: Worker.StopReason
    STOP_REASON_PREEMPTED: Worker.StopReason
    STOP_REASON_SUPERSEDED: Worker.StopReason
    STOP_REASON_JOB_TERMINATED: Worker.StopReason
    STOP_REASON_TASK_TIMEOUT: Worker.StopReason
    STOP_REASON_WORKER_DRAIN: Worker.StopReason
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
        tasks: _containers.RepeatedCompositeFieldContainer[_job_pb2.TaskStatus]
        def __init__(self, tasks: _Optional[_Iterable[_Union[_job_pb2.TaskStatus, _Mapping]]] = ...) -> None: ...
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
    class PingRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class PingResponse(_message.Message):
        __slots__ = ("healthy", "health_error")
        HEALTHY_FIELD_NUMBER: _ClassVar[int]
        HEALTH_ERROR_FIELD_NUMBER: _ClassVar[int]
        healthy: bool
        health_error: str
        def __init__(self, healthy: _Optional[bool] = ..., health_error: _Optional[str] = ...) -> None: ...
    class StartTasksRequest(_message.Message):
        __slots__ = ("tasks",)
        TASKS_FIELD_NUMBER: _ClassVar[int]
        tasks: _containers.RepeatedCompositeFieldContainer[_job_pb2.RunTaskRequest]
        def __init__(self, tasks: _Optional[_Iterable[_Union[_job_pb2.RunTaskRequest, _Mapping]]] = ...) -> None: ...
    class StartTasksResponse(_message.Message):
        __slots__ = ("acks",)
        ACKS_FIELD_NUMBER: _ClassVar[int]
        acks: _containers.RepeatedCompositeFieldContainer[Worker.TaskAck]
        def __init__(self, acks: _Optional[_Iterable[_Union[Worker.TaskAck, _Mapping]]] = ...) -> None: ...
    class TaskAck(_message.Message):
        __slots__ = ("task_id", "accepted", "error")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        ACCEPTED_FIELD_NUMBER: _ClassVar[int]
        ERROR_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        accepted: bool
        error: str
        def __init__(self, task_id: _Optional[str] = ..., accepted: _Optional[bool] = ..., error: _Optional[str] = ...) -> None: ...
    class StopTasksRequest(_message.Message):
        __slots__ = ("task_ids",)
        TASK_IDS_FIELD_NUMBER: _ClassVar[int]
        task_ids: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, task_ids: _Optional[_Iterable[str]] = ...) -> None: ...
    class StopTasksResponse(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class PollTasksRequest(_message.Message):
        __slots__ = ("expected_tasks",)
        EXPECTED_TASKS_FIELD_NUMBER: _ClassVar[int]
        expected_tasks: _containers.RepeatedCompositeFieldContainer[_job_pb2.WorkerTaskStatus]
        def __init__(self, expected_tasks: _Optional[_Iterable[_Union[_job_pb2.WorkerTaskStatus, _Mapping]]] = ...) -> None: ...
    class PollTasksResponse(_message.Message):
        __slots__ = ("tasks",)
        TASKS_FIELD_NUMBER: _ClassVar[int]
        tasks: _containers.RepeatedCompositeFieldContainer[_job_pb2.WorkerTaskStatus]
        def __init__(self, tasks: _Optional[_Iterable[_Union[_job_pb2.WorkerTaskStatus, _Mapping]]] = ...) -> None: ...
    class ReconcileRequest(_message.Message):
        __slots__ = ("worker_id", "desired")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        DESIRED_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        desired: _containers.RepeatedCompositeFieldContainer[Worker.DesiredAttempt]
        def __init__(self, worker_id: _Optional[str] = ..., desired: _Optional[_Iterable[_Union[Worker.DesiredAttempt, _Mapping]]] = ...) -> None: ...
    class ReconcileResponse(_message.Message):
        __slots__ = ("worker_id", "observed", "health")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        OBSERVED_FIELD_NUMBER: _ClassVar[int]
        HEALTH_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        observed: _containers.RepeatedCompositeFieldContainer[Worker.AttemptObservation]
        health: Worker.WorkerHealth
        def __init__(self, worker_id: _Optional[str] = ..., observed: _Optional[_Iterable[_Union[Worker.AttemptObservation, _Mapping]]] = ..., health: _Optional[_Union[Worker.WorkerHealth, _Mapping]] = ...) -> None: ...
    class DesiredAttempt(_message.Message):
        __slots__ = ("attempt_uid", "run", "stop", "task_id", "attempt_id")
        ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
        RUN_FIELD_NUMBER: _ClassVar[int]
        STOP_FIELD_NUMBER: _ClassVar[int]
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
        attempt_uid: str
        run: Worker.AttemptSpec
        stop: Worker.StopReason
        task_id: str
        attempt_id: int
        def __init__(self, attempt_uid: _Optional[str] = ..., run: _Optional[_Union[Worker.AttemptSpec, _Mapping]] = ..., stop: _Optional[_Union[Worker.StopReason, str]] = ..., task_id: _Optional[str] = ..., attempt_id: _Optional[int] = ...) -> None: ...
    class AttemptSpec(_message.Message):
        __slots__ = ("request",)
        REQUEST_FIELD_NUMBER: _ClassVar[int]
        request: _job_pb2.RunTaskRequest
        def __init__(self, request: _Optional[_Union[_job_pb2.RunTaskRequest, _Mapping]] = ...) -> None: ...
    class AttemptObservation(_message.Message):
        __slots__ = ("attempt_uid", "state", "exit_code", "error", "container_id", "finished_at", "resource_usage", "task_id", "attempt_id")
        ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
        STATE_FIELD_NUMBER: _ClassVar[int]
        EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
        ERROR_FIELD_NUMBER: _ClassVar[int]
        CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
        FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
        RESOURCE_USAGE_FIELD_NUMBER: _ClassVar[int]
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
        attempt_uid: str
        state: _job_pb2.TaskState
        exit_code: int
        error: str
        container_id: str
        finished_at: _time_pb2.Timestamp
        resource_usage: _job_pb2.ResourceUsage
        task_id: str
        attempt_id: int
        def __init__(self, attempt_uid: _Optional[str] = ..., state: _Optional[_Union[_job_pb2.TaskState, str]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ..., container_id: _Optional[str] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., resource_usage: _Optional[_Union[_job_pb2.ResourceUsage, _Mapping]] = ..., task_id: _Optional[str] = ..., attempt_id: _Optional[int] = ...) -> None: ...
    class WorkerHealth(_message.Message):
        __slots__ = ("healthy", "health_error", "resources")
        HEALTHY_FIELD_NUMBER: _ClassVar[int]
        HEALTH_ERROR_FIELD_NUMBER: _ClassVar[int]
        RESOURCES_FIELD_NUMBER: _ClassVar[int]
        healthy: bool
        health_error: str
        resources: _job_pb2.WorkerResourceSnapshot
        def __init__(self, healthy: _Optional[bool] = ..., health_error: _Optional[str] = ..., resources: _Optional[_Union[_job_pb2.WorkerResourceSnapshot, _Mapping]] = ...) -> None: ...
    def __init__(self) -> None: ...
