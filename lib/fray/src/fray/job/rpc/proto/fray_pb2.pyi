from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TaskStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TASK_STATUS_UNSPECIFIED: _ClassVar[TaskStatus]
    TASK_STATUS_PENDING: _ClassVar[TaskStatus]
    TASK_STATUS_RUNNING: _ClassVar[TaskStatus]
    TASK_STATUS_COMPLETED: _ClassVar[TaskStatus]
    TASK_STATUS_FAILED: _ClassVar[TaskStatus]
TASK_STATUS_UNSPECIFIED: TaskStatus
TASK_STATUS_PENDING: TaskStatus
TASK_STATUS_RUNNING: TaskStatus
TASK_STATUS_COMPLETED: TaskStatus
TASK_STATUS_FAILED: TaskStatus

class TaskSpec(_message.Message):
    __slots__ = ()
    class ResourcesEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_FN_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRIES_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    serialized_fn: bytes
    resources: _containers.ScalarMap[str, int]
    max_retries: int
    def __init__(self, task_id: _Optional[str] = ..., serialized_fn: _Optional[bytes] = ..., resources: _Optional[_Mapping[str, int]] = ..., max_retries: _Optional[int] = ...) -> None: ...

class TaskHandle(_message.Message):
    __slots__ = ()
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    status: TaskStatus
    worker_id: str
    error: str
    def __init__(self, task_id: _Optional[str] = ..., status: _Optional[_Union[TaskStatus, str]] = ..., worker_id: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class TaskResult(_message.Message):
    __slots__ = ()
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_RESULT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    serialized_result: bytes
    error: str
    def __init__(self, task_id: _Optional[str] = ..., serialized_result: _Optional[bytes] = ..., error: _Optional[str] = ...) -> None: ...

class WorkerInfo(_message.Message):
    __slots__ = ()
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    NUM_CPUS_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    address: str
    num_cpus: int
    memory_bytes: int
    def __init__(self, worker_id: _Optional[str] = ..., address: _Optional[str] = ..., num_cpus: _Optional[int] = ..., memory_bytes: _Optional[int] = ...) -> None: ...

class GetTaskRequest(_message.Message):
    __slots__ = ()
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    def __init__(self, worker_id: _Optional[str] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class WorkerTask(_message.Message):
    __slots__ = ()
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    status: TaskStatus
    started_at_ms: int
    def __init__(self, task_id: _Optional[str] = ..., status: _Optional[_Union[TaskStatus, str]] = ..., started_at_ms: _Optional[int] = ...) -> None: ...

class WorkerStatus(_message.Message):
    __slots__ = ()
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    CURRENT_TASKS_FIELD_NUMBER: _ClassVar[int]
    UPTIME_MS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    healthy: bool
    current_tasks: _containers.RepeatedCompositeFieldContainer[WorkerTask]
    uptime_ms: int
    def __init__(self, worker_id: _Optional[str] = ..., healthy: _Optional[bool] = ..., current_tasks: _Optional[_Iterable[_Union[WorkerTask, _Mapping]]] = ..., uptime_ms: _Optional[int] = ...) -> None: ...

class FrayController(_service.service): ...

class FrayController_Stub(FrayController): ...

class FrayWorker(_service.service): ...

class FrayWorker_Stub(FrayWorker): ...
