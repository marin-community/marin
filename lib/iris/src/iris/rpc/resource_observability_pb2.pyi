from . import iris_logging_pb2 as _iris_logging_pb2
from . import resource_pb2 as _resource_pb2
from . import resource_identity_pb2 as _resource_identity_pb2
from . import time_pb2 as _time_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class UserSummary(_message.Message):
    __slots__ = ("user_id", "task_state_counts", "job_state_counts", "role", "budget_limit", "budget_spent", "max_band", "budget_configured")
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
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_STATE_COUNTS_FIELD_NUMBER: _ClassVar[int]
    JOB_STATE_COUNTS_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    BUDGET_LIMIT_FIELD_NUMBER: _ClassVar[int]
    BUDGET_SPENT_FIELD_NUMBER: _ClassVar[int]
    MAX_BAND_FIELD_NUMBER: _ClassVar[int]
    BUDGET_CONFIGURED_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    task_state_counts: _containers.ScalarMap[str, int]
    job_state_counts: _containers.ScalarMap[str, int]
    role: str
    budget_limit: int
    budget_spent: int
    max_band: int
    budget_configured: bool
    def __init__(self, user_id: _Optional[str] = ..., task_state_counts: _Optional[_Mapping[str, int]] = ..., job_state_counts: _Optional[_Mapping[str, int]] = ..., role: _Optional[str] = ..., budget_limit: _Optional[int] = ..., budget_spent: _Optional[int] = ..., max_band: _Optional[int] = ..., budget_configured: _Optional[bool] = ...) -> None: ...

class UserQuery(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ActivityQuery(_message.Message):
    __slots__ = ("target", "attempt_uid", "after", "page")
    TARGET_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    AFTER_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    target: _resource_identity_pb2.ResourceKey
    attempt_uid: str
    after: _time_pb2.Timestamp
    page: _resource_pb2.PageRequest
    def __init__(self, target: _Optional[_Union[_resource_identity_pb2.ResourceKey, _Mapping]] = ..., attempt_uid: _Optional[str] = ..., after: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., page: _Optional[_Union[_resource_pb2.PageRequest, _Mapping]] = ...) -> None: ...

class ActivityEntry(_message.Message):
    __slots__ = ("entry_id", "occurred_at", "source", "severity", "kind", "message", "target", "attempt_uid", "correlation_id", "attributes")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ENTRY_ID_FIELD_NUMBER: _ClassVar[int]
    OCCURRED_AT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    entry_id: str
    occurred_at: _time_pb2.Timestamp
    source: str
    severity: str
    kind: str
    message: str
    target: _resource_identity_pb2.ResourceKey
    attempt_uid: str
    correlation_id: str
    attributes: _containers.ScalarMap[str, str]
    def __init__(self, entry_id: _Optional[str] = ..., occurred_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., source: _Optional[str] = ..., severity: _Optional[str] = ..., kind: _Optional[str] = ..., message: _Optional[str] = ..., target: _Optional[_Union[_resource_identity_pb2.ResourceKey, _Mapping]] = ..., attempt_uid: _Optional[str] = ..., correlation_id: _Optional[str] = ..., attributes: _Optional[_Mapping[str, str]] = ...) -> None: ...

class LogFilter(_message.Message):
    __slots__ = ("after", "cursor", "max_lines", "substring", "minimum_level", "tail")
    AFTER_FIELD_NUMBER: _ClassVar[int]
    CURSOR_FIELD_NUMBER: _ClassVar[int]
    MAX_LINES_FIELD_NUMBER: _ClassVar[int]
    SUBSTRING_FIELD_NUMBER: _ClassVar[int]
    MINIMUM_LEVEL_FIELD_NUMBER: _ClassVar[int]
    TAIL_FIELD_NUMBER: _ClassVar[int]
    after: _time_pb2.Timestamp
    cursor: int
    max_lines: int
    substring: str
    minimum_level: _iris_logging_pb2.LogLevel
    tail: bool
    def __init__(self, after: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., cursor: _Optional[int] = ..., max_lines: _Optional[int] = ..., substring: _Optional[str] = ..., minimum_level: _Optional[_Union[_iris_logging_pb2.LogLevel, str]] = ..., tail: _Optional[bool] = ...) -> None: ...

class LogTarget(_message.Message):
    __slots__ = ("job", "task", "attempt")
    JOB_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    job: _resource_identity_pb2.JobIdentity
    task: _resource_identity_pb2.TaskIdentity
    attempt: _resource_identity_pb2.AttemptIdentity
    def __init__(self, job: _Optional[_Union[_resource_identity_pb2.JobIdentity, _Mapping]] = ..., task: _Optional[_Union[_resource_identity_pb2.TaskIdentity, _Mapping]] = ..., attempt: _Optional[_Union[_resource_identity_pb2.AttemptIdentity, _Mapping]] = ...) -> None: ...

class LogQuery(_message.Message):
    __slots__ = ("target", "filter")
    TARGET_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    target: LogTarget
    filter: LogFilter
    def __init__(self, target: _Optional[_Union[LogTarget, _Mapping]] = ..., filter: _Optional[_Union[LogFilter, _Mapping]] = ...) -> None: ...
