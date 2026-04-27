from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LogLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LOG_LEVEL_UNKNOWN: _ClassVar[LogLevel]
    LOG_LEVEL_DEBUG: _ClassVar[LogLevel]
    LOG_LEVEL_INFO: _ClassVar[LogLevel]
    LOG_LEVEL_WARNING: _ClassVar[LogLevel]
    LOG_LEVEL_ERROR: _ClassVar[LogLevel]
    LOG_LEVEL_CRITICAL: _ClassVar[LogLevel]
LOG_LEVEL_UNKNOWN: LogLevel
LOG_LEVEL_DEBUG: LogLevel
LOG_LEVEL_INFO: LogLevel
LOG_LEVEL_WARNING: LogLevel
LOG_LEVEL_ERROR: LogLevel
LOG_LEVEL_CRITICAL: LogLevel

class Timestamp(_message.Message):
    __slots__ = ("epoch_ms",)
    EPOCH_MS_FIELD_NUMBER: _ClassVar[int]
    epoch_ms: int
    def __init__(self, epoch_ms: _Optional[int] = ...) -> None: ...

class LogEntry(_message.Message):
    __slots__ = ("timestamp", "source", "data", "attempt_id", "level", "key")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    timestamp: Timestamp
    source: str
    data: str
    attempt_id: int
    level: LogLevel
    key: str
    def __init__(self, timestamp: _Optional[_Union[Timestamp, _Mapping]] = ..., source: _Optional[str] = ..., data: _Optional[str] = ..., attempt_id: _Optional[int] = ..., level: _Optional[_Union[LogLevel, str]] = ..., key: _Optional[str] = ...) -> None: ...

class LogBatch(_message.Message):
    __slots__ = ("entries",)
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[LogEntry]
    def __init__(self, entries: _Optional[_Iterable[_Union[LogEntry, _Mapping]]] = ...) -> None: ...

class PushLogsRequest(_message.Message):
    __slots__ = ("key", "entries")
    KEY_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    key: str
    entries: _containers.RepeatedCompositeFieldContainer[LogEntry]
    def __init__(self, key: _Optional[str] = ..., entries: _Optional[_Iterable[_Union[LogEntry, _Mapping]]] = ...) -> None: ...

class PushLogsResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class FetchLogsRequest(_message.Message):
    __slots__ = ("source", "since_ms", "cursor", "substring", "max_lines", "tail", "min_level")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SINCE_MS_FIELD_NUMBER: _ClassVar[int]
    CURSOR_FIELD_NUMBER: _ClassVar[int]
    SUBSTRING_FIELD_NUMBER: _ClassVar[int]
    MAX_LINES_FIELD_NUMBER: _ClassVar[int]
    TAIL_FIELD_NUMBER: _ClassVar[int]
    MIN_LEVEL_FIELD_NUMBER: _ClassVar[int]
    source: str
    since_ms: int
    cursor: int
    substring: str
    max_lines: int
    tail: bool
    min_level: str
    def __init__(self, source: _Optional[str] = ..., since_ms: _Optional[int] = ..., cursor: _Optional[int] = ..., substring: _Optional[str] = ..., max_lines: _Optional[int] = ..., tail: _Optional[bool] = ..., min_level: _Optional[str] = ...) -> None: ...

class FetchLogsResponse(_message.Message):
    __slots__ = ("entries", "cursor")
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    CURSOR_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[LogEntry]
    cursor: int
    def __init__(self, entries: _Optional[_Iterable[_Union[LogEntry, _Mapping]]] = ..., cursor: _Optional[int] = ...) -> None: ...
