from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ErrorDetails(_message.Message):
    __slots__ = ("exception_type", "message", "traceback", "timestamp_ms")
    EXCEPTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TRACEBACK_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    exception_type: str
    message: str
    traceback: str
    timestamp_ms: int
    def __init__(self, exception_type: _Optional[str] = ..., message: _Optional[str] = ..., traceback: _Optional[str] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...
