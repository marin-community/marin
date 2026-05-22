from . import time_pb2 as _time_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ErrorDetails(_message.Message):
    __slots__ = ("exception_type", "message", "traceback", "timestamp")
    EXCEPTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TRACEBACK_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    exception_type: str
    message: str
    traceback: str
    timestamp: _time_pb2.Timestamp
    def __init__(self, exception_type: _Optional[str] = ..., message: _Optional[str] = ..., traceback: _Optional[str] = ..., timestamp: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...
