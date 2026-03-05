from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Timestamp(_message.Message):
    __slots__ = ("epoch_ms",)
    EPOCH_MS_FIELD_NUMBER: _ClassVar[int]
    epoch_ms: int
    def __init__(self, epoch_ms: _Optional[int] = ...) -> None: ...

class Duration(_message.Message):
    __slots__ = ("milliseconds",)
    MILLISECONDS_FIELD_NUMBER: _ClassVar[int]
    milliseconds: int
    def __init__(self, milliseconds: _Optional[int] = ...) -> None: ...
