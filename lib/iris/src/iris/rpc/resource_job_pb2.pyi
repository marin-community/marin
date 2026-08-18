from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CancelJob(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class JobUpdate(_message.Message):
    __slots__ = ("cancel",)
    CANCEL_FIELD_NUMBER: _ClassVar[int]
    cancel: CancelJob
    def __init__(self, cancel: _Optional[_Union[CancelJob, _Mapping]] = ...) -> None: ...
