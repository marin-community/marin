from google.protobuf import any_pb2 as _any_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ResourceRequest(_message.Message):
    __slots__ = ("resource_type", "input")
    RESOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    resource_type: str
    input: _any_pb2.Any
    def __init__(self, resource_type: _Optional[str] = ..., input: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class ResourceResponse(_message.Message):
    __slots__ = ("output",)
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    output: _any_pb2.Any
    def __init__(self, output: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...
