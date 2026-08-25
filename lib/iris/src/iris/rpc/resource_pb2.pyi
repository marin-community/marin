from google.protobuf import any_pb2 as _any_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ResourceRequest(_message.Message):
    __slots__ = ("resource_type", "input")
    RESOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    resource_type: str
    input: _any_pb2.Any
    def __init__(self, resource_type: _Optional[str] = ..., input: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class Resource(_message.Message):
    __slots__ = ("body",)
    BODY_FIELD_NUMBER: _ClassVar[int]
    body: _any_pb2.Any
    def __init__(self, body: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class PageInfo(_message.Message):
    __slots__ = ("total_count", "has_more")
    TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    HAS_MORE_FIELD_NUMBER: _ClassVar[int]
    total_count: int
    has_more: bool
    def __init__(self, total_count: _Optional[int] = ..., has_more: _Optional[bool] = ...) -> None: ...

class GetResponse(_message.Message):
    __slots__ = ("resource",)
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    resource: Resource
    def __init__(self, resource: _Optional[_Union[Resource, _Mapping]] = ...) -> None: ...

class ListResponse(_message.Message):
    __slots__ = ("resources", "page")
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    resources: _containers.RepeatedCompositeFieldContainer[Resource]
    page: PageInfo
    def __init__(self, resources: _Optional[_Iterable[_Union[Resource, _Mapping]]] = ..., page: _Optional[_Union[PageInfo, _Mapping]] = ...) -> None: ...

class BatchGetResponse(_message.Message):
    __slots__ = ("resources",)
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    resources: _containers.RepeatedCompositeFieldContainer[Resource]
    def __init__(self, resources: _Optional[_Iterable[_Union[Resource, _Mapping]]] = ...) -> None: ...
