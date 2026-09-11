from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ColumnMeta(_message.Message):
    __slots__ = ("name", "type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    def __init__(self, name: _Optional[str] = ..., type: _Optional[str] = ...) -> None: ...

class RawQueryRequest(_message.Message):
    __slots__ = ("sql",)
    SQL_FIELD_NUMBER: _ClassVar[int]
    sql: str
    def __init__(self, sql: _Optional[str] = ...) -> None: ...

class RawQueryResponse(_message.Message):
    __slots__ = ("columns", "rows")
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    columns: _containers.RepeatedCompositeFieldContainer[ColumnMeta]
    rows: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, columns: _Optional[_Iterable[_Union[ColumnMeta, _Mapping]]] = ..., rows: _Optional[_Iterable[str]] = ...) -> None: ...
