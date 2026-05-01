from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ColumnType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COLUMN_TYPE_UNKNOWN: _ClassVar[ColumnType]
    COLUMN_TYPE_STRING: _ClassVar[ColumnType]
    COLUMN_TYPE_INT64: _ClassVar[ColumnType]
    COLUMN_TYPE_FLOAT64: _ClassVar[ColumnType]
    COLUMN_TYPE_BOOL: _ClassVar[ColumnType]
    COLUMN_TYPE_TIMESTAMP_MS: _ClassVar[ColumnType]
    COLUMN_TYPE_BYTES: _ClassVar[ColumnType]
    COLUMN_TYPE_INT32: _ClassVar[ColumnType]
COLUMN_TYPE_UNKNOWN: ColumnType
COLUMN_TYPE_STRING: ColumnType
COLUMN_TYPE_INT64: ColumnType
COLUMN_TYPE_FLOAT64: ColumnType
COLUMN_TYPE_BOOL: ColumnType
COLUMN_TYPE_TIMESTAMP_MS: ColumnType
COLUMN_TYPE_BYTES: ColumnType
COLUMN_TYPE_INT32: ColumnType

class Column(_message.Message):
    __slots__ = ("name", "type", "nullable")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NULLABLE_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: ColumnType
    nullable: bool
    def __init__(self, name: _Optional[str] = ..., type: _Optional[_Union[ColumnType, str]] = ..., nullable: _Optional[bool] = ...) -> None: ...

class Schema(_message.Message):
    __slots__ = ("columns", "key_column")
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    KEY_COLUMN_FIELD_NUMBER: _ClassVar[int]
    columns: _containers.RepeatedCompositeFieldContainer[Column]
    key_column: str
    def __init__(self, columns: _Optional[_Iterable[_Union[Column, _Mapping]]] = ..., key_column: _Optional[str] = ...) -> None: ...

class RegisterTableRequest(_message.Message):
    __slots__ = ("namespace", "schema")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    schema: Schema
    def __init__(self, namespace: _Optional[str] = ..., schema: _Optional[_Union[Schema, _Mapping]] = ...) -> None: ...

class RegisterTableResponse(_message.Message):
    __slots__ = ("effective_schema",)
    EFFECTIVE_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    effective_schema: Schema
    def __init__(self, effective_schema: _Optional[_Union[Schema, _Mapping]] = ...) -> None: ...

class WriteRowsRequest(_message.Message):
    __slots__ = ("namespace", "arrow_ipc")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    ARROW_IPC_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    arrow_ipc: bytes
    def __init__(self, namespace: _Optional[str] = ..., arrow_ipc: _Optional[bytes] = ...) -> None: ...

class WriteRowsResponse(_message.Message):
    __slots__ = ("rows_written",)
    ROWS_WRITTEN_FIELD_NUMBER: _ClassVar[int]
    rows_written: int
    def __init__(self, rows_written: _Optional[int] = ...) -> None: ...

class QueryRequest(_message.Message):
    __slots__ = ("sql",)
    SQL_FIELD_NUMBER: _ClassVar[int]
    sql: str
    def __init__(self, sql: _Optional[str] = ...) -> None: ...

class QueryResponse(_message.Message):
    __slots__ = ("arrow_ipc", "row_count")
    ARROW_IPC_FIELD_NUMBER: _ClassVar[int]
    ROW_COUNT_FIELD_NUMBER: _ClassVar[int]
    arrow_ipc: bytes
    row_count: int
    def __init__(self, arrow_ipc: _Optional[bytes] = ..., row_count: _Optional[int] = ...) -> None: ...

class DropTableRequest(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    def __init__(self, namespace: _Optional[str] = ...) -> None: ...

class DropTableResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class NamespaceInfo(_message.Message):
    __slots__ = ("namespace", "schema")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    schema: Schema
    def __init__(self, namespace: _Optional[str] = ..., schema: _Optional[_Union[Schema, _Mapping]] = ...) -> None: ...

class ListNamespacesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListNamespacesResponse(_message.Message):
    __slots__ = ("namespaces",)
    NAMESPACES_FIELD_NUMBER: _ClassVar[int]
    namespaces: _containers.RepeatedCompositeFieldContainer[NamespaceInfo]
    def __init__(self, namespaces: _Optional[_Iterable[_Union[NamespaceInfo, _Mapping]]] = ...) -> None: ...

class GetTableSchemaRequest(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    def __init__(self, namespace: _Optional[str] = ...) -> None: ...

class GetTableSchemaResponse(_message.Message):
    __slots__ = ("schema",)
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    schema: Schema
    def __init__(self, schema: _Optional[_Union[Schema, _Mapping]] = ...) -> None: ...
