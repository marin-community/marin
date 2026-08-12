from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ResourceKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESOURCE_KIND_UNSPECIFIED: _ClassVar[ResourceKind]
    RESOURCE_KIND_JOB: _ClassVar[ResourceKind]
    RESOURCE_KIND_TASK: _ClassVar[ResourceKind]
    RESOURCE_KIND_ATTEMPT: _ClassVar[ResourceKind]
    RESOURCE_KIND_ENDPOINT: _ClassVar[ResourceKind]
    RESOURCE_KIND_NODE: _ClassVar[ResourceKind]
    RESOURCE_KIND_SLICE: _ClassVar[ResourceKind]
RESOURCE_KIND_UNSPECIFIED: ResourceKind
RESOURCE_KIND_JOB: ResourceKind
RESOURCE_KIND_TASK: ResourceKind
RESOURCE_KIND_ATTEMPT: ResourceKind
RESOURCE_KIND_ENDPOINT: ResourceKind
RESOURCE_KIND_NODE: ResourceKind
RESOURCE_KIND_SLICE: ResourceKind

class ResourceKey(_message.Message):
    __slots__ = ("cluster_id", "kind", "resource_id")
    CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    cluster_id: str
    kind: ResourceKind
    resource_id: str
    def __init__(self, cluster_id: _Optional[str] = ..., kind: _Optional[_Union[ResourceKind, str]] = ..., resource_id: _Optional[str] = ...) -> None: ...

class JobIdentity(_message.Message):
    __slots__ = ("key", "job_uid")
    KEY_FIELD_NUMBER: _ClassVar[int]
    JOB_UID_FIELD_NUMBER: _ClassVar[int]
    key: ResourceKey
    job_uid: str
    def __init__(self, key: _Optional[_Union[ResourceKey, _Mapping]] = ..., job_uid: _Optional[str] = ...) -> None: ...

class TaskIdentity(_message.Message):
    __slots__ = ("key", "task_uid")
    KEY_FIELD_NUMBER: _ClassVar[int]
    TASK_UID_FIELD_NUMBER: _ClassVar[int]
    key: ResourceKey
    task_uid: str
    def __init__(self, key: _Optional[_Union[ResourceKey, _Mapping]] = ..., task_uid: _Optional[str] = ...) -> None: ...

class AttemptLocator(_message.Message):
    __slots__ = ("task", "attempt_number")
    TASK_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_NUMBER_FIELD_NUMBER: _ClassVar[int]
    task: ResourceKey
    attempt_number: int
    def __init__(self, task: _Optional[_Union[ResourceKey, _Mapping]] = ..., attempt_number: _Optional[int] = ...) -> None: ...

class AttemptIdentity(_message.Message):
    __slots__ = ("task", "attempt_number", "attempt_uid")
    TASK_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_NUMBER_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    task: ResourceKey
    attempt_number: int
    attempt_uid: str
    def __init__(self, task: _Optional[_Union[ResourceKey, _Mapping]] = ..., attempt_number: _Optional[int] = ..., attempt_uid: _Optional[str] = ...) -> None: ...

class NodeIdentity(_message.Message):
    __slots__ = ("key", "backend_id", "node_uid")
    KEY_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_UID_FIELD_NUMBER: _ClassVar[int]
    key: ResourceKey
    backend_id: str
    node_uid: str
    def __init__(self, key: _Optional[_Union[ResourceKey, _Mapping]] = ..., backend_id: _Optional[str] = ..., node_uid: _Optional[str] = ...) -> None: ...

class NodeLocator(_message.Message):
    __slots__ = ("key", "backend_id", "node_uid")
    KEY_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_UID_FIELD_NUMBER: _ClassVar[int]
    key: ResourceKey
    backend_id: str
    node_uid: str
    def __init__(self, key: _Optional[_Union[ResourceKey, _Mapping]] = ..., backend_id: _Optional[str] = ..., node_uid: _Optional[str] = ...) -> None: ...

class SliceIdentity(_message.Message):
    __slots__ = ("key", "backend_id", "slice_uid")
    KEY_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    SLICE_UID_FIELD_NUMBER: _ClassVar[int]
    key: ResourceKey
    backend_id: str
    slice_uid: str
    def __init__(self, key: _Optional[_Union[ResourceKey, _Mapping]] = ..., backend_id: _Optional[str] = ..., slice_uid: _Optional[str] = ...) -> None: ...

class SliceLocator(_message.Message):
    __slots__ = ("key", "backend_id", "slice_uid")
    KEY_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    SLICE_UID_FIELD_NUMBER: _ClassVar[int]
    key: ResourceKey
    backend_id: str
    slice_uid: str
    def __init__(self, key: _Optional[_Union[ResourceKey, _Mapping]] = ..., backend_id: _Optional[str] = ..., slice_uid: _Optional[str] = ...) -> None: ...
