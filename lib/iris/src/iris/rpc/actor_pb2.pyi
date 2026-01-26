from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class ActorCall(_message.Message):
    __slots__ = ("actor_name", "method_name", "serialized_args", "serialized_kwargs")
    METHOD_NAME_FIELD_NUMBER: _ClassVar[int]
    ACTOR_NAME_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_ARGS_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_KWARGS_FIELD_NUMBER: _ClassVar[int]
    method_name: str
    actor_name: str
    serialized_args: bytes
    serialized_kwargs: bytes
    def __init__(
        self,
        method_name: str | None = ...,
        actor_name: str | None = ...,
        serialized_args: bytes | None = ...,
        serialized_kwargs: bytes | None = ...,
    ) -> None: ...

class ActorResponse(_message.Message):
    __slots__ = ("error", "serialized_value")
    SERIALIZED_VALUE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    serialized_value: bytes
    error: ActorError
    def __init__(self, serialized_value: bytes | None = ..., error: ActorError | _Mapping | None = ...) -> None: ...

class ActorError(_message.Message):
    __slots__ = ("error_type", "message", "serialized_exception")
    ERROR_TYPE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_EXCEPTION_FIELD_NUMBER: _ClassVar[int]
    error_type: str
    message: str
    serialized_exception: bytes
    def __init__(
        self, error_type: str | None = ..., message: str | None = ..., serialized_exception: bytes | None = ...
    ) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("healthy",)
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    def __init__(self, healthy: bool | None = ...) -> None: ...

class ListMethodsRequest(_message.Message):
    __slots__ = ("actor_name",)
    ACTOR_NAME_FIELD_NUMBER: _ClassVar[int]
    actor_name: str
    def __init__(self, actor_name: str | None = ...) -> None: ...

class MethodInfo(_message.Message):
    __slots__ = ("docstring", "name", "signature")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    DOCSTRING_FIELD_NUMBER: _ClassVar[int]
    name: str
    signature: str
    docstring: str
    def __init__(self, name: str | None = ..., signature: str | None = ..., docstring: str | None = ...) -> None: ...

class ListMethodsResponse(_message.Message):
    __slots__ = ("methods",)
    METHODS_FIELD_NUMBER: _ClassVar[int]
    methods: _containers.RepeatedCompositeFieldContainer[MethodInfo]
    def __init__(self, methods: _Iterable[MethodInfo | _Mapping] | None = ...) -> None: ...

class ListActorsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ActorInfo(_message.Message):
    __slots__ = ("actor_id", "metadata", "name", "registered_at_ms")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...

    NAME_FIELD_NUMBER: _ClassVar[int]
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    REGISTERED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    name: str
    actor_id: str
    registered_at_ms: int
    metadata: _containers.ScalarMap[str, str]
    def __init__(
        self,
        name: str | None = ...,
        actor_id: str | None = ...,
        registered_at_ms: int | None = ...,
        metadata: _Mapping[str, str] | None = ...,
    ) -> None: ...

class ListActorsResponse(_message.Message):
    __slots__ = ("actors",)
    ACTORS_FIELD_NUMBER: _ClassVar[int]
    actors: _containers.RepeatedCompositeFieldContainer[ActorInfo]
    def __init__(self, actors: _Iterable[ActorInfo | _Mapping] | None = ...) -> None: ...

class ActorService(_service.service): ...
class ActorService_Stub(ActorService): ...
