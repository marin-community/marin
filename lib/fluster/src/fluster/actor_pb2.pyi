from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActorCall(_message.Message):
    __slots__ = ("method_name", "serialized_args", "serialized_kwargs")
    METHOD_NAME_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_ARGS_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_KWARGS_FIELD_NUMBER: _ClassVar[int]
    method_name: str
    serialized_args: bytes
    serialized_kwargs: bytes
    def __init__(self, method_name: _Optional[str] = ..., serialized_args: _Optional[bytes] = ..., serialized_kwargs: _Optional[bytes] = ...) -> None: ...

class ActorResponse(_message.Message):
    __slots__ = ("serialized_value", "error")
    SERIALIZED_VALUE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    serialized_value: bytes
    error: ActorError
    def __init__(self, serialized_value: _Optional[bytes] = ..., error: _Optional[_Union[ActorError, _Mapping]] = ...) -> None: ...

class ActorError(_message.Message):
    __slots__ = ("error_type", "message", "serialized_exception")
    ERROR_TYPE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_EXCEPTION_FIELD_NUMBER: _ClassVar[int]
    error_type: str
    message: str
    serialized_exception: bytes
    def __init__(self, error_type: _Optional[str] = ..., message: _Optional[str] = ..., serialized_exception: _Optional[bytes] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("healthy",)
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    def __init__(self, healthy: _Optional[bool] = ...) -> None: ...

class ActorService(_service.service): ...

class ActorService_Stub(ActorService): ...
