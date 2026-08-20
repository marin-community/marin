from . import resource_identity_pb2 as _resource_identity_pb2
from . import time_pb2 as _time_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CpuProfile(_message.Message):
    __slots__ = ("format", "rate_hz", "native")
    class Format(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        FORMAT_UNSPECIFIED: _ClassVar[CpuProfile.Format]
        FLAMEGRAPH: _ClassVar[CpuProfile.Format]
        SPEEDSCOPE: _ClassVar[CpuProfile.Format]
        RAW: _ClassVar[CpuProfile.Format]
    FORMAT_UNSPECIFIED: CpuProfile.Format
    FLAMEGRAPH: CpuProfile.Format
    SPEEDSCOPE: CpuProfile.Format
    RAW: CpuProfile.Format
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    RATE_HZ_FIELD_NUMBER: _ClassVar[int]
    NATIVE_FIELD_NUMBER: _ClassVar[int]
    format: CpuProfile.Format
    rate_hz: int
    native: bool
    def __init__(self, format: _Optional[_Union[CpuProfile.Format, str]] = ..., rate_hz: _Optional[int] = ..., native: _Optional[bool] = ...) -> None: ...

class MemoryProfile(_message.Message):
    __slots__ = ("format", "leaks")
    class Format(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        FORMAT_UNSPECIFIED: _ClassVar[MemoryProfile.Format]
        FLAMEGRAPH: _ClassVar[MemoryProfile.Format]
        TABLE: _ClassVar[MemoryProfile.Format]
        STATS: _ClassVar[MemoryProfile.Format]
        RAW: _ClassVar[MemoryProfile.Format]
    FORMAT_UNSPECIFIED: MemoryProfile.Format
    FLAMEGRAPH: MemoryProfile.Format
    TABLE: MemoryProfile.Format
    STATS: MemoryProfile.Format
    RAW: MemoryProfile.Format
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    LEAKS_FIELD_NUMBER: _ClassVar[int]
    format: MemoryProfile.Format
    leaks: bool
    def __init__(self, format: _Optional[_Union[MemoryProfile.Format, str]] = ..., leaks: _Optional[bool] = ...) -> None: ...

class ThreadsProfile(_message.Message):
    __slots__ = ("locals", "native")
    LOCALS_FIELD_NUMBER: _ClassVar[int]
    NATIVE_FIELD_NUMBER: _ClassVar[int]
    locals: bool
    native: bool
    def __init__(self, locals: _Optional[bool] = ..., native: _Optional[bool] = ...) -> None: ...

class ProfileType(_message.Message):
    __slots__ = ("cpu", "memory", "threads")
    CPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    THREADS_FIELD_NUMBER: _ClassVar[int]
    cpu: CpuProfile
    memory: MemoryProfile
    threads: ThreadsProfile
    def __init__(self, cpu: _Optional[_Union[CpuProfile, _Mapping]] = ..., memory: _Optional[_Union[MemoryProfile, _Mapping]] = ..., threads: _Optional[_Union[ThreadsProfile, _Mapping]] = ...) -> None: ...

class CreateExecSession(_message.Message):
    __slots__ = ("attempt", "command", "timeout")
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    attempt: _resource_identity_pb2.AttemptIdentity
    command: _containers.RepeatedScalarFieldContainer[str]
    timeout: _time_pb2.Duration
    def __init__(self, attempt: _Optional[_Union[_resource_identity_pb2.AttemptIdentity, _Mapping]] = ..., command: _Optional[_Iterable[str]] = ..., timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ...) -> None: ...

class ExecSessionResult(_message.Message):
    __slots__ = ("exit_code", "stdout", "stderr", "error_message")
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    STDOUT_FIELD_NUMBER: _ClassVar[int]
    STDERR_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    exit_code: int
    stdout: str
    stderr: str
    error_message: str
    def __init__(self, exit_code: _Optional[int] = ..., stdout: _Optional[str] = ..., stderr: _Optional[str] = ..., error_message: _Optional[str] = ...) -> None: ...

class CreateProfileCapture(_message.Message):
    __slots__ = ("attempt", "profile", "duration")
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    attempt: _resource_identity_pb2.AttemptIdentity
    profile: ProfileType
    duration: _time_pb2.Duration
    def __init__(self, attempt: _Optional[_Union[_resource_identity_pb2.AttemptIdentity, _Mapping]] = ..., profile: _Optional[_Union[ProfileType, _Mapping]] = ..., duration: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ...) -> None: ...

class ProfileCaptureResult(_message.Message):
    __slots__ = ("profile_data", "error_message")
    PROFILE_DATA_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    profile_data: bytes
    error_message: str
    def __init__(self, profile_data: _Optional[bytes] = ..., error_message: _Optional[str] = ...) -> None: ...
