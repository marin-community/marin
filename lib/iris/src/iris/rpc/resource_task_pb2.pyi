from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PreemptAttempt(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class FailAttempt(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TerminateAttempt(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TaskUpdate(_message.Message):
    __slots__ = ("preempt", "fail", "terminate")
    PREEMPT_FIELD_NUMBER: _ClassVar[int]
    FAIL_FIELD_NUMBER: _ClassVar[int]
    TERMINATE_FIELD_NUMBER: _ClassVar[int]
    preempt: PreemptAttempt
    fail: FailAttempt
    terminate: TerminateAttempt
    def __init__(self, preempt: _Optional[_Union[PreemptAttempt, _Mapping]] = ..., fail: _Optional[_Union[FailAttempt, _Mapping]] = ..., terminate: _Optional[_Union[TerminateAttempt, _Mapping]] = ...) -> None: ...

class AttemptUpdate(_message.Message):
    __slots__ = ("preempt", "fail", "terminate")
    PREEMPT_FIELD_NUMBER: _ClassVar[int]
    FAIL_FIELD_NUMBER: _ClassVar[int]
    TERMINATE_FIELD_NUMBER: _ClassVar[int]
    preempt: PreemptAttempt
    fail: FailAttempt
    terminate: TerminateAttempt
    def __init__(self, preempt: _Optional[_Union[PreemptAttempt, _Mapping]] = ..., fail: _Optional[_Union[FailAttempt, _Mapping]] = ..., terminate: _Optional[_Union[TerminateAttempt, _Mapping]] = ...) -> None: ...
