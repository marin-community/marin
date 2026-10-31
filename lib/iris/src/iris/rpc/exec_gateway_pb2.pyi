from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class OpenRequest(_message.Message):
    __slots__ = ("grant",)
    GRANT_FIELD_NUMBER: _ClassVar[int]
    grant: str
    def __init__(self, grant: _Optional[str] = ...) -> None: ...

class OpenResponse(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class SendRequest(_message.Message):
    __slots__ = ("grant", "session_id", "stdin", "eof")
    GRANT_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    STDIN_FIELD_NUMBER: _ClassVar[int]
    EOF_FIELD_NUMBER: _ClassVar[int]
    grant: str
    session_id: str
    stdin: bytes
    eof: bool
    def __init__(self, grant: _Optional[str] = ..., session_id: _Optional[str] = ..., stdin: _Optional[bytes] = ..., eof: _Optional[bool] = ...) -> None: ...

class SendResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ReceiveRequest(_message.Message):
    __slots__ = ("grant", "session_id", "max_bytes", "wait_ms")
    GRANT_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    WAIT_MS_FIELD_NUMBER: _ClassVar[int]
    grant: str
    session_id: str
    max_bytes: int
    wait_ms: int
    def __init__(self, grant: _Optional[str] = ..., session_id: _Optional[str] = ..., max_bytes: _Optional[int] = ..., wait_ms: _Optional[int] = ...) -> None: ...

class ReceiveResponse(_message.Message):
    __slots__ = ("stdout", "stderr", "exited", "exit_code", "error")
    STDOUT_FIELD_NUMBER: _ClassVar[int]
    STDERR_FIELD_NUMBER: _ClassVar[int]
    EXITED_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    stdout: bytes
    stderr: bytes
    exited: bool
    exit_code: int
    error: str
    def __init__(self, stdout: _Optional[bytes] = ..., stderr: _Optional[bytes] = ..., exited: _Optional[bool] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ...) -> None: ...

class CloseRequest(_message.Message):
    __slots__ = ("grant", "session_id")
    GRANT_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    grant: str
    session_id: str
    def __init__(self, grant: _Optional[str] = ..., session_id: _Optional[str] = ...) -> None: ...

class CloseResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
