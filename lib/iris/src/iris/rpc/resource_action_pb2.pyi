from . import resource_identity_pb2 as _resource_identity_pb2
from . import time_pb2 as _time_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActionKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_KIND_UNSPECIFIED: _ClassVar[ActionKind]
    ACTION_KIND_CANCEL_JOB: _ClassVar[ActionKind]
    ACTION_KIND_RETRY_TASK: _ClassVar[ActionKind]
    ACTION_KIND_TERMINATE_ATTEMPT: _ClassVar[ActionKind]
    ACTION_KIND_FAIL_ATTEMPT: _ClassVar[ActionKind]

class ActionState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_STATE_UNSPECIFIED: _ClassVar[ActionState]
    ACTION_STATE_ACCEPTED: _ClassVar[ActionState]
    ACTION_STATE_VERIFYING: _ClassVar[ActionState]
    ACTION_STATE_SUCCEEDED: _ClassVar[ActionState]
    ACTION_STATE_FAILED: _ClassVar[ActionState]

class ActionResult(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_RESULT_UNSPECIFIED: _ClassVar[ActionResult]
    ACTION_RESULT_NONE: _ClassVar[ActionResult]
    ACTION_RESULT_SATISFIED: _ClassVar[ActionResult]
    ACTION_RESULT_TARGET_ABSENT: _ClassVar[ActionResult]
    ACTION_RESULT_PROVIDER_REJECTED: _ClassVar[ActionResult]
    ACTION_RESULT_INTERNAL_ERROR: _ClassVar[ActionResult]
ACTION_KIND_UNSPECIFIED: ActionKind
ACTION_KIND_CANCEL_JOB: ActionKind
ACTION_KIND_RETRY_TASK: ActionKind
ACTION_KIND_TERMINATE_ATTEMPT: ActionKind
ACTION_KIND_FAIL_ATTEMPT: ActionKind
ACTION_STATE_UNSPECIFIED: ActionState
ACTION_STATE_ACCEPTED: ActionState
ACTION_STATE_VERIFYING: ActionState
ACTION_STATE_SUCCEEDED: ActionState
ACTION_STATE_FAILED: ActionState
ACTION_RESULT_UNSPECIFIED: ActionResult
ACTION_RESULT_NONE: ActionResult
ACTION_RESULT_SATISFIED: ActionResult
ACTION_RESULT_TARGET_ABSENT: ActionResult
ACTION_RESULT_PROVIDER_REJECTED: ActionResult
ACTION_RESULT_INTERNAL_ERROR: ActionResult

class ActionReceipt(_message.Message):
    __slots__ = ("action_id", "kind", "target", "expected_target_uid", "expected_attempt_uid", "state", "result_code", "result_message", "created_at", "updated_at", "completed_at", "expected_attempt_number")
    ACTION_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_TARGET_UID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    RESULT_CODE_FIELD_NUMBER: _ClassVar[int]
    RESULT_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_ATTEMPT_NUMBER_FIELD_NUMBER: _ClassVar[int]
    action_id: str
    kind: ActionKind
    target: _resource_identity_pb2.ResourceKey
    expected_target_uid: str
    expected_attempt_uid: str
    state: ActionState
    result_code: ActionResult
    result_message: str
    created_at: _time_pb2.Timestamp
    updated_at: _time_pb2.Timestamp
    completed_at: _time_pb2.Timestamp
    expected_attempt_number: int
    def __init__(self, action_id: _Optional[str] = ..., kind: _Optional[_Union[ActionKind, str]] = ..., target: _Optional[_Union[_resource_identity_pb2.ResourceKey, _Mapping]] = ..., expected_target_uid: _Optional[str] = ..., expected_attempt_uid: _Optional[str] = ..., state: _Optional[_Union[ActionState, str]] = ..., result_code: _Optional[_Union[ActionResult, str]] = ..., result_message: _Optional[str] = ..., created_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., completed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., expected_attempt_number: _Optional[int] = ...) -> None: ...
