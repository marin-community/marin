from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExpertIdDtype(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXPERT_ID_DTYPE_UNSPECIFIED: _ClassVar[ExpertIdDtype]
    EXPERT_ID_DTYPE_UINT8: _ClassVar[ExpertIdDtype]
    EXPERT_ID_DTYPE_UINT16_LE: _ClassVar[ExpertIdDtype]

class LogprobMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LOGPROB_MODE_UNSPECIFIED: _ClassVar[LogprobMode]
    LOGPROB_MODE_BEHAVIOR: _ClassVar[LogprobMode]
    LOGPROB_MODE_POLICY: _ClassVar[LogprobMode]

class FinishReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FINISH_REASON_UNSPECIFIED: _ClassVar[FinishReason]
    FINISH_REASON_STOP: _ClassVar[FinishReason]
    FINISH_REASON_LENGTH: _ClassVar[FinishReason]
    FINISH_REASON_ABORT: _ClassVar[FinishReason]
EXPERT_ID_DTYPE_UNSPECIFIED: ExpertIdDtype
EXPERT_ID_DTYPE_UINT8: ExpertIdDtype
EXPERT_ID_DTYPE_UINT16_LE: ExpertIdDtype
LOGPROB_MODE_UNSPECIFIED: LogprobMode
LOGPROB_MODE_BEHAVIOR: LogprobMode
LOGPROB_MODE_POLICY: LogprobMode
FINISH_REASON_UNSPECIFIED: FinishReason
FINISH_REASON_STOP: FinishReason
FINISH_REASON_LENGTH: FinishReason
FINISH_REASON_ABORT: FinishReason

class GenerateRequest(_message.Message):
    __slots__ = ("prompt_token_ids", "max_new_tokens", "temperature", "top_p", "seed", "stop_sequences", "num_choices", "expected_policy_id", "expected_logprob_mode", "capture_router_replay")
    PROMPT_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    MAX_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    STOP_SEQUENCES_FIELD_NUMBER: _ClassVar[int]
    NUM_CHOICES_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_POLICY_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_LOGPROB_MODE_FIELD_NUMBER: _ClassVar[int]
    CAPTURE_ROUTER_REPLAY_FIELD_NUMBER: _ClassVar[int]
    prompt_token_ids: _containers.RepeatedScalarFieldContainer[int]
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    stop_sequences: _containers.RepeatedCompositeFieldContainer[TokenSequence]
    num_choices: int
    expected_policy_id: str
    expected_logprob_mode: LogprobMode
    capture_router_replay: bool
    def __init__(self, prompt_token_ids: _Optional[_Iterable[int]] = ..., max_new_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., seed: _Optional[int] = ..., stop_sequences: _Optional[_Iterable[_Union[TokenSequence, _Mapping]]] = ..., num_choices: _Optional[int] = ..., expected_policy_id: _Optional[str] = ..., expected_logprob_mode: _Optional[_Union[LogprobMode, str]] = ..., capture_router_replay: bool = ...) -> None: ...

class TokenSequence(_message.Message):
    __slots__ = ("token_ids",)
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, token_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("prompt_token_ids", "choices", "logprob_mode")
    PROMPT_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    CHOICES_FIELD_NUMBER: _ClassVar[int]
    LOGPROB_MODE_FIELD_NUMBER: _ClassVar[int]
    prompt_token_ids: _containers.RepeatedScalarFieldContainer[int]
    choices: _containers.RepeatedCompositeFieldContainer[Choice]
    logprob_mode: LogprobMode
    def __init__(self, prompt_token_ids: _Optional[_Iterable[int]] = ..., choices: _Optional[_Iterable[_Union[Choice, _Mapping]]] = ..., logprob_mode: _Optional[_Union[LogprobMode, str]] = ...) -> None: ...

class Choice(_message.Message):
    __slots__ = ("token_ids", "behavior_logprobs", "finish_reason", "policy_id", "model_version", "policy_logprobs", "router_replay")
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    BEHAVIOR_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    POLICY_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    POLICY_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    ROUTER_REPLAY_FIELD_NUMBER: _ClassVar[int]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    behavior_logprobs: _containers.RepeatedScalarFieldContainer[float]
    finish_reason: FinishReason
    policy_id: str
    model_version: int
    policy_logprobs: _containers.RepeatedScalarFieldContainer[float]
    router_replay: RouterReplay
    def __init__(self, token_ids: _Optional[_Iterable[int]] = ..., behavior_logprobs: _Optional[_Iterable[float]] = ..., finish_reason: _Optional[_Union[FinishReason, str]] = ..., policy_id: _Optional[str] = ..., model_version: _Optional[int] = ..., policy_logprobs: _Optional[_Iterable[float]] = ..., router_replay: _Optional[_Union[RouterReplay, _Mapping]] = ...) -> None: ...

class RouterReplay(_message.Message):
    __slots__ = ("num_layers", "num_positions", "experts_per_token", "valid", "expert_ids", "expert_id_dtype")
    NUM_LAYERS_FIELD_NUMBER: _ClassVar[int]
    NUM_POSITIONS_FIELD_NUMBER: _ClassVar[int]
    EXPERTS_PER_TOKEN_FIELD_NUMBER: _ClassVar[int]
    VALID_FIELD_NUMBER: _ClassVar[int]
    EXPERT_IDS_FIELD_NUMBER: _ClassVar[int]
    EXPERT_ID_DTYPE_FIELD_NUMBER: _ClassVar[int]
    num_layers: int
    num_positions: int
    experts_per_token: int
    valid: _containers.RepeatedScalarFieldContainer[bool]
    expert_ids: bytes
    expert_id_dtype: ExpertIdDtype
    def __init__(self, num_layers: _Optional[int] = ..., num_positions: _Optional[int] = ..., experts_per_token: _Optional[int] = ..., valid: _Optional[_Iterable[bool]] = ..., expert_ids: _Optional[bytes] = ..., expert_id_dtype: _Optional[_Union[ExpertIdDtype, str]] = ...) -> None: ...
