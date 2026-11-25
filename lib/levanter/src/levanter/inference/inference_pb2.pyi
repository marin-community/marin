from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GenerateRequest(_message.Message):
    __slots__ = ("prompt_token_ids", "max_new_tokens", "temperature", "top_p", "seed", "stop_sequences", "num_choices")
    PROMPT_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    MAX_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    STOP_SEQUENCES_FIELD_NUMBER: _ClassVar[int]
    NUM_CHOICES_FIELD_NUMBER: _ClassVar[int]
    prompt_token_ids: _containers.RepeatedScalarFieldContainer[int]
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    stop_sequences: _containers.RepeatedCompositeFieldContainer[TokenSequence]
    num_choices: int
    def __init__(self, prompt_token_ids: _Optional[_Iterable[int]] = ..., max_new_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., seed: _Optional[int] = ..., stop_sequences: _Optional[_Iterable[_Union[TokenSequence, _Mapping]]] = ..., num_choices: _Optional[int] = ...) -> None: ...

class TokenSequence(_message.Message):
    __slots__ = ("token_ids",)
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, token_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("prompt_token_ids", "choices")
    PROMPT_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    CHOICES_FIELD_NUMBER: _ClassVar[int]
    prompt_token_ids: _containers.RepeatedScalarFieldContainer[int]
    choices: _containers.RepeatedCompositeFieldContainer[Choice]
    def __init__(self, prompt_token_ids: _Optional[_Iterable[int]] = ..., choices: _Optional[_Iterable[_Union[Choice, _Mapping]]] = ...) -> None: ...

class Choice(_message.Message):
    __slots__ = ("token_ids", "behavior_logprobs")
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    BEHAVIOR_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    behavior_logprobs: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, token_ids: _Optional[_Iterable[int]] = ..., behavior_logprobs: _Optional[_Iterable[float]] = ...) -> None: ...
