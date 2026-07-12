from . import time_pb2 as _time_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetRpcStatsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RpcMethodStats(_message.Message):
    __slots__ = ("method", "count", "error_count", "total_duration_ms", "max_duration_ms", "p50_ms", "p95_ms", "p99_ms", "bucket_upper_bounds_ms", "bucket_counts", "last_call")
    METHOD_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    ERROR_COUNT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    P50_MS_FIELD_NUMBER: _ClassVar[int]
    P95_MS_FIELD_NUMBER: _ClassVar[int]
    P99_MS_FIELD_NUMBER: _ClassVar[int]
    BUCKET_UPPER_BOUNDS_MS_FIELD_NUMBER: _ClassVar[int]
    BUCKET_COUNTS_FIELD_NUMBER: _ClassVar[int]
    LAST_CALL_FIELD_NUMBER: _ClassVar[int]
    method: str
    count: int
    error_count: int
    total_duration_ms: float
    max_duration_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    bucket_upper_bounds_ms: _containers.RepeatedScalarFieldContainer[int]
    bucket_counts: _containers.RepeatedScalarFieldContainer[int]
    last_call: _time_pb2.Timestamp
    def __init__(self, method: _Optional[str] = ..., count: _Optional[int] = ..., error_count: _Optional[int] = ..., total_duration_ms: _Optional[float] = ..., max_duration_ms: _Optional[float] = ..., p50_ms: _Optional[float] = ..., p95_ms: _Optional[float] = ..., p99_ms: _Optional[float] = ..., bucket_upper_bounds_ms: _Optional[_Iterable[int]] = ..., bucket_counts: _Optional[_Iterable[int]] = ..., last_call: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class RpcCallSample(_message.Message):
    __slots__ = ("method", "timestamp", "duration_ms", "peer", "user_agent", "caller", "error_code", "error_message", "request_preview")
    METHOD_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    PEER_FIELD_NUMBER: _ClassVar[int]
    USER_AGENT_FIELD_NUMBER: _ClassVar[int]
    CALLER_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_PREVIEW_FIELD_NUMBER: _ClassVar[int]
    method: str
    timestamp: _time_pb2.Timestamp
    duration_ms: float
    peer: str
    user_agent: str
    caller: str
    error_code: str
    error_message: str
    request_preview: str
    def __init__(self, method: _Optional[str] = ..., timestamp: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., duration_ms: _Optional[float] = ..., peer: _Optional[str] = ..., user_agent: _Optional[str] = ..., caller: _Optional[str] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., request_preview: _Optional[str] = ...) -> None: ...

class GetRpcStatsResponse(_message.Message):
    __slots__ = ("methods", "slow_samples", "discovery_samples", "collector_started_at")
    METHODS_FIELD_NUMBER: _ClassVar[int]
    SLOW_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    DISCOVERY_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    COLLECTOR_STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    methods: _containers.RepeatedCompositeFieldContainer[RpcMethodStats]
    slow_samples: _containers.RepeatedCompositeFieldContainer[RpcCallSample]
    discovery_samples: _containers.RepeatedCompositeFieldContainer[RpcCallSample]
    collector_started_at: _time_pb2.Timestamp
    def __init__(self, methods: _Optional[_Iterable[_Union[RpcMethodStats, _Mapping]]] = ..., slow_samples: _Optional[_Iterable[_Union[RpcCallSample, _Mapping]]] = ..., discovery_samples: _Optional[_Iterable[_Union[RpcCallSample, _Mapping]]] = ..., collector_started_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...
