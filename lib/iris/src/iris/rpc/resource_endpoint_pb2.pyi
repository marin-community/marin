from . import resource_pb2 as _resource_pb2
from . import resource_identity_pb2 as _resource_identity_pb2
from . import time_pb2 as _time_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EndpointAccess(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ENDPOINT_ACCESS_PRIVATE: _ClassVar[EndpointAccess]
    ENDPOINT_ACCESS_LINK: _ClassVar[EndpointAccess]
ENDPOINT_ACCESS_PRIVATE: EndpointAccess
ENDPOINT_ACCESS_LINK: EndpointAccess

class EndpointQuery(_message.Message):
    __slots__ = ("name_prefix", "task", "page")
    NAME_PREFIX_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    name_prefix: str
    task: _resource_identity_pb2.ResourceKey
    page: _resource_pb2.PageRequest
    def __init__(self, name_prefix: _Optional[str] = ..., task: _Optional[_Union[_resource_identity_pb2.ResourceKey, _Mapping]] = ..., page: _Optional[_Union[_resource_pb2.PageRequest, _Mapping]] = ...) -> None: ...

class EndpointSummary(_message.Message):
    __slots__ = ("key", "endpoint_id", "name", "task", "execution_cluster_id", "access", "lease_deadline")
    KEY_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    ACCESS_FIELD_NUMBER: _ClassVar[int]
    LEASE_DEADLINE_FIELD_NUMBER: _ClassVar[int]
    key: _resource_identity_pb2.ResourceKey
    endpoint_id: str
    name: str
    task: _resource_identity_pb2.ResourceKey
    execution_cluster_id: str
    access: EndpointAccess
    lease_deadline: _time_pb2.Timestamp
    def __init__(self, key: _Optional[_Union[_resource_identity_pb2.ResourceKey, _Mapping]] = ..., endpoint_id: _Optional[str] = ..., name: _Optional[str] = ..., task: _Optional[_Union[_resource_identity_pb2.ResourceKey, _Mapping]] = ..., execution_cluster_id: _Optional[str] = ..., access: _Optional[_Union[EndpointAccess, str]] = ..., lease_deadline: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class EndpointDetail(_message.Message):
    __slots__ = ("summary", "address", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    summary: EndpointSummary
    address: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, summary: _Optional[_Union[EndpointSummary, _Mapping]] = ..., address: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class CreateEndpointCapability(_message.Message):
    __slots__ = ("endpoint", "endpoint_name", "ttl")
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_NAME_FIELD_NUMBER: _ClassVar[int]
    TTL_FIELD_NUMBER: _ClassVar[int]
    endpoint: _resource_identity_pb2.ResourceKey
    endpoint_name: str
    ttl: _time_pb2.Duration
    def __init__(self, endpoint: _Optional[_Union[_resource_identity_pb2.ResourceKey, _Mapping]] = ..., endpoint_name: _Optional[str] = ..., ttl: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ...) -> None: ...

class EndpointCapability(_message.Message):
    __slots__ = ("token", "expires_at", "capability_url")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    CAPABILITY_URL_FIELD_NUMBER: _ClassVar[int]
    token: str
    expires_at: _time_pb2.Timestamp
    capability_url: str
    def __init__(self, token: _Optional[str] = ..., expires_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., capability_url: _Optional[str] = ...) -> None: ...
