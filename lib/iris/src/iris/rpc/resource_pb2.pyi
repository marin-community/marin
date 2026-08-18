from google.protobuf import any_pb2 as _any_pb2
from . import time_pb2 as _time_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OperationPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OPERATION_PHASE_UNSPECIFIED: _ClassVar[OperationPhase]
    OPERATION_PHASE_ACCEPTED: _ClassVar[OperationPhase]
    OPERATION_PHASE_VERIFYING: _ClassVar[OperationPhase]
    OPERATION_PHASE_VERIFIED: _ClassVar[OperationPhase]
    OPERATION_PHASE_FAILED: _ClassVar[OperationPhase]
OPERATION_PHASE_UNSPECIFIED: OperationPhase
OPERATION_PHASE_ACCEPTED: OperationPhase
OPERATION_PHASE_VERIFYING: OperationPhase
OPERATION_PHASE_VERIFIED: OperationPhase
OPERATION_PHASE_FAILED: OperationPhase

class ResourceRef(_message.Message):
    __slots__ = ("authority_cluster_id", "type", "id", "uid")
    AUTHORITY_CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    UID_FIELD_NUMBER: _ClassVar[int]
    authority_cluster_id: str
    type: str
    id: str
    uid: str
    def __init__(self, authority_cluster_id: _Optional[str] = ..., type: _Optional[str] = ..., id: _Optional[str] = ..., uid: _Optional[str] = ...) -> None: ...

class Resource(_message.Message):
    __slots__ = ("ref", "body")
    REF_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    ref: ResourceRef
    body: _any_pb2.Any
    def __init__(self, ref: _Optional[_Union[ResourceRef, _Mapping]] = ..., body: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class MutationMetadata(_message.Message):
    __slots__ = ("request_id", "reason")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    reason: str
    def __init__(self, request_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class OperationError(_message.Message):
    __slots__ = ("code", "message")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    code: int
    message: str
    def __init__(self, code: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class Operation(_message.Message):
    __slots__ = ("ref", "phase", "verb", "requested_ref", "resolved_ref", "affected", "error", "accepted_at", "applied_at", "completed_at")
    REF_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    VERB_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_REF_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_REF_FIELD_NUMBER: _ClassVar[int]
    AFFECTED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_AT_FIELD_NUMBER: _ClassVar[int]
    APPLIED_AT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    ref: ResourceRef
    phase: OperationPhase
    verb: str
    requested_ref: ResourceRef
    resolved_ref: ResourceRef
    affected: _containers.RepeatedCompositeFieldContainer[ResourceRef]
    error: OperationError
    accepted_at: _time_pb2.Timestamp
    applied_at: _time_pb2.Timestamp
    completed_at: _time_pb2.Timestamp
    def __init__(self, ref: _Optional[_Union[ResourceRef, _Mapping]] = ..., phase: _Optional[_Union[OperationPhase, str]] = ..., verb: _Optional[str] = ..., requested_ref: _Optional[_Union[ResourceRef, _Mapping]] = ..., resolved_ref: _Optional[_Union[ResourceRef, _Mapping]] = ..., affected: _Optional[_Iterable[_Union[ResourceRef, _Mapping]]] = ..., error: _Optional[_Union[OperationError, _Mapping]] = ..., accepted_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., applied_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., completed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GetResourceRequest(_message.Message):
    __slots__ = ("ref",)
    REF_FIELD_NUMBER: _ClassVar[int]
    ref: ResourceRef
    def __init__(self, ref: _Optional[_Union[ResourceRef, _Mapping]] = ...) -> None: ...

class GetResourceResponse(_message.Message):
    __slots__ = ("resource",)
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    resource: Resource
    def __init__(self, resource: _Optional[_Union[Resource, _Mapping]] = ...) -> None: ...

class UpdateResourceRequest(_message.Message):
    __slots__ = ("mutation", "ref", "update")
    MUTATION_FIELD_NUMBER: _ClassVar[int]
    REF_FIELD_NUMBER: _ClassVar[int]
    UPDATE_FIELD_NUMBER: _ClassVar[int]
    mutation: MutationMetadata
    ref: ResourceRef
    update: _any_pb2.Any
    def __init__(self, mutation: _Optional[_Union[MutationMetadata, _Mapping]] = ..., ref: _Optional[_Union[ResourceRef, _Mapping]] = ..., update: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class ResourceCapability(_message.Message):
    __slots__ = ("type", "verbs", "update_type_urls")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VERBS_FIELD_NUMBER: _ClassVar[int]
    UPDATE_TYPE_URLS_FIELD_NUMBER: _ClassVar[int]
    type: str
    verbs: _containers.RepeatedScalarFieldContainer[str]
    update_type_urls: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, type: _Optional[str] = ..., verbs: _Optional[_Iterable[str]] = ..., update_type_urls: _Optional[_Iterable[str]] = ...) -> None: ...

class GetServiceInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetServiceInfoResponse(_message.Message):
    __slots__ = ("protocol_version", "resources")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    protocol_version: str
    resources: _containers.RepeatedCompositeFieldContainer[ResourceCapability]
    def __init__(self, protocol_version: _Optional[str] = ..., resources: _Optional[_Iterable[_Union[ResourceCapability, _Mapping]]] = ...) -> None: ...
