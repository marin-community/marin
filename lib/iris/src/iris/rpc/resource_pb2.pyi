from google.protobuf import any_pb2 as _any_pb2
from . import time_pb2 as _time_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SourceState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SOURCE_STATE_UNSPECIFIED: _ClassVar[SourceState]
    SOURCE_STATE_AVAILABLE: _ClassVar[SourceState]
    SOURCE_STATE_UNAVAILABLE: _ClassVar[SourceState]
    SOURCE_STATE_UNSUPPORTED: _ClassVar[SourceState]

class Freshness(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FRESHNESS_UNSPECIFIED: _ClassVar[Freshness]
    FRESHNESS_CURRENT: _ClassVar[Freshness]
    FRESHNESS_STALE: _ClassVar[Freshness]
    FRESHNESS_UNKNOWN: _ClassVar[Freshness]

class ResourceView(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESOURCE_VIEW_UNSPECIFIED: _ClassVar[ResourceView]
    RESOURCE_VIEW_BASIC: _ClassVar[ResourceView]
    RESOURCE_VIEW_FULL: _ClassVar[ResourceView]

class OperationPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OPERATION_PHASE_UNSPECIFIED: _ClassVar[OperationPhase]
    OPERATION_PHASE_ACCEPTED: _ClassVar[OperationPhase]
    OPERATION_PHASE_APPLIED: _ClassVar[OperationPhase]
    OPERATION_PHASE_VERIFYING: _ClassVar[OperationPhase]
    OPERATION_PHASE_VERIFIED: _ClassVar[OperationPhase]
    OPERATION_PHASE_FAILED: _ClassVar[OperationPhase]
SOURCE_STATE_UNSPECIFIED: SourceState
SOURCE_STATE_AVAILABLE: SourceState
SOURCE_STATE_UNAVAILABLE: SourceState
SOURCE_STATE_UNSUPPORTED: SourceState
FRESHNESS_UNSPECIFIED: Freshness
FRESHNESS_CURRENT: Freshness
FRESHNESS_STALE: Freshness
FRESHNESS_UNKNOWN: Freshness
RESOURCE_VIEW_UNSPECIFIED: ResourceView
RESOURCE_VIEW_BASIC: ResourceView
RESOURCE_VIEW_FULL: ResourceView
OPERATION_PHASE_UNSPECIFIED: OperationPhase
OPERATION_PHASE_ACCEPTED: OperationPhase
OPERATION_PHASE_APPLIED: OperationPhase
OPERATION_PHASE_VERIFYING: OperationPhase
OPERATION_PHASE_VERIFIED: OperationPhase
OPERATION_PHASE_FAILED: OperationPhase

class ResourceSourceStatus(_message.Message):
    __slots__ = ("source_id", "backend_id", "state", "freshness", "observed_at", "error_code", "error_message")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    FRESHNESS_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    backend_id: str
    state: SourceState
    freshness: Freshness
    observed_at: _time_pb2.Timestamp
    error_code: str
    error_message: str
    def __init__(self, source_id: _Optional[str] = ..., backend_id: _Optional[str] = ..., state: _Optional[_Union[SourceState, str]] = ..., freshness: _Optional[_Union[Freshness, str]] = ..., observed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ...) -> None: ...

class PageRequest(_message.Message):
    __slots__ = ("page_size", "page_token")
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    page_size: int
    page_token: str
    def __init__(self, page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class PageInfo(_message.Message):
    __slots__ = ("next_page_token", "source_statuses")
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    next_page_token: str
    source_statuses: _containers.RepeatedCompositeFieldContainer[ResourceSourceStatus]
    def __init__(self, next_page_token: _Optional[str] = ..., source_statuses: _Optional[_Iterable[_Union[ResourceSourceStatus, _Mapping]]] = ...) -> None: ...

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
    __slots__ = ("ref", "body", "etag")
    REF_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    ETAG_FIELD_NUMBER: _ClassVar[int]
    ref: ResourceRef
    body: _any_pb2.Any
    etag: str
    def __init__(self, ref: _Optional[_Union[ResourceRef, _Mapping]] = ..., body: _Optional[_Union[_any_pb2.Any, _Mapping]] = ..., etag: _Optional[str] = ...) -> None: ...

class MutationMetadata(_message.Message):
    __slots__ = ("request_id", "reason")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    reason: str
    def __init__(self, request_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class ResourceError(_message.Message):
    __slots__ = ("code", "reason", "message")
    CODE_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    code: int
    reason: str
    message: str
    def __init__(self, code: _Optional[int] = ..., reason: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class GetResourceRequest(_message.Message):
    __slots__ = ("ref", "view")
    REF_FIELD_NUMBER: _ClassVar[int]
    VIEW_FIELD_NUMBER: _ClassVar[int]
    ref: ResourceRef
    view: ResourceView
    def __init__(self, ref: _Optional[_Union[ResourceRef, _Mapping]] = ..., view: _Optional[_Union[ResourceView, str]] = ...) -> None: ...

class GetResourceResponse(_message.Message):
    __slots__ = ("resource", "source_statuses")
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    resource: Resource
    source_statuses: _containers.RepeatedCompositeFieldContainer[ResourceSourceStatus]
    def __init__(self, resource: _Optional[_Union[Resource, _Mapping]] = ..., source_statuses: _Optional[_Iterable[_Union[ResourceSourceStatus, _Mapping]]] = ...) -> None: ...

class ListResourcesRequest(_message.Message):
    __slots__ = ("type", "query", "page_size", "page_token", "view")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    VIEW_FIELD_NUMBER: _ClassVar[int]
    type: str
    query: _any_pb2.Any
    page_size: int
    page_token: str
    view: ResourceView
    def __init__(self, type: _Optional[str] = ..., query: _Optional[_Union[_any_pb2.Any, _Mapping]] = ..., page_size: _Optional[int] = ..., page_token: _Optional[str] = ..., view: _Optional[_Union[ResourceView, str]] = ...) -> None: ...

class ListResourcesResponse(_message.Message):
    __slots__ = ("resources", "next_page_token", "source_statuses")
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    resources: _containers.RepeatedCompositeFieldContainer[Resource]
    next_page_token: str
    source_statuses: _containers.RepeatedCompositeFieldContainer[ResourceSourceStatus]
    def __init__(self, resources: _Optional[_Iterable[_Union[Resource, _Mapping]]] = ..., next_page_token: _Optional[str] = ..., source_statuses: _Optional[_Iterable[_Union[ResourceSourceStatus, _Mapping]]] = ...) -> None: ...

class BatchGetResourcesRequest(_message.Message):
    __slots__ = ("type", "refs", "view")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    REFS_FIELD_NUMBER: _ClassVar[int]
    VIEW_FIELD_NUMBER: _ClassVar[int]
    type: str
    refs: _containers.RepeatedCompositeFieldContainer[ResourceRef]
    view: ResourceView
    def __init__(self, type: _Optional[str] = ..., refs: _Optional[_Iterable[_Union[ResourceRef, _Mapping]]] = ..., view: _Optional[_Union[ResourceView, str]] = ...) -> None: ...

class BatchGetResourceResult(_message.Message):
    __slots__ = ("resource", "error")
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    resource: Resource
    error: ResourceError
    def __init__(self, resource: _Optional[_Union[Resource, _Mapping]] = ..., error: _Optional[_Union[ResourceError, _Mapping]] = ...) -> None: ...

class BatchGetResourcesResponse(_message.Message):
    __slots__ = ("results", "source_statuses")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[BatchGetResourceResult]
    source_statuses: _containers.RepeatedCompositeFieldContainer[ResourceSourceStatus]
    def __init__(self, results: _Optional[_Iterable[_Union[BatchGetResourceResult, _Mapping]]] = ..., source_statuses: _Optional[_Iterable[_Union[ResourceSourceStatus, _Mapping]]] = ...) -> None: ...

class CreateResourceRequest(_message.Message):
    __slots__ = ("mutation", "type", "parent", "id", "body")
    MUTATION_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PARENT_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    mutation: MutationMetadata
    type: str
    parent: ResourceRef
    id: str
    body: _any_pb2.Any
    def __init__(self, mutation: _Optional[_Union[MutationMetadata, _Mapping]] = ..., type: _Optional[str] = ..., parent: _Optional[_Union[ResourceRef, _Mapping]] = ..., id: _Optional[str] = ..., body: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class UpdateResourceRequest(_message.Message):
    __slots__ = ("mutation", "ref", "update", "if_match")
    MUTATION_FIELD_NUMBER: _ClassVar[int]
    REF_FIELD_NUMBER: _ClassVar[int]
    UPDATE_FIELD_NUMBER: _ClassVar[int]
    IF_MATCH_FIELD_NUMBER: _ClassVar[int]
    mutation: MutationMetadata
    ref: ResourceRef
    update: _any_pb2.Any
    if_match: str
    def __init__(self, mutation: _Optional[_Union[MutationMetadata, _Mapping]] = ..., ref: _Optional[_Union[ResourceRef, _Mapping]] = ..., update: _Optional[_Union[_any_pb2.Any, _Mapping]] = ..., if_match: _Optional[str] = ...) -> None: ...

class DeleteResourceRequest(_message.Message):
    __slots__ = ("mutation", "ref", "if_match")
    MUTATION_FIELD_NUMBER: _ClassVar[int]
    REF_FIELD_NUMBER: _ClassVar[int]
    IF_MATCH_FIELD_NUMBER: _ClassVar[int]
    mutation: MutationMetadata
    ref: ResourceRef
    if_match: str
    def __init__(self, mutation: _Optional[_Union[MutationMetadata, _Mapping]] = ..., ref: _Optional[_Union[ResourceRef, _Mapping]] = ..., if_match: _Optional[str] = ...) -> None: ...

class Operation(_message.Message):
    __slots__ = ("ref", "phase", "verb", "requested_ref", "resolved_ref", "affected", "result", "error", "accepted_at", "applied_at", "completed_at")
    REF_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    VERB_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_REF_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_REF_FIELD_NUMBER: _ClassVar[int]
    AFFECTED_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
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
    result: _any_pb2.Any
    error: ResourceError
    accepted_at: _time_pb2.Timestamp
    applied_at: _time_pb2.Timestamp
    completed_at: _time_pb2.Timestamp
    def __init__(self, ref: _Optional[_Union[ResourceRef, _Mapping]] = ..., phase: _Optional[_Union[OperationPhase, str]] = ..., verb: _Optional[str] = ..., requested_ref: _Optional[_Union[ResourceRef, _Mapping]] = ..., resolved_ref: _Optional[_Union[ResourceRef, _Mapping]] = ..., affected: _Optional[_Iterable[_Union[ResourceRef, _Mapping]]] = ..., result: _Optional[_Union[_any_pb2.Any, _Mapping]] = ..., error: _Optional[_Union[ResourceError, _Mapping]] = ..., accepted_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., applied_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., completed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ResourceCapability(_message.Message):
    __slots__ = ("type", "verbs", "views", "body_type_urls", "query_type_urls", "update_type_urls", "features", "create_type_urls")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VERBS_FIELD_NUMBER: _ClassVar[int]
    VIEWS_FIELD_NUMBER: _ClassVar[int]
    BODY_TYPE_URLS_FIELD_NUMBER: _ClassVar[int]
    QUERY_TYPE_URLS_FIELD_NUMBER: _ClassVar[int]
    UPDATE_TYPE_URLS_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    CREATE_TYPE_URLS_FIELD_NUMBER: _ClassVar[int]
    type: str
    verbs: _containers.RepeatedScalarFieldContainer[str]
    views: _containers.RepeatedScalarFieldContainer[ResourceView]
    body_type_urls: _containers.RepeatedScalarFieldContainer[str]
    query_type_urls: _containers.RepeatedScalarFieldContainer[str]
    update_type_urls: _containers.RepeatedScalarFieldContainer[str]
    features: _containers.RepeatedScalarFieldContainer[str]
    create_type_urls: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, type: _Optional[str] = ..., verbs: _Optional[_Iterable[str]] = ..., views: _Optional[_Iterable[_Union[ResourceView, str]]] = ..., body_type_urls: _Optional[_Iterable[str]] = ..., query_type_urls: _Optional[_Iterable[str]] = ..., update_type_urls: _Optional[_Iterable[str]] = ..., features: _Optional[_Iterable[str]] = ..., create_type_urls: _Optional[_Iterable[str]] = ...) -> None: ...

class BackendResourceCapability(_message.Message):
    __slots__ = ("backend_id", "type", "verbs", "features")
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VERBS_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    backend_id: str
    type: str
    verbs: _containers.RepeatedScalarFieldContainer[str]
    features: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, backend_id: _Optional[str] = ..., type: _Optional[str] = ..., verbs: _Optional[_Iterable[str]] = ..., features: _Optional[_Iterable[str]] = ...) -> None: ...

class GetServiceInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetServiceInfoResponse(_message.Message):
    __slots__ = ("protocol_version", "controller_generation", "resources", "backend_resources")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_GENERATION_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    BACKEND_RESOURCES_FIELD_NUMBER: _ClassVar[int]
    protocol_version: str
    controller_generation: str
    resources: _containers.RepeatedCompositeFieldContainer[ResourceCapability]
    backend_resources: _containers.RepeatedCompositeFieldContainer[BackendResourceCapability]
    def __init__(self, protocol_version: _Optional[str] = ..., controller_generation: _Optional[str] = ..., resources: _Optional[_Iterable[_Union[ResourceCapability, _Mapping]]] = ..., backend_resources: _Optional[_Iterable[_Union[BackendResourceCapability, _Mapping]]] = ...) -> None: ...
