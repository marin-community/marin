from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class JobStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_STATUS_UNSPECIFIED: _ClassVar[JobStatus]
    JOB_STATUS_PENDING: _ClassVar[JobStatus]
    JOB_STATUS_BUILDING: _ClassVar[JobStatus]
    JOB_STATUS_RUNNING: _ClassVar[JobStatus]
    JOB_STATUS_SUCCEEDED: _ClassVar[JobStatus]
    JOB_STATUS_FAILED: _ClassVar[JobStatus]
    JOB_STATUS_KILLED: _ClassVar[JobStatus]
JOB_STATUS_UNSPECIFIED: JobStatus
JOB_STATUS_PENDING: JobStatus
JOB_STATUS_BUILDING: JobStatus
JOB_STATUS_RUNNING: JobStatus
JOB_STATUS_SUCCEEDED: JobStatus
JOB_STATUS_FAILED: JobStatus
JOB_STATUS_KILLED: JobStatus

class JobSpec(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_RESOURCES_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_GCS_PATH_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_HASH_FIELD_NUMBER: _ClassVar[int]
    name: str
    serialized_entrypoint: bytes
    serialized_resources: bytes
    serialized_environment: bytes
    bundle_gcs_path: str
    bundle_hash: str
    def __init__(self, name: _Optional[str] = ..., serialized_entrypoint: _Optional[bytes] = ..., serialized_resources: _Optional[bytes] = ..., serialized_environment: _Optional[bytes] = ..., bundle_gcs_path: _Optional[str] = ..., bundle_hash: _Optional[str] = ...) -> None: ...

class JobHandle(_message.Message):
    __slots__ = ()
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: JobStatus
    worker_id: str
    error: str
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[_Union[JobStatus, str]] = ..., worker_id: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class Endpoint(_message.Message):
    __slots__ = ()
    class MetadataEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    endpoint_id: str
    name: str
    address: str
    job_id: str
    namespace: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, endpoint_id: _Optional[str] = ..., name: _Optional[str] = ..., address: _Optional[str] = ..., job_id: _Optional[str] = ..., namespace: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class RegisterEndpointRequest(_message.Message):
    __slots__ = ()
    class MetadataEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    name: str
    address: str
    job_id: str
    namespace: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, name: _Optional[str] = ..., address: _Optional[str] = ..., job_id: _Optional[str] = ..., namespace: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class LookupRequest(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    name: str
    namespace: str
    def __init__(self, name: _Optional[str] = ..., namespace: _Optional[str] = ...) -> None: ...

class LookupResponse(_message.Message):
    __slots__ = ()
    ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    endpoints: _containers.RepeatedCompositeFieldContainer[Endpoint]
    def __init__(self, endpoints: _Optional[_Iterable[_Union[Endpoint, _Mapping]]] = ...) -> None: ...

class ListJobsRequest(_message.Message):
    __slots__ = ()
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    def __init__(self, namespace: _Optional[str] = ...) -> None: ...

class ListJobsResponse(_message.Message):
    __slots__ = ()
    JOBS_FIELD_NUMBER: _ClassVar[int]
    jobs: _containers.RepeatedCompositeFieldContainer[JobHandle]
    def __init__(self, jobs: _Optional[_Iterable[_Union[JobHandle, _Mapping]]] = ...) -> None: ...

class ListEndpointsRequest(_message.Message):
    __slots__ = ()
    PREFIX_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    prefix: str
    namespace: str
    def __init__(self, prefix: _Optional[str] = ..., namespace: _Optional[str] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ControllerService(_service.service): ...

class ControllerService_Stub(ControllerService): ...
