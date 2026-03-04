from . import time_pb2 as _time_pb2
from . import cluster_pb2 as _cluster_pb2
from . import config_pb2 as _config_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ControllerSnapshot(_message.Message):
    __slots__ = ("schema_version", "created_at", "jobs", "endpoints", "workers", "scaling_groups", "tracked_workers", "reservation_claims")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    JOBS_FIELD_NUMBER: _ClassVar[int]
    ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    WORKERS_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUPS_FIELD_NUMBER: _ClassVar[int]
    TRACKED_WORKERS_FIELD_NUMBER: _ClassVar[int]
    RESERVATION_CLAIMS_FIELD_NUMBER: _ClassVar[int]
    schema_version: int
    created_at: _time_pb2.Timestamp
    jobs: _containers.RepeatedCompositeFieldContainer[JobSnapshot]
    endpoints: _containers.RepeatedCompositeFieldContainer[EndpointSnapshot]
    workers: _containers.RepeatedCompositeFieldContainer[WorkerSnapshot]
    scaling_groups: _containers.RepeatedCompositeFieldContainer[ScalingGroupSnapshot]
    tracked_workers: _containers.RepeatedCompositeFieldContainer[TrackedWorkerSnapshot]
    reservation_claims: _containers.RepeatedCompositeFieldContainer[ReservationClaimSnapshot]
    def __init__(self, schema_version: _Optional[int] = ..., created_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., jobs: _Optional[_Iterable[_Union[JobSnapshot, _Mapping]]] = ..., endpoints: _Optional[_Iterable[_Union[EndpointSnapshot, _Mapping]]] = ..., workers: _Optional[_Iterable[_Union[WorkerSnapshot, _Mapping]]] = ..., scaling_groups: _Optional[_Iterable[_Union[ScalingGroupSnapshot, _Mapping]]] = ..., tracked_workers: _Optional[_Iterable[_Union[TrackedWorkerSnapshot, _Mapping]]] = ..., reservation_claims: _Optional[_Iterable[_Union[ReservationClaimSnapshot, _Mapping]]] = ...) -> None: ...

class JobSnapshot(_message.Message):
    __slots__ = ("job_id", "request", "state", "submitted_at", "root_submitted_at", "started_at", "finished_at", "error", "exit_code", "num_tasks", "scheduling_deadline_epoch_ms", "tasks")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    SUBMITTED_AT_FIELD_NUMBER: _ClassVar[int]
    ROOT_SUBMITTED_AT_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    NUM_TASKS_FIELD_NUMBER: _ClassVar[int]
    SCHEDULING_DEADLINE_EPOCH_MS_FIELD_NUMBER: _ClassVar[int]
    TASKS_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    request: _cluster_pb2.Controller.LaunchJobRequest
    state: int
    submitted_at: _time_pb2.Timestamp
    root_submitted_at: _time_pb2.Timestamp
    started_at: _time_pb2.Timestamp
    finished_at: _time_pb2.Timestamp
    error: str
    exit_code: int
    num_tasks: int
    scheduling_deadline_epoch_ms: int
    tasks: _containers.RepeatedCompositeFieldContainer[TaskSnapshot]
    def __init__(self, job_id: _Optional[str] = ..., request: _Optional[_Union[_cluster_pb2.Controller.LaunchJobRequest, _Mapping]] = ..., state: _Optional[int] = ..., submitted_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., root_submitted_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., started_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., error: _Optional[str] = ..., exit_code: _Optional[int] = ..., num_tasks: _Optional[int] = ..., scheduling_deadline_epoch_ms: _Optional[int] = ..., tasks: _Optional[_Iterable[_Union[TaskSnapshot, _Mapping]]] = ...) -> None: ...

class TaskSnapshot(_message.Message):
    __slots__ = ("task_id", "job_id", "state", "error", "exit_code", "started_at", "finished_at", "submitted_at", "max_retries_failure", "max_retries_preemption", "failure_count", "preemption_count", "attempts", "reservation_entry_idx")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    SUBMITTED_AT_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRIES_FAILURE_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRIES_PREEMPTION_FIELD_NUMBER: _ClassVar[int]
    FAILURE_COUNT_FIELD_NUMBER: _ClassVar[int]
    PREEMPTION_COUNT_FIELD_NUMBER: _ClassVar[int]
    ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    RESERVATION_ENTRY_IDX_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    job_id: str
    state: int
    error: str
    exit_code: int
    started_at: _time_pb2.Timestamp
    finished_at: _time_pb2.Timestamp
    submitted_at: _time_pb2.Timestamp
    max_retries_failure: int
    max_retries_preemption: int
    failure_count: int
    preemption_count: int
    attempts: _containers.RepeatedCompositeFieldContainer[TaskAttemptSnapshot]
    reservation_entry_idx: int
    def __init__(self, task_id: _Optional[str] = ..., job_id: _Optional[str] = ..., state: _Optional[int] = ..., error: _Optional[str] = ..., exit_code: _Optional[int] = ..., started_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., submitted_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., max_retries_failure: _Optional[int] = ..., max_retries_preemption: _Optional[int] = ..., failure_count: _Optional[int] = ..., preemption_count: _Optional[int] = ..., attempts: _Optional[_Iterable[_Union[TaskAttemptSnapshot, _Mapping]]] = ..., reservation_entry_idx: _Optional[int] = ...) -> None: ...

class TaskAttemptSnapshot(_message.Message):
    __slots__ = ("attempt_id", "worker_id", "state", "log_directory", "created_at", "started_at", "finished_at", "exit_code", "error")
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    LOG_DIRECTORY_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    attempt_id: int
    worker_id: str
    state: int
    log_directory: str
    created_at: _time_pb2.Timestamp
    started_at: _time_pb2.Timestamp
    finished_at: _time_pb2.Timestamp
    exit_code: int
    error: str
    def __init__(self, attempt_id: _Optional[int] = ..., worker_id: _Optional[str] = ..., state: _Optional[int] = ..., log_directory: _Optional[str] = ..., created_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., started_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., exit_code: _Optional[int] = ..., error: _Optional[str] = ...) -> None: ...

class EndpointSnapshot(_message.Message):
    __slots__ = ("endpoint_id", "name", "address", "job_id", "metadata", "registered_at", "task_id")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    REGISTERED_AT_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    endpoint_id: str
    name: str
    address: str
    job_id: str
    metadata: _containers.ScalarMap[str, str]
    registered_at: _time_pb2.Timestamp
    task_id: str
    def __init__(self, endpoint_id: _Optional[str] = ..., name: _Optional[str] = ..., address: _Optional[str] = ..., job_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., registered_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., task_id: _Optional[str] = ...) -> None: ...

class WorkerSnapshot(_message.Message):
    __slots__ = ("worker_id", "address", "metadata", "attributes")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _cluster_pb2.AttributeValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_cluster_pb2.AttributeValue, _Mapping]] = ...) -> None: ...
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    address: str
    metadata: _cluster_pb2.WorkerMetadata
    attributes: _containers.MessageMap[str, _cluster_pb2.AttributeValue]
    def __init__(self, worker_id: _Optional[str] = ..., address: _Optional[str] = ..., metadata: _Optional[_Union[_cluster_pb2.WorkerMetadata, _Mapping]] = ..., attributes: _Optional[_Mapping[str, _cluster_pb2.AttributeValue]] = ...) -> None: ...

class ScalingGroupSnapshot(_message.Message):
    __slots__ = ("name", "slices", "consecutive_failures", "backoff_until", "last_scale_up", "last_scale_down", "quota_exceeded_until", "quota_reason")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SLICES_FIELD_NUMBER: _ClassVar[int]
    CONSECUTIVE_FAILURES_FIELD_NUMBER: _ClassVar[int]
    BACKOFF_UNTIL_FIELD_NUMBER: _ClassVar[int]
    LAST_SCALE_UP_FIELD_NUMBER: _ClassVar[int]
    LAST_SCALE_DOWN_FIELD_NUMBER: _ClassVar[int]
    QUOTA_EXCEEDED_UNTIL_FIELD_NUMBER: _ClassVar[int]
    QUOTA_REASON_FIELD_NUMBER: _ClassVar[int]
    name: str
    slices: _containers.RepeatedCompositeFieldContainer[SliceSnapshot]
    consecutive_failures: int
    backoff_until: _time_pb2.Timestamp
    last_scale_up: _time_pb2.Timestamp
    last_scale_down: _time_pb2.Timestamp
    quota_exceeded_until: _time_pb2.Timestamp
    quota_reason: str
    def __init__(self, name: _Optional[str] = ..., slices: _Optional[_Iterable[_Union[SliceSnapshot, _Mapping]]] = ..., consecutive_failures: _Optional[int] = ..., backoff_until: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., last_scale_up: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., last_scale_down: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., quota_exceeded_until: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., quota_reason: _Optional[str] = ...) -> None: ...

class SliceSnapshot(_message.Message):
    __slots__ = ("slice_id", "scale_group", "lifecycle", "vm_addresses", "created_at", "last_active", "error_message")
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
    LIFECYCLE_FIELD_NUMBER: _ClassVar[int]
    VM_ADDRESSES_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_ACTIVE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    slice_id: str
    scale_group: str
    lifecycle: str
    vm_addresses: _containers.RepeatedScalarFieldContainer[str]
    created_at: _time_pb2.Timestamp
    last_active: _time_pb2.Timestamp
    error_message: str
    def __init__(self, slice_id: _Optional[str] = ..., scale_group: _Optional[str] = ..., lifecycle: _Optional[str] = ..., vm_addresses: _Optional[_Iterable[str]] = ..., created_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., last_active: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., error_message: _Optional[str] = ...) -> None: ...

class TrackedWorkerSnapshot(_message.Message):
    __slots__ = ("worker_id", "slice_id", "scale_group", "internal_address")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
    INTERNAL_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    slice_id: str
    scale_group: str
    internal_address: str
    def __init__(self, worker_id: _Optional[str] = ..., slice_id: _Optional[str] = ..., scale_group: _Optional[str] = ..., internal_address: _Optional[str] = ...) -> None: ...

class ReservationClaimSnapshot(_message.Message):
    __slots__ = ("worker_id", "job_id", "entry_idx")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    ENTRY_IDX_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    job_id: str
    entry_idx: int
    def __init__(self, worker_id: _Optional[str] = ..., job_id: _Optional[str] = ..., entry_idx: _Optional[int] = ...) -> None: ...
