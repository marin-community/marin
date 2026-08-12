from . import resource_pb2 as _resource_pb2
from . import resource_identity_pb2 as _resource_identity_pb2
from . import time_pb2 as _time_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TaskState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TASK_STATE_UNSPECIFIED: _ClassVar[TaskState]
    TASK_STATE_PENDING: _ClassVar[TaskState]
    TASK_STATE_BUILDING: _ClassVar[TaskState]
    TASK_STATE_RUNNING: _ClassVar[TaskState]
    TASK_STATE_SUCCEEDED: _ClassVar[TaskState]
    TASK_STATE_FAILED: _ClassVar[TaskState]
    TASK_STATE_KILLED: _ClassVar[TaskState]
    TASK_STATE_WORKER_FAILED: _ClassVar[TaskState]
    TASK_STATE_UNSCHEDULABLE: _ClassVar[TaskState]
    TASK_STATE_ASSIGNED: _ClassVar[TaskState]
    TASK_STATE_PREEMPTED: _ClassVar[TaskState]
    TASK_STATE_COSCHED_FAILED: _ClassVar[TaskState]
    TASK_STATE_MISSING: _ClassVar[TaskState]
TASK_STATE_UNSPECIFIED: TaskState
TASK_STATE_PENDING: TaskState
TASK_STATE_BUILDING: TaskState
TASK_STATE_RUNNING: TaskState
TASK_STATE_SUCCEEDED: TaskState
TASK_STATE_FAILED: TaskState
TASK_STATE_KILLED: TaskState
TASK_STATE_WORKER_FAILED: TaskState
TASK_STATE_UNSCHEDULABLE: TaskState
TASK_STATE_ASSIGNED: TaskState
TASK_STATE_PREEMPTED: TaskState
TASK_STATE_COSCHED_FAILED: TaskState
TASK_STATE_MISSING: TaskState

class TaskQuery(_message.Message):
    __slots__ = ("job", "job_id_prefix", "states", "backend_id", "authority_cluster_id", "execution_cluster_id", "page")
    JOB_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_PREFIX_FIELD_NUMBER: _ClassVar[int]
    STATES_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    AUTHORITY_CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    job: _resource_identity_pb2.ResourceKey
    job_id_prefix: str
    states: _containers.RepeatedScalarFieldContainer[TaskState]
    backend_id: str
    authority_cluster_id: str
    execution_cluster_id: str
    page: _resource_pb2.PageRequest
    def __init__(self, job: _Optional[_Union[_resource_identity_pb2.ResourceKey, _Mapping]] = ..., job_id_prefix: _Optional[str] = ..., states: _Optional[_Iterable[_Union[TaskState, str]]] = ..., backend_id: _Optional[str] = ..., authority_cluster_id: _Optional[str] = ..., execution_cluster_id: _Optional[str] = ..., page: _Optional[_Union[_resource_pb2.PageRequest, _Mapping]] = ...) -> None: ...

class TaskSummary(_message.Message):
    __slots__ = ("identity", "job", "task_index", "state", "execution_cluster_id", "backend_id", "current_attempt", "current_node", "failure_count", "preemption_count", "submitted_at", "started_at", "finished_at", "status_message", "error_message")
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    JOB_FIELD_NUMBER: _ClassVar[int]
    TASK_INDEX_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    CURRENT_NODE_FIELD_NUMBER: _ClassVar[int]
    FAILURE_COUNT_FIELD_NUMBER: _ClassVar[int]
    PREEMPTION_COUNT_FIELD_NUMBER: _ClassVar[int]
    SUBMITTED_AT_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    identity: _resource_identity_pb2.TaskIdentity
    job: _resource_identity_pb2.JobIdentity
    task_index: int
    state: TaskState
    execution_cluster_id: str
    backend_id: str
    current_attempt: _resource_identity_pb2.AttemptIdentity
    current_node: _resource_identity_pb2.NodeIdentity
    failure_count: int
    preemption_count: int
    submitted_at: _time_pb2.Timestamp
    started_at: _time_pb2.Timestamp
    finished_at: _time_pb2.Timestamp
    status_message: str
    error_message: str
    def __init__(self, identity: _Optional[_Union[_resource_identity_pb2.TaskIdentity, _Mapping]] = ..., job: _Optional[_Union[_resource_identity_pb2.JobIdentity, _Mapping]] = ..., task_index: _Optional[int] = ..., state: _Optional[_Union[TaskState, str]] = ..., execution_cluster_id: _Optional[str] = ..., backend_id: _Optional[str] = ..., current_attempt: _Optional[_Union[_resource_identity_pb2.AttemptIdentity, _Mapping]] = ..., current_node: _Optional[_Union[_resource_identity_pb2.NodeIdentity, _Mapping]] = ..., failure_count: _Optional[int] = ..., preemption_count: _Optional[int] = ..., submitted_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., started_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., status_message: _Optional[str] = ..., error_message: _Optional[str] = ...) -> None: ...

class AttemptSummary(_message.Message):
    __slots__ = ("identity", "state", "execution_cluster_id", "backend_id", "node", "created_at", "started_at", "finished_at", "exit_code", "error_message", "terminal_reason")
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_REASON_FIELD_NUMBER: _ClassVar[int]
    identity: _resource_identity_pb2.AttemptIdentity
    state: TaskState
    execution_cluster_id: str
    backend_id: str
    node: _resource_identity_pb2.NodeIdentity
    created_at: _time_pb2.Timestamp
    started_at: _time_pb2.Timestamp
    finished_at: _time_pb2.Timestamp
    exit_code: int
    error_message: str
    terminal_reason: str
    def __init__(self, identity: _Optional[_Union[_resource_identity_pb2.AttemptIdentity, _Mapping]] = ..., state: _Optional[_Union[TaskState, str]] = ..., execution_cluster_id: _Optional[str] = ..., backend_id: _Optional[str] = ..., node: _Optional[_Union[_resource_identity_pb2.NodeIdentity, _Mapping]] = ..., created_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., started_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., exit_code: _Optional[int] = ..., error_message: _Optional[str] = ..., terminal_reason: _Optional[str] = ...) -> None: ...

class AttemptRuntimeObject(_message.Message):
    __slots__ = ("provider_kind", "namespace", "name", "provider_uid", "provider_node_id", "provider_node_uid", "container_id", "observed_at")
    PROVIDER_KIND_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_UID_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NODE_UID_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    provider_kind: str
    namespace: str
    name: str
    provider_uid: str
    provider_node_id: str
    provider_node_uid: str
    container_id: str
    observed_at: _time_pb2.Timestamp
    def __init__(self, provider_kind: _Optional[str] = ..., namespace: _Optional[str] = ..., name: _Optional[str] = ..., provider_uid: _Optional[str] = ..., provider_node_id: _Optional[str] = ..., provider_node_uid: _Optional[str] = ..., container_id: _Optional[str] = ..., observed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class TaskDetail(_message.Message):
    __slots__ = ("summary", "attempts", "source_statuses", "root_cause_highlights")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    ROOT_CAUSE_HIGHLIGHTS_FIELD_NUMBER: _ClassVar[int]
    summary: TaskSummary
    attempts: _containers.RepeatedCompositeFieldContainer[AttemptSummary]
    source_statuses: _containers.RepeatedCompositeFieldContainer[_resource_pb2.ResourceSourceStatus]
    root_cause_highlights: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, summary: _Optional[_Union[TaskSummary, _Mapping]] = ..., attempts: _Optional[_Iterable[_Union[AttemptSummary, _Mapping]]] = ..., source_statuses: _Optional[_Iterable[_Union[_resource_pb2.ResourceSourceStatus, _Mapping]]] = ..., root_cause_highlights: _Optional[_Iterable[str]] = ...) -> None: ...

class AttemptDetail(_message.Message):
    __slots__ = ("summary", "runtime", "source_statuses")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    summary: AttemptSummary
    runtime: AttemptRuntimeObject
    source_statuses: _containers.RepeatedCompositeFieldContainer[_resource_pb2.ResourceSourceStatus]
    def __init__(self, summary: _Optional[_Union[AttemptSummary, _Mapping]] = ..., runtime: _Optional[_Union[AttemptRuntimeObject, _Mapping]] = ..., source_statuses: _Optional[_Iterable[_Union[_resource_pb2.ResourceSourceStatus, _Mapping]]] = ...) -> None: ...

class PreemptTaskUpdate(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TerminateTaskUpdate(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class FailTaskUpdate(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TaskUpdate(_message.Message):
    __slots__ = ("expected_attempt_uid", "preempt", "terminate", "fail")
    EXPECTED_ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    PREEMPT_FIELD_NUMBER: _ClassVar[int]
    TERMINATE_FIELD_NUMBER: _ClassVar[int]
    FAIL_FIELD_NUMBER: _ClassVar[int]
    expected_attempt_uid: str
    preempt: PreemptTaskUpdate
    terminate: TerminateTaskUpdate
    fail: FailTaskUpdate
    def __init__(self, expected_attempt_uid: _Optional[str] = ..., preempt: _Optional[_Union[PreemptTaskUpdate, _Mapping]] = ..., terminate: _Optional[_Union[TerminateTaskUpdate, _Mapping]] = ..., fail: _Optional[_Union[FailTaskUpdate, _Mapping]] = ...) -> None: ...

class PreemptAttemptUpdate(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TerminateAttemptUpdate(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class FailAttemptUpdate(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class AttemptUpdate(_message.Message):
    __slots__ = ("preempt", "terminate", "fail")
    PREEMPT_FIELD_NUMBER: _ClassVar[int]
    TERMINATE_FIELD_NUMBER: _ClassVar[int]
    FAIL_FIELD_NUMBER: _ClassVar[int]
    preempt: PreemptAttemptUpdate
    terminate: TerminateAttemptUpdate
    fail: FailAttemptUpdate
    def __init__(self, preempt: _Optional[_Union[PreemptAttemptUpdate, _Mapping]] = ..., terminate: _Optional[_Union[TerminateAttemptUpdate, _Mapping]] = ..., fail: _Optional[_Union[FailAttemptUpdate, _Mapping]] = ...) -> None: ...
