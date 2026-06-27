import datetime

from google.protobuf import duration_pb2 as _duration_pb2
from . import job_pb2 as _job_pb2
from . import worker_pb2 as _worker_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AckDisposition(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACK_DISPOSITION_APPLIED: _ClassVar[AckDisposition]
    ACK_DISPOSITION_STALE_DISCARDED: _ClassVar[AckDisposition]
    ACK_DISPOSITION_RETRY_LATER: _ClassVar[AckDisposition]
ACK_DISPOSITION_APPLIED: AckDisposition
ACK_DISPOSITION_STALE_DISCARDED: AckDisposition
ACK_DISPOSITION_RETRY_LATER: AckDisposition

class PollRequest(_message.Message):
    __slots__ = ("backend_id", "root_epoch_seen", "last_sync_id", "caps", "observations", "rolled_up_health", "capacity", "command_results")
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    ROOT_EPOCH_SEEN_FIELD_NUMBER: _ClassVar[int]
    LAST_SYNC_ID_FIELD_NUMBER: _ClassVar[int]
    CAPS_FIELD_NUMBER: _ClassVar[int]
    OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    ROLLED_UP_HEALTH_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_FIELD_NUMBER: _ClassVar[int]
    COMMAND_RESULTS_FIELD_NUMBER: _ClassVar[int]
    backend_id: str
    root_epoch_seen: int
    last_sync_id: int
    caps: AgentCapabilities
    observations: _containers.RepeatedCompositeFieldContainer[AgentObservation]
    rolled_up_health: _worker_pb2.Worker.WorkerHealth
    capacity: CapacitySummary
    command_results: _containers.RepeatedCompositeFieldContainer[CommandResult]
    def __init__(self, backend_id: _Optional[str] = ..., root_epoch_seen: _Optional[int] = ..., last_sync_id: _Optional[int] = ..., caps: _Optional[_Union[AgentCapabilities, _Mapping]] = ..., observations: _Optional[_Iterable[_Union[AgentObservation, _Mapping]]] = ..., rolled_up_health: _Optional[_Union[_worker_pb2.Worker.WorkerHealth, _Mapping]] = ..., capacity: _Optional[_Union[CapacitySummary, _Mapping]] = ..., command_results: _Optional[_Iterable[_Union[CommandResult, _Mapping]]] = ...) -> None: ...

class PollResponse(_message.Message):
    __slots__ = ("root_epoch", "new_sync_id", "snapshot", "lease_duration", "upserts", "removals", "autoscale", "acks", "pending_commands")
    ROOT_EPOCH_FIELD_NUMBER: _ClassVar[int]
    NEW_SYNC_ID_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    LEASE_DURATION_FIELD_NUMBER: _ClassVar[int]
    UPSERTS_FIELD_NUMBER: _ClassVar[int]
    REMOVALS_FIELD_NUMBER: _ClassVar[int]
    AUTOSCALE_FIELD_NUMBER: _ClassVar[int]
    ACKS_FIELD_NUMBER: _ClassVar[int]
    PENDING_COMMANDS_FIELD_NUMBER: _ClassVar[int]
    root_epoch: int
    new_sync_id: int
    snapshot: bool
    lease_duration: _duration_pb2.Duration
    upserts: _containers.RepeatedCompositeFieldContainer[DesiredAttempt]
    removals: _containers.RepeatedScalarFieldContainer[str]
    autoscale: _containers.RepeatedCompositeFieldContainer[DesiredCapacity]
    acks: _containers.RepeatedCompositeFieldContainer[AckObservation]
    pending_commands: _containers.RepeatedCompositeFieldContainer[InteractiveCommand]
    def __init__(self, root_epoch: _Optional[int] = ..., new_sync_id: _Optional[int] = ..., snapshot: _Optional[bool] = ..., lease_duration: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., upserts: _Optional[_Iterable[_Union[DesiredAttempt, _Mapping]]] = ..., removals: _Optional[_Iterable[str]] = ..., autoscale: _Optional[_Iterable[_Union[DesiredCapacity, _Mapping]]] = ..., acks: _Optional[_Iterable[_Union[AckObservation, _Mapping]]] = ..., pending_commands: _Optional[_Iterable[_Union[InteractiveCommand, _Mapping]]] = ...) -> None: ...

class DesiredAttempt(_message.Message):
    __slots__ = ("attempt_uid", "desired_generation", "spec", "constraints")
    ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    DESIRED_GENERATION_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    attempt_uid: str
    desired_generation: int
    spec: _worker_pb2.Worker.AttemptSpec
    constraints: _containers.RepeatedCompositeFieldContainer[_job_pb2.Constraint]
    def __init__(self, attempt_uid: _Optional[str] = ..., desired_generation: _Optional[int] = ..., spec: _Optional[_Union[_worker_pb2.Worker.AttemptSpec, _Mapping]] = ..., constraints: _Optional[_Iterable[_Union[_job_pb2.Constraint, _Mapping]]] = ...) -> None: ...

class AgentObservation(_message.Message):
    __slots__ = ("attempt_uid", "acted_root_epoch", "desired_generation", "state", "observed_worker", "exit_code", "message")
    ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    ACTED_ROOT_EPOCH_FIELD_NUMBER: _ClassVar[int]
    DESIRED_GENERATION_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_WORKER_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    attempt_uid: str
    acted_root_epoch: int
    desired_generation: int
    state: _job_pb2.TaskState
    observed_worker: str
    exit_code: int
    message: str
    def __init__(self, attempt_uid: _Optional[str] = ..., acted_root_epoch: _Optional[int] = ..., desired_generation: _Optional[int] = ..., state: _Optional[_Union[_job_pb2.TaskState, str]] = ..., observed_worker: _Optional[str] = ..., exit_code: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class AckObservation(_message.Message):
    __slots__ = ("attempt_uid", "disposition")
    ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    DISPOSITION_FIELD_NUMBER: _ClassVar[int]
    attempt_uid: str
    disposition: AckDisposition
    def __init__(self, attempt_uid: _Optional[str] = ..., disposition: _Optional[_Union[AckDisposition, str]] = ...) -> None: ...

class AgentCapabilities(_message.Message):
    __slots__ = ("agent_version", "device_variants")
    AGENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    DEVICE_VARIANTS_FIELD_NUMBER: _ClassVar[int]
    agent_version: str
    device_variants: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, agent_version: _Optional[str] = ..., device_variants: _Optional[_Iterable[str]] = ...) -> None: ...

class CapacitySummary(_message.Message):
    __slots__ = ("allocatable", "max_free_cpu_millicores", "max_free_memory_bytes", "largest_gang", "stale_ms", "backoff_until_ms")
    class AllocatableEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class LargestGangEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class BackoffUntilMsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    ALLOCATABLE_FIELD_NUMBER: _ClassVar[int]
    MAX_FREE_CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MAX_FREE_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    LARGEST_GANG_FIELD_NUMBER: _ClassVar[int]
    STALE_MS_FIELD_NUMBER: _ClassVar[int]
    BACKOFF_UNTIL_MS_FIELD_NUMBER: _ClassVar[int]
    allocatable: _containers.ScalarMap[str, int]
    max_free_cpu_millicores: int
    max_free_memory_bytes: int
    largest_gang: _containers.ScalarMap[str, int]
    stale_ms: int
    backoff_until_ms: _containers.ScalarMap[str, int]
    def __init__(self, allocatable: _Optional[_Mapping[str, int]] = ..., max_free_cpu_millicores: _Optional[int] = ..., max_free_memory_bytes: _Optional[int] = ..., largest_gang: _Optional[_Mapping[str, int]] = ..., stale_ms: _Optional[int] = ..., backoff_until_ms: _Optional[_Mapping[str, int]] = ...) -> None: ...

class DesiredCapacity(_message.Message):
    __slots__ = ("scale_group", "target_slices")
    SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
    TARGET_SLICES_FIELD_NUMBER: _ClassVar[int]
    scale_group: str
    target_slices: int
    def __init__(self, scale_group: _Optional[str] = ..., target_slices: _Optional[int] = ...) -> None: ...

class TaskTarget(_message.Message):
    __slots__ = ("full_task_id", "attempt_uid")
    FULL_TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    full_task_id: str
    attempt_uid: str
    def __init__(self, full_task_id: _Optional[str] = ..., attempt_uid: _Optional[str] = ...) -> None: ...

class InteractiveCommand(_message.Message):
    __slots__ = ("command_id", "target", "origin_user", "exec", "profile", "status")
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_USER_FIELD_NUMBER: _ClassVar[int]
    EXEC_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    command_id: str
    target: TaskTarget
    origin_user: str
    exec: _worker_pb2.Worker.ExecInContainerRequest
    profile: _job_pb2.ProfileTaskRequest
    status: _job_pb2.GetProcessStatusRequest
    def __init__(self, command_id: _Optional[str] = ..., target: _Optional[_Union[TaskTarget, _Mapping]] = ..., origin_user: _Optional[str] = ..., exec: _Optional[_Union[_worker_pb2.Worker.ExecInContainerRequest, _Mapping]] = ..., profile: _Optional[_Union[_job_pb2.ProfileTaskRequest, _Mapping]] = ..., status: _Optional[_Union[_job_pb2.GetProcessStatusRequest, _Mapping]] = ...) -> None: ...

class CommandResult(_message.Message):
    __slots__ = ("command_id", "error", "exec", "profile", "status")
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    EXEC_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    command_id: str
    error: str
    exec: _worker_pb2.Worker.ExecInContainerResponse
    profile: _job_pb2.ProfileTaskResponse
    status: _job_pb2.GetProcessStatusResponse
    def __init__(self, command_id: _Optional[str] = ..., error: _Optional[str] = ..., exec: _Optional[_Union[_worker_pb2.Worker.ExecInContainerResponse, _Mapping]] = ..., profile: _Optional[_Union[_job_pb2.ProfileTaskResponse, _Mapping]] = ..., status: _Optional[_Union[_job_pb2.GetProcessStatusResponse, _Mapping]] = ...) -> None: ...
