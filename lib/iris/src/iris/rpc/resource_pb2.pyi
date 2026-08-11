from . import iris_logging_pb2 as _iris_logging_pb2
from . import time_pb2 as _time_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ResourceKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESOURCE_KIND_UNSPECIFIED: _ClassVar[ResourceKind]
    RESOURCE_KIND_JOB: _ClassVar[ResourceKind]
    RESOURCE_KIND_TASK: _ClassVar[ResourceKind]
    RESOURCE_KIND_ATTEMPT: _ClassVar[ResourceKind]
    RESOURCE_KIND_ENDPOINT: _ClassVar[ResourceKind]
    RESOURCE_KIND_NODE: _ClassVar[ResourceKind]
    RESOURCE_KIND_SLICE: _ClassVar[ResourceKind]

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

class NodeHealth(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NODE_HEALTH_UNSPECIFIED: _ClassVar[NodeHealth]
    NODE_HEALTH_READY: _ClassVar[NodeHealth]
    NODE_HEALTH_DEGRADED: _ClassVar[NodeHealth]
    NODE_HEALTH_UNAVAILABLE: _ClassVar[NodeHealth]
    NODE_HEALTH_RETIRED: _ClassVar[NodeHealth]

class SliceLifecycle(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SLICE_LIFECYCLE_UNSPECIFIED: _ClassVar[SliceLifecycle]
    SLICE_LIFECYCLE_CREATING: _ClassVar[SliceLifecycle]
    SLICE_LIFECYCLE_READY: _ClassVar[SliceLifecycle]
    SLICE_LIFECYCLE_DELETING: _ClassVar[SliceLifecycle]
    SLICE_LIFECYCLE_FAILED: _ClassVar[SliceLifecycle]

class MembershipState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MEMBERSHIP_STATE_UNSPECIFIED: _ClassVar[MembershipState]
    MEMBERSHIP_STATE_UNKNOWN: _ClassVar[MembershipState]
    MEMBERSHIP_STATE_OBSERVED: _ClassVar[MembershipState]

class SliceCapacityState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SLICE_CAPACITY_STATE_UNSPECIFIED: _ClassVar[SliceCapacityState]
    SLICE_CAPACITY_STATE_UNKNOWN: _ClassVar[SliceCapacityState]
    SLICE_CAPACITY_STATE_AVAILABLE: _ClassVar[SliceCapacityState]
    SLICE_CAPACITY_STATE_IN_USE: _ClassVar[SliceCapacityState]
    SLICE_CAPACITY_STATE_IDLE: _ClassVar[SliceCapacityState]
    SLICE_CAPACITY_STATE_DEGRADED: _ClassVar[SliceCapacityState]

class ActionKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_KIND_UNSPECIFIED: _ClassVar[ActionKind]
    ACTION_KIND_CANCEL_JOB: _ClassVar[ActionKind]
    ACTION_KIND_RETRY_TASK: _ClassVar[ActionKind]
    ACTION_KIND_TERMINATE_ATTEMPT: _ClassVar[ActionKind]

class ActionState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_STATE_UNSPECIFIED: _ClassVar[ActionState]
    ACTION_STATE_ACCEPTED: _ClassVar[ActionState]
    ACTION_STATE_VERIFYING: _ClassVar[ActionState]
    ACTION_STATE_SUCCEEDED: _ClassVar[ActionState]
    ACTION_STATE_FAILED: _ClassVar[ActionState]

class ActionResult(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_RESULT_UNSPECIFIED: _ClassVar[ActionResult]
    ACTION_RESULT_NONE: _ClassVar[ActionResult]
    ACTION_RESULT_SATISFIED: _ClassVar[ActionResult]
    ACTION_RESULT_TARGET_ABSENT: _ClassVar[ActionResult]
    ACTION_RESULT_PROVIDER_REJECTED: _ClassVar[ActionResult]
    ACTION_RESULT_INTERNAL_ERROR: _ClassVar[ActionResult]

class EndpointAccess(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ENDPOINT_ACCESS_PRIVATE: _ClassVar[EndpointAccess]
    ENDPOINT_ACCESS_LINK: _ClassVar[EndpointAccess]

class JobState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_STATE_UNSPECIFIED: _ClassVar[JobState]
    JOB_STATE_PENDING: _ClassVar[JobState]
    JOB_STATE_BUILDING: _ClassVar[JobState]
    JOB_STATE_RUNNING: _ClassVar[JobState]
    JOB_STATE_SUCCEEDED: _ClassVar[JobState]
    JOB_STATE_FAILED: _ClassVar[JobState]
    JOB_STATE_KILLED: _ClassVar[JobState]
    JOB_STATE_WORKER_FAILED: _ClassVar[JobState]
    JOB_STATE_UNSCHEDULABLE: _ClassVar[JobState]

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

class ConstraintOp(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONSTRAINT_OP_EQ: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_NE: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_EXISTS: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_NOT_EXISTS: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_GT: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_GE: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_LT: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_LE: _ClassVar[ConstraintOp]
    CONSTRAINT_OP_IN: _ClassVar[ConstraintOp]

class ConstraintMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONSTRAINT_MODE_REQUIRED: _ClassVar[ConstraintMode]
    CONSTRAINT_MODE_PREFERRED: _ClassVar[ConstraintMode]

class JobPreemptionPolicy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_PREEMPTION_POLICY_UNSPECIFIED: _ClassVar[JobPreemptionPolicy]
    JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN: _ClassVar[JobPreemptionPolicy]
    JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN: _ClassVar[JobPreemptionPolicy]

class ExistingJobPolicy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXISTING_JOB_POLICY_UNSPECIFIED: _ClassVar[ExistingJobPolicy]
    EXISTING_JOB_POLICY_ERROR: _ClassVar[ExistingJobPolicy]
    EXISTING_JOB_POLICY_KEEP: _ClassVar[ExistingJobPolicy]
    EXISTING_JOB_POLICY_RECREATE: _ClassVar[ExistingJobPolicy]

class PriorityBand(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PRIORITY_BAND_INHERIT: _ClassVar[PriorityBand]
    PRIORITY_BAND_PRODUCTION: _ClassVar[PriorityBand]
    PRIORITY_BAND_INTERACTIVE: _ClassVar[PriorityBand]
    PRIORITY_BAND_BATCH: _ClassVar[PriorityBand]

class ContainerProfile(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONTAINER_PROFILE_UNSPECIFIED: _ClassVar[ContainerProfile]
    CONTAINER_PROFILE_RESTRICTED: _ClassVar[ContainerProfile]
    CONTAINER_PROFILE_DEFAULT: _ClassVar[ContainerProfile]
    CONTAINER_PROFILE_DOCKER_ACCESS: _ClassVar[ContainerProfile]
    CONTAINER_PROFILE_PRIVILEGED: _ClassVar[ContainerProfile]
    CONTAINER_PROFILE_GVISOR: _ClassVar[ContainerProfile]
RESOURCE_KIND_UNSPECIFIED: ResourceKind
RESOURCE_KIND_JOB: ResourceKind
RESOURCE_KIND_TASK: ResourceKind
RESOURCE_KIND_ATTEMPT: ResourceKind
RESOURCE_KIND_ENDPOINT: ResourceKind
RESOURCE_KIND_NODE: ResourceKind
RESOURCE_KIND_SLICE: ResourceKind
SOURCE_STATE_UNSPECIFIED: SourceState
SOURCE_STATE_AVAILABLE: SourceState
SOURCE_STATE_UNAVAILABLE: SourceState
SOURCE_STATE_UNSUPPORTED: SourceState
FRESHNESS_UNSPECIFIED: Freshness
FRESHNESS_CURRENT: Freshness
FRESHNESS_STALE: Freshness
FRESHNESS_UNKNOWN: Freshness
NODE_HEALTH_UNSPECIFIED: NodeHealth
NODE_HEALTH_READY: NodeHealth
NODE_HEALTH_DEGRADED: NodeHealth
NODE_HEALTH_UNAVAILABLE: NodeHealth
NODE_HEALTH_RETIRED: NodeHealth
SLICE_LIFECYCLE_UNSPECIFIED: SliceLifecycle
SLICE_LIFECYCLE_CREATING: SliceLifecycle
SLICE_LIFECYCLE_READY: SliceLifecycle
SLICE_LIFECYCLE_DELETING: SliceLifecycle
SLICE_LIFECYCLE_FAILED: SliceLifecycle
MEMBERSHIP_STATE_UNSPECIFIED: MembershipState
MEMBERSHIP_STATE_UNKNOWN: MembershipState
MEMBERSHIP_STATE_OBSERVED: MembershipState
SLICE_CAPACITY_STATE_UNSPECIFIED: SliceCapacityState
SLICE_CAPACITY_STATE_UNKNOWN: SliceCapacityState
SLICE_CAPACITY_STATE_AVAILABLE: SliceCapacityState
SLICE_CAPACITY_STATE_IN_USE: SliceCapacityState
SLICE_CAPACITY_STATE_IDLE: SliceCapacityState
SLICE_CAPACITY_STATE_DEGRADED: SliceCapacityState
ACTION_KIND_UNSPECIFIED: ActionKind
ACTION_KIND_CANCEL_JOB: ActionKind
ACTION_KIND_RETRY_TASK: ActionKind
ACTION_KIND_TERMINATE_ATTEMPT: ActionKind
ACTION_STATE_UNSPECIFIED: ActionState
ACTION_STATE_ACCEPTED: ActionState
ACTION_STATE_VERIFYING: ActionState
ACTION_STATE_SUCCEEDED: ActionState
ACTION_STATE_FAILED: ActionState
ACTION_RESULT_UNSPECIFIED: ActionResult
ACTION_RESULT_NONE: ActionResult
ACTION_RESULT_SATISFIED: ActionResult
ACTION_RESULT_TARGET_ABSENT: ActionResult
ACTION_RESULT_PROVIDER_REJECTED: ActionResult
ACTION_RESULT_INTERNAL_ERROR: ActionResult
ENDPOINT_ACCESS_PRIVATE: EndpointAccess
ENDPOINT_ACCESS_LINK: EndpointAccess
JOB_STATE_UNSPECIFIED: JobState
JOB_STATE_PENDING: JobState
JOB_STATE_BUILDING: JobState
JOB_STATE_RUNNING: JobState
JOB_STATE_SUCCEEDED: JobState
JOB_STATE_FAILED: JobState
JOB_STATE_KILLED: JobState
JOB_STATE_WORKER_FAILED: JobState
JOB_STATE_UNSCHEDULABLE: JobState
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
CONSTRAINT_OP_EQ: ConstraintOp
CONSTRAINT_OP_NE: ConstraintOp
CONSTRAINT_OP_EXISTS: ConstraintOp
CONSTRAINT_OP_NOT_EXISTS: ConstraintOp
CONSTRAINT_OP_GT: ConstraintOp
CONSTRAINT_OP_GE: ConstraintOp
CONSTRAINT_OP_LT: ConstraintOp
CONSTRAINT_OP_LE: ConstraintOp
CONSTRAINT_OP_IN: ConstraintOp
CONSTRAINT_MODE_REQUIRED: ConstraintMode
CONSTRAINT_MODE_PREFERRED: ConstraintMode
JOB_PREEMPTION_POLICY_UNSPECIFIED: JobPreemptionPolicy
JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN: JobPreemptionPolicy
JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN: JobPreemptionPolicy
EXISTING_JOB_POLICY_UNSPECIFIED: ExistingJobPolicy
EXISTING_JOB_POLICY_ERROR: ExistingJobPolicy
EXISTING_JOB_POLICY_KEEP: ExistingJobPolicy
EXISTING_JOB_POLICY_RECREATE: ExistingJobPolicy
PRIORITY_BAND_INHERIT: PriorityBand
PRIORITY_BAND_PRODUCTION: PriorityBand
PRIORITY_BAND_INTERACTIVE: PriorityBand
PRIORITY_BAND_BATCH: PriorityBand
CONTAINER_PROFILE_UNSPECIFIED: ContainerProfile
CONTAINER_PROFILE_RESTRICTED: ContainerProfile
CONTAINER_PROFILE_DEFAULT: ContainerProfile
CONTAINER_PROFILE_DOCKER_ACCESS: ContainerProfile
CONTAINER_PROFILE_PRIVILEGED: ContainerProfile
CONTAINER_PROFILE_GVISOR: ContainerProfile

class CpuProfile(_message.Message):
    __slots__ = ("format", "rate_hz", "native")
    class Format(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        FORMAT_UNSPECIFIED: _ClassVar[CpuProfile.Format]
        FLAMEGRAPH: _ClassVar[CpuProfile.Format]
        SPEEDSCOPE: _ClassVar[CpuProfile.Format]
        RAW: _ClassVar[CpuProfile.Format]
    FORMAT_UNSPECIFIED: CpuProfile.Format
    FLAMEGRAPH: CpuProfile.Format
    SPEEDSCOPE: CpuProfile.Format
    RAW: CpuProfile.Format
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    RATE_HZ_FIELD_NUMBER: _ClassVar[int]
    NATIVE_FIELD_NUMBER: _ClassVar[int]
    format: CpuProfile.Format
    rate_hz: int
    native: bool
    def __init__(self, format: _Optional[_Union[CpuProfile.Format, str]] = ..., rate_hz: _Optional[int] = ..., native: _Optional[bool] = ...) -> None: ...

class MemoryProfile(_message.Message):
    __slots__ = ("format", "leaks")
    class Format(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        FORMAT_UNSPECIFIED: _ClassVar[MemoryProfile.Format]
        FLAMEGRAPH: _ClassVar[MemoryProfile.Format]
        TABLE: _ClassVar[MemoryProfile.Format]
        STATS: _ClassVar[MemoryProfile.Format]
        RAW: _ClassVar[MemoryProfile.Format]
    FORMAT_UNSPECIFIED: MemoryProfile.Format
    FLAMEGRAPH: MemoryProfile.Format
    TABLE: MemoryProfile.Format
    STATS: MemoryProfile.Format
    RAW: MemoryProfile.Format
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    LEAKS_FIELD_NUMBER: _ClassVar[int]
    format: MemoryProfile.Format
    leaks: bool
    def __init__(self, format: _Optional[_Union[MemoryProfile.Format, str]] = ..., leaks: _Optional[bool] = ...) -> None: ...

class ThreadsProfile(_message.Message):
    __slots__ = ("locals",)
    LOCALS_FIELD_NUMBER: _ClassVar[int]
    locals: bool
    def __init__(self, locals: _Optional[bool] = ...) -> None: ...

class ProfileType(_message.Message):
    __slots__ = ("cpu", "memory", "threads")
    CPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    THREADS_FIELD_NUMBER: _ClassVar[int]
    cpu: CpuProfile
    memory: MemoryProfile
    threads: ThreadsProfile
    def __init__(self, cpu: _Optional[_Union[CpuProfile, _Mapping]] = ..., memory: _Optional[_Union[MemoryProfile, _Mapping]] = ..., threads: _Optional[_Union[ThreadsProfile, _Mapping]] = ...) -> None: ...

class DeviceConfig(_message.Message):
    __slots__ = ("cpu", "gpu", "tpu")
    CPU_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    TPU_FIELD_NUMBER: _ClassVar[int]
    cpu: CpuDevice
    gpu: GpuDevice
    tpu: TpuDevice
    def __init__(self, cpu: _Optional[_Union[CpuDevice, _Mapping]] = ..., gpu: _Optional[_Union[GpuDevice, _Mapping]] = ..., tpu: _Optional[_Union[TpuDevice, _Mapping]] = ...) -> None: ...

class CpuDevice(_message.Message):
    __slots__ = ("variant",)
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    variant: str
    def __init__(self, variant: _Optional[str] = ...) -> None: ...

class GpuDevice(_message.Message):
    __slots__ = ("variant", "count")
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    variant: str
    count: int
    def __init__(self, variant: _Optional[str] = ..., count: _Optional[int] = ...) -> None: ...

class TpuDevice(_message.Message):
    __slots__ = ("variant", "topology", "count")
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    TOPOLOGY_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    variant: str
    topology: str
    count: int
    def __init__(self, variant: _Optional[str] = ..., topology: _Optional[str] = ..., count: _Optional[int] = ...) -> None: ...

class ResourceSpecProto(_message.Message):
    __slots__ = ("cpu_millicores", "memory_bytes", "disk_bytes", "device")
    CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int
    device: DeviceConfig
    def __init__(self, cpu_millicores: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., device: _Optional[_Union[DeviceConfig, _Mapping]] = ...) -> None: ...

class EnvironmentConfig(_message.Message):
    __slots__ = ("env_vars", "setup_scripts")
    class EnvVarsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ENV_VARS_FIELD_NUMBER: _ClassVar[int]
    SETUP_SCRIPTS_FIELD_NUMBER: _ClassVar[int]
    env_vars: _containers.ScalarMap[str, str]
    setup_scripts: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, env_vars: _Optional[_Mapping[str, str]] = ..., setup_scripts: _Optional[_Iterable[str]] = ...) -> None: ...

class CommandEntrypoint(_message.Message):
    __slots__ = ("argv",)
    ARGV_FIELD_NUMBER: _ClassVar[int]
    argv: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, argv: _Optional[_Iterable[str]] = ...) -> None: ...

class RuntimeEntrypoint(_message.Message):
    __slots__ = ("setup_commands", "run_command", "workdir_files", "workdir_file_refs")
    class WorkdirFilesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bytes
        def __init__(self, key: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...
    class WorkdirFileRefsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SETUP_COMMANDS_FIELD_NUMBER: _ClassVar[int]
    RUN_COMMAND_FIELD_NUMBER: _ClassVar[int]
    WORKDIR_FILES_FIELD_NUMBER: _ClassVar[int]
    WORKDIR_FILE_REFS_FIELD_NUMBER: _ClassVar[int]
    setup_commands: _containers.RepeatedScalarFieldContainer[str]
    run_command: CommandEntrypoint
    workdir_files: _containers.ScalarMap[str, bytes]
    workdir_file_refs: _containers.ScalarMap[str, str]
    def __init__(self, setup_commands: _Optional[_Iterable[str]] = ..., run_command: _Optional[_Union[CommandEntrypoint, _Mapping]] = ..., workdir_files: _Optional[_Mapping[str, bytes]] = ..., workdir_file_refs: _Optional[_Mapping[str, str]] = ...) -> None: ...

class AttributeValue(_message.Message):
    __slots__ = ("string_value", "int_value", "float_value")
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    string_value: str
    int_value: int
    float_value: float
    def __init__(self, string_value: _Optional[str] = ..., int_value: _Optional[int] = ..., float_value: _Optional[float] = ...) -> None: ...

class Constraint(_message.Message):
    __slots__ = ("key", "op", "value", "values", "mode")
    KEY_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    key: str
    op: ConstraintOp
    value: AttributeValue
    values: _containers.RepeatedCompositeFieldContainer[AttributeValue]
    mode: ConstraintMode
    def __init__(self, key: _Optional[str] = ..., op: _Optional[_Union[ConstraintOp, str]] = ..., value: _Optional[_Union[AttributeValue, _Mapping]] = ..., values: _Optional[_Iterable[_Union[AttributeValue, _Mapping]]] = ..., mode: _Optional[_Union[ConstraintMode, str]] = ...) -> None: ...

class CoschedulingConfig(_message.Message):
    __slots__ = ("group_by",)
    GROUP_BY_FIELD_NUMBER: _ClassVar[int]
    group_by: str
    def __init__(self, group_by: _Optional[str] = ...) -> None: ...

class ResourceKey(_message.Message):
    __slots__ = ("cluster_id", "kind", "resource_id")
    CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    cluster_id: str
    kind: ResourceKind
    resource_id: str
    def __init__(self, cluster_id: _Optional[str] = ..., kind: _Optional[_Union[ResourceKind, str]] = ..., resource_id: _Optional[str] = ...) -> None: ...

class JobIdentity(_message.Message):
    __slots__ = ("key", "job_uid")
    KEY_FIELD_NUMBER: _ClassVar[int]
    JOB_UID_FIELD_NUMBER: _ClassVar[int]
    key: ResourceKey
    job_uid: str
    def __init__(self, key: _Optional[_Union[ResourceKey, _Mapping]] = ..., job_uid: _Optional[str] = ...) -> None: ...

class TaskIdentity(_message.Message):
    __slots__ = ("key", "task_uid")
    KEY_FIELD_NUMBER: _ClassVar[int]
    TASK_UID_FIELD_NUMBER: _ClassVar[int]
    key: ResourceKey
    task_uid: str
    def __init__(self, key: _Optional[_Union[ResourceKey, _Mapping]] = ..., task_uid: _Optional[str] = ...) -> None: ...

class AttemptLocator(_message.Message):
    __slots__ = ("task", "attempt_number")
    TASK_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_NUMBER_FIELD_NUMBER: _ClassVar[int]
    task: ResourceKey
    attempt_number: int
    def __init__(self, task: _Optional[_Union[ResourceKey, _Mapping]] = ..., attempt_number: _Optional[int] = ...) -> None: ...

class AttemptIdentity(_message.Message):
    __slots__ = ("task", "attempt_number", "attempt_uid")
    TASK_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_NUMBER_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    task: ResourceKey
    attempt_number: int
    attempt_uid: str
    def __init__(self, task: _Optional[_Union[ResourceKey, _Mapping]] = ..., attempt_number: _Optional[int] = ..., attempt_uid: _Optional[str] = ...) -> None: ...

class NodeIdentity(_message.Message):
    __slots__ = ("key", "backend_id", "node_uid")
    KEY_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_UID_FIELD_NUMBER: _ClassVar[int]
    key: ResourceKey
    backend_id: str
    node_uid: str
    def __init__(self, key: _Optional[_Union[ResourceKey, _Mapping]] = ..., backend_id: _Optional[str] = ..., node_uid: _Optional[str] = ...) -> None: ...

class NodeLocator(_message.Message):
    __slots__ = ("key", "backend_id", "node_uid")
    KEY_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_UID_FIELD_NUMBER: _ClassVar[int]
    key: ResourceKey
    backend_id: str
    node_uid: str
    def __init__(self, key: _Optional[_Union[ResourceKey, _Mapping]] = ..., backend_id: _Optional[str] = ..., node_uid: _Optional[str] = ...) -> None: ...

class SliceIdentity(_message.Message):
    __slots__ = ("key", "backend_id", "slice_uid")
    KEY_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    SLICE_UID_FIELD_NUMBER: _ClassVar[int]
    key: ResourceKey
    backend_id: str
    slice_uid: str
    def __init__(self, key: _Optional[_Union[ResourceKey, _Mapping]] = ..., backend_id: _Optional[str] = ..., slice_uid: _Optional[str] = ...) -> None: ...

class SliceLocator(_message.Message):
    __slots__ = ("key", "backend_id", "slice_uid")
    KEY_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    SLICE_UID_FIELD_NUMBER: _ClassVar[int]
    key: ResourceKey
    backend_id: str
    slice_uid: str
    def __init__(self, key: _Optional[_Union[ResourceKey, _Mapping]] = ..., backend_id: _Optional[str] = ..., slice_uid: _Optional[str] = ...) -> None: ...

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

class JobSpec(_message.Message):
    __slots__ = ("version", "name", "entrypoint", "resources", "environment", "bundle_id", "scheduling_timeout", "ports", "max_task_failures", "max_retries_failure", "max_retries_preemption", "constraints", "coscheduling", "replicas", "timeout", "fail_if_exists", "preemption_policy", "existing_job_policy", "priority_band", "task_image", "submit_argv", "client_revision_date", "container_profile")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_ID_FIELD_NUMBER: _ClassVar[int]
    SCHEDULING_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    PORTS_FIELD_NUMBER: _ClassVar[int]
    MAX_TASK_FAILURES_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRIES_FAILURE_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRIES_PREEMPTION_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    COSCHEDULING_FIELD_NUMBER: _ClassVar[int]
    REPLICAS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    FAIL_IF_EXISTS_FIELD_NUMBER: _ClassVar[int]
    PREEMPTION_POLICY_FIELD_NUMBER: _ClassVar[int]
    EXISTING_JOB_POLICY_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_BAND_FIELD_NUMBER: _ClassVar[int]
    TASK_IMAGE_FIELD_NUMBER: _ClassVar[int]
    SUBMIT_ARGV_FIELD_NUMBER: _ClassVar[int]
    CLIENT_REVISION_DATE_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_PROFILE_FIELD_NUMBER: _ClassVar[int]
    version: int
    name: str
    entrypoint: RuntimeEntrypoint
    resources: ResourceSpecProto
    environment: EnvironmentConfig
    bundle_id: str
    scheduling_timeout: _time_pb2.Duration
    ports: _containers.RepeatedScalarFieldContainer[str]
    max_task_failures: int
    max_retries_failure: int
    max_retries_preemption: int
    constraints: _containers.RepeatedCompositeFieldContainer[Constraint]
    coscheduling: CoschedulingConfig
    replicas: int
    timeout: _time_pb2.Duration
    fail_if_exists: bool
    preemption_policy: JobPreemptionPolicy
    existing_job_policy: ExistingJobPolicy
    priority_band: PriorityBand
    task_image: str
    submit_argv: _containers.RepeatedScalarFieldContainer[str]
    client_revision_date: str
    container_profile: ContainerProfile
    def __init__(self, version: _Optional[int] = ..., name: _Optional[str] = ..., entrypoint: _Optional[_Union[RuntimeEntrypoint, _Mapping]] = ..., resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ..., environment: _Optional[_Union[EnvironmentConfig, _Mapping]] = ..., bundle_id: _Optional[str] = ..., scheduling_timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., ports: _Optional[_Iterable[str]] = ..., max_task_failures: _Optional[int] = ..., max_retries_failure: _Optional[int] = ..., max_retries_preemption: _Optional[int] = ..., constraints: _Optional[_Iterable[_Union[Constraint, _Mapping]]] = ..., coscheduling: _Optional[_Union[CoschedulingConfig, _Mapping]] = ..., replicas: _Optional[int] = ..., timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., fail_if_exists: _Optional[bool] = ..., preemption_policy: _Optional[_Union[JobPreemptionPolicy, str]] = ..., existing_job_policy: _Optional[_Union[ExistingJobPolicy, str]] = ..., priority_band: _Optional[_Union[PriorityBand, str]] = ..., task_image: _Optional[str] = ..., submit_argv: _Optional[_Iterable[str]] = ..., client_revision_date: _Optional[str] = ..., container_profile: _Optional[_Union[ContainerProfile, str]] = ...) -> None: ...

class SubmitJobRequest(_message.Message):
    __slots__ = ("spec", "bundle_blob")
    SPEC_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_BLOB_FIELD_NUMBER: _ClassVar[int]
    spec: JobSpec
    bundle_blob: bytes
    def __init__(self, spec: _Optional[_Union[JobSpec, _Mapping]] = ..., bundle_blob: _Optional[bytes] = ...) -> None: ...

class SubmitJobResponse(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: JobIdentity
    def __init__(self, job: _Optional[_Union[JobIdentity, _Mapping]] = ...) -> None: ...

class JobQuery(_message.Message):
    __slots__ = ("owner_id", "parent", "job_id_prefix", "states", "backend_id", "execution_cluster_id", "page", "resource_id", "top_level_only")
    OWNER_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_PREFIX_FIELD_NUMBER: _ClassVar[int]
    STATES_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    TOP_LEVEL_ONLY_FIELD_NUMBER: _ClassVar[int]
    owner_id: str
    parent: ResourceKey
    job_id_prefix: str
    states: _containers.RepeatedScalarFieldContainer[JobState]
    backend_id: str
    execution_cluster_id: str
    page: PageRequest
    resource_id: str
    top_level_only: bool
    def __init__(self, owner_id: _Optional[str] = ..., parent: _Optional[_Union[ResourceKey, _Mapping]] = ..., job_id_prefix: _Optional[str] = ..., states: _Optional[_Iterable[_Union[JobState, str]]] = ..., backend_id: _Optional[str] = ..., execution_cluster_id: _Optional[str] = ..., page: _Optional[_Union[PageRequest, _Mapping]] = ..., resource_id: _Optional[str] = ..., top_level_only: _Optional[bool] = ...) -> None: ...

class JobSummary(_message.Message):
    __slots__ = ("identity", "owner_id", "parent", "state", "execution_cluster_id", "backend_id", "num_tasks", "submitted_at", "started_at", "finished_at", "error_message", "pending_reason", "exit_code", "resources")
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    OWNER_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_TASKS_FIELD_NUMBER: _ClassVar[int]
    SUBMITTED_AT_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    PENDING_REASON_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    identity: JobIdentity
    owner_id: str
    parent: JobIdentity
    state: JobState
    execution_cluster_id: str
    backend_id: str
    num_tasks: int
    submitted_at: _time_pb2.Timestamp
    started_at: _time_pb2.Timestamp
    finished_at: _time_pb2.Timestamp
    error_message: str
    pending_reason: str
    exit_code: int
    resources: ResourceSpecProto
    def __init__(self, identity: _Optional[_Union[JobIdentity, _Mapping]] = ..., owner_id: _Optional[str] = ..., parent: _Optional[_Union[JobIdentity, _Mapping]] = ..., state: _Optional[_Union[JobState, str]] = ..., execution_cluster_id: _Optional[str] = ..., backend_id: _Optional[str] = ..., num_tasks: _Optional[int] = ..., submitted_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., started_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., error_message: _Optional[str] = ..., pending_reason: _Optional[str] = ..., exit_code: _Optional[int] = ..., resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ...) -> None: ...

class JobDetail(_message.Message):
    __slots__ = ("summary", "spec")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    summary: JobSummary
    spec: JobSpec
    def __init__(self, summary: _Optional[_Union[JobSummary, _Mapping]] = ..., spec: _Optional[_Union[JobSpec, _Mapping]] = ...) -> None: ...

class ListJobsRequest(_message.Message):
    __slots__ = ("query",)
    QUERY_FIELD_NUMBER: _ClassVar[int]
    query: JobQuery
    def __init__(self, query: _Optional[_Union[JobQuery, _Mapping]] = ...) -> None: ...

class ListJobsResponse(_message.Message):
    __slots__ = ("jobs", "page")
    JOBS_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    jobs: _containers.RepeatedCompositeFieldContainer[JobSummary]
    page: PageInfo
    def __init__(self, jobs: _Optional[_Iterable[_Union[JobSummary, _Mapping]]] = ..., page: _Optional[_Union[PageInfo, _Mapping]] = ...) -> None: ...

class UserSummary(_message.Message):
    __slots__ = ("user_id", "task_state_counts", "job_state_counts", "role", "budget_limit", "budget_spent", "max_band", "budget_configured")
    class TaskStateCountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class JobStateCountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_STATE_COUNTS_FIELD_NUMBER: _ClassVar[int]
    JOB_STATE_COUNTS_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    BUDGET_LIMIT_FIELD_NUMBER: _ClassVar[int]
    BUDGET_SPENT_FIELD_NUMBER: _ClassVar[int]
    MAX_BAND_FIELD_NUMBER: _ClassVar[int]
    BUDGET_CONFIGURED_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    task_state_counts: _containers.ScalarMap[str, int]
    job_state_counts: _containers.ScalarMap[str, int]
    role: str
    budget_limit: int
    budget_spent: int
    max_band: int
    budget_configured: bool
    def __init__(self, user_id: _Optional[str] = ..., task_state_counts: _Optional[_Mapping[str, int]] = ..., job_state_counts: _Optional[_Mapping[str, int]] = ..., role: _Optional[str] = ..., budget_limit: _Optional[int] = ..., budget_spent: _Optional[int] = ..., max_band: _Optional[int] = ..., budget_configured: _Optional[bool] = ...) -> None: ...

class ListUsersRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListUsersResponse(_message.Message):
    __slots__ = ("users",)
    USERS_FIELD_NUMBER: _ClassVar[int]
    users: _containers.RepeatedCompositeFieldContainer[UserSummary]
    def __init__(self, users: _Optional[_Iterable[_Union[UserSummary, _Mapping]]] = ...) -> None: ...

class DescribeJobRequest(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: ResourceKey
    def __init__(self, job: _Optional[_Union[ResourceKey, _Mapping]] = ...) -> None: ...

class DescribeJobResponse(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: JobDetail
    def __init__(self, job: _Optional[_Union[JobDetail, _Mapping]] = ...) -> None: ...

class GetJobStateRequest(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: JobIdentity
    def __init__(self, job: _Optional[_Union[JobIdentity, _Mapping]] = ...) -> None: ...

class GetJobStateResponse(_message.Message):
    __slots__ = ("state",)
    STATE_FIELD_NUMBER: _ClassVar[int]
    state: JobState
    def __init__(self, state: _Optional[_Union[JobState, str]] = ...) -> None: ...

class TaskQuery(_message.Message):
    __slots__ = ("job", "job_id_prefix", "states", "backend_id", "authority_cluster_id", "execution_cluster_id", "page")
    JOB_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_PREFIX_FIELD_NUMBER: _ClassVar[int]
    STATES_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    AUTHORITY_CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    job: ResourceKey
    job_id_prefix: str
    states: _containers.RepeatedScalarFieldContainer[TaskState]
    backend_id: str
    authority_cluster_id: str
    execution_cluster_id: str
    page: PageRequest
    def __init__(self, job: _Optional[_Union[ResourceKey, _Mapping]] = ..., job_id_prefix: _Optional[str] = ..., states: _Optional[_Iterable[_Union[TaskState, str]]] = ..., backend_id: _Optional[str] = ..., authority_cluster_id: _Optional[str] = ..., execution_cluster_id: _Optional[str] = ..., page: _Optional[_Union[PageRequest, _Mapping]] = ...) -> None: ...

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
    identity: TaskIdentity
    job: JobIdentity
    task_index: int
    state: TaskState
    execution_cluster_id: str
    backend_id: str
    current_attempt: AttemptIdentity
    current_node: NodeIdentity
    failure_count: int
    preemption_count: int
    submitted_at: _time_pb2.Timestamp
    started_at: _time_pb2.Timestamp
    finished_at: _time_pb2.Timestamp
    status_message: str
    error_message: str
    def __init__(self, identity: _Optional[_Union[TaskIdentity, _Mapping]] = ..., job: _Optional[_Union[JobIdentity, _Mapping]] = ..., task_index: _Optional[int] = ..., state: _Optional[_Union[TaskState, str]] = ..., execution_cluster_id: _Optional[str] = ..., backend_id: _Optional[str] = ..., current_attempt: _Optional[_Union[AttemptIdentity, _Mapping]] = ..., current_node: _Optional[_Union[NodeIdentity, _Mapping]] = ..., failure_count: _Optional[int] = ..., preemption_count: _Optional[int] = ..., submitted_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., started_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., status_message: _Optional[str] = ..., error_message: _Optional[str] = ...) -> None: ...

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
    identity: AttemptIdentity
    state: TaskState
    execution_cluster_id: str
    backend_id: str
    node: NodeIdentity
    created_at: _time_pb2.Timestamp
    started_at: _time_pb2.Timestamp
    finished_at: _time_pb2.Timestamp
    exit_code: int
    error_message: str
    terminal_reason: str
    def __init__(self, identity: _Optional[_Union[AttemptIdentity, _Mapping]] = ..., state: _Optional[_Union[TaskState, str]] = ..., execution_cluster_id: _Optional[str] = ..., backend_id: _Optional[str] = ..., node: _Optional[_Union[NodeIdentity, _Mapping]] = ..., created_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., started_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., exit_code: _Optional[int] = ..., error_message: _Optional[str] = ..., terminal_reason: _Optional[str] = ...) -> None: ...

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
    source_statuses: _containers.RepeatedCompositeFieldContainer[ResourceSourceStatus]
    root_cause_highlights: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, summary: _Optional[_Union[TaskSummary, _Mapping]] = ..., attempts: _Optional[_Iterable[_Union[AttemptSummary, _Mapping]]] = ..., source_statuses: _Optional[_Iterable[_Union[ResourceSourceStatus, _Mapping]]] = ..., root_cause_highlights: _Optional[_Iterable[str]] = ...) -> None: ...

class AttemptDetail(_message.Message):
    __slots__ = ("summary", "runtime", "source_statuses")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    summary: AttemptSummary
    runtime: AttemptRuntimeObject
    source_statuses: _containers.RepeatedCompositeFieldContainer[ResourceSourceStatus]
    def __init__(self, summary: _Optional[_Union[AttemptSummary, _Mapping]] = ..., runtime: _Optional[_Union[AttemptRuntimeObject, _Mapping]] = ..., source_statuses: _Optional[_Iterable[_Union[ResourceSourceStatus, _Mapping]]] = ...) -> None: ...

class ListTasksRequest(_message.Message):
    __slots__ = ("query",)
    QUERY_FIELD_NUMBER: _ClassVar[int]
    query: TaskQuery
    def __init__(self, query: _Optional[_Union[TaskQuery, _Mapping]] = ...) -> None: ...

class ListTasksResponse(_message.Message):
    __slots__ = ("tasks", "page")
    TASKS_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    tasks: _containers.RepeatedCompositeFieldContainer[TaskSummary]
    page: PageInfo
    def __init__(self, tasks: _Optional[_Iterable[_Union[TaskSummary, _Mapping]]] = ..., page: _Optional[_Union[PageInfo, _Mapping]] = ...) -> None: ...

class DescribeTaskRequest(_message.Message):
    __slots__ = ("task",)
    TASK_FIELD_NUMBER: _ClassVar[int]
    task: ResourceKey
    def __init__(self, task: _Optional[_Union[ResourceKey, _Mapping]] = ...) -> None: ...

class DescribeTaskResponse(_message.Message):
    __slots__ = ("task",)
    TASK_FIELD_NUMBER: _ClassVar[int]
    task: TaskDetail
    def __init__(self, task: _Optional[_Union[TaskDetail, _Mapping]] = ...) -> None: ...

class BatchDescribeTasksRequest(_message.Message):
    __slots__ = ("tasks",)
    TASKS_FIELD_NUMBER: _ClassVar[int]
    tasks: _containers.RepeatedCompositeFieldContainer[ResourceKey]
    def __init__(self, tasks: _Optional[_Iterable[_Union[ResourceKey, _Mapping]]] = ...) -> None: ...

class BatchDescribeTasksResponse(_message.Message):
    __slots__ = ("tasks",)
    TASKS_FIELD_NUMBER: _ClassVar[int]
    tasks: _containers.RepeatedCompositeFieldContainer[TaskDetail]
    def __init__(self, tasks: _Optional[_Iterable[_Union[TaskDetail, _Mapping]]] = ...) -> None: ...

class DescribeAttemptRequest(_message.Message):
    __slots__ = ("attempt",)
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    attempt: AttemptLocator
    def __init__(self, attempt: _Optional[_Union[AttemptLocator, _Mapping]] = ...) -> None: ...

class DescribeAttemptResponse(_message.Message):
    __slots__ = ("attempt",)
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    attempt: AttemptDetail
    def __init__(self, attempt: _Optional[_Union[AttemptDetail, _Mapping]] = ...) -> None: ...

class NodeQuery(_message.Message):
    __slots__ = ("backend_id", "contains", "health", "page")
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    CONTAINS_FIELD_NUMBER: _ClassVar[int]
    HEALTH_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    backend_id: str
    contains: str
    health: _containers.RepeatedScalarFieldContainer[NodeHealth]
    page: PageRequest
    def __init__(self, backend_id: _Optional[str] = ..., contains: _Optional[str] = ..., health: _Optional[_Iterable[_Union[NodeHealth, str]]] = ..., page: _Optional[_Union[PageRequest, _Mapping]] = ...) -> None: ...

class NodeCapacity(_message.Message):
    __slots__ = ("cpu_millicores", "memory_bytes", "disk_bytes", "accelerator_kind", "accelerator_variant", "accelerator_count")
    CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_KIND_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_VARIANT_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_COUNT_FIELD_NUMBER: _ClassVar[int]
    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int
    accelerator_kind: str
    accelerator_variant: str
    accelerator_count: int
    def __init__(self, cpu_millicores: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., accelerator_kind: _Optional[str] = ..., accelerator_variant: _Optional[str] = ..., accelerator_count: _Optional[int] = ...) -> None: ...

class NodeSummary(_message.Message):
    __slots__ = ("identity", "health", "schedulable", "capacity", "scaling_group_id", "slice", "running_task_count", "observed_at", "region")
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    HEALTH_FIELD_NUMBER: _ClassVar[int]
    SCHEDULABLE_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    SLICE_FIELD_NUMBER: _ClassVar[int]
    RUNNING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    identity: NodeIdentity
    health: NodeHealth
    schedulable: bool
    capacity: NodeCapacity
    scaling_group_id: str
    slice: SliceIdentity
    running_task_count: int
    observed_at: _time_pb2.Timestamp
    region: str
    def __init__(self, identity: _Optional[_Union[NodeIdentity, _Mapping]] = ..., health: _Optional[_Union[NodeHealth, str]] = ..., schedulable: _Optional[bool] = ..., capacity: _Optional[_Union[NodeCapacity, _Mapping]] = ..., scaling_group_id: _Optional[str] = ..., slice: _Optional[_Union[SliceIdentity, _Mapping]] = ..., running_task_count: _Optional[int] = ..., observed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., region: _Optional[str] = ...) -> None: ...

class NodeAttribute(_message.Message):
    __slots__ = ("key", "string_value", "integer_value", "float_value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    INTEGER_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    string_value: str
    integer_value: int
    float_value: float
    def __init__(self, key: _Optional[str] = ..., string_value: _Optional[str] = ..., integer_value: _Optional[int] = ..., float_value: _Optional[float] = ...) -> None: ...

class NodeDetail(_message.Message):
    __slots__ = ("summary", "address", "attributes", "recent_attempts", "bootstrap_logs", "source_statuses")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    RECENT_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    BOOTSTRAP_LOGS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    summary: NodeSummary
    address: str
    attributes: _containers.RepeatedCompositeFieldContainer[NodeAttribute]
    recent_attempts: _containers.RepeatedCompositeFieldContainer[AttemptSummary]
    bootstrap_logs: str
    source_statuses: _containers.RepeatedCompositeFieldContainer[ResourceSourceStatus]
    def __init__(self, summary: _Optional[_Union[NodeSummary, _Mapping]] = ..., address: _Optional[str] = ..., attributes: _Optional[_Iterable[_Union[NodeAttribute, _Mapping]]] = ..., recent_attempts: _Optional[_Iterable[_Union[AttemptSummary, _Mapping]]] = ..., bootstrap_logs: _Optional[str] = ..., source_statuses: _Optional[_Iterable[_Union[ResourceSourceStatus, _Mapping]]] = ...) -> None: ...

class ListNodesRequest(_message.Message):
    __slots__ = ("query",)
    QUERY_FIELD_NUMBER: _ClassVar[int]
    query: NodeQuery
    def __init__(self, query: _Optional[_Union[NodeQuery, _Mapping]] = ...) -> None: ...

class ListNodesResponse(_message.Message):
    __slots__ = ("nodes", "page")
    NODES_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[NodeSummary]
    page: PageInfo
    def __init__(self, nodes: _Optional[_Iterable[_Union[NodeSummary, _Mapping]]] = ..., page: _Optional[_Union[PageInfo, _Mapping]] = ...) -> None: ...

class DescribeNodeRequest(_message.Message):
    __slots__ = ("node",)
    NODE_FIELD_NUMBER: _ClassVar[int]
    node: NodeLocator
    def __init__(self, node: _Optional[_Union[NodeLocator, _Mapping]] = ...) -> None: ...

class DescribeNodeResponse(_message.Message):
    __slots__ = ("node",)
    NODE_FIELD_NUMBER: _ClassVar[int]
    node: NodeDetail
    def __init__(self, node: _Optional[_Union[NodeDetail, _Mapping]] = ...) -> None: ...

class SliceQuery(_message.Message):
    __slots__ = ("backend_id", "scaling_group_id", "page")
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    backend_id: str
    scaling_group_id: str
    page: PageRequest
    def __init__(self, backend_id: _Optional[str] = ..., scaling_group_id: _Optional[str] = ..., page: _Optional[_Union[PageRequest, _Mapping]] = ...) -> None: ...

class SliceSummary(_message.Message):
    __slots__ = ("identity", "scaling_group_id", "lifecycle", "membership_state", "observed_member_count", "observed_at", "error_message", "created_at", "last_active_at", "capacity_state", "healthy_member_count", "degraded_member_count", "running_task_count")
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    LIFECYCLE_FIELD_NUMBER: _ClassVar[int]
    MEMBERSHIP_STATE_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_MEMBER_COUNT_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_ACTIVE_AT_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_STATE_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_MEMBER_COUNT_FIELD_NUMBER: _ClassVar[int]
    DEGRADED_MEMBER_COUNT_FIELD_NUMBER: _ClassVar[int]
    RUNNING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    identity: SliceIdentity
    scaling_group_id: str
    lifecycle: SliceLifecycle
    membership_state: MembershipState
    observed_member_count: int
    observed_at: _time_pb2.Timestamp
    error_message: str
    created_at: _time_pb2.Timestamp
    last_active_at: _time_pb2.Timestamp
    capacity_state: SliceCapacityState
    healthy_member_count: int
    degraded_member_count: int
    running_task_count: int
    def __init__(self, identity: _Optional[_Union[SliceIdentity, _Mapping]] = ..., scaling_group_id: _Optional[str] = ..., lifecycle: _Optional[_Union[SliceLifecycle, str]] = ..., membership_state: _Optional[_Union[MembershipState, str]] = ..., observed_member_count: _Optional[int] = ..., observed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., error_message: _Optional[str] = ..., created_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., last_active_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., capacity_state: _Optional[_Union[SliceCapacityState, str]] = ..., healthy_member_count: _Optional[int] = ..., degraded_member_count: _Optional[int] = ..., running_task_count: _Optional[int] = ...) -> None: ...

class SliceMember(_message.Message):
    __slots__ = ("provider_node_id", "node", "observed_at", "worker_id", "healthy", "usability", "running_task_count", "zone")
    PROVIDER_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    USABILITY_FIELD_NUMBER: _ClassVar[int]
    RUNNING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    ZONE_FIELD_NUMBER: _ClassVar[int]
    provider_node_id: str
    node: NodeIdentity
    observed_at: _time_pb2.Timestamp
    worker_id: str
    healthy: bool
    usability: str
    running_task_count: int
    zone: str
    def __init__(self, provider_node_id: _Optional[str] = ..., node: _Optional[_Union[NodeIdentity, _Mapping]] = ..., observed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., worker_id: _Optional[str] = ..., healthy: _Optional[bool] = ..., usability: _Optional[str] = ..., running_task_count: _Optional[int] = ..., zone: _Optional[str] = ...) -> None: ...

class SliceDetail(_message.Message):
    __slots__ = ("summary", "members", "source_statuses")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    MEMBERS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    summary: SliceSummary
    members: _containers.RepeatedCompositeFieldContainer[SliceMember]
    source_statuses: _containers.RepeatedCompositeFieldContainer[ResourceSourceStatus]
    def __init__(self, summary: _Optional[_Union[SliceSummary, _Mapping]] = ..., members: _Optional[_Iterable[_Union[SliceMember, _Mapping]]] = ..., source_statuses: _Optional[_Iterable[_Union[ResourceSourceStatus, _Mapping]]] = ...) -> None: ...

class ListSlicesRequest(_message.Message):
    __slots__ = ("query",)
    QUERY_FIELD_NUMBER: _ClassVar[int]
    query: SliceQuery
    def __init__(self, query: _Optional[_Union[SliceQuery, _Mapping]] = ...) -> None: ...

class ListSlicesResponse(_message.Message):
    __slots__ = ("slices", "page")
    SLICES_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    slices: _containers.RepeatedCompositeFieldContainer[SliceSummary]
    page: PageInfo
    def __init__(self, slices: _Optional[_Iterable[_Union[SliceSummary, _Mapping]]] = ..., page: _Optional[_Union[PageInfo, _Mapping]] = ...) -> None: ...

class DescribeSliceRequest(_message.Message):
    __slots__ = ("slice",)
    SLICE_FIELD_NUMBER: _ClassVar[int]
    slice: SliceLocator
    def __init__(self, slice: _Optional[_Union[SliceLocator, _Mapping]] = ...) -> None: ...

class DescribeSliceResponse(_message.Message):
    __slots__ = ("slice",)
    SLICE_FIELD_NUMBER: _ClassVar[int]
    slice: SliceDetail
    def __init__(self, slice: _Optional[_Union[SliceDetail, _Mapping]] = ...) -> None: ...

class StringValues(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, values: _Optional[_Iterable[str]] = ...) -> None: ...

class CapacityResourceAvailability(_message.Message):
    __slots__ = ("version", "observed_at", "amounts", "total_amounts", "held_by_band")
    class AmountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class TotalAmountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    VERSION_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    AMOUNTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_AMOUNTS_FIELD_NUMBER: _ClassVar[int]
    HELD_BY_BAND_FIELD_NUMBER: _ClassVar[int]
    version: int
    observed_at: _time_pb2.Timestamp
    amounts: _containers.ScalarMap[str, int]
    total_amounts: _containers.ScalarMap[str, int]
    held_by_band: _containers.RepeatedCompositeFieldContainer[CapacityBandAvailability]
    def __init__(self, version: _Optional[int] = ..., observed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., amounts: _Optional[_Mapping[str, int]] = ..., total_amounts: _Optional[_Mapping[str, int]] = ..., held_by_band: _Optional[_Iterable[_Union[CapacityBandAvailability, _Mapping]]] = ...) -> None: ...

class CapacityBandAvailability(_message.Message):
    __slots__ = ("band", "amounts")
    class AmountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    BAND_FIELD_NUMBER: _ClassVar[int]
    AMOUNTS_FIELD_NUMBER: _ClassVar[int]
    band: int
    amounts: _containers.ScalarMap[str, int]
    def __init__(self, band: _Optional[int] = ..., amounts: _Optional[_Mapping[str, int]] = ...) -> None: ...

class CapacitySlice(_message.Message):
    __slots__ = ("summary", "members")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    MEMBERS_FIELD_NUMBER: _ClassVar[int]
    summary: SliceSummary
    members: _containers.RepeatedCompositeFieldContainer[SliceMember]
    def __init__(self, summary: _Optional[_Union[SliceSummary, _Mapping]] = ..., members: _Optional[_Iterable[_Union[SliceMember, _Mapping]]] = ...) -> None: ...

class CapacityScalingGroup(_message.Message):
    __slots__ = ("name", "backend_id", "device_type", "device_variant", "quota_pool", "allocation_tier", "region", "current_demand", "peak_demand", "backoff_until", "consecutive_failures", "last_scale_up", "last_scale_down", "slices", "slice_state_counts", "availability_status", "availability_reason", "blocked_until", "scale_up_cooldown_until", "idle_threshold_ms")
    class SliceStateCountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_VARIANT_FIELD_NUMBER: _ClassVar[int]
    QUOTA_POOL_FIELD_NUMBER: _ClassVar[int]
    ALLOCATION_TIER_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    CURRENT_DEMAND_FIELD_NUMBER: _ClassVar[int]
    PEAK_DEMAND_FIELD_NUMBER: _ClassVar[int]
    BACKOFF_UNTIL_FIELD_NUMBER: _ClassVar[int]
    CONSECUTIVE_FAILURES_FIELD_NUMBER: _ClassVar[int]
    LAST_SCALE_UP_FIELD_NUMBER: _ClassVar[int]
    LAST_SCALE_DOWN_FIELD_NUMBER: _ClassVar[int]
    SLICES_FIELD_NUMBER: _ClassVar[int]
    SLICE_STATE_COUNTS_FIELD_NUMBER: _ClassVar[int]
    AVAILABILITY_STATUS_FIELD_NUMBER: _ClassVar[int]
    AVAILABILITY_REASON_FIELD_NUMBER: _ClassVar[int]
    BLOCKED_UNTIL_FIELD_NUMBER: _ClassVar[int]
    SCALE_UP_COOLDOWN_UNTIL_FIELD_NUMBER: _ClassVar[int]
    IDLE_THRESHOLD_MS_FIELD_NUMBER: _ClassVar[int]
    name: str
    backend_id: str
    device_type: str
    device_variant: str
    quota_pool: str
    allocation_tier: int
    region: str
    current_demand: int
    peak_demand: int
    backoff_until: _time_pb2.Timestamp
    consecutive_failures: int
    last_scale_up: _time_pb2.Timestamp
    last_scale_down: _time_pb2.Timestamp
    slices: _containers.RepeatedCompositeFieldContainer[CapacitySlice]
    slice_state_counts: _containers.ScalarMap[str, int]
    availability_status: str
    availability_reason: str
    blocked_until: _time_pb2.Timestamp
    scale_up_cooldown_until: _time_pb2.Timestamp
    idle_threshold_ms: int
    def __init__(self, name: _Optional[str] = ..., backend_id: _Optional[str] = ..., device_type: _Optional[str] = ..., device_variant: _Optional[str] = ..., quota_pool: _Optional[str] = ..., allocation_tier: _Optional[int] = ..., region: _Optional[str] = ..., current_demand: _Optional[int] = ..., peak_demand: _Optional[int] = ..., backoff_until: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., consecutive_failures: _Optional[int] = ..., last_scale_up: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., last_scale_down: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., slices: _Optional[_Iterable[_Union[CapacitySlice, _Mapping]]] = ..., slice_state_counts: _Optional[_Mapping[str, int]] = ..., availability_status: _Optional[str] = ..., availability_reason: _Optional[str] = ..., blocked_until: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., scale_up_cooldown_until: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., idle_threshold_ms: _Optional[int] = ...) -> None: ...

class CapacityAction(_message.Message):
    __slots__ = ("timestamp", "action_type", "scaling_group_id", "slice_id", "reason", "status")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    ACTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    SLICE_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    timestamp: _time_pb2.Timestamp
    action_type: str
    scaling_group_id: str
    slice_id: str
    reason: str
    status: str
    def __init__(self, timestamp: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., action_type: _Optional[str] = ..., scaling_group_id: _Optional[str] = ..., slice_id: _Optional[str] = ..., reason: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class CapacityDemandEntry(_message.Message):
    __slots__ = ("task_ids", "coschedule_group_id", "device_type", "device_variant", "preemptible")
    TASK_IDS_FIELD_NUMBER: _ClassVar[int]
    COSCHEDULE_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_VARIANT_FIELD_NUMBER: _ClassVar[int]
    PREEMPTIBLE_FIELD_NUMBER: _ClassVar[int]
    task_ids: _containers.RepeatedScalarFieldContainer[str]
    coschedule_group_id: str
    device_type: str
    device_variant: str
    preemptible: bool
    def __init__(self, task_ids: _Optional[_Iterable[str]] = ..., coschedule_group_id: _Optional[str] = ..., device_type: _Optional[str] = ..., device_variant: _Optional[str] = ..., preemptible: _Optional[bool] = ...) -> None: ...

class CapacityUnmetDemand(_message.Message):
    __slots__ = ("entry", "reason")
    ENTRY_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    entry: CapacityDemandEntry
    reason: str
    def __init__(self, entry: _Optional[_Union[CapacityDemandEntry, _Mapping]] = ..., reason: _Optional[str] = ...) -> None: ...

class CapacityGroupRouting(_message.Message):
    __slots__ = ("scaling_group_id", "priority", "assigned", "launch", "decision", "reason")
    SCALING_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    ASSIGNED_FIELD_NUMBER: _ClassVar[int]
    LAUNCH_FIELD_NUMBER: _ClassVar[int]
    DECISION_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    scaling_group_id: str
    priority: int
    assigned: int
    launch: int
    decision: str
    reason: str
    def __init__(self, scaling_group_id: _Optional[str] = ..., priority: _Optional[int] = ..., assigned: _Optional[int] = ..., launch: _Optional[int] = ..., decision: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class CapacityRouting(_message.Message):
    __slots__ = ("unmet", "groups")
    UNMET_FIELD_NUMBER: _ClassVar[int]
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    unmet: _containers.RepeatedCompositeFieldContainer[CapacityUnmetDemand]
    groups: _containers.RepeatedCompositeFieldContainer[CapacityGroupRouting]
    def __init__(self, unmet: _Optional[_Iterable[_Union[CapacityUnmetDemand, _Mapping]]] = ..., groups: _Optional[_Iterable[_Union[CapacityGroupRouting, _Mapping]]] = ...) -> None: ...

class CapacityKubernetesPod(_message.Message):
    __slots__ = ("pod_name", "task_id", "phase", "reason", "message", "last_transition", "node_name")
    POD_NAME_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    LAST_TRANSITION_FIELD_NUMBER: _ClassVar[int]
    NODE_NAME_FIELD_NUMBER: _ClassVar[int]
    pod_name: str
    task_id: str
    phase: str
    reason: str
    message: str
    last_transition: _time_pb2.Timestamp
    node_name: str
    def __init__(self, pod_name: _Optional[str] = ..., task_id: _Optional[str] = ..., phase: _Optional[str] = ..., reason: _Optional[str] = ..., message: _Optional[str] = ..., last_transition: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., node_name: _Optional[str] = ...) -> None: ...

class CapacityKubernetesPool(_message.Message):
    __slots__ = ("name", "instance_type", "scaling_group_id", "target_nodes", "current_nodes", "queued_nodes", "in_progress_nodes", "autoscaling", "min_nodes", "max_nodes", "capacity", "quota")
    NAME_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_NODES_FIELD_NUMBER: _ClassVar[int]
    CURRENT_NODES_FIELD_NUMBER: _ClassVar[int]
    QUEUED_NODES_FIELD_NUMBER: _ClassVar[int]
    IN_PROGRESS_NODES_FIELD_NUMBER: _ClassVar[int]
    AUTOSCALING_FIELD_NUMBER: _ClassVar[int]
    MIN_NODES_FIELD_NUMBER: _ClassVar[int]
    MAX_NODES_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_FIELD_NUMBER: _ClassVar[int]
    QUOTA_FIELD_NUMBER: _ClassVar[int]
    name: str
    instance_type: str
    scaling_group_id: str
    target_nodes: int
    current_nodes: int
    queued_nodes: int
    in_progress_nodes: int
    autoscaling: bool
    min_nodes: int
    max_nodes: int
    capacity: str
    quota: str
    def __init__(self, name: _Optional[str] = ..., instance_type: _Optional[str] = ..., scaling_group_id: _Optional[str] = ..., target_nodes: _Optional[int] = ..., current_nodes: _Optional[int] = ..., queued_nodes: _Optional[int] = ..., in_progress_nodes: _Optional[int] = ..., autoscaling: _Optional[bool] = ..., min_nodes: _Optional[int] = ..., max_nodes: _Optional[int] = ..., capacity: _Optional[str] = ..., quota: _Optional[str] = ...) -> None: ...

class CapacityKubernetesNode(_message.Message):
    __slots__ = ("name", "ready", "schedulable", "status_summary", "instance_type", "region", "accelerator_count", "accelerator_variant", "cpu_millicores", "memory_bytes", "disk_bytes", "running_pods", "created")
    NAME_FIELD_NUMBER: _ClassVar[int]
    READY_FIELD_NUMBER: _ClassVar[int]
    SCHEDULABLE_FIELD_NUMBER: _ClassVar[int]
    STATUS_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_COUNT_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_VARIANT_FIELD_NUMBER: _ClassVar[int]
    CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_BYTES_FIELD_NUMBER: _ClassVar[int]
    RUNNING_PODS_FIELD_NUMBER: _ClassVar[int]
    CREATED_FIELD_NUMBER: _ClassVar[int]
    name: str
    ready: bool
    schedulable: bool
    status_summary: str
    instance_type: str
    region: str
    accelerator_count: int
    accelerator_variant: str
    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int
    running_pods: int
    created: str
    def __init__(self, name: _Optional[str] = ..., ready: _Optional[bool] = ..., schedulable: _Optional[bool] = ..., status_summary: _Optional[str] = ..., instance_type: _Optional[str] = ..., region: _Optional[str] = ..., accelerator_count: _Optional[int] = ..., accelerator_variant: _Optional[str] = ..., cpu_millicores: _Optional[int] = ..., memory_bytes: _Optional[int] = ..., disk_bytes: _Optional[int] = ..., running_pods: _Optional[int] = ..., created: _Optional[str] = ...) -> None: ...

class CapacityKubernetesStatus(_message.Message):
    __slots__ = ("namespace", "total_nodes", "schedulable_nodes", "allocatable_cpu", "allocatable_memory", "pods", "provider_version", "pools", "nodes")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_NODES_FIELD_NUMBER: _ClassVar[int]
    SCHEDULABLE_NODES_FIELD_NUMBER: _ClassVar[int]
    ALLOCATABLE_CPU_FIELD_NUMBER: _ClassVar[int]
    ALLOCATABLE_MEMORY_FIELD_NUMBER: _ClassVar[int]
    PODS_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_VERSION_FIELD_NUMBER: _ClassVar[int]
    POOLS_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    total_nodes: int
    schedulable_nodes: int
    allocatable_cpu: str
    allocatable_memory: str
    pods: _containers.RepeatedCompositeFieldContainer[CapacityKubernetesPod]
    provider_version: str
    pools: _containers.RepeatedCompositeFieldContainer[CapacityKubernetesPool]
    nodes: _containers.RepeatedCompositeFieldContainer[CapacityKubernetesNode]
    def __init__(self, namespace: _Optional[str] = ..., total_nodes: _Optional[int] = ..., schedulable_nodes: _Optional[int] = ..., allocatable_cpu: _Optional[str] = ..., allocatable_memory: _Optional[str] = ..., pods: _Optional[_Iterable[_Union[CapacityKubernetesPod, _Mapping]]] = ..., provider_version: _Optional[str] = ..., pools: _Optional[_Iterable[_Union[CapacityKubernetesPool, _Mapping]]] = ..., nodes: _Optional[_Iterable[_Union[CapacityKubernetesNode, _Mapping]]] = ...) -> None: ...

class CapacityBackend(_message.Message):
    __slots__ = ("backend_id", "name", "kind", "capabilities", "advertised_attributes", "worker_count", "pending_task_count", "running_task_count", "has_autoscaler", "capacity_health", "availability", "scaling_groups", "recent_actions", "routing", "last_evaluation", "healthy_worker_count", "kubernetes")
    class AdvertisedAttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: StringValues
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[StringValues, _Mapping]] = ...) -> None: ...
    class CapacityHealthEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    ADVERTISED_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    WORKER_COUNT_FIELD_NUMBER: _ClassVar[int]
    PENDING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    RUNNING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    HAS_AUTOSCALER_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_HEALTH_FIELD_NUMBER: _ClassVar[int]
    AVAILABILITY_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUPS_FIELD_NUMBER: _ClassVar[int]
    RECENT_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    ROUTING_FIELD_NUMBER: _ClassVar[int]
    LAST_EVALUATION_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_WORKER_COUNT_FIELD_NUMBER: _ClassVar[int]
    KUBERNETES_FIELD_NUMBER: _ClassVar[int]
    backend_id: str
    name: str
    kind: str
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    advertised_attributes: _containers.MessageMap[str, StringValues]
    worker_count: int
    pending_task_count: int
    running_task_count: int
    has_autoscaler: bool
    capacity_health: _containers.ScalarMap[str, int]
    availability: CapacityResourceAvailability
    scaling_groups: _containers.RepeatedCompositeFieldContainer[CapacityScalingGroup]
    recent_actions: _containers.RepeatedCompositeFieldContainer[CapacityAction]
    routing: CapacityRouting
    last_evaluation: _time_pb2.Timestamp
    healthy_worker_count: int
    kubernetes: CapacityKubernetesStatus
    def __init__(self, backend_id: _Optional[str] = ..., name: _Optional[str] = ..., kind: _Optional[str] = ..., capabilities: _Optional[_Iterable[str]] = ..., advertised_attributes: _Optional[_Mapping[str, StringValues]] = ..., worker_count: _Optional[int] = ..., pending_task_count: _Optional[int] = ..., running_task_count: _Optional[int] = ..., has_autoscaler: _Optional[bool] = ..., capacity_health: _Optional[_Mapping[str, int]] = ..., availability: _Optional[_Union[CapacityResourceAvailability, _Mapping]] = ..., scaling_groups: _Optional[_Iterable[_Union[CapacityScalingGroup, _Mapping]]] = ..., recent_actions: _Optional[_Iterable[_Union[CapacityAction, _Mapping]]] = ..., routing: _Optional[_Union[CapacityRouting, _Mapping]] = ..., last_evaluation: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., healthy_worker_count: _Optional[int] = ..., kubernetes: _Optional[_Union[CapacityKubernetesStatus, _Mapping]] = ...) -> None: ...

class CapacityPeerBackend(_message.Message):
    __slots__ = ("backend_id", "name", "kind", "capabilities", "advertised_attributes", "scaling_groups", "worker_count", "pending_task_count", "running_task_count", "has_autoscaler", "capacity_health", "availability")
    class AdvertisedAttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: StringValues
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[StringValues, _Mapping]] = ...) -> None: ...
    class CapacityHealthEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    ADVERTISED_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    SCALING_GROUPS_FIELD_NUMBER: _ClassVar[int]
    WORKER_COUNT_FIELD_NUMBER: _ClassVar[int]
    PENDING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    RUNNING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    HAS_AUTOSCALER_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_HEALTH_FIELD_NUMBER: _ClassVar[int]
    AVAILABILITY_FIELD_NUMBER: _ClassVar[int]
    backend_id: str
    name: str
    kind: str
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    advertised_attributes: _containers.MessageMap[str, StringValues]
    scaling_groups: _containers.RepeatedScalarFieldContainer[str]
    worker_count: int
    pending_task_count: int
    running_task_count: int
    has_autoscaler: bool
    capacity_health: _containers.ScalarMap[str, int]
    availability: CapacityResourceAvailability
    def __init__(self, backend_id: _Optional[str] = ..., name: _Optional[str] = ..., kind: _Optional[str] = ..., capabilities: _Optional[_Iterable[str]] = ..., advertised_attributes: _Optional[_Mapping[str, StringValues]] = ..., scaling_groups: _Optional[_Iterable[str]] = ..., worker_count: _Optional[int] = ..., pending_task_count: _Optional[int] = ..., running_task_count: _Optional[int] = ..., has_autoscaler: _Optional[bool] = ..., capacity_health: _Optional[_Mapping[str, int]] = ..., availability: _Optional[_Union[CapacityResourceAvailability, _Mapping]] = ...) -> None: ...

class CapacityPeer(_message.Message):
    __slots__ = ("peer_id", "controller_address", "reachable", "last_contact_ms", "active_federated_jobs", "backends")
    PEER_ID_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    REACHABLE_FIELD_NUMBER: _ClassVar[int]
    LAST_CONTACT_MS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_FEDERATED_JOBS_FIELD_NUMBER: _ClassVar[int]
    BACKENDS_FIELD_NUMBER: _ClassVar[int]
    peer_id: str
    controller_address: str
    reachable: bool
    last_contact_ms: int
    active_federated_jobs: int
    backends: _containers.RepeatedCompositeFieldContainer[CapacityPeerBackend]
    def __init__(self, peer_id: _Optional[str] = ..., controller_address: _Optional[str] = ..., reachable: _Optional[bool] = ..., last_contact_ms: _Optional[int] = ..., active_federated_jobs: _Optional[int] = ..., backends: _Optional[_Iterable[_Union[CapacityPeerBackend, _Mapping]]] = ...) -> None: ...

class CapacityRunningPlacement(_message.Message):
    __slots__ = ("backend_id", "worker_id", "job_id", "user_id", "task_count")
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    backend_id: str
    worker_id: str
    job_id: str
    user_id: str
    task_count: int
    def __init__(self, backend_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., job_id: _Optional[str] = ..., user_id: _Optional[str] = ..., task_count: _Optional[int] = ...) -> None: ...

class CapacityUnroutableJob(_message.Message):
    __slots__ = ("job_id", "reason")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    reason: str
    def __init__(self, job_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class GetCapacityStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetCapacityStatusResponse(_message.Message):
    __slots__ = ("backends", "peers", "running_placements", "unroutable_job_count", "unroutable_jobs", "source_statuses")
    BACKENDS_FIELD_NUMBER: _ClassVar[int]
    PEERS_FIELD_NUMBER: _ClassVar[int]
    RUNNING_PLACEMENTS_FIELD_NUMBER: _ClassVar[int]
    UNROUTABLE_JOB_COUNT_FIELD_NUMBER: _ClassVar[int]
    UNROUTABLE_JOBS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    backends: _containers.RepeatedCompositeFieldContainer[CapacityBackend]
    peers: _containers.RepeatedCompositeFieldContainer[CapacityPeer]
    running_placements: _containers.RepeatedCompositeFieldContainer[CapacityRunningPlacement]
    unroutable_job_count: int
    unroutable_jobs: _containers.RepeatedCompositeFieldContainer[CapacityUnroutableJob]
    source_statuses: _containers.RepeatedCompositeFieldContainer[ResourceSourceStatus]
    def __init__(self, backends: _Optional[_Iterable[_Union[CapacityBackend, _Mapping]]] = ..., peers: _Optional[_Iterable[_Union[CapacityPeer, _Mapping]]] = ..., running_placements: _Optional[_Iterable[_Union[CapacityRunningPlacement, _Mapping]]] = ..., unroutable_job_count: _Optional[int] = ..., unroutable_jobs: _Optional[_Iterable[_Union[CapacityUnroutableJob, _Mapping]]] = ..., source_statuses: _Optional[_Iterable[_Union[ResourceSourceStatus, _Mapping]]] = ...) -> None: ...

class EndpointQuery(_message.Message):
    __slots__ = ("name_prefix", "task", "page")
    NAME_PREFIX_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    name_prefix: str
    task: ResourceKey
    page: PageRequest
    def __init__(self, name_prefix: _Optional[str] = ..., task: _Optional[_Union[ResourceKey, _Mapping]] = ..., page: _Optional[_Union[PageRequest, _Mapping]] = ...) -> None: ...

class EndpointSummary(_message.Message):
    __slots__ = ("key", "endpoint_id", "name", "task", "execution_cluster_id", "access", "lease_deadline")
    KEY_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    ACCESS_FIELD_NUMBER: _ClassVar[int]
    LEASE_DEADLINE_FIELD_NUMBER: _ClassVar[int]
    key: ResourceKey
    endpoint_id: str
    name: str
    task: ResourceKey
    execution_cluster_id: str
    access: EndpointAccess
    lease_deadline: _time_pb2.Timestamp
    def __init__(self, key: _Optional[_Union[ResourceKey, _Mapping]] = ..., endpoint_id: _Optional[str] = ..., name: _Optional[str] = ..., task: _Optional[_Union[ResourceKey, _Mapping]] = ..., execution_cluster_id: _Optional[str] = ..., access: _Optional[_Union[EndpointAccess, str]] = ..., lease_deadline: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...

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

class ListEndpointsRequest(_message.Message):
    __slots__ = ("query",)
    QUERY_FIELD_NUMBER: _ClassVar[int]
    query: EndpointQuery
    def __init__(self, query: _Optional[_Union[EndpointQuery, _Mapping]] = ...) -> None: ...

class ListEndpointsResponse(_message.Message):
    __slots__ = ("endpoints", "page")
    ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    endpoints: _containers.RepeatedCompositeFieldContainer[EndpointSummary]
    page: PageInfo
    def __init__(self, endpoints: _Optional[_Iterable[_Union[EndpointSummary, _Mapping]]] = ..., page: _Optional[_Union[PageInfo, _Mapping]] = ...) -> None: ...

class DescribeEndpointRequest(_message.Message):
    __slots__ = ("endpoint",)
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    endpoint: ResourceKey
    def __init__(self, endpoint: _Optional[_Union[ResourceKey, _Mapping]] = ...) -> None: ...

class DescribeEndpointResponse(_message.Message):
    __slots__ = ("endpoint",)
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    endpoint: EndpointDetail
    def __init__(self, endpoint: _Optional[_Union[EndpointDetail, _Mapping]] = ...) -> None: ...

class BatchDescribeEndpointsRequest(_message.Message):
    __slots__ = ("endpoints",)
    ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    endpoints: _containers.RepeatedCompositeFieldContainer[ResourceKey]
    def __init__(self, endpoints: _Optional[_Iterable[_Union[ResourceKey, _Mapping]]] = ...) -> None: ...

class BatchDescribeEndpointsResponse(_message.Message):
    __slots__ = ("endpoints",)
    ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    endpoints: _containers.RepeatedCompositeFieldContainer[EndpointDetail]
    def __init__(self, endpoints: _Optional[_Iterable[_Union[EndpointDetail, _Mapping]]] = ...) -> None: ...

class MintEndpointTokenRequest(_message.Message):
    __slots__ = ("endpoint", "ttl")
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    TTL_FIELD_NUMBER: _ClassVar[int]
    endpoint: ResourceKey
    ttl: _time_pb2.Duration
    def __init__(self, endpoint: _Optional[_Union[ResourceKey, _Mapping]] = ..., ttl: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ...) -> None: ...

class MintEndpointTokenResponse(_message.Message):
    __slots__ = ("token", "expires_at", "capability_url")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    CAPABILITY_URL_FIELD_NUMBER: _ClassVar[int]
    token: str
    expires_at: _time_pb2.Timestamp
    capability_url: str
    def __init__(self, token: _Optional[str] = ..., expires_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., capability_url: _Optional[str] = ...) -> None: ...

class ActivityQuery(_message.Message):
    __slots__ = ("target", "attempt_uid", "after", "page")
    TARGET_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    AFTER_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    target: ResourceKey
    attempt_uid: str
    after: _time_pb2.Timestamp
    page: PageRequest
    def __init__(self, target: _Optional[_Union[ResourceKey, _Mapping]] = ..., attempt_uid: _Optional[str] = ..., after: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., page: _Optional[_Union[PageRequest, _Mapping]] = ...) -> None: ...

class ActivityEntry(_message.Message):
    __slots__ = ("entry_id", "occurred_at", "source", "severity", "kind", "message", "target", "attempt_uid", "correlation_id", "attributes")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ENTRY_ID_FIELD_NUMBER: _ClassVar[int]
    OCCURRED_AT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    entry_id: str
    occurred_at: _time_pb2.Timestamp
    source: str
    severity: str
    kind: str
    message: str
    target: ResourceKey
    attempt_uid: str
    correlation_id: str
    attributes: _containers.ScalarMap[str, str]
    def __init__(self, entry_id: _Optional[str] = ..., occurred_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., source: _Optional[str] = ..., severity: _Optional[str] = ..., kind: _Optional[str] = ..., message: _Optional[str] = ..., target: _Optional[_Union[ResourceKey, _Mapping]] = ..., attempt_uid: _Optional[str] = ..., correlation_id: _Optional[str] = ..., attributes: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ListActivityRequest(_message.Message):
    __slots__ = ("query",)
    QUERY_FIELD_NUMBER: _ClassVar[int]
    query: ActivityQuery
    def __init__(self, query: _Optional[_Union[ActivityQuery, _Mapping]] = ...) -> None: ...

class ListActivityResponse(_message.Message):
    __slots__ = ("entries", "page")
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[ActivityEntry]
    page: PageInfo
    def __init__(self, entries: _Optional[_Iterable[_Union[ActivityEntry, _Mapping]]] = ..., page: _Optional[_Union[PageInfo, _Mapping]] = ...) -> None: ...

class LogQuery(_message.Message):
    __slots__ = ("after", "cursor", "max_lines", "substring", "minimum_level", "tail")
    AFTER_FIELD_NUMBER: _ClassVar[int]
    CURSOR_FIELD_NUMBER: _ClassVar[int]
    MAX_LINES_FIELD_NUMBER: _ClassVar[int]
    SUBSTRING_FIELD_NUMBER: _ClassVar[int]
    MINIMUM_LEVEL_FIELD_NUMBER: _ClassVar[int]
    TAIL_FIELD_NUMBER: _ClassVar[int]
    after: _time_pb2.Timestamp
    cursor: int
    max_lines: int
    substring: str
    minimum_level: _iris_logging_pb2.LogLevel
    tail: bool
    def __init__(self, after: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., cursor: _Optional[int] = ..., max_lines: _Optional[int] = ..., substring: _Optional[str] = ..., minimum_level: _Optional[_Union[_iris_logging_pb2.LogLevel, str]] = ..., tail: _Optional[bool] = ...) -> None: ...

class LogTarget(_message.Message):
    __slots__ = ("job", "task", "attempt")
    JOB_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    job: JobIdentity
    task: TaskIdentity
    attempt: AttemptIdentity
    def __init__(self, job: _Optional[_Union[JobIdentity, _Mapping]] = ..., task: _Optional[_Union[TaskIdentity, _Mapping]] = ..., attempt: _Optional[_Union[AttemptIdentity, _Mapping]] = ...) -> None: ...

class FetchLogsRequest(_message.Message):
    __slots__ = ("target", "query")
    TARGET_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    target: LogTarget
    query: LogQuery
    def __init__(self, target: _Optional[_Union[LogTarget, _Mapping]] = ..., query: _Optional[_Union[LogQuery, _Mapping]] = ...) -> None: ...

class FetchLogsResponse(_message.Message):
    __slots__ = ("entries", "next_cursor", "source_statuses")
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    NEXT_CURSOR_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[_iris_logging_pb2.LogEntry]
    next_cursor: int
    source_statuses: _containers.RepeatedCompositeFieldContainer[ResourceSourceStatus]
    def __init__(self, entries: _Optional[_Iterable[_Union[_iris_logging_pb2.LogEntry, _Mapping]]] = ..., next_cursor: _Optional[int] = ..., source_statuses: _Optional[_Iterable[_Union[ResourceSourceStatus, _Mapping]]] = ...) -> None: ...

class ActionReceipt(_message.Message):
    __slots__ = ("action_id", "kind", "target", "expected_target_uid", "expected_attempt_uid", "state", "result_code", "result_message", "created_at", "updated_at", "completed_at")
    ACTION_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_TARGET_UID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    RESULT_CODE_FIELD_NUMBER: _ClassVar[int]
    RESULT_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    action_id: str
    kind: ActionKind
    target: ResourceKey
    expected_target_uid: str
    expected_attempt_uid: str
    state: ActionState
    result_code: ActionResult
    result_message: str
    created_at: _time_pb2.Timestamp
    updated_at: _time_pb2.Timestamp
    completed_at: _time_pb2.Timestamp
    def __init__(self, action_id: _Optional[str] = ..., kind: _Optional[_Union[ActionKind, str]] = ..., target: _Optional[_Union[ResourceKey, _Mapping]] = ..., expected_target_uid: _Optional[str] = ..., expected_attempt_uid: _Optional[str] = ..., state: _Optional[_Union[ActionState, str]] = ..., result_code: _Optional[_Union[ActionResult, str]] = ..., result_message: _Optional[str] = ..., created_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., completed_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class CancelJobRequest(_message.Message):
    __slots__ = ("job", "idempotency_key")
    JOB_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_KEY_FIELD_NUMBER: _ClassVar[int]
    job: JobIdentity
    idempotency_key: str
    def __init__(self, job: _Optional[_Union[JobIdentity, _Mapping]] = ..., idempotency_key: _Optional[str] = ...) -> None: ...

class RetryTaskRequest(_message.Message):
    __slots__ = ("task", "expected_attempt_uid", "idempotency_key")
    TASK_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_ATTEMPT_UID_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_KEY_FIELD_NUMBER: _ClassVar[int]
    task: TaskIdentity
    expected_attempt_uid: str
    idempotency_key: str
    def __init__(self, task: _Optional[_Union[TaskIdentity, _Mapping]] = ..., expected_attempt_uid: _Optional[str] = ..., idempotency_key: _Optional[str] = ...) -> None: ...

class TerminateAttemptRequest(_message.Message):
    __slots__ = ("attempt", "idempotency_key")
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_KEY_FIELD_NUMBER: _ClassVar[int]
    attempt: AttemptIdentity
    idempotency_key: str
    def __init__(self, attempt: _Optional[_Union[AttemptIdentity, _Mapping]] = ..., idempotency_key: _Optional[str] = ...) -> None: ...

class ActionResponse(_message.Message):
    __slots__ = ("receipt",)
    RECEIPT_FIELD_NUMBER: _ClassVar[int]
    receipt: ActionReceipt
    def __init__(self, receipt: _Optional[_Union[ActionReceipt, _Mapping]] = ...) -> None: ...

class GetActionReceiptRequest(_message.Message):
    __slots__ = ("action_id",)
    ACTION_ID_FIELD_NUMBER: _ClassVar[int]
    action_id: str
    def __init__(self, action_id: _Optional[str] = ...) -> None: ...

class ExecAttemptRequest(_message.Message):
    __slots__ = ("attempt", "command", "timeout")
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    attempt: AttemptIdentity
    command: _containers.RepeatedScalarFieldContainer[str]
    timeout: _time_pb2.Duration
    def __init__(self, attempt: _Optional[_Union[AttemptIdentity, _Mapping]] = ..., command: _Optional[_Iterable[str]] = ..., timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ...) -> None: ...

class ExecAttemptResponse(_message.Message):
    __slots__ = ("exit_code", "stdout", "stderr", "error_message")
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    STDOUT_FIELD_NUMBER: _ClassVar[int]
    STDERR_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    exit_code: int
    stdout: str
    stderr: str
    error_message: str
    def __init__(self, exit_code: _Optional[int] = ..., stdout: _Optional[str] = ..., stderr: _Optional[str] = ..., error_message: _Optional[str] = ...) -> None: ...

class ProfileAttemptRequest(_message.Message):
    __slots__ = ("attempt", "profile", "duration")
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    attempt: AttemptIdentity
    profile: ProfileType
    duration: _time_pb2.Duration
    def __init__(self, attempt: _Optional[_Union[AttemptIdentity, _Mapping]] = ..., profile: _Optional[_Union[ProfileType, _Mapping]] = ..., duration: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ...) -> None: ...

class ProfileAttemptResponse(_message.Message):
    __slots__ = ("profile_data", "error_message")
    PROFILE_DATA_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    profile_data: bytes
    error_message: str
    def __init__(self, profile_data: _Optional[bytes] = ..., error_message: _Optional[str] = ...) -> None: ...
