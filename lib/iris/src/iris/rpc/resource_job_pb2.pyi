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
JOB_STATE_UNSPECIFIED: JobState
JOB_STATE_PENDING: JobState
JOB_STATE_BUILDING: JobState
JOB_STATE_RUNNING: JobState
JOB_STATE_SUCCEEDED: JobState
JOB_STATE_FAILED: JobState
JOB_STATE_KILLED: JobState
JOB_STATE_WORKER_FAILED: JobState
JOB_STATE_UNSCHEDULABLE: JobState
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

class CreateJob(_message.Message):
    __slots__ = ("spec", "bundle_blob")
    SPEC_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_BLOB_FIELD_NUMBER: _ClassVar[int]
    spec: JobSpec
    bundle_blob: bytes
    def __init__(self, spec: _Optional[_Union[JobSpec, _Mapping]] = ..., bundle_blob: _Optional[bytes] = ...) -> None: ...

class CreatedJob(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: _resource_identity_pb2.JobIdentity
    def __init__(self, job: _Optional[_Union[_resource_identity_pb2.JobIdentity, _Mapping]] = ...) -> None: ...

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
    parent: _resource_identity_pb2.ResourceKey
    job_id_prefix: str
    states: _containers.RepeatedScalarFieldContainer[JobState]
    backend_id: str
    execution_cluster_id: str
    page: _resource_pb2.PageRequest
    resource_id: str
    top_level_only: bool
    def __init__(self, owner_id: _Optional[str] = ..., parent: _Optional[_Union[_resource_identity_pb2.ResourceKey, _Mapping]] = ..., job_id_prefix: _Optional[str] = ..., states: _Optional[_Iterable[_Union[JobState, str]]] = ..., backend_id: _Optional[str] = ..., execution_cluster_id: _Optional[str] = ..., page: _Optional[_Union[_resource_pb2.PageRequest, _Mapping]] = ..., resource_id: _Optional[str] = ..., top_level_only: _Optional[bool] = ...) -> None: ...

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
    identity: _resource_identity_pb2.JobIdentity
    owner_id: str
    parent: _resource_identity_pb2.JobIdentity
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
    def __init__(self, identity: _Optional[_Union[_resource_identity_pb2.JobIdentity, _Mapping]] = ..., owner_id: _Optional[str] = ..., parent: _Optional[_Union[_resource_identity_pb2.JobIdentity, _Mapping]] = ..., state: _Optional[_Union[JobState, str]] = ..., execution_cluster_id: _Optional[str] = ..., backend_id: _Optional[str] = ..., num_tasks: _Optional[int] = ..., submitted_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., started_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., error_message: _Optional[str] = ..., pending_reason: _Optional[str] = ..., exit_code: _Optional[int] = ..., resources: _Optional[_Union[ResourceSpecProto, _Mapping]] = ...) -> None: ...

class JobDetail(_message.Message):
    __slots__ = ("summary", "spec")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    summary: JobSummary
    spec: JobSpec
    def __init__(self, summary: _Optional[_Union[JobSummary, _Mapping]] = ..., spec: _Optional[_Union[JobSpec, _Mapping]] = ...) -> None: ...

class CancelJobUpdate(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class JobUpdate(_message.Message):
    __slots__ = ("cancel",)
    CANCEL_FIELD_NUMBER: _ClassVar[int]
    cancel: CancelJobUpdate
    def __init__(self, cancel: _Optional[_Union[CancelJobUpdate, _Mapping]] = ...) -> None: ...
