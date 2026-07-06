from . import job_pb2 as _job_pb2
from . import query_pb2 as _query_pb2
from . import time_pb2 as _time_pb2
from . import vm_pb2 as _vm_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Controller(_message.Message):
    __slots__ = ()
    class JobSortField(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        JOB_SORT_FIELD_UNSPECIFIED: _ClassVar[Controller.JobSortField]
        JOB_SORT_FIELD_DATE: _ClassVar[Controller.JobSortField]
        JOB_SORT_FIELD_NAME: _ClassVar[Controller.JobSortField]
        JOB_SORT_FIELD_STATE: _ClassVar[Controller.JobSortField]
        JOB_SORT_FIELD_FAILURES: _ClassVar[Controller.JobSortField]
        JOB_SORT_FIELD_PREEMPTIONS: _ClassVar[Controller.JobSortField]
    JOB_SORT_FIELD_UNSPECIFIED: Controller.JobSortField
    JOB_SORT_FIELD_DATE: Controller.JobSortField
    JOB_SORT_FIELD_NAME: Controller.JobSortField
    JOB_SORT_FIELD_STATE: Controller.JobSortField
    JOB_SORT_FIELD_FAILURES: Controller.JobSortField
    JOB_SORT_FIELD_PREEMPTIONS: Controller.JobSortField
    class SortDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SORT_DIRECTION_UNSPECIFIED: _ClassVar[Controller.SortDirection]
        SORT_DIRECTION_ASC: _ClassVar[Controller.SortDirection]
        SORT_DIRECTION_DESC: _ClassVar[Controller.SortDirection]
    SORT_DIRECTION_UNSPECIFIED: Controller.SortDirection
    SORT_DIRECTION_ASC: Controller.SortDirection
    SORT_DIRECTION_DESC: Controller.SortDirection
    class JobQueryScope(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        JOB_QUERY_SCOPE_UNSPECIFIED: _ClassVar[Controller.JobQueryScope]
        JOB_QUERY_SCOPE_ALL: _ClassVar[Controller.JobQueryScope]
        JOB_QUERY_SCOPE_ROOTS: _ClassVar[Controller.JobQueryScope]
        JOB_QUERY_SCOPE_CHILDREN: _ClassVar[Controller.JobQueryScope]
    JOB_QUERY_SCOPE_UNSPECIFIED: Controller.JobQueryScope
    JOB_QUERY_SCOPE_ALL: Controller.JobQueryScope
    JOB_QUERY_SCOPE_ROOTS: Controller.JobQueryScope
    JOB_QUERY_SCOPE_CHILDREN: Controller.JobQueryScope
    class WorkerSortField(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        WORKER_SORT_FIELD_UNSPECIFIED: _ClassVar[Controller.WorkerSortField]
        WORKER_SORT_FIELD_WORKER_ID: _ClassVar[Controller.WorkerSortField]
        WORKER_SORT_FIELD_LAST_HEARTBEAT: _ClassVar[Controller.WorkerSortField]
        WORKER_SORT_FIELD_DEVICE_TYPE: _ClassVar[Controller.WorkerSortField]
    WORKER_SORT_FIELD_UNSPECIFIED: Controller.WorkerSortField
    WORKER_SORT_FIELD_WORKER_ID: Controller.WorkerSortField
    WORKER_SORT_FIELD_LAST_HEARTBEAT: Controller.WorkerSortField
    WORKER_SORT_FIELD_DEVICE_TYPE: Controller.WorkerSortField
    class EndpointAccess(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        ENDPOINT_ACCESS_PRIVATE: _ClassVar[Controller.EndpointAccess]
        ENDPOINT_ACCESS_LINK: _ClassVar[Controller.EndpointAccess]
    ENDPOINT_ACCESS_PRIVATE: Controller.EndpointAccess
    ENDPOINT_ACCESS_LINK: Controller.EndpointAccess
    class LaunchJobRequest(_message.Message):
        __slots__ = ("name", "entrypoint", "resources", "environment", "bundle_id", "bundle_blob", "scheduling_timeout", "ports", "max_task_failures", "max_retries_failure", "max_retries_preemption", "constraints", "coscheduling", "replicas", "timeout", "fail_if_exists", "preemption_policy", "existing_job_policy", "priority_band", "task_image", "submit_argv", "client_revision_date", "container_profile", "federation")
        NAME_FIELD_NUMBER: _ClassVar[int]
        ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
        RESOURCES_FIELD_NUMBER: _ClassVar[int]
        ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
        BUNDLE_ID_FIELD_NUMBER: _ClassVar[int]
        BUNDLE_BLOB_FIELD_NUMBER: _ClassVar[int]
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
        FEDERATION_FIELD_NUMBER: _ClassVar[int]
        name: str
        entrypoint: _job_pb2.RuntimeEntrypoint
        resources: _job_pb2.ResourceSpecProto
        environment: _job_pb2.EnvironmentConfig
        bundle_id: str
        bundle_blob: bytes
        scheduling_timeout: _time_pb2.Duration
        ports: _containers.RepeatedScalarFieldContainer[str]
        max_task_failures: int
        max_retries_failure: int
        max_retries_preemption: int
        constraints: _containers.RepeatedCompositeFieldContainer[_job_pb2.Constraint]
        coscheduling: _job_pb2.CoschedulingConfig
        replicas: int
        timeout: _time_pb2.Duration
        fail_if_exists: bool
        preemption_policy: _job_pb2.JobPreemptionPolicy
        existing_job_policy: _job_pb2.ExistingJobPolicy
        priority_band: _job_pb2.PriorityBand
        task_image: str
        submit_argv: _containers.RepeatedScalarFieldContainer[str]
        client_revision_date: str
        container_profile: _job_pb2.ContainerProfile
        federation: Controller.FederationHandoff
        def __init__(self, name: _Optional[str] = ..., entrypoint: _Optional[_Union[_job_pb2.RuntimeEntrypoint, _Mapping]] = ..., resources: _Optional[_Union[_job_pb2.ResourceSpecProto, _Mapping]] = ..., environment: _Optional[_Union[_job_pb2.EnvironmentConfig, _Mapping]] = ..., bundle_id: _Optional[str] = ..., bundle_blob: _Optional[bytes] = ..., scheduling_timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., ports: _Optional[_Iterable[str]] = ..., max_task_failures: _Optional[int] = ..., max_retries_failure: _Optional[int] = ..., max_retries_preemption: _Optional[int] = ..., constraints: _Optional[_Iterable[_Union[_job_pb2.Constraint, _Mapping]]] = ..., coscheduling: _Optional[_Union[_job_pb2.CoschedulingConfig, _Mapping]] = ..., replicas: _Optional[int] = ..., timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., fail_if_exists: _Optional[bool] = ..., preemption_policy: _Optional[_Union[_job_pb2.JobPreemptionPolicy, str]] = ..., existing_job_policy: _Optional[_Union[_job_pb2.ExistingJobPolicy, str]] = ..., priority_band: _Optional[_Union[_job_pb2.PriorityBand, str]] = ..., task_image: _Optional[str] = ..., submit_argv: _Optional[_Iterable[str]] = ..., client_revision_date: _Optional[str] = ..., container_profile: _Optional[_Union[_job_pb2.ContainerProfile, str]] = ..., federation: _Optional[_Union[Controller.FederationHandoff, _Mapping]] = ...) -> None: ...
    class FederationHandoff(_message.Message):
        __slots__ = ("requester_id", "owner_principal")
        REQUESTER_ID_FIELD_NUMBER: _ClassVar[int]
        OWNER_PRINCIPAL_FIELD_NUMBER: _ClassVar[int]
        requester_id: str
        owner_principal: str
        def __init__(self, requester_id: _Optional[str] = ..., owner_principal: _Optional[str] = ...) -> None: ...
    class LaunchJobResponse(_message.Message):
        __slots__ = ("job_id",)
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        def __init__(self, job_id: _Optional[str] = ...) -> None: ...
    class GetJobStatusRequest(_message.Message):
        __slots__ = ("job_id",)
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        def __init__(self, job_id: _Optional[str] = ...) -> None: ...
    class GetJobStatusResponse(_message.Message):
        __slots__ = ("job", "request")
        JOB_FIELD_NUMBER: _ClassVar[int]
        REQUEST_FIELD_NUMBER: _ClassVar[int]
        job: _job_pb2.JobStatus
        request: Controller.LaunchJobRequest
        def __init__(self, job: _Optional[_Union[_job_pb2.JobStatus, _Mapping]] = ..., request: _Optional[_Union[Controller.LaunchJobRequest, _Mapping]] = ...) -> None: ...
    class GetJobStateRequest(_message.Message):
        __slots__ = ("job_ids",)
        JOB_IDS_FIELD_NUMBER: _ClassVar[int]
        job_ids: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, job_ids: _Optional[_Iterable[str]] = ...) -> None: ...
    class GetJobStateResponse(_message.Message):
        __slots__ = ("states",)
        class StatesEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: _job_pb2.JobState
            def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_job_pb2.JobState, str]] = ...) -> None: ...
        STATES_FIELD_NUMBER: _ClassVar[int]
        states: _containers.ScalarMap[str, _job_pb2.JobState]
        def __init__(self, states: _Optional[_Mapping[str, _job_pb2.JobState]] = ...) -> None: ...
    class TerminateJobRequest(_message.Message):
        __slots__ = ("job_id",)
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        def __init__(self, job_id: _Optional[str] = ...) -> None: ...
    class JobQuery(_message.Message):
        __slots__ = ("scope", "parent_job_id", "name_filter", "state_filter", "sort_field", "sort_direction", "offset", "limit", "job_id_prefix", "backend_id", "cluster")
        SCOPE_FIELD_NUMBER: _ClassVar[int]
        PARENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
        NAME_FILTER_FIELD_NUMBER: _ClassVar[int]
        STATE_FILTER_FIELD_NUMBER: _ClassVar[int]
        SORT_FIELD_FIELD_NUMBER: _ClassVar[int]
        SORT_DIRECTION_FIELD_NUMBER: _ClassVar[int]
        OFFSET_FIELD_NUMBER: _ClassVar[int]
        LIMIT_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_PREFIX_FIELD_NUMBER: _ClassVar[int]
        BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
        CLUSTER_FIELD_NUMBER: _ClassVar[int]
        scope: Controller.JobQueryScope
        parent_job_id: str
        name_filter: str
        state_filter: str
        sort_field: Controller.JobSortField
        sort_direction: Controller.SortDirection
        offset: int
        limit: int
        job_id_prefix: str
        backend_id: str
        cluster: str
        def __init__(self, scope: _Optional[_Union[Controller.JobQueryScope, str]] = ..., parent_job_id: _Optional[str] = ..., name_filter: _Optional[str] = ..., state_filter: _Optional[str] = ..., sort_field: _Optional[_Union[Controller.JobSortField, str]] = ..., sort_direction: _Optional[_Union[Controller.SortDirection, str]] = ..., offset: _Optional[int] = ..., limit: _Optional[int] = ..., job_id_prefix: _Optional[str] = ..., backend_id: _Optional[str] = ..., cluster: _Optional[str] = ...) -> None: ...
    class ListJobsRequest(_message.Message):
        __slots__ = ("query",)
        QUERY_FIELD_NUMBER: _ClassVar[int]
        query: Controller.JobQuery
        def __init__(self, query: _Optional[_Union[Controller.JobQuery, _Mapping]] = ...) -> None: ...
    class ListJobsResponse(_message.Message):
        __slots__ = ("jobs", "total_count", "has_more")
        JOBS_FIELD_NUMBER: _ClassVar[int]
        TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
        HAS_MORE_FIELD_NUMBER: _ClassVar[int]
        jobs: _containers.RepeatedCompositeFieldContainer[_job_pb2.JobStatus]
        total_count: int
        has_more: bool
        def __init__(self, jobs: _Optional[_Iterable[_Union[_job_pb2.JobStatus, _Mapping]]] = ..., total_count: _Optional[int] = ..., has_more: _Optional[bool] = ...) -> None: ...
    class GetTaskStatusRequest(_message.Message):
        __slots__ = ("task_id",)
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        def __init__(self, task_id: _Optional[str] = ...) -> None: ...
    class GetTaskStatusResponse(_message.Message):
        __slots__ = ("task", "job_resources", "root_cause_highlights")
        TASK_FIELD_NUMBER: _ClassVar[int]
        JOB_RESOURCES_FIELD_NUMBER: _ClassVar[int]
        ROOT_CAUSE_HIGHLIGHTS_FIELD_NUMBER: _ClassVar[int]
        task: _job_pb2.TaskStatus
        job_resources: _job_pb2.ResourceSpecProto
        root_cause_highlights: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, task: _Optional[_Union[_job_pb2.TaskStatus, _Mapping]] = ..., job_resources: _Optional[_Union[_job_pb2.ResourceSpecProto, _Mapping]] = ..., root_cause_highlights: _Optional[_Iterable[str]] = ...) -> None: ...
    class ListTasksRequest(_message.Message):
        __slots__ = ("job_id",)
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        def __init__(self, job_id: _Optional[str] = ...) -> None: ...
    class ListTasksResponse(_message.Message):
        __slots__ = ("tasks",)
        TASKS_FIELD_NUMBER: _ClassVar[int]
        tasks: _containers.RepeatedCompositeFieldContainer[_job_pb2.TaskStatus]
        def __init__(self, tasks: _Optional[_Iterable[_Union[_job_pb2.TaskStatus, _Mapping]]] = ...) -> None: ...
    class KickTasksRequest(_message.Message):
        __slots__ = ("targets", "desired_state", "reason")
        TARGETS_FIELD_NUMBER: _ClassVar[int]
        DESIRED_STATE_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        targets: _containers.RepeatedScalarFieldContainer[str]
        desired_state: _job_pb2.TaskState
        reason: str
        def __init__(self, targets: _Optional[_Iterable[str]] = ..., desired_state: _Optional[_Union[_job_pb2.TaskState, str]] = ..., reason: _Optional[str] = ...) -> None: ...
    class KickResult(_message.Message):
        __slots__ = ("target", "task_id", "queued", "detail")
        TARGET_FIELD_NUMBER: _ClassVar[int]
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        QUEUED_FIELD_NUMBER: _ClassVar[int]
        DETAIL_FIELD_NUMBER: _ClassVar[int]
        target: str
        task_id: str
        queued: bool
        detail: str
        def __init__(self, target: _Optional[str] = ..., task_id: _Optional[str] = ..., queued: _Optional[bool] = ..., detail: _Optional[str] = ...) -> None: ...
    class KickTasksResponse(_message.Message):
        __slots__ = ("results",)
        RESULTS_FIELD_NUMBER: _ClassVar[int]
        results: _containers.RepeatedCompositeFieldContainer[Controller.KickResult]
        def __init__(self, results: _Optional[_Iterable[_Union[Controller.KickResult, _Mapping]]] = ...) -> None: ...
    class ExecInContainerRequest(_message.Message):
        __slots__ = ("task_id", "command", "timeout_seconds")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        COMMAND_FIELD_NUMBER: _ClassVar[int]
        TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        command: _containers.RepeatedScalarFieldContainer[str]
        timeout_seconds: int
        def __init__(self, task_id: _Optional[str] = ..., command: _Optional[_Iterable[str]] = ..., timeout_seconds: _Optional[int] = ...) -> None: ...
    class ExecInContainerResponse(_message.Message):
        __slots__ = ("exit_code", "stdout", "stderr", "error")
        EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
        STDOUT_FIELD_NUMBER: _ClassVar[int]
        STDERR_FIELD_NUMBER: _ClassVar[int]
        ERROR_FIELD_NUMBER: _ClassVar[int]
        exit_code: int
        stdout: str
        stderr: str
        error: str
        def __init__(self, exit_code: _Optional[int] = ..., stdout: _Optional[str] = ..., stderr: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...
    class WorkerInfo(_message.Message):
        __slots__ = ("worker_id", "address", "metadata", "registered_at")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        REGISTERED_AT_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        address: str
        metadata: _job_pb2.WorkerMetadata
        registered_at: _time_pb2.Timestamp
        def __init__(self, worker_id: _Optional[str] = ..., address: _Optional[str] = ..., metadata: _Optional[_Union[_job_pb2.WorkerMetadata, _Mapping]] = ..., registered_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...
    class WorkerHealthStatus(_message.Message):
        __slots__ = ("worker_id", "healthy", "consecutive_failures", "last_heartbeat", "running_job_ids", "address", "metadata", "status_message", "backend_id", "scale_group")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        HEALTHY_FIELD_NUMBER: _ClassVar[int]
        CONSECUTIVE_FAILURES_FIELD_NUMBER: _ClassVar[int]
        LAST_HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
        RUNNING_JOB_IDS_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
        BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
        SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        healthy: bool
        consecutive_failures: int
        last_heartbeat: _time_pb2.Timestamp
        running_job_ids: _containers.RepeatedScalarFieldContainer[str]
        address: str
        metadata: _job_pb2.WorkerMetadata
        status_message: str
        backend_id: str
        scale_group: str
        def __init__(self, worker_id: _Optional[str] = ..., healthy: _Optional[bool] = ..., consecutive_failures: _Optional[int] = ..., last_heartbeat: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., running_job_ids: _Optional[_Iterable[str]] = ..., address: _Optional[str] = ..., metadata: _Optional[_Union[_job_pb2.WorkerMetadata, _Mapping]] = ..., status_message: _Optional[str] = ..., backend_id: _Optional[str] = ..., scale_group: _Optional[str] = ...) -> None: ...
    class WorkerQuery(_message.Message):
        __slots__ = ("contains", "sort_field", "sort_direction", "offset", "limit", "backend_id")
        CONTAINS_FIELD_NUMBER: _ClassVar[int]
        SORT_FIELD_FIELD_NUMBER: _ClassVar[int]
        SORT_DIRECTION_FIELD_NUMBER: _ClassVar[int]
        OFFSET_FIELD_NUMBER: _ClassVar[int]
        LIMIT_FIELD_NUMBER: _ClassVar[int]
        BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
        contains: str
        sort_field: Controller.WorkerSortField
        sort_direction: Controller.SortDirection
        offset: int
        limit: int
        backend_id: str
        def __init__(self, contains: _Optional[str] = ..., sort_field: _Optional[_Union[Controller.WorkerSortField, str]] = ..., sort_direction: _Optional[_Union[Controller.SortDirection, str]] = ..., offset: _Optional[int] = ..., limit: _Optional[int] = ..., backend_id: _Optional[str] = ...) -> None: ...
    class ListWorkersRequest(_message.Message):
        __slots__ = ("query",)
        QUERY_FIELD_NUMBER: _ClassVar[int]
        query: Controller.WorkerQuery
        def __init__(self, query: _Optional[_Union[Controller.WorkerQuery, _Mapping]] = ...) -> None: ...
    class ListWorkersResponse(_message.Message):
        __slots__ = ("workers", "total_count", "has_more")
        WORKERS_FIELD_NUMBER: _ClassVar[int]
        TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
        HAS_MORE_FIELD_NUMBER: _ClassVar[int]
        workers: _containers.RepeatedCompositeFieldContainer[Controller.WorkerHealthStatus]
        total_count: int
        has_more: bool
        def __init__(self, workers: _Optional[_Iterable[_Union[Controller.WorkerHealthStatus, _Mapping]]] = ..., total_count: _Optional[int] = ..., has_more: _Optional[bool] = ...) -> None: ...
    class RegisterRequest(_message.Message):
        __slots__ = ("address", "metadata", "worker_id", "slice_id", "scale_group")
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        SLICE_ID_FIELD_NUMBER: _ClassVar[int]
        SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
        address: str
        metadata: _job_pb2.WorkerMetadata
        worker_id: str
        slice_id: str
        scale_group: str
        def __init__(self, address: _Optional[str] = ..., metadata: _Optional[_Union[_job_pb2.WorkerMetadata, _Mapping]] = ..., worker_id: _Optional[str] = ..., slice_id: _Optional[str] = ..., scale_group: _Optional[str] = ...) -> None: ...
    class RegisterResponse(_message.Message):
        __slots__ = ("worker_id", "accepted")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        ACCEPTED_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        accepted: bool
        def __init__(self, worker_id: _Optional[str] = ..., accepted: _Optional[bool] = ...) -> None: ...
    class Endpoint(_message.Message):
        __slots__ = ("endpoint_id", "name", "address", "task_id", "metadata", "access")
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
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        ACCESS_FIELD_NUMBER: _ClassVar[int]
        endpoint_id: str
        name: str
        address: str
        task_id: str
        metadata: _containers.ScalarMap[str, str]
        access: Controller.EndpointAccess
        def __init__(self, endpoint_id: _Optional[str] = ..., name: _Optional[str] = ..., address: _Optional[str] = ..., task_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., access: _Optional[_Union[Controller.EndpointAccess, str]] = ...) -> None: ...
    class RegisterEndpointRequest(_message.Message):
        __slots__ = ("name", "address", "task_id", "metadata", "attempt_id", "endpoint_id", "lease_duration", "access")
        class MetadataEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
        NAME_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
        ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
        LEASE_DURATION_FIELD_NUMBER: _ClassVar[int]
        ACCESS_FIELD_NUMBER: _ClassVar[int]
        name: str
        address: str
        task_id: str
        metadata: _containers.ScalarMap[str, str]
        attempt_id: int
        endpoint_id: str
        lease_duration: _time_pb2.Duration
        access: Controller.EndpointAccess
        def __init__(self, name: _Optional[str] = ..., address: _Optional[str] = ..., task_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., attempt_id: _Optional[int] = ..., endpoint_id: _Optional[str] = ..., lease_duration: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., access: _Optional[_Union[Controller.EndpointAccess, str]] = ...) -> None: ...
    class MintEndpointTokenRequest(_message.Message):
        __slots__ = ("endpoint_name", "ttl")
        ENDPOINT_NAME_FIELD_NUMBER: _ClassVar[int]
        TTL_FIELD_NUMBER: _ClassVar[int]
        endpoint_name: str
        ttl: _time_pb2.Duration
        def __init__(self, endpoint_name: _Optional[str] = ..., ttl: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ...) -> None: ...
    class MintEndpointTokenResponse(_message.Message):
        __slots__ = ("token", "expires_at")
        TOKEN_FIELD_NUMBER: _ClassVar[int]
        EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
        token: str
        expires_at: _time_pb2.Timestamp
        def __init__(self, token: _Optional[str] = ..., expires_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...
    class RegisterEndpointResponse(_message.Message):
        __slots__ = ("endpoint_id", "lease_duration")
        ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
        LEASE_DURATION_FIELD_NUMBER: _ClassVar[int]
        endpoint_id: str
        lease_duration: _time_pb2.Duration
        def __init__(self, endpoint_id: _Optional[str] = ..., lease_duration: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ...) -> None: ...
    class UnregisterEndpointRequest(_message.Message):
        __slots__ = ("endpoint_id",)
        ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
        endpoint_id: str
        def __init__(self, endpoint_id: _Optional[str] = ...) -> None: ...
    class ListEndpointsRequest(_message.Message):
        __slots__ = ("prefix", "exact", "task_ids")
        PREFIX_FIELD_NUMBER: _ClassVar[int]
        EXACT_FIELD_NUMBER: _ClassVar[int]
        TASK_IDS_FIELD_NUMBER: _ClassVar[int]
        prefix: str
        exact: bool
        task_ids: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, prefix: _Optional[str] = ..., exact: _Optional[bool] = ..., task_ids: _Optional[_Iterable[str]] = ...) -> None: ...
    class ListEndpointsResponse(_message.Message):
        __slots__ = ("endpoints",)
        ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
        endpoints: _containers.RepeatedCompositeFieldContainer[Controller.Endpoint]
        def __init__(self, endpoints: _Optional[_Iterable[_Union[Controller.Endpoint, _Mapping]]] = ...) -> None: ...
    class GetAutoscalerStatusRequest(_message.Message):
        __slots__ = ("backend_id",)
        BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
        backend_id: str
        def __init__(self, backend_id: _Optional[str] = ...) -> None: ...
    class GetAutoscalerStatusResponse(_message.Message):
        __slots__ = ("status",)
        STATUS_FIELD_NUMBER: _ClassVar[int]
        status: _vm_pb2.AutoscalerStatus
        def __init__(self, status: _Optional[_Union[_vm_pb2.AutoscalerStatus, _Mapping]] = ...) -> None: ...
    class BeginCheckpointRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class BeginCheckpointResponse(_message.Message):
        __slots__ = ("checkpoint_path", "created_at", "job_count", "task_count", "worker_count")
        CHECKPOINT_PATH_FIELD_NUMBER: _ClassVar[int]
        CREATED_AT_FIELD_NUMBER: _ClassVar[int]
        JOB_COUNT_FIELD_NUMBER: _ClassVar[int]
        TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
        WORKER_COUNT_FIELD_NUMBER: _ClassVar[int]
        checkpoint_path: str
        created_at: _time_pb2.Timestamp
        job_count: int
        task_count: int
        worker_count: int
        def __init__(self, checkpoint_path: _Optional[str] = ..., created_at: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., job_count: _Optional[int] = ..., task_count: _Optional[int] = ..., worker_count: _Optional[int] = ...) -> None: ...
    class UserSummary(_message.Message):
        __slots__ = ("user", "task_state_counts", "job_state_counts", "role")
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
        USER_FIELD_NUMBER: _ClassVar[int]
        TASK_STATE_COUNTS_FIELD_NUMBER: _ClassVar[int]
        JOB_STATE_COUNTS_FIELD_NUMBER: _ClassVar[int]
        ROLE_FIELD_NUMBER: _ClassVar[int]
        user: str
        task_state_counts: _containers.ScalarMap[str, int]
        job_state_counts: _containers.ScalarMap[str, int]
        role: str
        def __init__(self, user: _Optional[str] = ..., task_state_counts: _Optional[_Mapping[str, int]] = ..., job_state_counts: _Optional[_Mapping[str, int]] = ..., role: _Optional[str] = ...) -> None: ...
    class ListUsersRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class ListUsersResponse(_message.Message):
        __slots__ = ("users",)
        USERS_FIELD_NUMBER: _ClassVar[int]
        users: _containers.RepeatedCompositeFieldContainer[Controller.UserSummary]
        def __init__(self, users: _Optional[_Iterable[_Union[Controller.UserSummary, _Mapping]]] = ...) -> None: ...
    class GetWorkerStatusRequest(_message.Message):
        __slots__ = ("id",)
        ID_FIELD_NUMBER: _ClassVar[int]
        id: str
        def __init__(self, id: _Optional[str] = ...) -> None: ...
    class GetWorkerStatusResponse(_message.Message):
        __slots__ = ("vm", "scale_group", "worker", "bootstrap_logs", "recent_attempts")
        VM_FIELD_NUMBER: _ClassVar[int]
        SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
        WORKER_FIELD_NUMBER: _ClassVar[int]
        BOOTSTRAP_LOGS_FIELD_NUMBER: _ClassVar[int]
        RECENT_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
        vm: _vm_pb2.VmInfo
        scale_group: str
        worker: Controller.WorkerHealthStatus
        bootstrap_logs: str
        recent_attempts: _containers.RepeatedCompositeFieldContainer[Controller.WorkerTaskAttempt]
        def __init__(self, vm: _Optional[_Union[_vm_pb2.VmInfo, _Mapping]] = ..., scale_group: _Optional[str] = ..., worker: _Optional[_Union[Controller.WorkerHealthStatus, _Mapping]] = ..., bootstrap_logs: _Optional[str] = ..., recent_attempts: _Optional[_Iterable[_Union[Controller.WorkerTaskAttempt, _Mapping]]] = ...) -> None: ...
    class WorkerTaskAttempt(_message.Message):
        __slots__ = ("task_id", "attempt", "resources")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        ATTEMPT_FIELD_NUMBER: _ClassVar[int]
        RESOURCES_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        attempt: _job_pb2.TaskAttempt
        resources: _job_pb2.ResourceSpecProto
        def __init__(self, task_id: _Optional[str] = ..., attempt: _Optional[_Union[_job_pb2.TaskAttempt, _Mapping]] = ..., resources: _Optional[_Union[_job_pb2.ResourceSpecProto, _Mapping]] = ...) -> None: ...
    class GetKubernetesClusterStatusRequest(_message.Message):
        __slots__ = ("backend_id",)
        BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
        backend_id: str
        def __init__(self, backend_id: _Optional[str] = ...) -> None: ...
    class KubernetesPodStatus(_message.Message):
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
    class NodePoolStatus(_message.Message):
        __slots__ = ("name", "instance_type", "scale_group", "target_nodes", "current_nodes", "queued_nodes", "in_progress_nodes", "autoscaling", "min_nodes", "max_nodes", "capacity", "quota")
        NAME_FIELD_NUMBER: _ClassVar[int]
        INSTANCE_TYPE_FIELD_NUMBER: _ClassVar[int]
        SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
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
        scale_group: str
        target_nodes: int
        current_nodes: int
        queued_nodes: int
        in_progress_nodes: int
        autoscaling: bool
        min_nodes: int
        max_nodes: int
        capacity: str
        quota: str
        def __init__(self, name: _Optional[str] = ..., instance_type: _Optional[str] = ..., scale_group: _Optional[str] = ..., target_nodes: _Optional[int] = ..., current_nodes: _Optional[int] = ..., queued_nodes: _Optional[int] = ..., in_progress_nodes: _Optional[int] = ..., autoscaling: _Optional[bool] = ..., min_nodes: _Optional[int] = ..., max_nodes: _Optional[int] = ..., capacity: _Optional[str] = ..., quota: _Optional[str] = ...) -> None: ...
    class GetKubernetesClusterStatusResponse(_message.Message):
        __slots__ = ("namespace", "total_nodes", "schedulable_nodes", "allocatable_cpu", "allocatable_memory", "pod_statuses", "provider_version", "node_pools")
        NAMESPACE_FIELD_NUMBER: _ClassVar[int]
        TOTAL_NODES_FIELD_NUMBER: _ClassVar[int]
        SCHEDULABLE_NODES_FIELD_NUMBER: _ClassVar[int]
        ALLOCATABLE_CPU_FIELD_NUMBER: _ClassVar[int]
        ALLOCATABLE_MEMORY_FIELD_NUMBER: _ClassVar[int]
        POD_STATUSES_FIELD_NUMBER: _ClassVar[int]
        PROVIDER_VERSION_FIELD_NUMBER: _ClassVar[int]
        NODE_POOLS_FIELD_NUMBER: _ClassVar[int]
        namespace: str
        total_nodes: int
        schedulable_nodes: int
        allocatable_cpu: str
        allocatable_memory: str
        pod_statuses: _containers.RepeatedCompositeFieldContainer[Controller.KubernetesPodStatus]
        provider_version: str
        node_pools: _containers.RepeatedCompositeFieldContainer[Controller.NodePoolStatus]
        def __init__(self, namespace: _Optional[str] = ..., total_nodes: _Optional[int] = ..., schedulable_nodes: _Optional[int] = ..., allocatable_cpu: _Optional[str] = ..., allocatable_memory: _Optional[str] = ..., pod_statuses: _Optional[_Iterable[_Union[Controller.KubernetesPodStatus, _Mapping]]] = ..., provider_version: _Optional[str] = ..., node_pools: _Optional[_Iterable[_Union[Controller.NodePoolStatus, _Mapping]]] = ...) -> None: ...
    class SetUserBudgetRequest(_message.Message):
        __slots__ = ("user_id", "budget_limit", "max_band")
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        BUDGET_LIMIT_FIELD_NUMBER: _ClassVar[int]
        MAX_BAND_FIELD_NUMBER: _ClassVar[int]
        user_id: str
        budget_limit: int
        max_band: _job_pb2.PriorityBand
        def __init__(self, user_id: _Optional[str] = ..., budget_limit: _Optional[int] = ..., max_band: _Optional[_Union[_job_pb2.PriorityBand, str]] = ...) -> None: ...
    class SetUserBudgetResponse(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class GetUserBudgetRequest(_message.Message):
        __slots__ = ("user_id",)
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        user_id: str
        def __init__(self, user_id: _Optional[str] = ...) -> None: ...
    class GetUserBudgetResponse(_message.Message):
        __slots__ = ("user_id", "budget_limit", "budget_spent", "max_band")
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        BUDGET_LIMIT_FIELD_NUMBER: _ClassVar[int]
        BUDGET_SPENT_FIELD_NUMBER: _ClassVar[int]
        MAX_BAND_FIELD_NUMBER: _ClassVar[int]
        user_id: str
        budget_limit: int
        budget_spent: int
        max_band: _job_pb2.PriorityBand
        def __init__(self, user_id: _Optional[str] = ..., budget_limit: _Optional[int] = ..., budget_spent: _Optional[int] = ..., max_band: _Optional[_Union[_job_pb2.PriorityBand, str]] = ...) -> None: ...
    class ListUserBudgetsRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class ListUserBudgetsResponse(_message.Message):
        __slots__ = ("users",)
        USERS_FIELD_NUMBER: _ClassVar[int]
        users: _containers.RepeatedCompositeFieldContainer[Controller.GetUserBudgetResponse]
        def __init__(self, users: _Optional[_Iterable[_Union[Controller.GetUserBudgetResponse, _Mapping]]] = ...) -> None: ...
    class GetSchedulerStateRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class PendingTaskBucket(_message.Message):
        __slots__ = ("band", "user_id", "job_id", "count", "backend_id")
        BAND_FIELD_NUMBER: _ClassVar[int]
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        COUNT_FIELD_NUMBER: _ClassVar[int]
        BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
        band: _job_pb2.PriorityBand
        user_id: str
        job_id: str
        count: int
        backend_id: str
        def __init__(self, band: _Optional[_Union[_job_pb2.PriorityBand, str]] = ..., user_id: _Optional[str] = ..., job_id: _Optional[str] = ..., count: _Optional[int] = ..., backend_id: _Optional[str] = ...) -> None: ...
    class RunningTaskBucket(_message.Message):
        __slots__ = ("band", "user_id", "worker_id", "job_id", "count", "backend_id")
        BAND_FIELD_NUMBER: _ClassVar[int]
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        COUNT_FIELD_NUMBER: _ClassVar[int]
        BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
        band: _job_pb2.PriorityBand
        user_id: str
        worker_id: str
        job_id: str
        count: int
        backend_id: str
        def __init__(self, band: _Optional[_Union[_job_pb2.PriorityBand, str]] = ..., user_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., job_id: _Optional[str] = ..., count: _Optional[int] = ..., backend_id: _Optional[str] = ...) -> None: ...
    class SchedulerUserBudget(_message.Message):
        __slots__ = ("user_id", "budget_limit", "budget_spent", "max_band", "effective_band", "utilization_percent")
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        BUDGET_LIMIT_FIELD_NUMBER: _ClassVar[int]
        BUDGET_SPENT_FIELD_NUMBER: _ClassVar[int]
        MAX_BAND_FIELD_NUMBER: _ClassVar[int]
        EFFECTIVE_BAND_FIELD_NUMBER: _ClassVar[int]
        UTILIZATION_PERCENT_FIELD_NUMBER: _ClassVar[int]
        user_id: str
        budget_limit: int
        budget_spent: int
        max_band: _job_pb2.PriorityBand
        effective_band: _job_pb2.PriorityBand
        utilization_percent: float
        def __init__(self, user_id: _Optional[str] = ..., budget_limit: _Optional[int] = ..., budget_spent: _Optional[int] = ..., max_band: _Optional[_Union[_job_pb2.PriorityBand, str]] = ..., effective_band: _Optional[_Union[_job_pb2.PriorityBand, str]] = ..., utilization_percent: _Optional[float] = ...) -> None: ...
    class GetSchedulerStateResponse(_message.Message):
        __slots__ = ("user_budgets", "total_pending", "total_running", "pending_buckets", "running_buckets")
        USER_BUDGETS_FIELD_NUMBER: _ClassVar[int]
        TOTAL_PENDING_FIELD_NUMBER: _ClassVar[int]
        TOTAL_RUNNING_FIELD_NUMBER: _ClassVar[int]
        PENDING_BUCKETS_FIELD_NUMBER: _ClassVar[int]
        RUNNING_BUCKETS_FIELD_NUMBER: _ClassVar[int]
        user_budgets: _containers.RepeatedCompositeFieldContainer[Controller.SchedulerUserBudget]
        total_pending: int
        total_running: int
        pending_buckets: _containers.RepeatedCompositeFieldContainer[Controller.PendingTaskBucket]
        running_buckets: _containers.RepeatedCompositeFieldContainer[Controller.RunningTaskBucket]
        def __init__(self, user_budgets: _Optional[_Iterable[_Union[Controller.SchedulerUserBudget, _Mapping]]] = ..., total_pending: _Optional[int] = ..., total_running: _Optional[int] = ..., pending_buckets: _Optional[_Iterable[_Union[Controller.PendingTaskBucket, _Mapping]]] = ..., running_buckets: _Optional[_Iterable[_Union[Controller.RunningTaskBucket, _Mapping]]] = ...) -> None: ...
    class ListBackendsRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class WorkerFleetDetail(_message.Message):
        __slots__ = ("autoscaler", "healthy_worker_count", "total_worker_count")
        AUTOSCALER_FIELD_NUMBER: _ClassVar[int]
        HEALTHY_WORKER_COUNT_FIELD_NUMBER: _ClassVar[int]
        TOTAL_WORKER_COUNT_FIELD_NUMBER: _ClassVar[int]
        autoscaler: _vm_pb2.AutoscalerStatus
        healthy_worker_count: int
        total_worker_count: int
        def __init__(self, autoscaler: _Optional[_Union[_vm_pb2.AutoscalerStatus, _Mapping]] = ..., healthy_worker_count: _Optional[int] = ..., total_worker_count: _Optional[int] = ...) -> None: ...
    class BackendStatus(_message.Message):
        __slots__ = ("kubernetes", "worker")
        KUBERNETES_FIELD_NUMBER: _ClassVar[int]
        WORKER_FIELD_NUMBER: _ClassVar[int]
        kubernetes: Controller.GetKubernetesClusterStatusResponse
        worker: Controller.WorkerFleetDetail
        def __init__(self, kubernetes: _Optional[_Union[Controller.GetKubernetesClusterStatusResponse, _Mapping]] = ..., worker: _Optional[_Union[Controller.WorkerFleetDetail, _Mapping]] = ...) -> None: ...
    class BackendSummary(_message.Message):
        __slots__ = ("backend_id", "name", "kind", "capabilities", "advertised_attributes", "restricted", "allowed_user_count", "scale_groups", "worker_count", "pending_task_count", "running_task_count", "has_autoscaler", "capacity_health", "detail")
        class AdvertisedAttributesEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: StringList
            def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[StringList, _Mapping]] = ...) -> None: ...
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
        RESTRICTED_FIELD_NUMBER: _ClassVar[int]
        ALLOWED_USER_COUNT_FIELD_NUMBER: _ClassVar[int]
        SCALE_GROUPS_FIELD_NUMBER: _ClassVar[int]
        WORKER_COUNT_FIELD_NUMBER: _ClassVar[int]
        PENDING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
        RUNNING_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
        HAS_AUTOSCALER_FIELD_NUMBER: _ClassVar[int]
        CAPACITY_HEALTH_FIELD_NUMBER: _ClassVar[int]
        DETAIL_FIELD_NUMBER: _ClassVar[int]
        backend_id: str
        name: str
        kind: str
        capabilities: _containers.RepeatedScalarFieldContainer[str]
        advertised_attributes: _containers.MessageMap[str, StringList]
        restricted: bool
        allowed_user_count: int
        scale_groups: _containers.RepeatedScalarFieldContainer[str]
        worker_count: int
        pending_task_count: int
        running_task_count: int
        has_autoscaler: bool
        capacity_health: _containers.ScalarMap[str, int]
        detail: Controller.BackendStatus
        def __init__(self, backend_id: _Optional[str] = ..., name: _Optional[str] = ..., kind: _Optional[str] = ..., capabilities: _Optional[_Iterable[str]] = ..., advertised_attributes: _Optional[_Mapping[str, StringList]] = ..., restricted: _Optional[bool] = ..., allowed_user_count: _Optional[int] = ..., scale_groups: _Optional[_Iterable[str]] = ..., worker_count: _Optional[int] = ..., pending_task_count: _Optional[int] = ..., running_task_count: _Optional[int] = ..., has_autoscaler: _Optional[bool] = ..., capacity_health: _Optional[_Mapping[str, int]] = ..., detail: _Optional[_Union[Controller.BackendStatus, _Mapping]] = ...) -> None: ...
    class UnroutableJob(_message.Message):
        __slots__ = ("job_id", "reason")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        reason: str
        def __init__(self, job_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...
    class ListBackendsResponse(_message.Message):
        __slots__ = ("backends", "unroutable_job_count", "unroutable_sample")
        BACKENDS_FIELD_NUMBER: _ClassVar[int]
        UNROUTABLE_JOB_COUNT_FIELD_NUMBER: _ClassVar[int]
        UNROUTABLE_SAMPLE_FIELD_NUMBER: _ClassVar[int]
        backends: _containers.RepeatedCompositeFieldContainer[Controller.BackendSummary]
        unroutable_job_count: int
        unroutable_sample: _containers.RepeatedCompositeFieldContainer[Controller.UnroutableJob]
        def __init__(self, backends: _Optional[_Iterable[_Union[Controller.BackendSummary, _Mapping]]] = ..., unroutable_job_count: _Optional[int] = ..., unroutable_sample: _Optional[_Iterable[_Union[Controller.UnroutableJob, _Mapping]]] = ...) -> None: ...
    class ListPeersRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class PeerSummary(_message.Message):
        __slots__ = ("peer_id", "controller_address", "dashboard_url", "reachable", "last_contact_ms", "active_federated_jobs", "aggregate_spend_micros", "backends")
        PEER_ID_FIELD_NUMBER: _ClassVar[int]
        CONTROLLER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
        DASHBOARD_URL_FIELD_NUMBER: _ClassVar[int]
        REACHABLE_FIELD_NUMBER: _ClassVar[int]
        LAST_CONTACT_MS_FIELD_NUMBER: _ClassVar[int]
        ACTIVE_FEDERATED_JOBS_FIELD_NUMBER: _ClassVar[int]
        AGGREGATE_SPEND_MICROS_FIELD_NUMBER: _ClassVar[int]
        BACKENDS_FIELD_NUMBER: _ClassVar[int]
        peer_id: str
        controller_address: str
        dashboard_url: str
        reachable: bool
        last_contact_ms: int
        active_federated_jobs: int
        aggregate_spend_micros: int
        backends: _containers.RepeatedCompositeFieldContainer[Controller.BackendSummary]
        def __init__(self, peer_id: _Optional[str] = ..., controller_address: _Optional[str] = ..., dashboard_url: _Optional[str] = ..., reachable: _Optional[bool] = ..., last_contact_ms: _Optional[int] = ..., active_federated_jobs: _Optional[int] = ..., aggregate_spend_micros: _Optional[int] = ..., backends: _Optional[_Iterable[_Union[Controller.BackendSummary, _Mapping]]] = ...) -> None: ...
    class ListPeersResponse(_message.Message):
        __slots__ = ("peers",)
        PEERS_FIELD_NUMBER: _ClassVar[int]
        peers: _containers.RepeatedCompositeFieldContainer[Controller.PeerSummary]
        def __init__(self, peers: _Optional[_Iterable[_Union[Controller.PeerSummary, _Mapping]]] = ...) -> None: ...
    class FederationSyncRequest(_message.Message):
        __slots__ = ("requester_id", "cursor")
        REQUESTER_ID_FIELD_NUMBER: _ClassVar[int]
        CURSOR_FIELD_NUMBER: _ClassVar[int]
        requester_id: str
        cursor: str
        def __init__(self, requester_id: _Optional[str] = ..., cursor: _Optional[str] = ...) -> None: ...
    class FederationJobDelta(_message.Message):
        __slots__ = ("job_id", "summary", "changed_tasks", "tombstone")
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        SUMMARY_FIELD_NUMBER: _ClassVar[int]
        CHANGED_TASKS_FIELD_NUMBER: _ClassVar[int]
        TOMBSTONE_FIELD_NUMBER: _ClassVar[int]
        job_id: str
        summary: _job_pb2.JobStatus
        changed_tasks: _containers.RepeatedCompositeFieldContainer[_job_pb2.TaskStatus]
        tombstone: bool
        def __init__(self, job_id: _Optional[str] = ..., summary: _Optional[_Union[_job_pb2.JobStatus, _Mapping]] = ..., changed_tasks: _Optional[_Iterable[_Union[_job_pb2.TaskStatus, _Mapping]]] = ..., tombstone: _Optional[bool] = ...) -> None: ...
    class FederationSyncResponse(_message.Message):
        __slots__ = ("deltas", "next_cursor", "cursor_stale")
        DELTAS_FIELD_NUMBER: _ClassVar[int]
        NEXT_CURSOR_FIELD_NUMBER: _ClassVar[int]
        CURSOR_STALE_FIELD_NUMBER: _ClassVar[int]
        deltas: _containers.RepeatedCompositeFieldContainer[Controller.FederationJobDelta]
        next_cursor: str
        cursor_stale: bool
        def __init__(self, deltas: _Optional[_Iterable[_Union[Controller.FederationJobDelta, _Mapping]]] = ..., next_cursor: _Optional[str] = ..., cursor_stale: _Optional[bool] = ...) -> None: ...
    def __init__(self) -> None: ...

class StringList(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, values: _Optional[_Iterable[str]] = ...) -> None: ...
