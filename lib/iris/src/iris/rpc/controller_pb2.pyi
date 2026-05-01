from . import job_pb2 as _job_pb2
from . import iris_logging_pb2 as _iris_logging_pb2
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
    class LaunchJobRequest(_message.Message):
        __slots__ = ("name", "entrypoint", "resources", "environment", "bundle_id", "bundle_blob", "scheduling_timeout", "ports", "max_task_failures", "max_retries_failure", "max_retries_preemption", "constraints", "coscheduling", "replicas", "timeout", "fail_if_exists", "reservation", "preemption_policy", "existing_job_policy", "priority_band", "task_image", "submit_argv", "client_revision_date")
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
        RESERVATION_FIELD_NUMBER: _ClassVar[int]
        PREEMPTION_POLICY_FIELD_NUMBER: _ClassVar[int]
        EXISTING_JOB_POLICY_FIELD_NUMBER: _ClassVar[int]
        PRIORITY_BAND_FIELD_NUMBER: _ClassVar[int]
        TASK_IMAGE_FIELD_NUMBER: _ClassVar[int]
        SUBMIT_ARGV_FIELD_NUMBER: _ClassVar[int]
        CLIENT_REVISION_DATE_FIELD_NUMBER: _ClassVar[int]
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
        reservation: _job_pb2.ReservationConfig
        preemption_policy: _job_pb2.JobPreemptionPolicy
        existing_job_policy: _job_pb2.ExistingJobPolicy
        priority_band: _job_pb2.PriorityBand
        task_image: str
        submit_argv: _containers.RepeatedScalarFieldContainer[str]
        client_revision_date: str
        def __init__(self, name: _Optional[str] = ..., entrypoint: _Optional[_Union[_job_pb2.RuntimeEntrypoint, _Mapping]] = ..., resources: _Optional[_Union[_job_pb2.ResourceSpecProto, _Mapping]] = ..., environment: _Optional[_Union[_job_pb2.EnvironmentConfig, _Mapping]] = ..., bundle_id: _Optional[str] = ..., bundle_blob: _Optional[bytes] = ..., scheduling_timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., ports: _Optional[_Iterable[str]] = ..., max_task_failures: _Optional[int] = ..., max_retries_failure: _Optional[int] = ..., max_retries_preemption: _Optional[int] = ..., constraints: _Optional[_Iterable[_Union[_job_pb2.Constraint, _Mapping]]] = ..., coscheduling: _Optional[_Union[_job_pb2.CoschedulingConfig, _Mapping]] = ..., replicas: _Optional[int] = ..., timeout: _Optional[_Union[_time_pb2.Duration, _Mapping]] = ..., fail_if_exists: _Optional[bool] = ..., reservation: _Optional[_Union[_job_pb2.ReservationConfig, _Mapping]] = ..., preemption_policy: _Optional[_Union[_job_pb2.JobPreemptionPolicy, str]] = ..., existing_job_policy: _Optional[_Union[_job_pb2.ExistingJobPolicy, str]] = ..., priority_band: _Optional[_Union[_job_pb2.PriorityBand, str]] = ..., task_image: _Optional[str] = ..., submit_argv: _Optional[_Iterable[str]] = ..., client_revision_date: _Optional[str] = ...) -> None: ...
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
        __slots__ = ("job", "request", "resource_min", "resource_max")
        JOB_FIELD_NUMBER: _ClassVar[int]
        REQUEST_FIELD_NUMBER: _ClassVar[int]
        RESOURCE_MIN_FIELD_NUMBER: _ClassVar[int]
        RESOURCE_MAX_FIELD_NUMBER: _ClassVar[int]
        job: _job_pb2.JobStatus
        request: Controller.LaunchJobRequest
        resource_min: _job_pb2.ResourceUsage
        resource_max: _job_pb2.ResourceUsage
        def __init__(self, job: _Optional[_Union[_job_pb2.JobStatus, _Mapping]] = ..., request: _Optional[_Union[Controller.LaunchJobRequest, _Mapping]] = ..., resource_min: _Optional[_Union[_job_pb2.ResourceUsage, _Mapping]] = ..., resource_max: _Optional[_Union[_job_pb2.ResourceUsage, _Mapping]] = ...) -> None: ...
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
        __slots__ = ("scope", "parent_job_id", "name_filter", "state_filter", "sort_field", "sort_direction", "offset", "limit")
        SCOPE_FIELD_NUMBER: _ClassVar[int]
        PARENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
        NAME_FILTER_FIELD_NUMBER: _ClassVar[int]
        STATE_FILTER_FIELD_NUMBER: _ClassVar[int]
        SORT_FIELD_FIELD_NUMBER: _ClassVar[int]
        SORT_DIRECTION_FIELD_NUMBER: _ClassVar[int]
        OFFSET_FIELD_NUMBER: _ClassVar[int]
        LIMIT_FIELD_NUMBER: _ClassVar[int]
        scope: Controller.JobQueryScope
        parent_job_id: str
        name_filter: str
        state_filter: str
        sort_field: Controller.JobSortField
        sort_direction: Controller.SortDirection
        offset: int
        limit: int
        def __init__(self, scope: _Optional[_Union[Controller.JobQueryScope, str]] = ..., parent_job_id: _Optional[str] = ..., name_filter: _Optional[str] = ..., state_filter: _Optional[str] = ..., sort_field: _Optional[_Union[Controller.JobSortField, str]] = ..., sort_direction: _Optional[_Union[Controller.SortDirection, str]] = ..., offset: _Optional[int] = ..., limit: _Optional[int] = ...) -> None: ...
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
        __slots__ = ("task", "job_resources")
        TASK_FIELD_NUMBER: _ClassVar[int]
        JOB_RESOURCES_FIELD_NUMBER: _ClassVar[int]
        task: _job_pb2.TaskStatus
        job_resources: _job_pb2.ResourceSpecProto
        def __init__(self, task: _Optional[_Union[_job_pb2.TaskStatus, _Mapping]] = ..., job_resources: _Optional[_Union[_job_pb2.ResourceSpecProto, _Mapping]] = ...) -> None: ...
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
        __slots__ = ("worker_id", "healthy", "consecutive_failures", "last_heartbeat", "running_job_ids", "address", "metadata", "status_message")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        HEALTHY_FIELD_NUMBER: _ClassVar[int]
        CONSECUTIVE_FAILURES_FIELD_NUMBER: _ClassVar[int]
        LAST_HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
        RUNNING_JOB_IDS_FIELD_NUMBER: _ClassVar[int]
        ADDRESS_FIELD_NUMBER: _ClassVar[int]
        METADATA_FIELD_NUMBER: _ClassVar[int]
        STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        healthy: bool
        consecutive_failures: int
        last_heartbeat: _time_pb2.Timestamp
        running_job_ids: _containers.RepeatedScalarFieldContainer[str]
        address: str
        metadata: _job_pb2.WorkerMetadata
        status_message: str
        def __init__(self, worker_id: _Optional[str] = ..., healthy: _Optional[bool] = ..., consecutive_failures: _Optional[int] = ..., last_heartbeat: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ..., running_job_ids: _Optional[_Iterable[str]] = ..., address: _Optional[str] = ..., metadata: _Optional[_Union[_job_pb2.WorkerMetadata, _Mapping]] = ..., status_message: _Optional[str] = ...) -> None: ...
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
        __slots__ = ("endpoint_id", "name", "address", "task_id", "metadata")
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
        endpoint_id: str
        name: str
        address: str
        task_id: str
        metadata: _containers.ScalarMap[str, str]
        def __init__(self, endpoint_id: _Optional[str] = ..., name: _Optional[str] = ..., address: _Optional[str] = ..., task_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...
    class RegisterEndpointRequest(_message.Message):
        __slots__ = ("name", "address", "task_id", "metadata", "attempt_id", "endpoint_id")
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
        name: str
        address: str
        task_id: str
        metadata: _containers.ScalarMap[str, str]
        attempt_id: int
        endpoint_id: str
        def __init__(self, name: _Optional[str] = ..., address: _Optional[str] = ..., task_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., attempt_id: _Optional[int] = ..., endpoint_id: _Optional[str] = ...) -> None: ...
    class RegisterEndpointResponse(_message.Message):
        __slots__ = ("endpoint_id",)
        ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
        endpoint_id: str
        def __init__(self, endpoint_id: _Optional[str] = ...) -> None: ...
    class UnregisterEndpointRequest(_message.Message):
        __slots__ = ("endpoint_id",)
        ENDPOINT_ID_FIELD_NUMBER: _ClassVar[int]
        endpoint_id: str
        def __init__(self, endpoint_id: _Optional[str] = ...) -> None: ...
    class ListEndpointsRequest(_message.Message):
        __slots__ = ("prefix", "exact")
        PREFIX_FIELD_NUMBER: _ClassVar[int]
        EXACT_FIELD_NUMBER: _ClassVar[int]
        prefix: str
        exact: bool
        def __init__(self, prefix: _Optional[str] = ..., exact: _Optional[bool] = ...) -> None: ...
    class ListEndpointsResponse(_message.Message):
        __slots__ = ("endpoints",)
        ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
        endpoints: _containers.RepeatedCompositeFieldContainer[Controller.Endpoint]
        def __init__(self, endpoints: _Optional[_Iterable[_Union[Controller.Endpoint, _Mapping]]] = ...) -> None: ...
    class GetAutoscalerStatusRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
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
        __slots__ = ("user", "task_state_counts", "job_state_counts")
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
        user: str
        task_state_counts: _containers.ScalarMap[str, int]
        job_state_counts: _containers.ScalarMap[str, int]
        def __init__(self, user: _Optional[str] = ..., task_state_counts: _Optional[_Mapping[str, int]] = ..., job_state_counts: _Optional[_Mapping[str, int]] = ...) -> None: ...
    class ListUsersRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class ListUsersResponse(_message.Message):
        __slots__ = ("users",)
        USERS_FIELD_NUMBER: _ClassVar[int]
        users: _containers.RepeatedCompositeFieldContainer[Controller.UserSummary]
        def __init__(self, users: _Optional[_Iterable[_Union[Controller.UserSummary, _Mapping]]] = ...) -> None: ...
    class GetTaskLogsRequest(_message.Message):
        __slots__ = ("id", "include_children", "since_ms", "max_total_lines", "substring", "attempt_id", "min_level", "cursor", "tail")
        ID_FIELD_NUMBER: _ClassVar[int]
        INCLUDE_CHILDREN_FIELD_NUMBER: _ClassVar[int]
        SINCE_MS_FIELD_NUMBER: _ClassVar[int]
        MAX_TOTAL_LINES_FIELD_NUMBER: _ClassVar[int]
        SUBSTRING_FIELD_NUMBER: _ClassVar[int]
        ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
        MIN_LEVEL_FIELD_NUMBER: _ClassVar[int]
        CURSOR_FIELD_NUMBER: _ClassVar[int]
        TAIL_FIELD_NUMBER: _ClassVar[int]
        id: str
        include_children: bool
        since_ms: int
        max_total_lines: int
        substring: str
        attempt_id: int
        min_level: str
        cursor: int
        tail: bool
        def __init__(self, id: _Optional[str] = ..., include_children: _Optional[bool] = ..., since_ms: _Optional[int] = ..., max_total_lines: _Optional[int] = ..., substring: _Optional[str] = ..., attempt_id: _Optional[int] = ..., min_level: _Optional[str] = ..., cursor: _Optional[int] = ..., tail: _Optional[bool] = ...) -> None: ...
    class TaskLogBatch(_message.Message):
        __slots__ = ("task_id", "logs", "error", "worker_id")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        LOGS_FIELD_NUMBER: _ClassVar[int]
        ERROR_FIELD_NUMBER: _ClassVar[int]
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        logs: _containers.RepeatedCompositeFieldContainer[_iris_logging_pb2.LogEntry]
        error: str
        worker_id: str
        def __init__(self, task_id: _Optional[str] = ..., logs: _Optional[_Iterable[_Union[_iris_logging_pb2.LogEntry, _Mapping]]] = ..., error: _Optional[str] = ..., worker_id: _Optional[str] = ...) -> None: ...
    class GetTaskLogsResponse(_message.Message):
        __slots__ = ("task_logs", "truncated", "child_job_statuses", "cursor")
        TASK_LOGS_FIELD_NUMBER: _ClassVar[int]
        TRUNCATED_FIELD_NUMBER: _ClassVar[int]
        CHILD_JOB_STATUSES_FIELD_NUMBER: _ClassVar[int]
        CURSOR_FIELD_NUMBER: _ClassVar[int]
        task_logs: _containers.RepeatedCompositeFieldContainer[Controller.TaskLogBatch]
        truncated: bool
        child_job_statuses: _containers.RepeatedCompositeFieldContainer[_job_pb2.JobStatus]
        cursor: int
        def __init__(self, task_logs: _Optional[_Iterable[_Union[Controller.TaskLogBatch, _Mapping]]] = ..., truncated: _Optional[bool] = ..., child_job_statuses: _Optional[_Iterable[_Union[_job_pb2.JobStatus, _Mapping]]] = ..., cursor: _Optional[int] = ...) -> None: ...
    class GetWorkerStatusRequest(_message.Message):
        __slots__ = ("id",)
        ID_FIELD_NUMBER: _ClassVar[int]
        id: str
        def __init__(self, id: _Optional[str] = ...) -> None: ...
    class GetWorkerStatusResponse(_message.Message):
        __slots__ = ("vm", "scale_group", "worker", "bootstrap_logs", "recent_attempts", "current_resources", "resource_history")
        VM_FIELD_NUMBER: _ClassVar[int]
        SCALE_GROUP_FIELD_NUMBER: _ClassVar[int]
        WORKER_FIELD_NUMBER: _ClassVar[int]
        BOOTSTRAP_LOGS_FIELD_NUMBER: _ClassVar[int]
        RECENT_ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
        CURRENT_RESOURCES_FIELD_NUMBER: _ClassVar[int]
        RESOURCE_HISTORY_FIELD_NUMBER: _ClassVar[int]
        vm: _vm_pb2.VmInfo
        scale_group: str
        worker: Controller.WorkerHealthStatus
        bootstrap_logs: str
        recent_attempts: _containers.RepeatedCompositeFieldContainer[Controller.WorkerTaskAttempt]
        current_resources: _job_pb2.WorkerResourceSnapshot
        resource_history: _containers.RepeatedCompositeFieldContainer[_job_pb2.WorkerResourceSnapshot]
        def __init__(self, vm: _Optional[_Union[_vm_pb2.VmInfo, _Mapping]] = ..., scale_group: _Optional[str] = ..., worker: _Optional[_Union[Controller.WorkerHealthStatus, _Mapping]] = ..., bootstrap_logs: _Optional[str] = ..., recent_attempts: _Optional[_Iterable[_Union[Controller.WorkerTaskAttempt, _Mapping]]] = ..., current_resources: _Optional[_Union[_job_pb2.WorkerResourceSnapshot, _Mapping]] = ..., resource_history: _Optional[_Iterable[_Union[_job_pb2.WorkerResourceSnapshot, _Mapping]]] = ...) -> None: ...
    class WorkerTaskAttempt(_message.Message):
        __slots__ = ("task_id", "attempt")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        ATTEMPT_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        attempt: _job_pb2.TaskAttempt
        def __init__(self, task_id: _Optional[str] = ..., attempt: _Optional[_Union[_job_pb2.TaskAttempt, _Mapping]] = ...) -> None: ...
    class SchedulingEvent(_message.Message):
        __slots__ = ("task_id", "attempt_id", "event_type", "reason", "message", "timestamp")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
        EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        MESSAGE_FIELD_NUMBER: _ClassVar[int]
        TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        attempt_id: int
        event_type: str
        reason: str
        message: str
        timestamp: _time_pb2.Timestamp
        def __init__(self, task_id: _Optional[str] = ..., attempt_id: _Optional[int] = ..., event_type: _Optional[str] = ..., reason: _Optional[str] = ..., message: _Optional[str] = ..., timestamp: _Optional[_Union[_time_pb2.Timestamp, _Mapping]] = ...) -> None: ...
    class ClusterCapacity(_message.Message):
        __slots__ = ("schedulable_nodes", "total_cpu_millicores", "available_cpu_millicores", "total_memory_bytes", "available_memory_bytes")
        SCHEDULABLE_NODES_FIELD_NUMBER: _ClassVar[int]
        TOTAL_CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
        AVAILABLE_CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
        TOTAL_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
        AVAILABLE_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
        schedulable_nodes: int
        total_cpu_millicores: int
        available_cpu_millicores: int
        total_memory_bytes: int
        available_memory_bytes: int
        def __init__(self, schedulable_nodes: _Optional[int] = ..., total_cpu_millicores: _Optional[int] = ..., available_cpu_millicores: _Optional[int] = ..., total_memory_bytes: _Optional[int] = ..., available_memory_bytes: _Optional[int] = ...) -> None: ...
    class GetProviderStatusRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class GetProviderStatusResponse(_message.Message):
        __slots__ = ("has_direct_provider", "scheduling_events", "capacity")
        HAS_DIRECT_PROVIDER_FIELD_NUMBER: _ClassVar[int]
        SCHEDULING_EVENTS_FIELD_NUMBER: _ClassVar[int]
        CAPACITY_FIELD_NUMBER: _ClassVar[int]
        has_direct_provider: bool
        scheduling_events: _containers.RepeatedCompositeFieldContainer[Controller.SchedulingEvent]
        capacity: Controller.ClusterCapacity
        def __init__(self, has_direct_provider: _Optional[bool] = ..., scheduling_events: _Optional[_Iterable[_Union[Controller.SchedulingEvent, _Mapping]]] = ..., capacity: _Optional[_Union[Controller.ClusterCapacity, _Mapping]] = ...) -> None: ...
    class GetKubernetesClusterStatusRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
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
    class RestartWorkerRequest(_message.Message):
        __slots__ = ("worker_id",)
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        def __init__(self, worker_id: _Optional[str] = ...) -> None: ...
    class RestartWorkerResponse(_message.Message):
        __slots__ = ("accepted", "error")
        ACCEPTED_FIELD_NUMBER: _ClassVar[int]
        ERROR_FIELD_NUMBER: _ClassVar[int]
        accepted: bool
        error: str
        def __init__(self, accepted: _Optional[bool] = ..., error: _Optional[str] = ...) -> None: ...
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
    class UpdateTaskStatusRequest(_message.Message):
        __slots__ = ("worker_id", "updates")
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        UPDATES_FIELD_NUMBER: _ClassVar[int]
        worker_id: str
        updates: _containers.RepeatedCompositeFieldContainer[_job_pb2.WorkerTaskStatus]
        def __init__(self, worker_id: _Optional[str] = ..., updates: _Optional[_Iterable[_Union[_job_pb2.WorkerTaskStatus, _Mapping]]] = ...) -> None: ...
    class UpdateTaskStatusResponse(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class GetSchedulerStateRequest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class SchedulerTaskEntry(_message.Message):
        __slots__ = ("task_id", "job_id", "user_id", "original_band", "effective_band", "queue_position", "resource_value")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        ORIGINAL_BAND_FIELD_NUMBER: _ClassVar[int]
        EFFECTIVE_BAND_FIELD_NUMBER: _ClassVar[int]
        QUEUE_POSITION_FIELD_NUMBER: _ClassVar[int]
        RESOURCE_VALUE_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        job_id: str
        user_id: str
        original_band: _job_pb2.PriorityBand
        effective_band: _job_pb2.PriorityBand
        queue_position: int
        resource_value: int
        def __init__(self, task_id: _Optional[str] = ..., job_id: _Optional[str] = ..., user_id: _Optional[str] = ..., original_band: _Optional[_Union[_job_pb2.PriorityBand, str]] = ..., effective_band: _Optional[_Union[_job_pb2.PriorityBand, str]] = ..., queue_position: _Optional[int] = ..., resource_value: _Optional[int] = ...) -> None: ...
    class SchedulerBandGroup(_message.Message):
        __slots__ = ("band", "tasks", "total_in_band")
        BAND_FIELD_NUMBER: _ClassVar[int]
        TASKS_FIELD_NUMBER: _ClassVar[int]
        TOTAL_IN_BAND_FIELD_NUMBER: _ClassVar[int]
        band: _job_pb2.PriorityBand
        tasks: _containers.RepeatedCompositeFieldContainer[Controller.SchedulerTaskEntry]
        total_in_band: int
        def __init__(self, band: _Optional[_Union[_job_pb2.PriorityBand, str]] = ..., tasks: _Optional[_Iterable[_Union[Controller.SchedulerTaskEntry, _Mapping]]] = ..., total_in_band: _Optional[int] = ...) -> None: ...
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
    class SchedulerRunningTask(_message.Message):
        __slots__ = ("task_id", "job_id", "user_id", "worker_id", "effective_band", "resource_value", "preemptible", "preemptible_by", "is_coscheduled")
        TASK_ID_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
        USER_ID_FIELD_NUMBER: _ClassVar[int]
        WORKER_ID_FIELD_NUMBER: _ClassVar[int]
        EFFECTIVE_BAND_FIELD_NUMBER: _ClassVar[int]
        RESOURCE_VALUE_FIELD_NUMBER: _ClassVar[int]
        PREEMPTIBLE_FIELD_NUMBER: _ClassVar[int]
        PREEMPTIBLE_BY_FIELD_NUMBER: _ClassVar[int]
        IS_COSCHEDULED_FIELD_NUMBER: _ClassVar[int]
        task_id: str
        job_id: str
        user_id: str
        worker_id: str
        effective_band: _job_pb2.PriorityBand
        resource_value: int
        preemptible: bool
        preemptible_by: _containers.RepeatedScalarFieldContainer[_job_pb2.PriorityBand]
        is_coscheduled: bool
        def __init__(self, task_id: _Optional[str] = ..., job_id: _Optional[str] = ..., user_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., effective_band: _Optional[_Union[_job_pb2.PriorityBand, str]] = ..., resource_value: _Optional[int] = ..., preemptible: _Optional[bool] = ..., preemptible_by: _Optional[_Iterable[_Union[_job_pb2.PriorityBand, str]]] = ..., is_coscheduled: _Optional[bool] = ...) -> None: ...
    class GetSchedulerStateResponse(_message.Message):
        __slots__ = ("pending_queue", "user_budgets", "running_tasks", "total_pending", "total_running")
        PENDING_QUEUE_FIELD_NUMBER: _ClassVar[int]
        USER_BUDGETS_FIELD_NUMBER: _ClassVar[int]
        RUNNING_TASKS_FIELD_NUMBER: _ClassVar[int]
        TOTAL_PENDING_FIELD_NUMBER: _ClassVar[int]
        TOTAL_RUNNING_FIELD_NUMBER: _ClassVar[int]
        pending_queue: _containers.RepeatedCompositeFieldContainer[Controller.SchedulerBandGroup]
        user_budgets: _containers.RepeatedCompositeFieldContainer[Controller.SchedulerUserBudget]
        running_tasks: _containers.RepeatedCompositeFieldContainer[Controller.SchedulerRunningTask]
        total_pending: int
        total_running: int
        def __init__(self, pending_queue: _Optional[_Iterable[_Union[Controller.SchedulerBandGroup, _Mapping]]] = ..., user_budgets: _Optional[_Iterable[_Union[Controller.SchedulerUserBudget, _Mapping]]] = ..., running_tasks: _Optional[_Iterable[_Union[Controller.SchedulerRunningTask, _Mapping]]] = ..., total_pending: _Optional[int] = ..., total_running: _Optional[int] = ...) -> None: ...
    def __init__(self) -> None: ...
