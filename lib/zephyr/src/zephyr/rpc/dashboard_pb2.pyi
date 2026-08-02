from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PipelinePhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PIPELINE_PHASE_UNSPECIFIED: _ClassVar[PipelinePhase]
    PIPELINE_PHASE_INITIALIZING: _ClassVar[PipelinePhase]
    PIPELINE_PHASE_WAITING_FOR_WORKERS: _ClassVar[PipelinePhase]
    PIPELINE_PHASE_RUNNING: _ClassVar[PipelinePhase]
    PIPELINE_PHASE_SUCCEEDED: _ClassVar[PipelinePhase]
    PIPELINE_PHASE_FAILED: _ClassVar[PipelinePhase]
    PIPELINE_PHASE_STOPPING: _ClassVar[PipelinePhase]

class PlanNodeState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PLAN_NODE_STATE_UNSPECIFIED: _ClassVar[PlanNodeState]
    PLAN_NODE_STATE_PENDING: _ClassVar[PlanNodeState]
    PLAN_NODE_STATE_RUNNING: _ClassVar[PlanNodeState]
    PLAN_NODE_STATE_SUCCEEDED: _ClassVar[PlanNodeState]
    PLAN_NODE_STATE_FAILED: _ClassVar[PlanNodeState]

class CounterAggregation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COUNTER_AGGREGATION_UNSPECIFIED: _ClassVar[CounterAggregation]
    COUNTER_AGGREGATION_SUM: _ClassVar[CounterAggregation]
    COUNTER_AGGREGATION_AVERAGE: _ClassVar[CounterAggregation]
    COUNTER_AGGREGATION_MAX: _ClassVar[CounterAggregation]
    COUNTER_AGGREGATION_MIN: _ClassVar[CounterAggregation]
PIPELINE_PHASE_UNSPECIFIED: PipelinePhase
PIPELINE_PHASE_INITIALIZING: PipelinePhase
PIPELINE_PHASE_WAITING_FOR_WORKERS: PipelinePhase
PIPELINE_PHASE_RUNNING: PipelinePhase
PIPELINE_PHASE_SUCCEEDED: PipelinePhase
PIPELINE_PHASE_FAILED: PipelinePhase
PIPELINE_PHASE_STOPPING: PipelinePhase
PLAN_NODE_STATE_UNSPECIFIED: PlanNodeState
PLAN_NODE_STATE_PENDING: PlanNodeState
PLAN_NODE_STATE_RUNNING: PlanNodeState
PLAN_NODE_STATE_SUCCEEDED: PlanNodeState
PLAN_NODE_STATE_FAILED: PlanNodeState
COUNTER_AGGREGATION_UNSPECIFIED: CounterAggregation
COUNTER_AGGREGATION_SUM: CounterAggregation
COUNTER_AGGREGATION_AVERAGE: CounterAggregation
COUNTER_AGGREGATION_MAX: CounterAggregation
COUNTER_AGGREGATION_MIN: CounterAggregation

class ListPipelinesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class PipelineSummary(_message.Message):
    __slots__ = ("execution_id", "pipeline_name", "phase", "current_stage", "completed_shards", "total_shards", "started_at_ms", "fatal_error")
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_NAME_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STAGE_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_SHARDS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_SHARDS_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    FATAL_ERROR_FIELD_NUMBER: _ClassVar[int]
    execution_id: str
    pipeline_name: str
    phase: PipelinePhase
    current_stage: str
    completed_shards: int
    total_shards: int
    started_at_ms: int
    fatal_error: str
    def __init__(self, execution_id: _Optional[str] = ..., pipeline_name: _Optional[str] = ..., phase: _Optional[_Union[PipelinePhase, str]] = ..., current_stage: _Optional[str] = ..., completed_shards: _Optional[int] = ..., total_shards: _Optional[int] = ..., started_at_ms: _Optional[int] = ..., fatal_error: _Optional[str] = ...) -> None: ...

class ListPipelinesResponse(_message.Message):
    __slots__ = ("pipelines",)
    PIPELINES_FIELD_NUMBER: _ClassVar[int]
    pipelines: _containers.RepeatedCompositeFieldContainer[PipelineSummary]
    def __init__(self, pipelines: _Optional[_Iterable[_Union[PipelineSummary, _Mapping]]] = ...) -> None: ...

class GetPlanRequest(_message.Message):
    __slots__ = ("execution_id",)
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    execution_id: str
    def __init__(self, execution_id: _Optional[str] = ...) -> None: ...

class GetStatusRequest(_message.Message):
    __slots__ = ("execution_id",)
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    execution_id: str
    def __init__(self, execution_id: _Optional[str] = ...) -> None: ...

class PlanNode(_message.Message):
    __slots__ = ("node_id", "label", "stage_type", "operation_types", "output_shards", "stage_index", "parent_node_id", "auxiliary")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    STAGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    OPERATION_TYPES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SHARDS_FIELD_NUMBER: _ClassVar[int]
    STAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    PARENT_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    AUXILIARY_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    label: str
    stage_type: str
    operation_types: _containers.RepeatedScalarFieldContainer[str]
    output_shards: int
    stage_index: int
    parent_node_id: str
    auxiliary: bool
    def __init__(self, node_id: _Optional[str] = ..., label: _Optional[str] = ..., stage_type: _Optional[str] = ..., operation_types: _Optional[_Iterable[str]] = ..., output_shards: _Optional[int] = ..., stage_index: _Optional[int] = ..., parent_node_id: _Optional[str] = ..., auxiliary: _Optional[bool] = ...) -> None: ...

class PlanEdge(_message.Message):
    __slots__ = ("source_node_id", "target_node_id", "label")
    SOURCE_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    source_node_id: str
    target_node_id: str
    label: str
    def __init__(self, source_node_id: _Optional[str] = ..., target_node_id: _Optional[str] = ..., label: _Optional[str] = ...) -> None: ...

class GetPlanResponse(_message.Message):
    __slots__ = ("pipeline_name", "pipeline_id", "execution_id", "source_item_count", "source_shard_count", "nodes", "edges")
    PIPELINE_NAME_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ITEM_COUNT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SHARD_COUNT_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    pipeline_name: str
    pipeline_id: int
    execution_id: str
    source_item_count: int
    source_shard_count: int
    nodes: _containers.RepeatedCompositeFieldContainer[PlanNode]
    edges: _containers.RepeatedCompositeFieldContainer[PlanEdge]
    def __init__(self, pipeline_name: _Optional[str] = ..., pipeline_id: _Optional[int] = ..., execution_id: _Optional[str] = ..., source_item_count: _Optional[int] = ..., source_shard_count: _Optional[int] = ..., nodes: _Optional[_Iterable[_Union[PlanNode, _Mapping]]] = ..., edges: _Optional[_Iterable[_Union[PlanEdge, _Mapping]]] = ...) -> None: ...

class PlanNodeStatus(_message.Message):
    __slots__ = ("node_id", "state", "started_at_ms", "finished_at_ms")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    state: PlanNodeState
    started_at_ms: int
    finished_at_ms: int
    def __init__(self, node_id: _Optional[str] = ..., state: _Optional[_Union[PlanNodeState, str]] = ..., started_at_ms: _Optional[int] = ..., finished_at_ms: _Optional[int] = ...) -> None: ...

class WorkerStateCount(_message.Message):
    __slots__ = ("state", "count")
    STATE_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    state: str
    count: int
    def __init__(self, state: _Optional[str] = ..., count: _Optional[int] = ...) -> None: ...

class ResourceUsage(_message.Message):
    __slots__ = ("cpu_cores", "cpu_capacity_cores", "cpu_utilization", "memory_bytes", "memory_capacity_bytes", "memory_utilization")
    CPU_CORES_FIELD_NUMBER: _ClassVar[int]
    CPU_CAPACITY_CORES_FIELD_NUMBER: _ClassVar[int]
    CPU_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_CAPACITY_BYTES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    cpu_cores: float
    cpu_capacity_cores: float
    cpu_utilization: float
    memory_bytes: int
    memory_capacity_bytes: int
    memory_utilization: float
    def __init__(self, cpu_cores: _Optional[float] = ..., cpu_capacity_cores: _Optional[float] = ..., cpu_utilization: _Optional[float] = ..., memory_bytes: _Optional[int] = ..., memory_capacity_bytes: _Optional[int] = ..., memory_utilization: _Optional[float] = ...) -> None: ...

class GetStatusResponse(_message.Message):
    __slots__ = ("phase", "current_node_id", "current_stage", "current_stage_index", "total_stages", "completed_shards", "total_shards", "in_flight_shards", "queued_shards", "retries", "started_at_ms", "finished_at_ms", "fatal_error", "coordinator_task_id", "expected_workers", "worker_states", "resources", "node_statuses", "execution_id")
    PHASE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STAGE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_STAGES_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_SHARDS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_SHARDS_FIELD_NUMBER: _ClassVar[int]
    IN_FLIGHT_SHARDS_FIELD_NUMBER: _ClassVar[int]
    QUEUED_SHARDS_FIELD_NUMBER: _ClassVar[int]
    RETRIES_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    FATAL_ERROR_FIELD_NUMBER: _ClassVar[int]
    COORDINATOR_TASK_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_WORKERS_FIELD_NUMBER: _ClassVar[int]
    WORKER_STATES_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    NODE_STATUSES_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    phase: PipelinePhase
    current_node_id: str
    current_stage: str
    current_stage_index: int
    total_stages: int
    completed_shards: int
    total_shards: int
    in_flight_shards: int
    queued_shards: int
    retries: int
    started_at_ms: int
    finished_at_ms: int
    fatal_error: str
    coordinator_task_id: str
    expected_workers: int
    worker_states: _containers.RepeatedCompositeFieldContainer[WorkerStateCount]
    resources: ResourceUsage
    node_statuses: _containers.RepeatedCompositeFieldContainer[PlanNodeStatus]
    execution_id: str
    def __init__(self, phase: _Optional[_Union[PipelinePhase, str]] = ..., current_node_id: _Optional[str] = ..., current_stage: _Optional[str] = ..., current_stage_index: _Optional[int] = ..., total_stages: _Optional[int] = ..., completed_shards: _Optional[int] = ..., total_shards: _Optional[int] = ..., in_flight_shards: _Optional[int] = ..., queued_shards: _Optional[int] = ..., retries: _Optional[int] = ..., started_at_ms: _Optional[int] = ..., finished_at_ms: _Optional[int] = ..., fatal_error: _Optional[str] = ..., coordinator_task_id: _Optional[str] = ..., expected_workers: _Optional[int] = ..., worker_states: _Optional[_Iterable[_Union[WorkerStateCount, _Mapping]]] = ..., resources: _Optional[_Union[ResourceUsage, _Mapping]] = ..., node_statuses: _Optional[_Iterable[_Union[PlanNodeStatus, _Mapping]]] = ..., execution_id: _Optional[str] = ...) -> None: ...

class GetMetricsRequest(_message.Message):
    __slots__ = ("max_points", "execution_id")
    MAX_POINTS_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    max_points: int
    execution_id: str
    def __init__(self, max_points: _Optional[int] = ..., execution_id: _Optional[str] = ...) -> None: ...

class MetricPoint(_message.Message):
    __slots__ = ("timestamp_ms", "stage", "item_rate", "byte_rate", "cpu_cores", "memory_bytes", "active_shards")
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    ITEM_RATE_FIELD_NUMBER: _ClassVar[int]
    BYTE_RATE_FIELD_NUMBER: _ClassVar[int]
    CPU_CORES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_SHARDS_FIELD_NUMBER: _ClassVar[int]
    timestamp_ms: int
    stage: str
    item_rate: float
    byte_rate: float
    cpu_cores: float
    memory_bytes: int
    active_shards: int
    def __init__(self, timestamp_ms: _Optional[int] = ..., stage: _Optional[str] = ..., item_rate: _Optional[float] = ..., byte_rate: _Optional[float] = ..., cpu_cores: _Optional[float] = ..., memory_bytes: _Optional[int] = ..., active_shards: _Optional[int] = ...) -> None: ...

class GetMetricsResponse(_message.Message):
    __slots__ = ("points", "warning")
    POINTS_FIELD_NUMBER: _ClassVar[int]
    WARNING_FIELD_NUMBER: _ClassVar[int]
    points: _containers.RepeatedCompositeFieldContainer[MetricPoint]
    warning: str
    def __init__(self, points: _Optional[_Iterable[_Union[MetricPoint, _Mapping]]] = ..., warning: _Optional[str] = ...) -> None: ...

class CounterValue(_message.Message):
    __slots__ = ("name", "int_value", "double_value", "aggregation", "stage", "observations")
    NAME_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_VALUE_FIELD_NUMBER: _ClassVar[int]
    AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    name: str
    int_value: int
    double_value: float
    aggregation: CounterAggregation
    stage: str
    observations: int
    def __init__(self, name: _Optional[str] = ..., int_value: _Optional[int] = ..., double_value: _Optional[float] = ..., aggregation: _Optional[_Union[CounterAggregation, str]] = ..., stage: _Optional[str] = ..., observations: _Optional[int] = ...) -> None: ...

class ListCountersRequest(_message.Message):
    __slots__ = ("stage", "search", "sort_field", "sort_descending", "offset", "limit", "execution_id")
    STAGE_FIELD_NUMBER: _ClassVar[int]
    SEARCH_FIELD_NUMBER: _ClassVar[int]
    SORT_FIELD_FIELD_NUMBER: _ClassVar[int]
    SORT_DESCENDING_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    stage: str
    search: str
    sort_field: str
    sort_descending: bool
    offset: int
    limit: int
    execution_id: str
    def __init__(self, stage: _Optional[str] = ..., search: _Optional[str] = ..., sort_field: _Optional[str] = ..., sort_descending: _Optional[bool] = ..., offset: _Optional[int] = ..., limit: _Optional[int] = ..., execution_id: _Optional[str] = ...) -> None: ...

class ListCountersResponse(_message.Message):
    __slots__ = ("counters", "total")
    COUNTERS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    counters: _containers.RepeatedCompositeFieldContainer[CounterValue]
    total: int
    def __init__(self, counters: _Optional[_Iterable[_Union[CounterValue, _Mapping]]] = ..., total: _Optional[int] = ...) -> None: ...

class ListWorkersRequest(_message.Message):
    __slots__ = ("search", "sort_field", "sort_descending", "offset", "limit")
    SEARCH_FIELD_NUMBER: _ClassVar[int]
    SORT_FIELD_FIELD_NUMBER: _ClassVar[int]
    SORT_DESCENDING_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    search: str
    sort_field: str
    sort_descending: bool
    offset: int
    limit: int
    def __init__(self, search: _Optional[str] = ..., sort_field: _Optional[str] = ..., sort_descending: _Optional[bool] = ..., offset: _Optional[int] = ..., limit: _Optional[int] = ...) -> None: ...

class WorkerStatus(_message.Message):
    __slots__ = ("worker_id", "task_id", "state", "last_seen_age_seconds", "assignments", "cpu_percent", "memory_bytes")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    LAST_SEEN_AGE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
    CPU_PERCENT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    task_id: str
    state: str
    last_seen_age_seconds: float
    assignments: _containers.RepeatedCompositeFieldContainer[WorkerAssignment]
    cpu_percent: float
    memory_bytes: int
    def __init__(self, worker_id: _Optional[str] = ..., task_id: _Optional[str] = ..., state: _Optional[str] = ..., last_seen_age_seconds: _Optional[float] = ..., assignments: _Optional[_Iterable[_Union[WorkerAssignment, _Mapping]]] = ..., cpu_percent: _Optional[float] = ..., memory_bytes: _Optional[int] = ...) -> None: ...

class WorkerAssignment(_message.Message):
    __slots__ = ("execution_id", "shard")
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    SHARD_FIELD_NUMBER: _ClassVar[int]
    execution_id: str
    shard: int
    def __init__(self, execution_id: _Optional[str] = ..., shard: _Optional[int] = ...) -> None: ...

class ListWorkersResponse(_message.Message):
    __slots__ = ("workers", "total")
    WORKERS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    workers: _containers.RepeatedCompositeFieldContainer[WorkerStatus]
    total: int
    def __init__(self, workers: _Optional[_Iterable[_Union[WorkerStatus, _Mapping]]] = ..., total: _Optional[int] = ...) -> None: ...
