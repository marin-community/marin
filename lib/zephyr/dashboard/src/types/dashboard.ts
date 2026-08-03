// These interfaces mirror the dataclasses in `zephyr/dashboard.py`. The
// coordinator and this dashboard ship together, so the two stay in step.

export interface PipelineSummary {
  execution_id: string
  pipeline_name: string
  current_stage: string
}

export interface PipelineList {
  pipelines: PipelineSummary[]
}

export interface PlanNode {
  node_id: string
  label: string
  stage_type: string
  output_shards: number
  stage_index: number
  parent_node_id: string
  auxiliary: boolean
  operation_types: string[]
}

export interface PipelinePlan {
  pipeline_name: string
  execution_id: string
  source_item_count: number
  nodes: PlanNode[]
}

export interface PlanNodeStatus {
  node_id: string
  state: string
}

export interface WorkerStateCount {
  state: string
  count: number
}

export interface ResourceUsage {
  cpu_cores: number
  cpu_utilization: number
  memory_bytes: number
  memory_utilization: number
}

export interface PipelineStatus {
  execution_id: string
  phase: string
  current_node_id: string
  current_stage: string
  completed_shards: number
  total_shards: number
  in_flight_shards: number
  queued_shards: number
  retries: number
  started_at_ms: number
  finished_at_ms: number
  fatal_error: string
  coordinator_task_id: string
  expected_workers: number
  worker_states: WorkerStateCount[]
  resources: ResourceUsage
  node_statuses: PlanNodeStatus[]
}

export interface MetricPoint {
  timestamp_ms: number
  stage: string
  item_rate: number
  byte_rate: number
  cpu_cores: number
  memory_bytes: number
}

export interface PipelineMetrics {
  points: MetricPoint[]
  warning: string
}

export interface CounterValue {
  name: string
  value: number
  aggregation: string
  stage: string
  observations: number
}

export interface CounterPage {
  counters: CounterValue[]
  total: number
}

export interface WorkerAssignment {
  execution_id: string
  shard: number
}

export interface WorkerStatus {
  worker_id: string
  task_id: string
  state: string
  last_seen_age_seconds: number
  assignments: WorkerAssignment[]
  cpu_percent: number
  memory_bytes: number
}

export interface WorkerPage {
  workers: WorkerStatus[]
  total: number
}
