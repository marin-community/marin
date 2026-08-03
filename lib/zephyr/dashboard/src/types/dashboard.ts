export type Integer = string | number

export interface PipelineSummary {
  executionId: string
  pipelineName?: string
  currentStage?: string
}

export interface ListPipelinesResponse {
  pipelines?: PipelineSummary[]
}

export interface PlanNode {
  nodeId: string
  label: string
  stageType: string
  operationTypes?: string[]
  outputShards?: number
  stageIndex?: number
  parentNodeId?: string
  auxiliary?: boolean
}

export interface PipelinePlan {
  pipelineName?: string
  executionId?: string
  sourceItemCount?: Integer
  nodes?: PlanNode[]
}

export interface PlanNodeStatus {
  nodeId: string
  state: string
}

export interface WorkerStateCount {
  state: string
  count: number
}

export interface ResourceUsage {
  cpuCores?: number
  cpuUtilization?: number
  memoryBytes?: Integer
  memoryUtilization?: number
}

export interface PipelineStatus {
  executionId?: string
  phase?: string
  currentNodeId?: string
  currentStage?: string
  completedShards?: number
  totalShards?: number
  inFlightShards?: number
  queuedShards?: number
  retries?: number
  startedAtMs?: Integer
  finishedAtMs?: Integer
  fatalError?: string
  coordinatorTaskId?: string
  expectedWorkers?: number
  workerStates?: WorkerStateCount[]
  resources?: ResourceUsage
  nodeStatuses?: PlanNodeStatus[]
}

export interface MetricPoint {
  timestampMs: Integer
  stage?: string
  itemRate?: number
  byteRate?: number
  cpuCores?: number
  memoryBytes?: Integer
}

export interface PipelineMetrics {
  points?: MetricPoint[]
  warning?: string
}

export interface CounterValue {
  name: string
  intValue?: Integer
  doubleValue?: number
  aggregation?: string
  stage?: string
  observations?: Integer
}

export interface ListCountersResponse {
  counters?: CounterValue[]
  total?: number
}

export interface WorkerStatus {
  workerId: string
  taskId?: string
  state?: string
  lastSeenAgeSeconds?: number
  assignments?: WorkerAssignment[]
  cpuPercent?: number
  memoryBytes?: Integer
}

export interface WorkerAssignment {
  executionId: string
  shard: number
}

export interface ListWorkersResponse {
  workers?: WorkerStatus[]
  total?: number
}
