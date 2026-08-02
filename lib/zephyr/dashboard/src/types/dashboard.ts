export type Integer = string | number

export interface PipelineSummary {
  executionId: string
  pipelineName?: string
  phase?: string
  currentStage?: string
  completedShards?: number
  totalShards?: number
  startedAtMs?: Integer
  fatalError?: string
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

export interface PlanEdge {
  sourceNodeId: string
  targetNodeId: string
  label?: string
}

export interface PipelinePlan {
  pipelineName?: string
  pipelineId?: number
  executionId?: string
  sourceItemCount?: Integer
  sourceShardCount?: number
  nodes?: PlanNode[]
  edges?: PlanEdge[]
}

export interface PlanNodeStatus {
  nodeId: string
  state: string
  startedAtMs?: Integer
  finishedAtMs?: Integer
}

export interface WorkerStateCount {
  state: string
  count: number
}

export interface ResourceUsage {
  cpuCores?: number
  cpuCapacityCores?: number
  cpuUtilization?: number
  memoryBytes?: Integer
  memoryCapacityBytes?: Integer
  memoryUtilization?: number
}

export interface PipelineStatus {
  executionId?: string
  phase?: string
  currentNodeId?: string
  currentStage?: string
  currentStageIndex?: number
  totalStages?: number
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
  activeShards?: number
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
