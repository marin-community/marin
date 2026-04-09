/**
 * TypeScript interfaces mirroring the protobuf messages used by the Iris dashboard.
 *
 * These are manually maintained to cover only the fields the dashboard actually reads.
 * Proto JSON serialization uses camelCase field names and serializes int64 as strings.
 * Timestamps use our custom iris.time.Timestamp with { epochMs: string }.
 */

// -- Primitives --

/** iris.time.Timestamp serialized as JSON. epochMs is a string (int64). */
export interface ProtoTimestamp {
  epochMs: string
}

/** iris.time.Duration serialized as JSON. */
export interface ProtoDuration {
  milliseconds: string
}

/** Typed attribute value for worker attributes and constraint matching. */
export interface AttributeValue {
  stringValue?: string
  intValue?: string
  floatValue?: string
}

// -- Resources --

export interface ResourceSpecProto {
  cpuMillicores?: number
  memoryBytes?: string
  diskBytes?: string
  device?: DeviceConfig
}

export interface DeviceConfig {
  cpu?: { variant?: string }
  gpu?: { variant?: string; count?: number }
  tpu?: { variant?: string; topology?: string; count?: number }
}

export interface ResourceUsage {
  memoryMb?: string
  diskMb?: string
  cpuMillicores?: number
  memoryPeakMb?: string
  processCount?: number
}

export interface BuildMetrics {
  buildStarted?: ProtoTimestamp
  buildFinished?: ProtoTimestamp
  fromCache?: boolean
  imageTag?: string
}

// -- Constraints --

export interface Constraint {
  key: string
  op: string
  value?: AttributeValue
  values?: AttributeValue[]
}

// -- Tasks --

export interface TaskAttempt {
  attemptId: number
  workerId?: string
  state: string
  exitCode?: number
  error?: string
  startedAt?: ProtoTimestamp
  finishedAt?: ProtoTimestamp
  isWorkerFailure?: boolean
}

export interface TaskStatus {
  taskId: string
  state: string
  workerId?: string
  workerAddress?: string
  exitCode?: number
  error?: string
  startedAt?: ProtoTimestamp
  finishedAt?: ProtoTimestamp
  ports?: Record<string, number>
  resourceUsage?: ResourceUsage
  buildMetrics?: BuildMetrics
  currentAttemptId?: number
  attempts?: TaskAttempt[]
  pendingReason?: string
  canBeScheduled?: boolean
  containerId?: string
}

// -- Jobs --

export interface JobStatus {
  jobId: string
  state: string
  exitCode?: number
  error?: string
  startedAt?: ProtoTimestamp
  finishedAt?: ProtoTimestamp
  ports?: Record<string, number>
  resourceUsage?: ResourceUsage
  statusMessage?: string
  buildMetrics?: BuildMetrics
  failureCount?: number
  preemptionCount?: number
  tasks?: TaskStatus[]
  name: string
  submittedAt?: ProtoTimestamp
  resources?: ResourceSpecProto
  taskStateCounts?: Record<string, number>
  taskCount?: number
  completedCount?: number
  pendingReason?: string
  hasChildren?: boolean
}

// -- Controller RPC Responses --

export interface ListJobsResponse {
  jobs: JobStatus[]
  totalCount: number
  hasMore: boolean
}

export interface GetJobStatusResponse {
  job: JobStatus
  request?: LaunchJobRequest
}

export interface LaunchJobRequest {
  name: string
  resources?: ResourceSpecProto
  constraints?: Constraint[]
  replicas?: number
}

export interface GetTaskStatusResponse {
  task: TaskStatus
}

export interface ListTasksResponse {
  tasks: TaskStatus[]
}

// -- Workers --

export interface WorkerMetadata {
  hostname?: string
  ipAddress?: string
  cpuCount?: number
  memoryBytes?: string
  diskBytes?: string
  device?: DeviceConfig
  tpuName?: string
  tpuWorkerHostnames?: string
  tpuWorkerId?: string
  tpuChipsPerHostBounds?: string
  gpuCount?: number
  gpuName?: string
  gpuMemoryMb?: number
  gceInstanceName?: string
  gceZone?: string
  attributes?: Record<string, AttributeValue>
  vmAddress?: string
  gitHash?: string
}

export interface WorkerHealthStatus {
  workerId: string
  healthy: boolean
  consecutiveFailures?: number
  lastHeartbeat?: ProtoTimestamp
  runningJobIds?: string[]
  address?: string
  metadata?: WorkerMetadata
  statusMessage?: string
}

export interface ListWorkersResponse {
  workers: WorkerHealthStatus[]
}

export interface WorkerResourceSnapshot {
  timestamp?: ProtoTimestamp
  hostCpuPercent?: number
  memoryUsedBytes?: string
  memoryTotalBytes?: string
  diskUsedBytes?: string
  diskTotalBytes?: string
  runningTaskCount?: number
  totalProcessCount?: number
  netRecvBps?: string
  netSentBps?: string
}

export interface GetWorkerStatusResponse {
  vm?: VmInfo
  scaleGroup?: string
  worker?: WorkerHealthStatus
  bootstrapLogs?: string
  workerLogEntries?: LogEntry[]
  recentTasks?: TaskStatus[]
  currentResources?: WorkerResourceSnapshot
  resourceHistory?: WorkerResourceSnapshot[]
}

// -- Endpoints --

export interface EndpointInfo {
  endpointId?: string
  name: string
  address: string
  taskId?: string
  metadata?: Record<string, string>
}

export interface ListEndpointsResponse {
  endpoints: EndpointInfo[]
}

// -- Autoscaler --

export interface VmInfo {
  vmId: string
  sliceId?: string
  scaleGroup?: string
  state: string
  address?: string
  zone?: string
  createdAt?: ProtoTimestamp
  stateChangedAt?: ProtoTimestamp
  workerId?: string
  workerHealthy?: boolean
  initPhase?: string
  initLogTail?: string
  initError?: string
  labels?: Record<string, string>
}

export interface SliceInfo {
  sliceId: string
  scaleGroup?: string
  createdAt?: ProtoTimestamp
  vms?: VmInfo[]
  errorMessage?: string
  lastActive?: ProtoTimestamp
  idle?: boolean
}

export interface ScaleGroupConfig {
  quotaPool?: string
  allocationTier?: number
}

export interface ScaleGroupStatus {
  name: string
  config?: ScaleGroupConfig
  currentDemand?: number
  peakDemand?: number
  backoffUntil?: ProtoTimestamp
  consecutiveFailures?: number
  lastScaleUp?: ProtoTimestamp
  lastScaleDown?: ProtoTimestamp
  slices?: SliceInfo[]
  sliceStateCounts?: Record<string, number>
  availabilityStatus?: string
  availabilityReason?: string
  blockedUntil?: ProtoTimestamp
  scaleUpCooldownUntil?: ProtoTimestamp
  idleThresholdMs?: string
}

export interface AutoscalerAction {
  timestamp?: ProtoTimestamp
  actionType?: string
  scaleGroup?: string
  sliceId?: string
  reason?: string
  status?: string
}

export interface GroupRoutingStatus {
  group: string
  priority?: number
  assigned?: number
  launch?: number
  decision?: string
  reason?: string
}

export interface RoutingDecision {
  groupToLaunch?: Record<string, number>
  groupReasons?: Record<string, string>
  unmetEntries?: UnmetDemand[]
  groupStatuses?: GroupRoutingStatus[]
}

export interface UnmetDemand {
  entry?: DemandEntryStatus
  reason?: string
}

export interface DemandEntryStatus {
  taskIds?: string[]
  coscheduleGroupId?: string
  deviceType?: string
  deviceVariant?: string
  preemptible?: boolean
}

export interface AutoscalerStatus {
  groups?: ScaleGroupStatus[]
  currentDemand?: Record<string, number>
  lastEvaluation?: ProtoTimestamp
  recentActions?: AutoscalerAction[]
  lastRoutingDecision?: RoutingDecision
}

export interface GetAutoscalerStatusResponse {
  status: AutoscalerStatus
}

// -- Kubernetes Cluster Status --

export interface KubernetesPodStatus {
  podName: string
  taskId: string
  phase: string
  reason: string
  message: string
  lastTransition?: ProtoTimestamp
  nodeName?: string
}

export interface NodePoolStatus {
  name: string
  instanceType: string
  scaleGroup: string
  targetNodes: number
  currentNodes: number
  queuedNodes: number
  inProgressNodes: number
  autoscaling: boolean
  minNodes: number
  maxNodes: number
  capacity: string
  quota: string
}

export interface GetKubernetesClusterStatusResponse {
  namespace?: string
  totalNodes?: number
  schedulableNodes?: number
  allocatableCpu?: string
  allocatableMemory?: string
  podStatuses?: KubernetesPodStatus[]
  providerVersion?: string
  nodePools?: NodePoolStatus[]
}

// -- Users --

export interface UserSummary {
  user: string
  taskStateCounts?: Record<string, number>
  jobStateCounts?: Record<string, number>
}

export interface ListUsersResponse {
  users: UserSummary[]
}

// -- Logs --

export interface LogEntry {
  timestamp?: ProtoTimestamp
  source?: string
  data: string
  attemptId?: number
  level?: string
  key?: string
}

export interface FetchLogsResponse {
  entries: LogEntry[]
  cursor?: string
}

// -- Process Status --

export interface ProcessInfo {
  hostname?: string
  pid?: number
  pythonVersion?: string
  uptimeMs?: string
  memoryRssBytes?: string
  memoryVmsBytes?: string
  cpuMillicores?: number
  threadCount?: number
  openFdCount?: number
  memoryTotalBytes?: string
  cpuCount?: number
  gitHash?: string
}

export interface GetProcessStatusResponse {
  processInfo?: ProcessInfo
  logEntries?: LogEntry[]
}

// -- Transactions --

export interface TransactionAction {
  timestamp?: ProtoTimestamp
  action?: string
  entityId?: string
  details?: string
}

export interface GetTransactionsResponse {
  actions: TransactionAction[]
}

// -- Task State Counts (used in job summaries and user summaries) --

/** Mapping from lowercase state name to count, e.g. { running: 2, pending: 5 } */
export type TaskStateCounts = Record<string, number>

// -- Current User --

export interface GetCurrentUserResponse {
  userId: string
  role: string
  displayName?: string
}

// -- API Keys --

export interface ApiKeyInfo {
  keyId: string
  keyPrefix: string
  userId: string
  name: string
  createdAtMs: string
  lastUsedAtMs: string
  expiresAtMs: string
  revoked: boolean
}

export interface ListApiKeysResponse {
  keys: ApiKeyInfo[]
}

// -- Scheduler State --

export interface SchedulerTaskEntry {
  taskId: string
  jobId: string
  userId: string
  originalBand: string
  effectiveBand: string
  queuePosition: number
  resourceValue: number
}

export interface SchedulerBandGroup {
  band: string
  tasks: SchedulerTaskEntry[]
  totalInBand: number
}

export interface SchedulerUserBudget {
  userId: string
  budgetLimit: string
  budgetSpent: string
  maxBand: string
  effectiveBand: string
  utilizationPercent: number
}

export interface SchedulerRunningTask {
  taskId: string
  jobId: string
  userId: string
  workerId: string
  effectiveBand: string
  resourceValue: number
  preemptible: boolean
  preemptibleBy: string[]
  isCoscheduled: boolean
}

export interface GetSchedulerStateResponse {
  pendingQueue: SchedulerBandGroup[]
  userBudgets: SchedulerUserBudget[]
  runningTasks: SchedulerRunningTask[]
  totalPending: number
  totalRunning: number
}
