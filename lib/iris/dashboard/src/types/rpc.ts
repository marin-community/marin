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

// -- Cluster coordinate --
//
// Mirrors iris.cluster.types: every job/task carries a `cluster` coordinate that
// is always set — `'local'` for a locally-owned row, a peer id when handed off.
// `'local'` is a reserved sentinel, not a real cluster id. The helpers tolerate
// an absent value (contexts without a cluster, e.g. worker/controller logs) as
// local, so a naive truthiness check never misclassifies a local row.

export const LOCAL_CLUSTER = 'local'

/** True when the row is locally owned (`'local'`, or no cluster in context). */
export function isLocal(cluster: string | undefined): boolean {
  return !cluster || cluster === LOCAL_CLUSTER
}

/** True when the row was handed off to a peer cluster. */
export function isFederated(cluster: string | undefined): boolean {
  return !!cluster && cluster !== LOCAL_CLUSTER
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
  attemptUid?: string
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
  // Worker-resident in-memory snapshot (Worker.GetTaskStatus only). The
  // controller-served TaskStatus carries no resourceUsage; query the
  // iris.task stats namespace via useLogServerStatsRpc for time series.
  resourceUsage?: ResourceUsage
  buildMetrics?: BuildMetrics
  currentAttemptId?: number
  attempts?: TaskAttempt[]
  pendingReason?: string
  // Human-readable status for a task waiting to run (e.g. the Kueue admission
  // detail explaining why a BUILDING/pending k8s task has not been placed).
  statusMessage?: string
  canBeScheduled?: boolean
  containerId?: string
  // No per-task failure/preemption count fields — derive them from `attempts`;
  // job-level totals live on JobStatus.
  backendId?: string
  // Cluster coordinate: always set — `'local'` for a locally-owned task, a peer
  // id when handed off to that peer cluster (backendId then empty).
  cluster?: string
  // Task submission time on the owning cluster. Absent (not epoch 0) for a
  // mirrored federated task the peer has not yet reported a real submit time for.
  submittedAt?: ProtoTimestamp
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
  parentJobId?: string
  backendId?: string
  // Cluster coordinate: always set — `'local'` for a locally-owned job, a peer
  // id when handed off to that peer cluster.
  cluster?: string
  // Handoff lifecycle for a federated job (gate on `cluster` first — a local job
  // and an old message both read as PEER_STATUS_NONE). One of PEER_STATUS_NONE |
  // PEER_STATUS_PENDING_SCHEDULING | PEER_STATUS_ASSIGNED | PEER_STATUS_SYNCED |
  // PEER_STATUS_REJECTED. This is the job's handoff state, not peer health.
  peerStatus?: string
}

export interface JobQuery {
  scope?: string
  parentJobId?: string
  nameFilter?: string
  stateFilter?: string
  sortField?: string
  sortDirection?: string
  offset?: number
  limit?: number
  // Anchored prefix match against the full wire job_id (e.g. "/alice/").
  jobIdPrefix?: string
  backendId?: string
  // Filter to jobs in one cluster (`'local'` or a peer id). Unset = all clusters.
  cluster?: string
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

export interface CommandEntrypoint {
  argv?: string[]
}

export interface RuntimeEntrypoint {
  setupCommands?: string[]
  runCommand?: CommandEntrypoint
  workdirFiles?: Record<string, string>
  workdirFileRefs?: Record<string, string>
}

export interface EnvironmentConfig {
  pipPackages?: string[]
  envVars?: Record<string, string>
  extras?: string[]
  pythonVersion?: string
  dockerfile?: string
}

export interface LaunchJobRequest {
  name: string
  entrypoint?: RuntimeEntrypoint
  environment?: EnvironmentConfig
  resources?: ResourceSpecProto
  constraints?: Constraint[]
  ports?: string[]
  bundleId?: string
  replicas?: number
  priorityBand?: string
  submitArgv?: string[]
  // Job aborts once more than this many tasks fail terminally (default 0).
  maxTaskFailures?: number
  // Per-task retry budget on failure (default 0) and on preemption.
  maxRetriesFailure?: number
  maxRetriesPreemption?: number
}

export interface GetTaskStatusResponse {
  task: TaskStatus
  jobResources?: ResourceSpecProto
  /** Likely root-cause log lines distilled from a failed task's logs. */
  rootCauseHighlights?: string[]
}

export interface ListTasksResponse {
  tasks: TaskStatus[]
}

// -- Workers --

export interface Provenance {
  treeHash?: string
  baseCommit?: string
  dirty?: boolean
  branch?: string
  builtBy?: string
}

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
  provenance?: Provenance
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
  backendId?: string
  scaleGroup?: string
}

export interface WorkerQuery {
  contains?: string
  sortField?: string
  sortDirection?: string
  offset?: number
  limit?: number
  backendId?: string
}

export interface ListWorkersResponse {
  workers: WorkerHealthStatus[]
  totalCount: number
  hasMore: boolean
}

export interface WorkerTaskAttempt {
  taskId: string
  attempt?: TaskAttempt
  // Static allocation inherited from the parent job; unset when no request.
  resources?: ResourceSpecProto
}

export interface GetWorkerStatusResponse {
  vm?: VmInfo
  scaleGroup?: string
  worker?: WorkerHealthStatus
  bootstrapLogs?: string
  // workerLogEntries removed from this response to avoid blocking the worker
  // page render on a slow LogService proxy. Fetched separately via
  // LogService.FetchLogs(source="/system/worker/<worker_id>").
  recentAttempts?: WorkerTaskAttempt[]
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
  /** WorkerUsability: "healthy" | "degraded" | "dead"; empty if not in the roster. */
  usability?: string
  initPhase?: string
  initLogTail?: string
  initError?: string
  /** Number of tasks currently assigned to this VM by the scheduler. */
  runningTaskCount?: number
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
  /**
   * Authoritative slice lifecycle state from the autoscaler:
   * "requesting" | "booting" | "initializing" | "ready" | "failed".
   * Render this directly (via sliceLifecycle()); do NOT infer state from `vms`,
   * which is empty until a slice's workers register — a booting slice has none.
   */
  state?: string
  /** Count of DEGRADED (reachable-but-failing) hosts among `vms`, for detail display. */
  degradedSlotCount?: number
  /**
   * Server-derived placement status of a ready slice: "available" | "in_use" |
   * "idle" | "degraded". Empty for non-ready slices.
   */
  capacityStatus?: string
}

export interface ScaleGroupStatus {
  name: string
  backendId?: string
  deviceType?: string
  deviceVariant?: string
  quotaPool?: string
  allocationTier?: number
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
  // Config-derived role from the controller's in-memory RolePolicy.
  role?: string
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
  /** Store row id, ascending in write order. int64, so proto JSON sends a string. */
  seq?: string
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
  provenance?: Provenance
}

export interface GetProcessStatusResponse {
  processInfo?: ProcessInfo
  logEntries?: LogEntry[]
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

// -- Scheduler State --

/** Aggregated pending-task count keyed by (band, user, job). */
export interface PendingTaskBucket {
  band: string
  userId: string
  jobId: string
  count: number
  backendId?: string
}

/** Aggregated running-task count keyed by (band, user, worker, job). */
export interface RunningTaskBucket {
  band: string
  userId: string
  workerId: string
  jobId: string
  count: number
  backendId?: string
}

export interface SchedulerUserBudget {
  userId: string
  budgetLimit: string
  budgetSpent: string
  maxBand: string
  effectiveBand: string
  utilizationPercent: number
}

export interface GetSchedulerStateResponse {
  userBudgets: SchedulerUserBudget[]
  totalPending: number
  totalRunning: number
  pendingBuckets: PendingTaskBucket[]
  runningBuckets: RunningTaskBucket[]
}

// -- RPC Statistics (iris.stats.StatsService) --

export interface RpcMethodStats {
  method: string
  count?: string
  errorCount?: string
  totalDurationMs?: number
  maxDurationMs?: number
  p50Ms?: number
  p95Ms?: number
  p99Ms?: number
  bucketUpperBoundsMs?: string[]
  bucketCounts?: string[]
  lastCall?: ProtoTimestamp
}

export interface RpcCallSample {
  method: string
  timestamp?: ProtoTimestamp
  durationMs?: number
  peer?: string
  userAgent?: string
  caller?: string
  errorCode?: string
  errorMessage?: string
  requestPreview?: string
}

export interface GetRpcStatsResponse {
  methods?: RpcMethodStats[]
  slowSamples?: RpcCallSample[]
  discoverySamples?: RpcCallSample[]
  collectorStartedAt?: ProtoTimestamp
}

// -- Multi-backend --

/** Lightweight backend descriptor from /auth/config `backends` array. */
export interface BackendInfo {
  id: string
  name: string
  capabilities: string[]
}

/** Worker-daemon fleet detail: the backend's autoscaler view plus DB-derived health counts. */
export interface WorkerFleetDetail {
  autoscaler?: AutoscalerStatus
  healthyWorkerCount?: number
  totalWorkerCount?: number
}

/** Backend-authored expanded status; exactly one variant is set per the backend's capability. */
export interface BackendStatus {
  kubernetes?: GetKubernetesClusterStatusResponse
  worker?: WorkerFleetDetail
}

/** Free vs. total consumable capacity per resource token (lowercased
 *  device-variant → chips). map<string,int64> values JSON-encode as strings. */
export interface ResourceAvailability {
  version?: number
  /** When the serving cluster computed the amounts, ms since epoch. int64 → string. */
  observationEpochMs?: string
  /** Free chips per variant, e.g. { h100: "24" }. */
  amounts?: Record<string, string>
  /** Total chips per variant over the same capacity; absent on a peer that
   *  predates the field. */
  totalAmounts?: Record<string, string>
}

/** Per-backend summary returned by the ListBackends RPC. */
export interface BackendSummary {
  backendId: string
  name: string
  kind: string
  capabilities: string[]
  /** Map of attribute key → list of string values. */
  advertisedAttributes: Record<string, { values: string[] }>
  scaleGroups: string[]
  workerCount: number
  pendingTaskCount: number
  runningTaskCount: number
  hasAutoscaler: boolean
  /** availability_status string → pool count. */
  capacityHealth: Record<string, number>
  /** Expanded per-backend status rendered in the Backends tab detail panel. */
  detail?: BackendStatus
  /** Free/total capacity metric; unset when the backend does not supply it. */
  availability?: ResourceAvailability
}

export interface UnroutableJob {
  jobId: string
  reason: string
}

export interface ListBackendsResponse {
  backends: BackendSummary[]
  unroutableJobCount: number
  unroutableSample: UnroutableJob[]
}

// -- Federation peers --

/** A federation peer returned by the ListPeers RPC: a remote Iris controller
 *  this cluster may hand whole jobs to, plus its forwarded backend topology. */
export interface PeerSummary {
  peerId: string
  // proto3 JSON omits default-valued fields, so string/bool/repeated fields are
  // absent on the wire when empty — hence optional here.
  controllerAddress?: string
  /** Last capability heartbeat succeeded. */
  reachable?: boolean
  /** Last successful contact, ms since epoch (0/absent if never contacted). int64 → string. */
  lastContactMs?: string
  activeFederatedJobs?: number
  /** Aggregate spend across this peer's federated jobs, micros. int64 → string. */
  aggregateSpendMicros?: string
  /** The peer's own backends, forwarded from its ListBackends. */
  backends?: BackendSummary[]
}

export interface ListPeersResponse {
  peers: PeerSummary[]
}
