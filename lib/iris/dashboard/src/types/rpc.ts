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
  // Bounded terminal cause, set only on a failed attempt.
  terminalReason?: string
}

/** Why a failed attempt ended, empty when it did not fail. `terminalReason`
 *  wins because an init-container failure sends `error` as an empty string,
 *  which `??` would keep. */
export function attemptFailureReason(attempt: TaskAttempt): string {
  return attempt.terminalReason || attempt.error || ''
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
  // Worker-resident in-memory snapshot (Worker.GetTaskStatus only). Query the
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
  setupScripts?: string[]
  extras?: string[]
  pythonVersion?: string
  dockerfile?: string
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

export interface VmInfo {
  vmId: string
  sliceId?: string
  scaleGroup?: string
  state: string
  address?: string
  zone?: string
  createdAt?: ProtoTimestamp
  workerId?: string
  workerHealthy?: boolean
  usability?: string
  runningTaskCount?: number
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

// -- Multi-backend --

/** Lightweight backend descriptor from /auth/config `backends` array. */
export interface BackendInfo {
  id: string
  name: string
  capabilities: string[]
}

// -- Typed ResourceService --

export interface ResourceKey {
  clusterId: string
  kind: string
  resourceId: string
}

export interface ResourceSourceStatus {
  sourceId: string
  backendId?: string
  state: string
  freshness: string
  observedAt?: ProtoTimestamp
  errorCode?: string
  errorMessage?: string
}

export interface ResourcePageInfo {
  nextPageToken?: string
  sourceStatuses?: ResourceSourceStatus[]
}

export interface ResourceJobIdentity {
  key: ResourceKey
  jobUid: string
}

export interface ResourceTaskIdentity {
  key: ResourceKey
  taskUid: string
}

export interface ResourceAttemptIdentity {
  task: ResourceKey
  attemptNumber: number
  attemptUid: string
}

export interface ResourceNodeIdentity {
  key: ResourceKey
  backendId: string
  nodeUid: string
}

export interface ResourceSliceIdentity {
  key: ResourceKey
  backendId: string
  sliceUid: string
}

export interface ResourceJobSummary {
  identity: ResourceJobIdentity
  ownerId: string
  parent?: ResourceJobIdentity
  state: string
  executionClusterId: string
  backendId: string
  numTasks: number
  submittedAt?: ProtoTimestamp
  startedAt?: ProtoTimestamp
  finishedAt?: ProtoTimestamp
  errorMessage?: string
  pendingReason?: string
}

export interface ResourceJobDetail {
  summary: ResourceJobSummary
  spec: {
    name: string
    resources?: ResourceSpecProto
    entrypoint?: RuntimeEntrypoint
    environment?: EnvironmentConfig
    constraints?: Constraint[]
    bundleId?: string
    replicas?: number
    priorityBand?: string
    taskImage?: string
    maxTaskFailures?: number
    maxRetriesFailure?: number
    maxRetriesPreemption?: number
    ports?: string[]
    submitArgv?: string[]
    clientRevisionDate?: string
    containerProfile?: string
  }
}

export interface ResourceTaskSummary {
  identity: ResourceTaskIdentity
  job: ResourceJobIdentity
  taskIndex: number
  state: string
  executionClusterId: string
  backendId: string
  currentAttempt?: ResourceAttemptIdentity
  currentNode?: ResourceNodeIdentity
  failureCount: number
  preemptionCount: number
  submittedAt?: ProtoTimestamp
  startedAt?: ProtoTimestamp
  finishedAt?: ProtoTimestamp
  statusMessage?: string
  errorMessage?: string
}

export interface ResourceAttemptSummary {
  identity: ResourceAttemptIdentity
  state: string
  executionClusterId: string
  backendId: string
  node?: ResourceNodeIdentity
  createdAt?: ProtoTimestamp
  startedAt?: ProtoTimestamp
  finishedAt?: ProtoTimestamp
  exitCode?: number
  errorMessage?: string
  terminalReason?: string
}

export interface AttemptRuntimeObject {
  providerKind: string
  namespace?: string
  name?: string
  providerUid?: string
  providerNodeId?: string
  providerNodeUid?: string
  containerId?: string
  observedAt?: ProtoTimestamp
}

export interface ResourceTaskDetail {
  summary: ResourceTaskSummary
  attempts?: ResourceAttemptSummary[]
  sourceStatuses?: ResourceSourceStatus[]
  rootCauseHighlights?: string[]
}

export interface ResourceAttemptDetail {
  summary: ResourceAttemptSummary
  runtime?: AttemptRuntimeObject
  sourceStatuses?: ResourceSourceStatus[]
}

export interface ResourceListJobsResponse {
  jobs?: ResourceJobSummary[]
  page?: ResourcePageInfo
}

export interface ResourceUserSummary {
  userId: string
  taskStateCounts?: Record<string, number>
  jobStateCounts?: Record<string, number>
  role?: string
  budgetLimit?: string
  budgetSpent?: string
  maxBand?: number
  budgetConfigured?: boolean
}

export interface ResourceListUsersResponse {
  users?: ResourceUserSummary[]
}

export interface ResourceDescribeJobResponse {
  job?: ResourceJobDetail
}

export interface ResourceListTasksResponse {
  tasks?: ResourceTaskSummary[]
  page?: ResourcePageInfo
}

export interface ResourceDescribeTaskResponse {
  task?: ResourceTaskDetail
}

export interface ResourceDescribeAttemptResponse {
  attempt?: ResourceAttemptDetail
}

export interface ResourceNodeCapacity {
  cpuMillicores?: string
  memoryBytes?: string
  diskBytes?: string
  acceleratorKind?: string
  acceleratorVariant?: string
  acceleratorCount?: number
}

export interface ResourceNodeSummary {
  identity: ResourceNodeIdentity
  health: string
  schedulable: boolean
  capacity?: ResourceNodeCapacity
  scalingGroupId?: string
  slice?: ResourceSliceIdentity
  runningTaskCount: number
  observedAt?: ProtoTimestamp
  region?: string
}

export interface ResourceNodeAttribute {
  key: string
  stringValue?: string
  integerValue?: string
  floatValue?: number
}

export interface ResourceNodeDetail {
  summary: ResourceNodeSummary
  address?: string
  attributes?: ResourceNodeAttribute[]
  recentAttempts?: ResourceAttemptSummary[]
  bootstrapLogs?: string
  sourceStatuses?: ResourceSourceStatus[]
}

export interface ResourceListNodesResponse {
  nodes?: ResourceNodeSummary[]
  page?: ResourcePageInfo
}

export interface ResourceDescribeNodeResponse {
  node?: ResourceNodeDetail
}

export interface ResourceSliceSummary {
  identity: ResourceSliceIdentity
  scalingGroupId: string
  lifecycle: string
  membershipState: string
  observedMemberCount: number
  observedAt?: ProtoTimestamp
  errorMessage?: string
  createdAt?: ProtoTimestamp
  lastActiveAt?: ProtoTimestamp
  capacityState?: string
  healthyMemberCount?: number
  degradedMemberCount?: number
  runningTaskCount?: number
}

export interface ResourceListSlicesResponse {
  slices?: ResourceSliceSummary[]
  page?: ResourcePageInfo
}

export interface ResourceSliceMember {
  providerNodeId: string
  node?: ResourceNodeIdentity
  observedAt?: ProtoTimestamp
  workerId?: string
  healthy?: boolean
  usability?: string
  runningTaskCount?: number
  zone?: string
}

export interface ResourceSliceDetail {
  summary: ResourceSliceSummary
  members?: ResourceSliceMember[]
  sourceStatuses?: ResourceSourceStatus[]
}

export interface ResourceDescribeSliceResponse {
  slice?: ResourceSliceDetail
}

export interface ResourceCapacityAvailability {
  version?: number
  observedAt?: ProtoTimestamp
  amounts?: Record<string, string>
  totalAmounts?: Record<string, string>
  heldByBand?: Array<{ band: number; amounts?: Record<string, string> }>
}

export interface ResourceCapacitySlice {
  summary: ResourceSliceSummary
  members?: ResourceSliceMember[]
}

export interface ResourceCapacityScalingGroup {
  name: string
  backendId: string
  deviceType?: string
  deviceVariant?: string
  quotaPool?: string
  allocationTier?: number
  region?: string
  currentDemand?: number
  peakDemand?: number
  backoffUntil?: ProtoTimestamp
  consecutiveFailures?: number
  lastScaleUp?: ProtoTimestamp
  lastScaleDown?: ProtoTimestamp
  slices?: ResourceCapacitySlice[]
  sliceStateCounts?: Record<string, number>
  availabilityStatus?: string
  availabilityReason?: string
  blockedUntil?: ProtoTimestamp
  scaleUpCooldownUntil?: ProtoTimestamp
  idleThresholdMs?: string
}

export interface ResourceCapacityAction {
  timestamp?: ProtoTimestamp
  actionType?: string
  scalingGroupId?: string
  sliceId?: string
  reason?: string
  status?: string
}

export interface ResourceCapacityDemandEntry {
  taskIds?: string[]
  coscheduleGroupId?: string
  deviceType?: string
  deviceVariant?: string
  preemptible?: boolean
}

export interface ResourceCapacityUnmetDemand {
  entry?: ResourceCapacityDemandEntry
  reason?: string
}

export interface ResourceCapacityGroupRouting {
  scalingGroupId: string
  priority?: number
  assigned?: number
  launch?: number
  decision?: string
  reason?: string
}

export interface ResourceCapacityRouting {
  unmet?: ResourceCapacityUnmetDemand[]
  groups?: ResourceCapacityGroupRouting[]
}

export interface ResourceCapacityKubernetesPool {
  name: string
  instanceType?: string
  scalingGroupId?: string
  targetNodes?: number
  currentNodes?: number
  queuedNodes?: number
  inProgressNodes?: number
  autoscaling?: boolean
  minNodes?: number
  maxNodes?: number
  capacity?: string
  quota?: string
}

export interface ResourceCapacityKubernetesPod {
  podName: string
  taskId?: string
  phase?: string
  reason?: string
  message?: string
  lastTransition?: ProtoTimestamp
  nodeName?: string
}

export interface ResourceCapacityKubernetesNode {
  name: string
  ready?: boolean
  schedulable?: boolean
  statusSummary?: string
  instanceType?: string
  region?: string
  acceleratorCount?: number
  acceleratorVariant?: string
  cpuMillicores?: string
  memoryBytes?: string
  diskBytes?: string
  runningPods?: number
  created?: string
}

export interface ResourceCapacityKubernetesStatus {
  namespace?: string
  totalNodes?: number
  schedulableNodes?: number
  allocatableCpu?: string
  allocatableMemory?: string
  providerVersion?: string
  pods?: ResourceCapacityKubernetesPod[]
  pools?: ResourceCapacityKubernetesPool[]
  nodes?: ResourceCapacityKubernetesNode[]
}

export interface ResourceCapacityBackend {
  backendId: string
  name: string
  kind: string
  capabilities?: string[]
  advertisedAttributes?: Record<string, { values?: string[] }>
  workerCount?: number
  pendingTaskCount?: number
  runningTaskCount?: number
  hasAutoscaler?: boolean
  capacityHealth?: Record<string, number>
  availability?: ResourceCapacityAvailability
  scalingGroups?: ResourceCapacityScalingGroup[]
  recentActions?: ResourceCapacityAction[]
  routing?: ResourceCapacityRouting
  lastEvaluation?: ProtoTimestamp
  healthyWorkerCount?: number
  kubernetes?: ResourceCapacityKubernetesStatus
}

export interface ResourceCapacityPeerBackend {
  backendId: string
  name: string
  kind: string
  capabilities?: string[]
  advertisedAttributes?: Record<string, { values?: string[] }>
  scalingGroups?: string[]
  workerCount?: number
  pendingTaskCount?: number
  runningTaskCount?: number
  hasAutoscaler?: boolean
  capacityHealth?: Record<string, number>
  availability?: ResourceCapacityAvailability
}

export interface ResourceCapacityPeer {
  peerId: string
  controllerAddress?: string
  reachable?: boolean
  lastContactMs?: string
  activeFederatedJobs?: number
  backends?: ResourceCapacityPeerBackend[]
}

export interface ResourceCapacityPlacement {
  backendId: string
  workerId: string
  jobId: string
  userId: string
  taskCount: number
}

export interface ResourceCapacityUnroutableJob {
  jobId: string
  reason: string
}

export interface ResourceGetCapacityStatusResponse {
  backends?: ResourceCapacityBackend[]
  peers?: ResourceCapacityPeer[]
  runningPlacements?: ResourceCapacityPlacement[]
  unroutableJobCount?: number
  unroutableJobs?: ResourceCapacityUnroutableJob[]
  sourceStatuses?: ResourceSourceStatus[]
}

export interface ResourceEndpointSummary {
  key: ResourceKey
  endpointId: string
  name: string
  task?: ResourceKey
  executionClusterId: string
  access: string
  leaseDeadline?: ProtoTimestamp
}

export interface ResourceEndpointDetail {
  summary: ResourceEndpointSummary
  address: string
  metadata?: Record<string, string>
}

export interface ResourceListEndpointsResponse {
  endpoints?: ResourceEndpointSummary[]
  page?: ResourcePageInfo
}

export interface ResourceDescribeEndpointResponse {
  endpoint?: ResourceEndpointDetail
}

export interface ResourceActivityEntry {
  entryId: string
  occurredAt?: ProtoTimestamp
  source: string
  severity: string
  kind: string
  message: string
  target: ResourceKey
  attemptUid?: string
  correlationId?: string
  attributes?: Record<string, string>
}

export interface ResourceListActivityResponse {
  entries?: ResourceActivityEntry[]
  page?: ResourcePageInfo
}

export interface ResourceActionReceipt {
  actionId: string
  kind: string
  target: ResourceKey
  expectedTargetUid: string
  expectedAttemptUid?: string
  expectedAttemptNumber?: number
  state: string
  resultCode: string
  resultMessage?: string
  createdAt?: ProtoTimestamp
  updatedAt?: ProtoTimestamp
  completedAt?: ProtoTimestamp
}

export interface ResourceActionResponse {
  receipt?: ResourceActionReceipt
}
