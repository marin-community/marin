<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import { controllerRpcCall, useLogServerStatsRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { stateToName, stateDisplayName } from '@/types/status'
import { useBackends } from '@/composables/useBackends'
import { useCopyToClipboard } from '@/composables/useCopyToClipboard'
import {
  LOCAL_CLUSTER, isFederated,
  type JobStatus, type TaskStatus, type LaunchJobRequest, type JobQuery,
  type GetJobStatusResponse, type GetTaskStatusResponse, type ListTasksResponse, type ListJobsResponse,
  type EndpointInfo, type ListEndpointsResponse,
} from '@/types/rpc'
import { timestampMs, formatTimestamp, formatDuration, formatRelativeTime, formatBytes, formatCpuMillicores, formatDeviceConfig, bandDisplayName, bandColor } from '@/utils/formatting'
import { decodeArrowIpc } from '@/utils/arrow'
import { getLeafJobName } from '@/utils/jobTree'
import { batchSummarySql } from '@/utils/taskStatus'
import { openSpeedscopeWindow } from '@/utils/speedscope'
import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import LogViewer from '@/components/shared/LogViewer.vue'
import MarkdownRenderer from '@/components/shared/MarkdownRenderer.vue'
import EndpointLink from '@/components/shared/EndpointLink.vue'
import ClusterLink from '@/components/shared/ClusterLink.vue'
import { useMediaQuery } from '@/composables/useMediaQuery'

// Tailwind's `sm` breakpoint is 640px. Cards on mobile, table on desktop.
// v-if-switched (not CSS-hidden) so only one variant is in the DOM.
const isMobile = useMediaQuery('(max-width: 639px)')

const props = defineProps<{
  jobId: string
}>()

const route = useRoute()
const { multiBackend, peers, ensurePeers } = useBackends()

const TERMINAL_STATES = new Set(['succeeded', 'failed', 'killed', 'worker_failed', 'cosched_failed', 'preempted', 'unschedulable'])
const FAILED_TERMINAL_STATES = new Set(['failed', 'worker_failed', 'cosched_failed', 'preempted', 'unschedulable'])

// -- State --

const job = ref<JobStatus | null>(null)
const jobRequest = ref<LaunchJobRequest | null>(null)
const tasks = ref<TaskStatus[]>([])
// Distilled log highlights for the failed job's root-cause task (see below).
const rootCauseHighlights = ref<string[]>([])
// Endpoints registered by this job's tasks, grouped by wire-format task id.
const endpointsByTask = ref<Map<string, EndpointInfo[]>>(new Map())
const childJobsByParent = ref<Map<string, JobStatus[]>>(new Map())
const expandedChildJobs = ref<Set<string>>(new Set())
const loadingChildJobs = ref<Set<string>>(new Set())
const loading = ref(true)
const error = ref<string | null>(null)
const profilingTaskId = ref<string | null>(null)
const { copied: copiedName, copy: copyToClipboard } = useCopyToClipboard()
const taskSearch = ref('')
const stateFilter = ref('')

type SortColumn = 'task' | 'state' | 'mem' | 'cpu' | 'duration'
type SortDir = 'asc' | 'desc'

const sortColumn = ref<SortColumn | null>(null)
const sortDir = ref<SortDir>('asc')

type ChildSortColumn = 'name' | 'state' | 'duration'
const childSortColumn = ref<ChildSortColumn | null>(null)
const childSortDir = ref<SortDir>('asc')

function toggleSort(col: SortColumn) {
  if (sortColumn.value === col) {
    if (sortDir.value === 'asc') sortDir.value = 'desc'
    else { sortColumn.value = null; sortDir.value = 'asc' }
  } else {
    sortColumn.value = col
    sortDir.value = 'asc'
  }
}

function toggleChildSort(col: ChildSortColumn) {
  if (childSortColumn.value === col) {
    if (childSortDir.value === 'asc') childSortDir.value = 'desc'
    else { childSortColumn.value = null; childSortDir.value = 'asc' }
  } else {
    childSortColumn.value = col
    childSortDir.value = 'asc'
  }
}

function copyJobName() {
  const name = job.value?.name
  if (!name) return
  copyToClipboard(name)
}

// -- Fetch --

let fetchGeneration = 0

async function fetchChildJobs(parentJobId: string): Promise<JobStatus[]> {
  const response = await controllerRpcCall<ListJobsResponse>('ListJobs', {
    query: {
      scope: 'JOB_QUERY_SCOPE_CHILDREN',
      parentJobId,
    } satisfies JobQuery,
  })
  return response.jobs ?? []
}

// --- Per-task resource samples sourced from finelog stats (iris.task) ---
//
// Latest sample per task_id, scoped to this job's tasks. Drives the
// Memory / Peak / CPU columns and their sort comparators. Empty until the
// stats query lands. The controller no longer populates
// TaskStatus.resource_usage, so this is the canonical source.
//
// The underlying iris.task columns are stored in MB (matching the
// ResourceUsage proto), but the SQL projection converts to bytes so the
// dashboard can call formatBytes directly without a per-call MB→B helper.
interface TaskStatRow {
  task_id?: string
  attempt_id?: number
  cpu_millicores?: number
  // Memory in bytes — converted in the SQL projection so the dashboard can call
  // formatBytes directly without re-multiplying at every render site.
  memory_bytes?: number
  memory_peak_bytes?: number
}

function buildTaskStatsSql(taskIds: readonly string[]): string {
  if (taskIds.length === 0) return ''
  // QueryRequest has no param binding; manual DuckDB single-quote escape.
  const list = taskIds.map(t => `'${t.replace(/'/g, "''")}'`).join(',')
  return `
SELECT
  task_id,
  attempt_id,
  cpu_millicores,
  memory_mb * 1024 * 1024 AS memory_bytes,
  memory_peak_mb * 1024 * 1024 AS memory_peak_bytes
FROM "iris.task"
WHERE task_id IN (${list})
QUALIFY row_number() OVER (PARTITION BY task_id ORDER BY ts DESC) = 1
`.trim()
}

const { data: taskStatsData, refresh: fetchTaskStats } = useLogServerStatsRpc<{ arrowIpc?: string }>(
  'Query',
  () => ({ sql: buildTaskStatsSql(tasks.value.map(t => t.taskId)) }),
)

const taskUsageMap = computed<Map<string, TaskStatRow>>(() => {
  const ipc = taskStatsData.value?.arrowIpc
  const m = new Map<string, TaskStatRow>()
  if (!ipc) return m
  const rows = decodeArrowIpc(ipc).rows as TaskStatRow[]
  for (const r of rows) {
    if (r.task_id) m.set(r.task_id, r)
  }
  return m
})

function taskMemBytes(taskId: string): number {
  return Number(taskUsageMap.value.get(taskId)?.memory_bytes ?? 0)
}
function taskPeakMemBytes(taskId: string): number {
  return Number(taskUsageMap.value.get(taskId)?.memory_peak_bytes ?? 0)
}
function taskCpuMillicores(taskId: string): number {
  return Number(taskUsageMap.value.get(taskId)?.cpu_millicores ?? 0)
}

// Batched status text query (see batchSummarySql in utils/taskStatus).
interface StatusTextRow {
  task_id?: string
  status_text_summary_md?: string
}

const { data: statusTextData, refresh: fetchStatusText } = useLogServerStatsRpc<{ arrowIpc?: string }>(
  'Query',
  () => ({ sql: batchSummarySql(tasks.value.map(t => t.taskId)) }),
)

const statusTextMap = computed<Map<string, string>>(() => {
  const ipc = statusTextData.value?.arrowIpc
  const m = new Map<string, string>()
  if (!ipc) return m
  const rows = decodeArrowIpc(ipc).rows as StatusTextRow[]
  for (const r of rows) {
    if (r.task_id) m.set(r.task_id, r.status_text_summary_md ?? '')
  }
  return m
})

function taskStatusTextSummary(taskId: string): string {
  return statusTextMap.value.get(taskId) ?? ''
}

// True when the task has finished with a non-zero exit code. Drives the
// merged status column: non-zero exits get prominent display; zero exits
// fall back to status text or the state badge.
function taskExitNonZero(t: TaskStatus): boolean {
  return TERMINAL_STATES.has(stateToName(t.state))
    && t.exitCode !== undefined
    && t.exitCode !== 0
}

// Min/max of the latest stat sample across currently-running tasks. Powers
// the "Live Resource Usage" panel; null when no running task has reported.
interface RunningResourceRange {
  cpuMillicoresMin: number
  cpuMillicoresMax: number
  memoryBytesMin: number
  memoryBytesMax: number
  memoryPeakBytesMax: number
}
const runningResourceRange = computed<RunningResourceRange | null>(() => {
  const samples: TaskStatRow[] = []
  for (const t of tasks.value) {
    if (stateToName(t.state) !== 'running') continue
    const r = taskUsageMap.value.get(t.taskId)
    if (r) samples.push(r)
  }
  if (samples.length === 0) return null
  const cpus = samples.map(r => Number(r.cpu_millicores ?? 0))
  const mems = samples.map(r => Number(r.memory_bytes ?? 0))
  const peaks = samples.map(r => Number(r.memory_peak_bytes ?? 0))
  return {
    cpuMillicoresMin: Math.min(...cpus),
    cpuMillicoresMax: Math.max(...cpus),
    memoryBytesMin: Math.min(...mems),
    memoryBytesMax: Math.max(...mems),
    memoryPeakBytesMax: Math.max(...peaks),
  }
})

async function fetchEndpoints(gen: number) {
  const taskIds = tasks.value.map(t => t.taskId)
  if (taskIds.length === 0) {
    endpointsByTask.value = new Map()
    return
  }
  // Endpoint links are an enhancement; a fetch failure should never block the
  // task table, so swallow the error and leave the prior grouping in place.
  try {
    const resp = await controllerRpcCall<ListEndpointsResponse>('ListEndpoints', { taskIds })
    if (gen !== fetchGeneration) return  // superseded by a newer fetchData()
    const grouped = new Map<string, EndpointInfo[]>()
    for (const ep of resp.endpoints ?? []) {
      if (!ep.taskId) continue
      const list = grouped.get(ep.taskId)
      if (list) list.push(ep)
      else grouped.set(ep.taskId, [ep])
    }
    endpointsByTask.value = grouped
  } catch (e) {
    console.warn('ListEndpoints failed', e)
  }
}

function taskEndpoints(taskId: string): EndpointInfo[] {
  return endpointsByTask.value.get(taskId) ?? []
}

// The desktop table drops the wide, usually-empty Status column and the cramped
// per-task endpoint stack into a subtle second row rendered only when there is
// something to show: a non-zero exit, a live status summary, a failure error, or
// registered endpoints.
function taskHasStatusDetail(t: TaskStatus): boolean {
  if (taskExitNonZero(t)) return true
  const name = stateToName(t.state)
  if (taskStatusTextSummary(t.taskId) && !TERMINAL_STATES.has(name)) return true
  return Boolean(t.error) && FAILED_TERMINAL_STATES.has(name)
}

function taskHasDetailRow(t: TaskStatus): boolean {
  return taskHasStatusDetail(t) || taskEndpoints(t.taskId).length > 0
}

async function fetchData() {
  const gen = ++fetchGeneration
  error.value = null
  try {
    const [jobResp, tasksResp] = await Promise.all([
      controllerRpcCall<GetJobStatusResponse>('GetJobStatus', { jobId: props.jobId }),
      controllerRpcCall<ListTasksResponse>('ListTasks', { jobId: props.jobId }),
    ])
    if (gen !== fetchGeneration) return  // superseded by a newer fetchData()
    if (!jobResp.job) {
      error.value = 'Job not found'
      return
    }
    job.value = jobResp.job
    jobRequest.value = jobResp.request ?? null
    tasks.value = tasksResp.tasks ?? []

    // Surface the crash for a failed job: one on-demand fetch for the single
    // root-cause task, gated so a healthy or still-running job pays nothing.
    if (jobResp.job.error && TERMINAL_STATES.has(stateToName(jobResp.job.state))) {
      void fetchRootCauseHighlights(gen)
    } else {
      rootCauseHighlights.value = []
    }

    // Refresh stats only once tasks are known so the SQL filter targets the
    // current job's tasks. Failures here surface as zero values — never block
    // the rest of the page.
    if (tasks.value.length > 0) {
      void fetchTaskStats()
      void fetchStatusText()
      void fetchEndpoints(gen)
    } else {
      endpointsByTask.value = new Map()
    }

    const parentIds = [props.jobId, ...expandedChildJobs.value]
    const childEntries = await Promise.all(
      parentIds.map(async parentJobId => [parentJobId, await fetchChildJobs(parentJobId)] as const),
    )
    if (gen !== fetchGeneration) return
    childJobsByParent.value = new Map(childEntries)
  } catch (e) {
    if (gen !== fetchGeneration) return  // superseded by a newer fetchData()
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (gen === fetchGeneration) {
      loading.value = false
    }
  }
}


onMounted(fetchData)
// Federation roster (for a handed-off job's peer reachability); lazy, shared.
onMounted(() => void ensurePeers())

// Auto-refresh while job is not terminal
const isTerminal = computed(() => {
  if (!job.value) return false
  return TERMINAL_STATES.has(stateToName(job.value.state))
})

interface HandoffBadge { label: string; classes: string }

// Badge next to the Cluster InfoRow surfacing where a federated job sits in the
// handoff lifecycle. Absent for a local job, a terminal job, or once the peer
// has both acked the handoff and reported at least one task (PEER_STATUS_SYNCED).
const handoffBadge = computed<HandoffBadge | null>(() => {
  const j = job.value
  if (!j || !isFederated(j.cluster) || isTerminal.value) return null
  if (j.peerStatus === 'PEER_STATUS_PENDING_SCHEDULING') {
    return { label: 'awaiting peer acceptance', classes: 'bg-status-warning-bg text-status-warning border-status-warning-border' }
  }
  if (j.peerStatus === 'PEER_STATUS_ASSIGNED') {
    return { label: 'awaiting first status report', classes: 'bg-surface-sunken text-text-muted border-surface-border' }
  }
  return null
})

// The peer this job was handed off to, from the shared federation roster
// (populated lazily via ensurePeers). Undefined for a local job or before the
// roster has loaded.
const jobPeer = computed(() => {
  const j = job.value
  if (!j || !isFederated(j.cluster)) return undefined
  return peers.value.find(p => p.peerId === j.cluster)
})

// A handed-off, non-terminal job whose peer failed its last capability
// heartbeat: the one signal that explains a stuck PENDING_SCHEDULING.
const peerUnreachable = computed(() => {
  if (isTerminal.value || !job.value || !isFederated(job.value.cluster)) return false
  return jobPeer.value?.reachable === false
})

const { stop: stopRefresh, start: startRefresh } = useAutoRefresh(fetchData, 10_000)

watch(isTerminal, (terminal) => {
  if (terminal) stopRefresh()
})

// Re-fetch when navigating between jobs (Vue Router reuses the component).
watch(() => props.jobId, () => {
  loading.value = true
  job.value = null
  jobRequest.value = null
  tasks.value = []
  endpointsByTask.value = new Map()
  childJobsByParent.value = new Map()
  expandedChildJobs.value = new Set()
  loadingChildJobs.value = new Set()
  error.value = null
  fetchData()
  startRefresh()
})

// -- Formatting helpers --

function jobDuration(j: JobStatus): string {
  const started = timestampMs(j.startedAt)
  if (!started) return '-'
  const ended = timestampMs(j.finishedAt) || Date.now()
  return formatDuration(started, ended)
}

function taskDuration(t: TaskStatus): string {
  const started = timestampMs(t.startedAt)
  if (!started) return '-'
  const ended = timestampMs(t.finishedAt) || Date.now()
  return formatDuration(started, ended)
}

function taskIndex(taskId: string): string {
  const last = taskId.split('/').pop()
  if (!last) return '-'
  const parsed = parseInt(last, 10)
  return isNaN(parsed) ? '-' : String(parsed)
}

// -- Child job helpers --

function childJobDurationMs(j: JobStatus): number {
  const started = timestampMs(j.startedAt)
  if (!started) return 0
  const ended = timestampMs(j.finishedAt) || Date.now()
  return ended - started
}

const childJobComparator = computed<((a: JobStatus, b: JobStatus) => number) | undefined>(() => {
  const col = childSortColumn.value
  if (!col) return undefined
  const dir = childSortDir.value === 'asc' ? 1 : -1
  return (a: JobStatus, b: JobStatus) => {
    let cmp = 0
    switch (col) {
      case 'name':
        cmp = getLeafJobName(a.name).localeCompare(getLeafJobName(b.name))
        break
      case 'state':
        cmp = (STATE_SORT_ORDER[stateToName(a.state)] ?? 99) - (STATE_SORT_ORDER[stateToName(b.state)] ?? 99)
        break
      case 'duration':
        cmp = childJobDurationMs(a) - childJobDurationMs(b)
        break
    }
    return cmp * dir
  }
})

const flattenedChildJobs = computed(() => {
  const result: Array<{ job: JobStatus; depth: number }> = []

  function walk(parentJobId: string, depth: number) {
    const children = childJobsByParent.value.get(parentJobId) ?? []
    const sorted = childJobComparator.value ? [...children].sort(childJobComparator.value) : children
    for (const child of sorted) {
      result.push({ job: child, depth })
      if (expandedChildJobs.value.has(child.jobId)) {
        walk(child.jobId, depth + 1)
      }
    }
  }

  walk(props.jobId, 0)
  return result
})

async function toggleExpandedChildJob(jobStatus: JobStatus) {
  const next = new Set(expandedChildJobs.value)
  if (next.has(jobStatus.jobId)) {
    next.delete(jobStatus.jobId)
    expandedChildJobs.value = next
    return
  }

  next.add(jobStatus.jobId)
  expandedChildJobs.value = next

  if (childJobsByParent.value.has(jobStatus.jobId)) {
    return
  }

  const nextLoading = new Set(loadingChildJobs.value)
  nextLoading.add(jobStatus.jobId)
  loadingChildJobs.value = nextLoading
  try {
    const children = await fetchChildJobs(jobStatus.jobId)
    const nextChildren = new Map(childJobsByParent.value)
    nextChildren.set(jobStatus.jobId, children)
    childJobsByParent.value = nextChildren
  } finally {
    const doneLoading = new Set(loadingChildJobs.value)
    doneLoading.delete(jobStatus.jobId)
    loadingChildJobs.value = doneLoading
  }
}

const SEGMENT_COLORS: Record<string, string> = {
  succeeded: 'bg-status-success',
  running: 'bg-accent',
  building: 'bg-status-purple',
  assigned: 'bg-status-orange',
  failed: 'bg-status-danger',
  worker_failed: 'bg-status-danger',
  preempted: 'bg-status-warning',
  killed: 'bg-text-muted',
  pending: 'bg-surface-border',
}

interface ProgressSegment {
  count: number
  colorClass: string
  label: string
}

function progressSegments(j: JobStatus): ProgressSegment[] {
  const counts = j.taskStateCounts ?? {}
  const total = j.taskCount ?? 0
  if (total === 0) return []
  const succeeded = counts['succeeded'] ?? 0
  const running = counts['running'] ?? 0
  const building = counts['building'] ?? 0
  const assigned = counts['assigned'] ?? 0
  const failed = counts['failed'] ?? 0
  const workerFailed = counts['worker_failed'] ?? 0
  const coschedFailed = counts['cosched_failed'] ?? 0
  const preempted = counts['preempted'] ?? 0
  const killed = counts['killed'] ?? 0
  const pending =
    total - succeeded - running - building - assigned - failed - workerFailed - coschedFailed - preempted - killed
  return [
    { count: succeeded, colorClass: SEGMENT_COLORS['succeeded'], label: 'succeeded' },
    { count: running, colorClass: SEGMENT_COLORS['running'], label: 'running' },
    { count: building, colorClass: SEGMENT_COLORS['building'], label: 'building' },
    { count: assigned, colorClass: SEGMENT_COLORS['assigned'], label: 'assigned' },
    { count: failed, colorClass: SEGMENT_COLORS['failed'], label: 'failed' },
    { count: workerFailed, colorClass: SEGMENT_COLORS['worker_failed'], label: 'worker_failed' },
    { count: coschedFailed, colorClass: SEGMENT_COLORS['cosched_failed'], label: 'cosched_failed' },
    { count: preempted, colorClass: SEGMENT_COLORS['preempted'], label: 'preempted' },
    { count: killed, colorClass: SEGMENT_COLORS['killed'], label: 'killed' },
    { count: Math.max(0, pending), colorClass: SEGMENT_COLORS['pending'], label: 'pending' },
  ].filter(s => s.count > 0)
}

function progressSummary(j: JobStatus): string {
  const counts = j.taskStateCounts ?? {}
  const running = counts['running'] ?? 0
  const total = j.taskCount ?? 0
  const succeeded = counts['succeeded'] ?? 0
  if (running > 0) return `${running} running`
  return `${succeeded}/${total}`
}

// -- Computed --

const pageTitle = computed(() => {
  if (!job.value) return `Job: ${props.jobId}`
  const name = job.value.name
  return (name && name !== props.jobId) ? name : `Job: ${props.jobId}`
})

// Child jobs link back to their parent job; root jobs link to the jobs list.
const backTo = computed(() => {
  const parentJobId = job.value?.parentJobId
  return {
    path: parentJobId ? `/job/${encodeURIComponent(parentJobId)}` : '/',
    query: route.query,
  }
})

const backLabel = computed(() => (job.value?.parentJobId ? 'Back to parent job' : 'Jobs'))

const subtitle = computed(() => {
  if (!job.value) return ''
  return (job.value.name && job.value.name !== props.jobId) ? `ID: ${props.jobId}` : ''
})

const taskCounts = computed(() => {
  const counts = { total: 0, succeeded: 0, running: 0, building: 0, assigned: 0, pending: 0, failed: 0 }
  for (const t of tasks.value) {
    counts.total++
    const state = stateToName(t.state)
    if (state === 'succeeded' || state === 'killed') counts.succeeded++
    else if (state === 'running') counts.running++
    else if (state === 'building') counts.building++
    else if (state === 'assigned') counts.assigned++
    else if (state === 'pending') counts.pending++
    else if (state === 'failed' || state === 'worker_failed' || state === 'cosched_failed' || state === 'preempted') counts.failed++
  }
  return counts
})

// The Scheduling pane auto-opens while tasks are pending — that's when placement
// constraints explain why the job can't run — but a manual collapse then sticks
// instead of being forced back open on the next auto-refresh.
const schedulingOpen = ref(false)
watch(() => taskCounts.value.pending > 0, pending => { if (pending) schedulingOpen.value = true }, { immediate: true })

const MAX_FAILURE_EXAMPLES = 8

interface TaskFailureSummary {
  taskId: string
  taskIndex: string
  attemptId: number     // most recent failed attempt of this state
  error: string
  finishedAtMs: number
  failureCount: number  // failed attempts of this state for the task
}

// One entry per task — its most recent failed attempt plus a retry count — so a
// single task that fails repeatedly doesn't crowd out other failing tasks.
// ListTasks attaches only the latest failed attempt per state (not the full
// history), so the failure stays visible after the task is retried back into a
// running/pending state. The badge counts the attached attempts, so it reflects
// the failures shown here, not the task's authoritative lifetime total (those
// job-level totals live on JobStatus).
function collectFailuresByState(stateName: string, count: (t: TaskStatus) => number): TaskFailureSummary[] {
  const out: TaskFailureSummary[] = []
  for (const task of tasks.value) {
    let latest: { attemptId: number; error: string; finishedAtMs: number } | null = null
    for (const attempt of task.attempts ?? []) {
      if (stateToName(attempt.state) !== stateName) continue
      const finishedAtMs = timestampMs(attempt.finishedAt)
      if (!latest || finishedAtMs >= latest.finishedAtMs) {
        latest = { attemptId: attempt.attemptId, error: attempt.error ?? '', finishedAtMs }
      }
    }
    if (!latest) continue
    out.push({
      taskId: task.taskId,
      taskIndex: taskIndex(task.taskId),
      attemptId: latest.attemptId,
      error: latest.error,
      finishedAtMs: latest.finishedAtMs,
      failureCount: count(task),
    })
  }
  return out.sort((a, b) => b.finishedAtMs - a.finishedAtMs)
}

const recentTaskFailures = computed<TaskFailureSummary[]>(() =>
  collectFailuresByState('failed', t => (t.attempts ?? []).filter(a => stateToName(a.state) === 'failed').length),
)
// Worker failures share the preemption budget with kills/preemptions, so there
// is no clean per-task "worker failure" counter; surface the latest one without
// a count badge.
const recentWorkerFailures = computed<TaskFailureSummary[]>(() =>
  collectFailuresByState('worker_failed', () => 0),
)

// The job's root cause is the earliest-finished failed attempt among the
// failures surfaced above (matching the controller's job.error pick). Its logs
// are the ones worth distilling into a stack trace.
const rootCauseFailure = computed<TaskFailureSummary | null>(() => {
  const candidates = [...recentTaskFailures.value, ...recentWorkerFailures.value]
  return candidates.length
    ? candidates.reduce((earliest, f) => (f.finishedAtMs < earliest.finishedAtMs ? f : earliest))
    : null
})

// Fetch the root-cause task's distilled log highlights on demand — one task per
// failed job, never a per-task fetch across the whole list. Best-effort: a
// failure here never blocks the page.
async function fetchRootCauseHighlights(gen: number): Promise<void> {
  rootCauseHighlights.value = []
  const taskId = rootCauseFailure.value?.taskId
  if (!taskId) return
  try {
    const resp = await controllerRpcCall<GetTaskStatusResponse>('GetTaskStatus', { taskId })
    if (gen !== fetchGeneration) return
    rootCauseHighlights.value = resp.rootCauseHighlights ?? []
  } catch {
    // Highlights are advisory; leave them empty on any fetch error.
  }
}

// Failure / preemption budgets from the launch request. The reconstructed
// request always carries these int32 fields, but proto3 omits zero defaults
// from the JSON, so coalesce to 0. maxTaskFailures is the job-level abort
// threshold (terminally-failed tasks tolerated); the retry budgets are per task.
const maxTaskFailures = computed(() => jobRequest.value?.maxTaskFailures ?? 0)
const maxRetriesFailure = computed(() => jobRequest.value?.maxRetriesFailure ?? 0)
const maxRetriesPreemption = computed(() => jobRequest.value?.maxRetriesPreemption ?? 0)

const failuresTitle = computed(() =>
  `${job.value?.failureCount ?? 0} failed task attempts (including retries). `
  + `Each task retries up to ${maxRetriesFailure.value}× on failure; `
  + `the job fails once more than ${maxTaskFailures.value} tasks fail.`,
)

const preemptionsTitle = computed(() =>
  `${job.value?.preemptionCount ?? 0} task preemptions (including retries). `
  + `Each task retries up to ${maxRetriesPreemption.value}× on preemption.`,
)

const acceleratorDisplay = computed(() => {
  const j = job.value
  const req = jobRequest.value
  const base = formatDeviceConfig(j?.resources?.device)
    ?? formatDeviceConfig(req?.resources?.device)
  return base ?? '-'
})

const cpuDisplay = computed(() => {
  const mc = job.value?.resources?.cpuMillicores
  if (!mc) return '-'
  return String(mc / 1000)
})

const memoryDisplay = computed(() => {
  const mb = job.value?.resources?.memoryBytes
  if (!mb) return '-'
  return formatBytes(parseInt(mb, 10))
})

const diskDisplay = computed(() => {
  const db = job.value?.resources?.diskBytes
  if (!db) return '-'
  return formatBytes(parseInt(db, 10))
})

const STATE_SORT_ORDER: Record<string, number> = {
  running: 0, building: 1, assigned: 2, pending: 3,
  succeeded: 4, killed: 5, failed: 6, worker_failed: 7, cosched_failed: 8, preempted: 9, unschedulable: 10,
}

function taskDurationMs(t: TaskStatus): number {
  const started = timestampMs(t.startedAt)
  if (!started) return 0
  const ended = timestampMs(t.finishedAt) || Date.now()
  return ended - started
}

const availableStates = computed(() => {
  const seen = new Set<string>()
  for (const t of tasks.value) seen.add(stateToName(t.state))
  return [...seen].sort((a, b) => (STATE_SORT_ORDER[a] ?? 99) - (STATE_SORT_ORDER[b] ?? 99))
})

const filteredTasks = computed(() => {
  const q = taskSearch.value.toLowerCase().trim()
  const sf = stateFilter.value
  const result = (!q && !sf)
    ? [...tasks.value]
    : tasks.value.filter(t => {
        if (sf && stateToName(t.state) !== sf) return false
        if (!q) return true
        return (t.workerId?.toLowerCase().includes(q))
          || taskIndex(t.taskId).includes(q)
      })

  const col = sortColumn.value
  if (!col) return result

  const dir = sortDir.value === 'asc' ? 1 : -1
  result.sort((a, b) => {
    let cmp = 0
    switch (col) {
      case 'task':
        cmp = parseInt(taskIndex(a.taskId)) - parseInt(taskIndex(b.taskId))
        break
      case 'state':
        cmp = (STATE_SORT_ORDER[stateToName(a.state)] ?? 99) - (STATE_SORT_ORDER[stateToName(b.state)] ?? 99)
        break
      case 'mem':
        // Sort by peak when present so the "biggest memory consumer" surfaces;
        // current usage breaks ties for tasks without a peak sample yet.
        cmp = (taskPeakMemBytes(a.taskId) || taskMemBytes(a.taskId))
            - (taskPeakMemBytes(b.taskId) || taskMemBytes(b.taskId))
        break
      case 'cpu':
        cmp = taskCpuMillicores(a.taskId) - taskCpuMillicores(b.taskId)
        break
      case 'duration':
        cmp = taskDurationMs(a) - taskDurationMs(b)
        break
    }
    return cmp * dir
  })
  return result
})

// -- Task Pagination --

const TASK_PAGE_SIZE = 50
const taskPage = ref(0)

const totalTaskPages = computed(() => Math.max(1, Math.ceil(filteredTasks.value.length / TASK_PAGE_SIZE)))

const paginatedTasks = computed(() => {
  // Clamp the effective page against the current filtered length so a shrink
  // during auto-refresh never yields an empty slice on a stale page. The
  // watcher below mirrors this into `taskPage` so the paginator footer stays
  // in sync.
  const effectivePage = Math.min(taskPage.value, totalTaskPages.value - 1)
  const start = Math.max(0, effectivePage) * TASK_PAGE_SIZE
  return filteredTasks.value.slice(start, start + TASK_PAGE_SIZE)
})

// Reset page when filters or sort change
watch([taskSearch, stateFilter, sortColumn, sortDir], () => { taskPage.value = 0 })

// Clamp taskPage when the filtered task list shrinks underneath us. This
// happens during the 10s auto-refresh when task state transitions change
// which tasks match the active state filter — without clamping, a user on
// a later page can be left with an empty table body and a stale footer
// range (e.g. "251-240 of 240"). The computed runs eagerly so the page is
// corrected before `paginatedTasks` slices against the new length.
watch(totalTaskPages, (pages) => {
  if (taskPage.value >= pages) {
    taskPage.value = Math.max(0, pages - 1)
  }
})

// -- Profiling --

function buildProfileType(profilerType: string, format: string | null): Record<string, unknown> {
  if (profilerType === 'cpu') return { cpu: { format: format ?? 'SPEEDSCOPE' } }
  if (profilerType === 'memory') return { memory: { format: format ?? 'RAW' } }
  return { threads: {} }
}

async function handleProfile(taskId: string, profilerType: string, format: string | null) {
  profilingTaskId.value = taskId
  // Open the viewer synchronously within the click gesture; CPU profiling takes
  // ~10s, after which a fresh window.open would be blocked as a popup.
  const pending = profilerType === 'cpu' && (format ?? 'SPEEDSCOPE') === 'SPEEDSCOPE' ? openSpeedscopeWindow() : null
  try {
    const body = {
      target: taskId,
      durationSeconds: 10,
      profileType: buildProfileType(profilerType, format),
    }
    const resp = await controllerRpcCall<{ profileData?: string; error?: string }>('ProfileTask', body)
    if (resp.error) {
      pending?.cancel()
      alert(`${profilerType.toUpperCase()} profile failed: ${resp.error}`)
      return
    }
    if (!resp.profileData) {
      pending?.cancel()
      return
    }
    const bin = atob(resp.profileData)
    const bytes = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) {
      bytes[i] = bin.charCodeAt(i)
    }
    if (pending) {
      pending.show(bytes, taskId)
      return
    }
    const blob = new Blob([bytes], { type: 'application/octet-stream' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const ts = new Date().toISOString().replace(/[T]/g, '_').replace(/:/g, '-').replace(/\.\d+Z$/, '')
    const ext = profilerType === 'memory' ? 'bin' : 'out'
    a.download = `${ts}_profile-${taskId.replace(/\//g, '_')}.${ext}`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    pending?.cancel()
    alert(`${profilerType.toUpperCase()} profile failed: ${e instanceof Error ? e.message : e}`)
  } finally {
    profilingTaskId.value = null
  }
}
</script>

<template>
  <PageShell :title="pageTitle" :back-to="backTo" :back-label="backLabel">
    <template v-if="job?.name" #title-suffix>
      <button
        class="inline-flex items-center gap-1 px-1.5 py-0.5 text-xs text-text-muted hover:text-text
               border border-surface-border rounded hover:bg-surface-raised transition-colors"
        title="Copy job name"
        @click="copyJobName"
      >
        <svg v-if="copiedName" class="w-3 h-3 text-status-success" viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
        </svg>
        <svg v-else class="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
          <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
        </svg>
        {{ copiedName ? 'Copied' : 'Copy name' }}
      </button>
    </template>

    <!-- Subtitle (job ID when name differs) -->
    <p v-if="subtitle" class="text-sm text-text-secondary font-mono -mt-4 mb-6">
      {{ subtitle }}
    </p>

    <!-- Loading -->
    <div v-if="loading" class="flex items-center justify-center py-12 text-text-muted text-sm">
      <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
      Loading...
    </div>

    <!-- Error -->
    <div
      v-else-if="error"
      class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <!-- Content -->
    <template v-else-if="job">
      <!-- Error banner -->
      <div
        v-if="job.error"
        class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
      >
        <span class="font-semibold">Error:</span> {{ job.error }}
      </div>

      <!-- Likely root cause: highlights distilled from the root-cause task's own logs -->
      <div
        v-if="rootCauseHighlights.length"
        class="mb-4 px-4 py-3 rounded-lg border border-status-danger-border bg-status-danger-bg"
      >
        <div class="text-sm font-semibold text-status-danger mb-2">Likely Root Cause</div>
        <pre class="text-xs font-mono text-status-danger whitespace-pre-wrap break-all">{{ rootCauseHighlights.join('\n') }}</pre>
      </div>

      <!-- Peer unreachable banner: this job's handoff target failed its last
           capability heartbeat, so its status updates may be stale — the one
           signal that explains a stuck "awaiting peer acceptance". -->
      <div
        v-if="peerUnreachable"
        class="mb-4 px-4 py-3 bg-status-warning-bg border border-status-warning-border rounded-lg text-sm text-status-warning"
      >
        <span class="font-semibold">Peer {{ job.cluster }} is unreachable.</span>
        <span v-if="jobPeer?.lastContactMs && jobPeer.lastContactMs !== '0'">
          Last contact {{ formatRelativeTime(Number(jobPeer.lastContactMs)) }}.
        </span>
        <RouterLink to="/backends" class="ml-1 text-accent hover:underline">View execution targets</RouterLink>
      </div>

      <!-- Pending reason banner -->
      <div
        v-if="job.pendingReason"
        class="mb-4 px-4 py-3 bg-status-warning-bg border border-status-warning-border rounded-lg"
      >
        <span class="font-semibold text-status-warning text-sm">Scheduling Diagnostic:</span>
        <pre class="mt-2 p-3 bg-surface rounded text-xs font-mono whitespace-pre-wrap">{{ job.pendingReason }}</pre>
      </div>

      <!-- Failed tasks callout: genuine task failures (non-zero exit). Shows each
           task's most recent failed attempt — surfaced even after the task has
           been retried back into a running/pending state — with its total
           failure count. One row per task. -->
      <div
        v-if="recentTaskFailures.length > 0"
        class="mb-4 px-4 py-3 bg-status-danger-bg border border-status-danger-border rounded-lg"
      >
        <span class="font-semibold text-status-danger text-sm">
          {{ recentTaskFailures.length }} task{{ recentTaskFailures.length !== 1 ? 's' : '' }} with failed attempts
        </span>
        <div class="mt-2 flex flex-col gap-1">
          <div
            v-for="f in recentTaskFailures.slice(0, MAX_FAILURE_EXAMPLES)"
            :key="f.taskId"
            class="text-xs text-text-secondary"
          >
            <RouterLink
              :to="`/job/${encodeURIComponent(props.jobId)}/task/${encodeURIComponent(f.taskId)}`"
              class="text-accent hover:underline font-mono"
            >
              task {{ f.taskIndex }}
            </RouterLink>
            <span class="text-text-muted"> attempt {{ f.attemptId }}</span>
            <span v-if="f.failureCount > 1" class="text-status-danger"> ·&times;{{ f.failureCount }}</span>
            <span v-if="f.finishedAtMs" class="text-text-muted"> · {{ formatRelativeTime(f.finishedAtMs) }}</span>
            <span v-if="f.error" class="text-status-danger"> · {{ f.error.length > 120 ? f.error.slice(0, 120) + '…' : f.error }}</span>
          </div>
          <span
            v-if="recentTaskFailures.length > MAX_FAILURE_EXAMPLES"
            class="text-xs text-text-muted"
          >
            … and {{ recentTaskFailures.length - MAX_FAILURE_EXAMPLES }} more
          </span>
        </div>
      </div>

      <!-- Worker-failure callout: tasks whose attempt died with the worker
           (infra death), as opposed to a genuine non-zero exit. -->
      <div
        v-if="recentWorkerFailures.length > 0"
        class="mb-4 px-4 py-3 bg-status-warning-bg border border-status-warning-border rounded-lg"
      >
        <span class="font-semibold text-status-warning text-sm">
          {{ recentWorkerFailures.length }} task{{ recentWorkerFailures.length !== 1 ? 's' : '' }} with worker failures
        </span>
        <div class="mt-2 flex flex-col gap-1">
          <div
            v-for="f in recentWorkerFailures.slice(0, MAX_FAILURE_EXAMPLES)"
            :key="f.taskId"
            class="text-xs text-text-secondary"
          >
            <RouterLink
              :to="`/job/${encodeURIComponent(props.jobId)}/task/${encodeURIComponent(f.taskId)}`"
              class="text-accent hover:underline font-mono"
            >
              task {{ f.taskIndex }}
            </RouterLink>
            <span class="text-text-muted"> attempt {{ f.attemptId }}</span>
            <span v-if="f.finishedAtMs" class="text-text-muted"> · {{ formatRelativeTime(f.finishedAtMs) }}</span>
            <span v-if="f.error" class="text-status-warning"> · {{ f.error.length > 120 ? f.error.slice(0, 120) + '…' : f.error }}</span>
          </div>
          <span
            v-if="recentWorkerFailures.length > MAX_FAILURE_EXAMPLES"
            class="text-xs text-text-muted"
          >
            … and {{ recentWorkerFailures.length - MAX_FAILURE_EXAMPLES }} more
          </span>
        </div>
      </div>

      <!-- Info cards -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <InfoCard title="Job Status">
          <InfoRow label="State">
            <StatusBadge :status="job.state" size="sm" />
          </InfoRow>
          <InfoRow label="Started">
            <span class="font-mono">{{ formatTimestamp(job.startedAt) }}</span>
          </InfoRow>
          <InfoRow label="Finished">
            <span class="font-mono">{{ isTerminal ? formatTimestamp(job.finishedAt) : '-' }}</span>
          </InfoRow>
          <InfoRow label="Duration">
            <span class="font-mono">{{ jobDuration(job) }}</span>
          </InfoRow>
          <InfoRow label="Failures">
            <span :title="failuresTitle">
              {{ job.failureCount ?? 0 }}<span class="text-text-muted"> / (max {{ maxTaskFailures }})</span>
            </span>
          </InfoRow>
          <InfoRow label="Preemptions">
            <span :title="preemptionsTitle">
              {{ job.preemptionCount ?? 0 }}<span class="text-text-muted"> / (max {{ maxRetriesPreemption }}/task)</span>
            </span>
          </InfoRow>
          <InfoRow v-if="jobRequest?.priorityBand" label="Priority">
            <span :class="bandColor(jobRequest.priorityBand)" class="font-semibold">
              {{ bandDisplayName(jobRequest.priorityBand) }}
            </span>
          </InfoRow>
          <InfoRow v-if="multiBackend && job?.backendId" label="Backend">
            <span class="font-mono">{{ job!.backendId }}</span>
          </InfoRow>
          <!-- Cluster: every job carries a cluster coordinate (`'local'` by
               default); links inward to the parent's jobs list filtered to it.
               For a federated job, a badge surfaces its handoff posture. -->
          <InfoRow label="Cluster">
            <span class="flex items-center gap-2">
              <ClusterLink :cluster="job?.cluster ?? LOCAL_CLUSTER" />
              <span
                v-if="handoffBadge"
                class="inline-flex items-center px-1.5 py-0.5 rounded text-xs border"
                :class="handoffBadge.classes"
              >
                {{ handoffBadge.label }}
              </span>
            </span>
          </InfoRow>
        </InfoCard>

        <InfoCard title="Task Summary">
          <InfoRow label="Total">{{ taskCounts.total }}</InfoRow>
          <InfoRow label="Completed">{{ taskCounts.succeeded }}</InfoRow>
          <InfoRow label="Running">{{ taskCounts.running }}</InfoRow>
          <InfoRow label="Building">{{ taskCounts.building }}</InfoRow>
          <InfoRow label="Assigned">{{ taskCounts.assigned }}</InfoRow>
          <InfoRow label="Pending">{{ taskCounts.pending }}</InfoRow>
          <InfoRow label="Failed">{{ taskCounts.failed }}</InfoRow>
        </InfoCard>

        <!-- Resources: per-VM reservation, annotated inline with live usage across the
             running tasks so the card reads as utilization (am I using what I booked,
             and how close to OOM) rather than two disconnected lists of numbers. -->
        <InfoCard title="Resources (per VM)">
          <InfoRow label="CPU">
            <span class="font-mono">{{ cpuDisplay }}</span>
            <span v-if="runningResourceRange" class="text-text-muted"> · {{ formatCpuMillicores(runningResourceRange.cpuMillicoresMin) }}&ndash;{{ formatCpuMillicores(runningResourceRange.cpuMillicoresMax) }} used</span>
          </InfoRow>
          <InfoRow label="Memory">
            <span class="font-mono">{{ memoryDisplay }}</span>
            <span v-if="runningResourceRange" class="text-text-muted"> · {{ formatBytes(runningResourceRange.memoryBytesMin) }}&ndash;{{ formatBytes(runningResourceRange.memoryBytesMax) }} used</span>
          </InfoRow>
          <InfoRow v-if="runningResourceRange?.memoryPeakBytesMax" label="Peak memory">
            <span class="font-mono">{{ formatBytes(runningResourceRange.memoryPeakBytesMax) }}</span>
          </InfoRow>
          <InfoRow label="Disk">{{ diskDisplay }}</InfoRow>
          <InfoRow label="Accelerator">{{ acceleratorDisplay }}</InfoRow>
          <!-- Falls back to the requested replica count (job.taskCount) for a
               federated job whose peer has not yet reported its task set. -->
          <InfoRow label="Replicas">{{ tasks.length || job.taskCount || '-' }}</InfoRow>
        </InfoCard>
      </div>

      <!-- Scheduling: placement constraints, auto-opened while tasks are pending so an
           operator sees what the job is asking for right when it can't be placed. -->
      <details
        v-if="jobRequest?.constraints && jobRequest.constraints.length > 0"
        class="mb-6 rounded-lg border border-surface-border bg-surface"
        :open="schedulingOpen"
        @toggle="schedulingOpen = ($event.target as HTMLDetailsElement).open"
      >
        <summary class="flex items-center gap-2 px-4 py-2 cursor-pointer select-none text-xs font-semibold uppercase tracking-wider text-text-secondary">
          Scheduling
          <span class="font-normal normal-case tracking-normal text-text-muted">
            — {{ jobRequest.constraints.length }} constraint{{ jobRequest.constraints.length > 1 ? 's' : '' }}<template v-if="taskCounts.pending > 0"> · {{ taskCounts.pending }} task{{ taskCounts.pending > 1 ? 's' : '' }} pending</template>
          </span>
        </summary>
        <div class="border-t border-surface-border px-4 py-3 flex flex-wrap gap-1.5">
          <span
            v-for="(c, i) in jobRequest.constraints"
            :key="i"
            class="inline-block rounded bg-surface-sunken px-2 py-0.5 font-mono text-xs text-text-secondary"
          >
            {{ c.key }} {{ c.op }} {{ c.value?.stringValue ?? c.value?.intValue ?? '' }}
          </span>
        </div>
      </details>

      <!-- Job Request Details (collapsed by default — open to inspect command/env) -->
      <details
        v-if="jobRequest?.entrypoint?.runCommand?.argv?.length || jobRequest?.submitArgv?.length || jobRequest?.environment?.envVars || jobRequest?.environment?.pipPackages?.length || jobRequest?.ports?.length"
        class="mb-6 rounded-lg border border-surface-border bg-surface"
      >
        <summary class="flex items-center gap-2 px-4 py-2 cursor-pointer select-none text-xs font-semibold uppercase tracking-wider text-text-secondary">
          Job Request
          <span class="font-normal normal-case tracking-normal text-text-muted">— command, setup &amp; environment</span>
        </summary>
        <div class="flex flex-col gap-2 text-sm border-t border-surface-border px-4 py-3">
          <div v-if="jobRequest.entrypoint?.runCommand?.argv?.length">
            <span class="text-text-muted text-xs">Command</span>
            <pre class="mt-0.5 px-2 py-1 bg-surface-sunken rounded font-mono text-xs whitespace-pre-wrap break-all">{{ jobRequest.entrypoint.runCommand.argv.join(' ') }}</pre>
          </div>
          <div v-if="jobRequest.submitArgv?.length">
            <span class="text-text-muted text-xs">Submitted via</span>
            <pre class="mt-0.5 px-2 py-1 bg-surface-sunken rounded font-mono text-xs whitespace-pre-wrap break-all">{{ jobRequest.submitArgv.join(' ') }}</pre>
          </div>
          <div v-if="jobRequest.entrypoint?.setupCommands?.length">
            <span class="text-text-muted text-xs">Setup Commands</span>
            <pre class="mt-0.5 px-2 py-1 bg-surface-sunken rounded font-mono text-xs whitespace-pre-wrap break-all">{{ jobRequest.entrypoint.setupCommands.join('\n') }}</pre>
          </div>
          <div v-if="jobRequest.environment?.envVars && Object.keys(jobRequest.environment.envVars).length">
            <span class="text-text-muted text-xs">Environment Variables</span>
            <div class="mt-0.5 flex flex-wrap gap-1.5">
              <span
                v-for="(val, key) in jobRequest.environment.envVars"
                :key="key"
                class="inline-block rounded bg-surface-sunken px-2 py-0.5 font-mono text-xs text-text-secondary"
              >
                {{ key }}={{ val }}
              </span>
            </div>
          </div>
          <div v-if="jobRequest.environment?.pipPackages?.length">
            <span class="text-text-muted text-xs">Pip Packages</span>
            <div class="mt-0.5 flex flex-wrap gap-1.5">
              <span
                v-for="(pkg, i) in jobRequest.environment.pipPackages"
                :key="i"
                class="inline-block rounded bg-surface-sunken px-2 py-0.5 font-mono text-xs text-text-secondary"
              >
                {{ pkg }}
              </span>
            </div>
          </div>
          <div v-if="jobRequest.ports?.length">
            <span class="text-text-muted text-xs">Ports</span>
            <div class="mt-0.5 flex flex-wrap gap-1.5">
              <span
                v-for="(port, i) in jobRequest.ports"
                :key="i"
                class="inline-block rounded bg-surface-sunken px-2 py-0.5 font-mono text-xs text-text-secondary"
              >
                {{ port }}
              </span>
            </div>
          </div>
        </div>
      </details>

      <!-- Child Jobs -->
      <div v-if="flattenedChildJobs.length > 0" class="mb-6">
        <div class="mb-3 flex items-center justify-between gap-3">
          <h3 class="text-sm font-semibold uppercase tracking-wider text-text-secondary">
            Child Jobs
          </h3>
        </div>
        <!-- Mobile: card grid (one card per child job) -->
        <div v-if="isMobile" class="grid grid-cols-1 gap-2">
          <div
            v-for="node in flattenedChildJobs"
            :key="'child-card-' + node.job.jobId"
            class="rounded-lg border border-surface-border bg-surface px-3 py-2"
            :style="node.depth > 0 ? { marginLeft: (Math.min(node.depth, 3) * 12) + 'px' } : undefined"
          >
            <div class="flex items-start gap-1.5">
              <button
                v-if="node.job.hasChildren"
                class="text-text-muted hover:text-text select-none w-4 text-center text-xs shrink-0 mt-0.5"
                @click.stop="toggleExpandedChildJob(node.job)"
              >
                {{ loadingChildJobs.has(node.job.jobId) ? '…' : (expandedChildJobs.has(node.job.jobId) ? '▼' : '▶') }}
              </button>
              <span v-else class="w-4 shrink-0" />
              <RouterLink
                :to="'/job/' + encodeURIComponent(node.job.jobId)"
                class="text-accent hover:underline font-mono text-[13px] flex-1 min-w-0 break-anywhere"
              >
                {{ getLeafJobName(node.job.name) }}
              </RouterLink>
            </div>
            <div class="mt-1.5 pl-5 flex items-center gap-2 flex-wrap">
              <StatusBadge :status="node.job.state" size="sm" />
              <span class="text-xs text-text-muted font-mono">{{ jobDuration(node.job) }}</span>
            </div>
            <div
              v-if="node.job.pendingReason"
              class="mt-1 pl-5 text-xs text-text-muted"
              :title="node.job.pendingReason"
            >
              {{ node.job.pendingReason }}
            </div>
            <div v-if="(node.job.taskCount ?? 0) > 0" class="mt-2 pl-5 flex items-center gap-2">
              <div class="flex h-2 flex-1 rounded-full overflow-hidden bg-surface-sunken">
                <div
                  v-for="(seg, i) in progressSegments(node.job)"
                  :key="i"
                  :class="seg.colorClass"
                  :style="{ width: (seg.count / (node.job.taskCount ?? 1) * 100).toFixed(1) + '%' }"
                  :title="seg.label + ': ' + seg.count"
                />
              </div>
              <span class="text-xs text-text-secondary whitespace-nowrap">
                {{ progressSummary(node.job) }}
              </span>
            </div>
          </div>
        </div>

        <!-- Desktop: table -->
        <div v-else class="overflow-x-auto">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border">
              <th class="px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleChildSort('name')">
                Name <span v-if="childSortColumn === 'name'" class="ml-0.5">{{ childSortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleChildSort('state')">
                State <span v-if="childSortColumn === 'state'" class="ml-0.5">{{ childSortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="hidden sm:table-cell px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleChildSort('duration')">
                Duration <span v-if="childSortColumn === 'duration'" class="ml-0.5">{{ childSortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Tasks</th>
              <th class="hidden lg:table-cell px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Diagnostic</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="node in flattenedChildJobs"
              :key="node.job.jobId"
              class="group/row border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td
                class="px-2 sm:px-3 py-2 text-[13px]"
                :style="{ paddingLeft: (node.depth * 20 + 12) + 'px' }"
              >
                <span class="inline-flex items-center gap-1 max-w-full">
                  <button
                    v-if="node.job.hasChildren"
                    class="text-text-muted hover:text-text select-none w-4 text-center text-xs shrink-0"
                    @click.stop="toggleExpandedChildJob(node.job)"
                  >
                    {{ loadingChildJobs.has(node.job.jobId) ? '…' : (expandedChildJobs.has(node.job.jobId) ? '▼' : '▶') }}
                  </button>
                  <span v-else class="w-4 shrink-0" />
                  <RouterLink
                    :to="'/job/' + encodeURIComponent(node.job.jobId)"
                    class="text-accent hover:underline font-mono break-anywhere"
                  >
                    {{ getLeafJobName(node.job.name) }}
                  </RouterLink>
                </span>
              </td>
              <td class="px-2 sm:px-3 py-2 text-[13px]">
                <StatusBadge :status="node.job.state" size="sm" />
              </td>
              <td class="hidden sm:table-cell px-2 sm:px-3 py-2 text-[13px] text-text-secondary font-mono">
                {{ jobDuration(node.job) }}
              </td>
              <td class="px-2 sm:px-3 py-2 text-[13px]">
                <div v-if="(node.job.taskCount ?? 0) === 0" class="text-xs text-text-muted">
                  no tasks
                </div>
                <div v-else class="flex items-center gap-1.5">
                  <div class="flex h-2 w-16 sm:w-28 rounded-full overflow-hidden bg-surface-sunken">
                    <div
                      v-for="(seg, i) in progressSegments(node.job)"
                      :key="i"
                      :class="seg.colorClass"
                      :style="{ width: (seg.count / (node.job.taskCount ?? 1) * 100).toFixed(1) + '%' }"
                      :title="seg.label + ': ' + seg.count"
                    />
                  </div>
                  <span class="hidden sm:inline text-xs text-text-secondary whitespace-nowrap">
                    {{ progressSummary(node.job) }}
                  </span>
                </div>
              </td>
              <td class="hidden lg:table-cell px-2 sm:px-3 py-2 text-xs text-text-muted max-w-xs truncate" :title="node.job.pendingReason ?? ''">
                {{ node.job.pendingReason || '—' }}
              </td>
            </tr>
          </tbody>
        </table>
        </div>
      </div>

      <!-- Tasks table -->
      <div class="flex flex-wrap items-center justify-between gap-2 mb-3">
        <h3 class="text-sm font-semibold uppercase tracking-wider text-text-secondary">
          Tasks
        </h3>
        <div v-if="tasks.length > 0" class="flex flex-wrap items-center gap-2 w-full sm:w-auto">
          <select
            v-model="stateFilter"
            class="px-3 py-1.5 text-sm rounded-md border border-surface-border bg-surface-primary text-text-primary focus:outline-none focus:ring-1 focus:ring-accent"
          >
            <option value="">All states</option>
            <option v-for="s in availableStates" :key="s" :value="s">{{ stateDisplayName(s) }}</option>
          </select>
          <input
            v-model="taskSearch"
            type="text"
            placeholder="Search workers..."
            class="flex-1 sm:flex-initial sm:w-64 px-3 py-1.5 text-sm rounded-md border border-surface-border bg-surface-primary text-text-primary placeholder-text-muted focus:outline-none focus:ring-1 focus:ring-accent"
          />
        </div>
      </div>

      <!-- A federated job's tasks live on the peer until the first FederationSync
           mirrors them back; name the peer and the requested count instead of a
           bare "No tasks" over an unexplained empty table. -->
      <p
        v-if="tasks.length === 0 && job.cluster && isFederated(job.cluster) && !isTerminal"
        class="py-16 text-center text-sm text-text-muted"
      >
        {{ job.taskCount ?? 0 }} task{{ (job.taskCount ?? 0) !== 1 ? 's' : '' }} on peer
        <ClusterLink :cluster="job.cluster" /> — awaiting first status report
      </p>
      <EmptyState v-else-if="tasks.length === 0" message="No tasks" />
      <EmptyState v-else-if="filteredTasks.length === 0" message="No matching tasks" />

      <template v-else>
      <!-- Mobile: card grid (one card per task) -->
      <div v-if="isMobile" class="grid grid-cols-1 gap-2">
        <div
          v-for="task in paginatedTasks"
          :key="'task-card-' + task.taskId"
          class="rounded-lg border border-surface-border bg-surface px-3 py-2"
        >
          <div class="flex items-start gap-2 flex-wrap">
            <RouterLink
              :to="`/job/${encodeURIComponent(props.jobId)}/task/${encodeURIComponent(task.taskId)}`"
              class="text-accent hover:underline font-mono text-[13px]"
            >
              Task {{ taskIndex(task.taskId) }}
            </RouterLink>
            <StatusBadge :status="task.state" size="sm" />
          </div>
          <div v-if="task.pendingReason" class="mt-1 text-xs text-status-warning" :title="task.pendingReason">
            {{ task.pendingReason }}
          </div>
          <div class="mt-1 text-xs text-text-muted font-mono break-anywhere">
            <RouterLink
              v-if="task.workerId"
              :to="`/job/${encodeURIComponent(props.jobId)}/task/${encodeURIComponent(task.taskId)}`"
              class="text-accent hover:underline"
            >
              {{ task.workerId }}
            </RouterLink>
            <template v-if="task.startedAt">
              <span v-if="task.workerId"> · </span>{{ formatTimestamp(task.startedAt) }}
            </template>
            <span> · {{ taskDuration(task) }}</span>
            <span v-if="taskExitNonZero(task)" class="text-status-danger">
              · exit {{ task.exitCode }}
            </span>
          </div>
          <div class="mt-1 text-xs break-anywhere">
            <MarkdownRenderer v-if="taskStatusTextSummary(task.taskId) && !TERMINAL_STATES.has(stateToName(task.state))" :content="taskStatusTextSummary(task.taskId)" class="text-text-secondary" />
            <span v-else-if="task.error && FAILED_TERMINAL_STATES.has(stateToName(task.state))" class="text-status-danger" :title="task.error">{{ task.error.length > 160 ? task.error.slice(0, 160) + '…' : task.error }}</span>
          </div>
          <div v-if="taskEndpoints(task.taskId).length" class="mt-1 flex flex-wrap gap-x-3 gap-y-0.5">
            <EndpointLink
              v-for="ep in taskEndpoints(task.taskId)"
              :key="ep.endpointId ?? ep.name"
              :name="ep.name"
              short
              class="text-[11px]"
            />
          </div>
          <div v-if="stateToName(task.state) === 'running'" class="mt-2 flex gap-1">
            <button
              class="px-2 py-0.5 text-[11px] font-semibold rounded bg-status-purple text-white hover:opacity-80 disabled:opacity-50"
              :disabled="profilingTaskId === task.taskId"
              @click="handleProfile(task.taskId, 'cpu', 'SPEEDSCOPE')"
            >
              {{ profilingTaskId === task.taskId ? '⏳' : 'CPU' }}
            </button>
            <button
              class="px-2 py-0.5 text-[11px] font-semibold rounded bg-status-success text-white hover:opacity-80 disabled:opacity-50"
              :disabled="profilingTaskId === task.taskId"
              @click="handleProfile(task.taskId, 'memory', 'RAW')"
            >
              {{ profilingTaskId === task.taskId ? '⏳' : 'MEM' }}
            </button>
            <RouterLink
              :to="`/job/${encodeURIComponent(props.jobId)}/task/${encodeURIComponent(task.taskId)}/threads`"
              class="px-2 py-0.5 text-[11px] font-semibold rounded bg-accent text-white hover:opacity-80 inline-block text-center no-underline"
            >
              THR
            </RouterLink>
          </div>
        </div>
      </div>

      <!-- Desktop: table.
           table-fixed only kicks in at lg, where every column is visible; below
           lg the auto layout sizes the reduced column set to content so nothing
           collides. Status text and registered endpoints live in a subtle second
           row (see taskHasDetailRow) instead of dedicated columns. -->
      <div v-else class="overflow-x-auto">
        <table class="w-full border-collapse lg:table-fixed">
          <colgroup class="hidden lg:table-column-group">
            <col class="w-[5%]" />   <!-- Task -->
            <col class="w-[12%]" />  <!-- State -->
            <col />                  <!-- Worker -->
            <col class="w-[13%]" />  <!-- Memory / Peak -->
            <col class="w-[6%]" />   <!-- CPU -->
            <col class="w-[14%]" />  <!-- Started -->
            <col class="w-[8%]" />   <!-- Duration -->
            <col class="w-[15%]" />  <!-- Profiling -->
          </colgroup>
          <thead>
            <tr class="border-b border-surface-border">
              <th class="px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleSort('task')">
                Task <span v-if="sortColumn === 'task'" class="ml-0.5">{{ sortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleSort('state')">
                State <span v-if="sortColumn === 'state'" class="ml-0.5">{{ sortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="hidden md:table-cell px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Worker</th>
              <th class="hidden lg:table-cell px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleSort('mem')">
                Memory / Peak <span v-if="sortColumn === 'mem'" class="ml-0.5">{{ sortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="hidden lg:table-cell px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleSort('cpu')">
                CPU <span v-if="sortColumn === 'cpu'" class="ml-0.5">{{ sortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="hidden md:table-cell px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Started</th>
              <th class="hidden sm:table-cell px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleSort('duration')">
                Duration <span v-if="sortColumn === 'duration'" class="ml-0.5">{{ sortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="hidden md:table-cell px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Profiling</th>
            </tr>
          </thead>
          <tbody>
            <template v-for="task in paginatedTasks" :key="task.taskId">
              <tr
                class="hover:bg-surface-raised transition-colors"
                :class="{ 'border-b border-surface-border-subtle': !taskHasDetailRow(task) }"
              >
                <td class="px-2 sm:px-3 py-2 text-[13px] font-mono align-top">
                  <RouterLink
                    :to="`/job/${encodeURIComponent(props.jobId)}/task/${encodeURIComponent(task.taskId)}`"
                    class="text-accent hover:underline"
                  >
                    {{ taskIndex(task.taskId) }}
                  </RouterLink>
                </td>
                <td class="px-2 sm:px-3 py-2 text-[13px] align-top">
                  <StatusBadge :status="task.state" size="sm" />
                  <div v-if="task.pendingReason" class="text-xs text-status-warning mt-0.5 max-w-xs truncate" :title="task.pendingReason">
                    {{ task.pendingReason }}
                  </div>
                </td>
                <td class="hidden md:table-cell px-2 sm:px-3 py-2 text-[13px] max-w-[150px] lg:max-w-none truncate align-top" :title="task.workerId ?? ''">
                  <RouterLink
                    v-if="task.workerId"
                    :to="`/job/${encodeURIComponent(props.jobId)}/task/${encodeURIComponent(task.taskId)}`"
                    class="text-accent hover:underline font-mono text-xs"
                  >
                    {{ task.workerId }}
                  </RouterLink>
                  <span v-else class="text-text-muted">&mdash;</span>
                </td>
                <td class="hidden lg:table-cell px-2 sm:px-3 py-2 text-[13px] font-mono align-top">
                  <template v-if="taskMemBytes(task.taskId) || taskPeakMemBytes(task.taskId)">
                    {{ taskMemBytes(task.taskId) ? formatBytes(taskMemBytes(task.taskId)) : '-' }}
                    <span class="text-text-muted">/</span>
                    {{ taskPeakMemBytes(task.taskId) ? formatBytes(taskPeakMemBytes(task.taskId)) : '-' }}
                  </template>
                  <span v-else class="text-text-muted">-</span>
                </td>
                <td class="hidden lg:table-cell px-2 sm:px-3 py-2 text-[13px] font-mono align-top">
                  {{ formatCpuMillicores(taskCpuMillicores(task.taskId)) }}
                </td>
                <td class="hidden md:table-cell px-2 sm:px-3 py-2 text-[13px] font-mono text-text-secondary align-top">
                  {{ formatTimestamp(task.startedAt) }}
                </td>
                <td class="hidden sm:table-cell px-2 sm:px-3 py-2 text-[13px] font-mono text-text-secondary align-top">
                  {{ taskDuration(task) }}
                </td>
                <td class="hidden md:table-cell px-2 sm:px-3 py-2 text-[13px] align-top">
                  <div
                    v-if="stateToName(task.state) === 'running'"
                    class="inline-flex items-center rounded-md border border-surface-border divide-x divide-surface-border overflow-hidden"
                  >
                    <button
                      class="px-1.5 py-1 text-[11px] font-semibold text-status-purple hover:bg-status-purple-bg transition-colors disabled:opacity-50"
                      :disabled="profilingTaskId === task.taskId"
                      title="CPU profile (10s)"
                      @click="handleProfile(task.taskId, 'cpu', 'SPEEDSCOPE')"
                    >
                      {{ profilingTaskId === task.taskId ? '⏳' : 'CPU' }}
                    </button>
                    <button
                      class="px-1.5 py-1 text-[11px] font-semibold text-status-success hover:bg-status-success-bg transition-colors disabled:opacity-50"
                      :disabled="profilingTaskId === task.taskId"
                      title="Memory profile (10s)"
                      @click="handleProfile(task.taskId, 'memory', 'RAW')"
                    >
                      {{ profilingTaskId === task.taskId ? '⏳' : 'MEM' }}
                    </button>
                    <RouterLink
                      :to="`/job/${encodeURIComponent(props.jobId)}/task/${encodeURIComponent(task.taskId)}/threads`"
                      class="px-1.5 py-1 text-[11px] font-semibold text-accent hover:bg-accent-subtle no-underline transition-colors"
                      title="Thread dump"
                    >
                      THR
                    </RouterLink>
                  </div>
                  <span v-else class="text-text-muted">&mdash;</span>
                </td>
              </tr>
              <!-- Subtle detail row: status summary and/or registered endpoints. -->
              <tr
                v-if="taskHasDetailRow(task)"
                class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
              >
                <td class="px-2 sm:px-3" />
                <td colspan="7" class="px-2 sm:px-3 pb-2 pt-0 text-[11px]">
                  <div class="space-y-1">
                    <div v-if="taskHasStatusDetail(task)" class="flex items-baseline gap-2">
                      <span class="shrink-0 inline-flex items-center gap-1 text-text-muted uppercase tracking-wider text-[10px] font-semibold">
                        <span aria-hidden="true">↳</span>Status
                      </span>
                      <span class="min-w-0">
                        <span v-if="taskExitNonZero(task)" class="text-status-danger font-mono">exit {{ task.exitCode }}</span>
                        <MarkdownRenderer v-else-if="taskStatusTextSummary(task.taskId) && !TERMINAL_STATES.has(stateToName(task.state))" :content="taskStatusTextSummary(task.taskId)" class="text-text-secondary" />
                        <span v-else-if="task.error && FAILED_TERMINAL_STATES.has(stateToName(task.state))" class="text-status-danger break-anywhere" :title="task.error">{{ task.error.length > 160 ? task.error.slice(0, 160) + '…' : task.error }}</span>
                      </span>
                    </div>
                    <div v-if="taskEndpoints(task.taskId).length" class="flex flex-wrap items-center gap-x-3 gap-y-1">
                      <span class="shrink-0 inline-flex items-center gap-1 text-text-muted uppercase tracking-wider text-[10px] font-semibold">
                        <span aria-hidden="true">↳</span>Endpoints
                      </span>
                      <EndpointLink
                        v-for="ep in taskEndpoints(task.taskId)"
                        :key="ep.endpointId ?? ep.name"
                        :name="ep.name"
                        short
                        class="text-[11px]"
                      />
                    </div>
                  </div>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>

      <!-- Pagination (shared between mobile cards and desktop table) -->
      <div v-if="totalTaskPages > 1" class="mt-2 flex items-center justify-between px-2 sm:px-3 py-2 text-xs text-text-secondary border-t border-surface-border">
        <span>
          {{ taskPage * TASK_PAGE_SIZE + 1 }}&ndash;{{ Math.min((taskPage + 1) * TASK_PAGE_SIZE, filteredTasks.length) }}
          of {{ filteredTasks.length }} tasks
        </span>
        <div class="flex items-center gap-1">
          <button
            :disabled="taskPage === 0"
            class="px-2 py-1 rounded hover:bg-surface-raised disabled:opacity-30 disabled:cursor-not-allowed"
            @click="taskPage--"
          >
            &larr; Prev
          </button>
          <span class="px-2 font-mono">{{ taskPage + 1 }} / {{ totalTaskPages }}</span>
          <button
            :disabled="taskPage >= totalTaskPages - 1"
            class="px-2 py-1 rounded hover:bg-surface-raised disabled:opacity-30 disabled:cursor-not-allowed"
            @click="taskPage++"
          >
            Next &rarr;
          </button>
        </div>
      </div>
      </template>

      <!-- Job logs -->
      <div class="mt-6 mb-6">
        <h3 class="text-sm font-semibold uppercase tracking-wider text-text-secondary mb-3">
          Job Logs
        </h3>
        <!-- Logs are served under the job id in the shared finelog; a federated
             job passes its cluster so LogViewer filters to the peer's rows. -->
        <LogViewer :task-id="jobId" :cluster="job?.cluster" />
      </div>
    </template>

  </PageShell>
</template>
