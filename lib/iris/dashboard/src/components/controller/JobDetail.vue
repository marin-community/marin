<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { controllerRpcCall } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { stateToName, stateDisplayName } from '@/types/status'
import type {
  JobStatus, TaskStatus, LaunchJobRequest, JobQuery,
  GetJobStatusResponse, ListTasksResponse, ListJobsResponse,
  ResourceUsage,
} from '@/types/rpc'
import { timestampMs, formatTimestamp, formatDuration, formatRelativeTime, formatBytes, formatCpuMillicores, formatDeviceConfig } from '@/utils/formatting'
import { getLeafJobName } from '@/utils/jobTree'
import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import LogViewer from '@/components/shared/LogViewer.vue'

const props = defineProps<{
  jobId: string
}>()

const TERMINAL_STATES = new Set(['succeeded', 'failed', 'killed', 'worker_failed', 'preempted', 'unschedulable'])

// -- State --

const job = ref<JobStatus | null>(null)
const jobRequest = ref<LaunchJobRequest | null>(null)
const tasks = ref<TaskStatus[]>([])
const childJobsByParent = ref<Map<string, JobStatus[]>>(new Map())
const expandedChildJobs = ref<Set<string>>(new Set())
const loadingChildJobs = ref<Set<string>>(new Set())
const loading = ref(true)
const error = ref<string | null>(null)
const profilingTaskId = ref<string | null>(null)
const copiedName = ref(false)
const taskSearch = ref('')
const stateFilter = ref('')

type SortColumn = 'task' | 'state' | 'mem' | 'peakMem' | 'cpu' | 'duration'
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

async function copyJobName() {
  const name = job.value?.name
  if (!name) return
  await navigator.clipboard.writeText(name)
  copiedName.value = true
  setTimeout(() => { copiedName.value = false }, 1500)
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

// Auto-refresh while job is not terminal
const isTerminal = computed(() => {
  if (!job.value) return false
  return TERMINAL_STATES.has(stateToName(job.value.state))
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

function formatMemMb(usage: ResourceUsage | undefined): string {
  if (!usage?.memoryMb) return '-'
  const mb = parseInt(usage.memoryMb, 10)
  return `${mb} MB`
}

function formatPeakMemMb(usage: ResourceUsage | undefined): string {
  if (!usage?.memoryPeakMb) return '-'
  const mb = parseInt(usage.memoryPeakMb, 10)
  return `${mb} MB`
}

function formatCpu(usage: ResourceUsage | undefined): string {
  return formatCpuMillicores(usage?.cpuMillicores)
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

async function loadChildJobs(parentJobId: string) {
  if (loadingChildJobs.value.has(parentJobId)) return
  const nextLoading = new Set(loadingChildJobs.value)
  nextLoading.add(parentJobId)
  loadingChildJobs.value = nextLoading
  try {
    const children = await fetchChildJobs(parentJobId)
    const nextChildren = new Map(childJobsByParent.value)
    nextChildren.set(parentJobId, children)
    childJobsByParent.value = nextChildren
  } finally {
    const doneLoading = new Set(loadingChildJobs.value)
    doneLoading.delete(parentJobId)
    loadingChildJobs.value = doneLoading
  }
}

function toggleExpandedChildJob(jobStatus: JobStatus) {
  const next = new Set(expandedChildJobs.value)
  if (next.has(jobStatus.jobId)) {
    next.delete(jobStatus.jobId)
  } else {
    next.add(jobStatus.jobId)
    if (!childJobsByParent.value.has(jobStatus.jobId)) {
      void loadChildJobs(jobStatus.jobId)
    }
  }
  expandedChildJobs.value = next
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
  const preempted = counts['preempted'] ?? 0
  const killed = counts['killed'] ?? 0
  const pending = total - succeeded - running - building - assigned - failed - workerFailed - preempted - killed
  return [
    { count: succeeded, colorClass: SEGMENT_COLORS['succeeded'], label: 'succeeded' },
    { count: running, colorClass: SEGMENT_COLORS['running'], label: 'running' },
    { count: building, colorClass: SEGMENT_COLORS['building'], label: 'building' },
    { count: assigned, colorClass: SEGMENT_COLORS['assigned'], label: 'assigned' },
    { count: failed, colorClass: SEGMENT_COLORS['failed'], label: 'failed' },
    { count: workerFailed, colorClass: SEGMENT_COLORS['worker_failed'], label: 'worker_failed' },
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
    else if (state === 'failed' || state === 'worker_failed' || state === 'preempted') counts.failed++
  }
  return counts
})

const FAILED_ATTEMPT_STATES = new Set(['failed', 'worker_failed'])
const MAX_FAILURE_EXAMPLES = 5

interface FailedAttemptSummary {
  taskId: string
  taskIndex: string
  attemptId: number
  error: string
  finishedAtMs: number
  isWorkerFailure: boolean
}

const recentFailedAttempts = computed<FailedAttemptSummary[]>(() => {
  const failures: FailedAttemptSummary[] = []
  for (const task of tasks.value) {
    for (const attempt of task.attempts ?? []) {
      if (!FAILED_ATTEMPT_STATES.has(stateToName(attempt.state))) continue
      failures.push({
        taskId: task.taskId,
        taskIndex: taskIndex(task.taskId),
        attemptId: attempt.attemptId,
        error: attempt.error ?? '',
        finishedAtMs: timestampMs(attempt.finishedAt),
        isWorkerFailure: attempt.isWorkerFailure ?? false,
      })
    }
  }
  failures.sort((a, b) => b.finishedAtMs - a.finishedAtMs)
  return failures
})

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
  succeeded: 4, killed: 5, failed: 6, worker_failed: 7, preempted: 8, unschedulable: 9,
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
        cmp = (parseInt(a.resourceUsage?.memoryMb ?? '0') || 0) - (parseInt(b.resourceUsage?.memoryMb ?? '0') || 0)
        break
      case 'peakMem':
        cmp = (parseInt(a.resourceUsage?.memoryPeakMb ?? '0') || 0) - (parseInt(b.resourceUsage?.memoryPeakMb ?? '0') || 0)
        break
      case 'cpu':
        cmp = (a.resourceUsage?.cpuMillicores ?? 0) - (b.resourceUsage?.cpuMillicores ?? 0)
        break
      case 'duration':
        cmp = taskDurationMs(a) - taskDurationMs(b)
        break
    }
    return cmp * dir
  })
  return result
})

// -- Profiling --

function buildProfileType(profilerType: string, format: string | null): Record<string, unknown> {
  if (profilerType === 'cpu') return { cpu: { format: format ?? 'SPEEDSCOPE' } }
  if (profilerType === 'memory') return { memory: { format: format ?? 'RAW' } }
  return { threads: {} }
}

async function handleProfile(taskId: string, profilerType: string, format: string | null) {
  profilingTaskId.value = taskId
  try {
    const body = {
      target: taskId,
      durationSeconds: 10,
      profileType: buildProfileType(profilerType, format),
    }
    const resp = await controllerRpcCall<{ profileData?: string; error?: string }>('ProfileTask', body)
    if (resp.error) {
      alert(`${profilerType.toUpperCase()} profile failed: ${resp.error}`)
      return
    }
    if (resp.profileData) {
      const bin = atob(resp.profileData)
      const bytes = new Uint8Array(bin.length)
      for (let i = 0; i < bin.length; i++) {
        bytes[i] = bin.charCodeAt(i)
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
    }
  } catch (e) {
    alert(`${profilerType.toUpperCase()} profile failed: ${e instanceof Error ? e.message : e}`)
  } finally {
    profilingTaskId.value = null
  }
}
</script>

<template>
  <PageShell :title="pageTitle" back-to="/" back-label="Jobs">
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

      <!-- Pending reason banner -->
      <div
        v-if="job.pendingReason"
        class="mb-4 px-4 py-3 bg-status-warning-bg border border-status-warning-border rounded-lg"
      >
        <span class="font-semibold text-status-warning text-sm">Scheduling Diagnostic:</span>
        <pre class="mt-2 p-3 bg-surface rounded text-xs font-mono whitespace-pre-wrap">{{ job.pendingReason }}</pre>
      </div>

      <!-- Recent task attempt failures callout -->
      <div
        v-if="recentFailedAttempts.length > 0"
        class="mb-4 px-4 py-3 bg-status-danger-bg border border-status-danger-border rounded-lg"
      >
        <span class="font-semibold text-status-danger text-sm">
          {{ recentFailedAttempts.length }} recent task attempt failure{{ recentFailedAttempts.length !== 1 ? 's' : '' }}
        </span>
        <div class="mt-2 flex flex-col gap-1">
          <div
            v-for="f in recentFailedAttempts.slice(0, MAX_FAILURE_EXAMPLES)"
            :key="`${f.taskId}-${f.attemptId}`"
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
            <span v-if="f.error" class="text-status-danger"> · {{ f.error.length > 120 ? f.error.slice(0, 120) + '…' : f.error }}</span>
          </div>
          <span
            v-if="recentFailedAttempts.length > MAX_FAILURE_EXAMPLES"
            class="text-xs text-text-muted"
          >
            … and {{ recentFailedAttempts.length - MAX_FAILURE_EXAMPLES }} more
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
            {{ job.failureCount ?? 0 }}
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

        <InfoCard title="Resources (per VM)">
          <InfoRow label="CPU">{{ cpuDisplay }}</InfoRow>
          <InfoRow label="Memory">{{ memoryDisplay }}</InfoRow>
          <InfoRow label="Disk">{{ diskDisplay }}</InfoRow>
          <InfoRow label="Accelerator">{{ acceleratorDisplay }}</InfoRow>
          <InfoRow label="Replicas">{{ tasks.length || '-' }}</InfoRow>
        </InfoCard>
      </div>

      <!-- Constraints -->
      <div
        v-if="jobRequest?.constraints && jobRequest.constraints.length > 0"
        class="mb-6 rounded-lg border border-surface-border bg-surface px-4 py-3"
      >
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">
          Constraints
        </h3>
        <div class="flex flex-wrap gap-1.5">
          <span
            v-for="(c, i) in jobRequest.constraints"
            :key="i"
            class="inline-block rounded bg-surface-sunken px-2 py-0.5 font-mono text-xs text-text-secondary"
          >
            {{ c.key }} {{ c.op }} {{ c.value?.stringValue ?? c.value?.intValue ?? '' }}
          </span>
        </div>
      </div>

      <!-- Child Jobs -->
      <div v-if="flattenedChildJobs.length > 0" class="mb-6">
        <div class="mb-3 flex items-center justify-between gap-3">
          <h3 class="text-sm font-semibold uppercase tracking-wider text-text-secondary">
            Child Jobs
          </h3>
        </div>
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border">
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleChildSort('name')">
                Name <span v-if="childSortColumn === 'name'" class="ml-0.5">{{ childSortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleChildSort('state')">
                State <span v-if="childSortColumn === 'state'" class="ml-0.5">{{ childSortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleChildSort('duration')">
                Duration <span v-if="childSortColumn === 'duration'" class="ml-0.5">{{ childSortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Tasks</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Diagnostic</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="node in flattenedChildJobs"
              :key="node.job.jobId"
              class="group/row border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td
                class="px-3 py-2 text-[13px]"
                :style="{ paddingLeft: (node.depth * 20 + 12) + 'px' }"
              >
                <span class="inline-flex items-center gap-1">
                  <button
                    v-if="node.job.hasChildren"
                    class="text-text-muted hover:text-text select-none w-4 text-center text-xs"
                    @click.stop="toggleExpandedChildJob(node.job)"
                  >
                    {{ loadingChildJobs.has(node.job.jobId) ? '…' : (expandedChildJobs.has(node.job.jobId) ? '▼' : '▶') }}
                  </button>
                  <span v-else class="w-4" />
                  <RouterLink
                    :to="'/job/' + encodeURIComponent(node.job.jobId)"
                    class="text-accent hover:underline font-mono"
                  >
                    {{ getLeafJobName(node.job.name) }}
                  </RouterLink>
                </span>
              </td>
              <td class="px-3 py-2 text-[13px]">
                <StatusBadge :status="node.job.state" size="sm" />
              </td>
              <td class="px-3 py-2 text-[13px] text-text-secondary font-mono">
                {{ jobDuration(node.job) }}
              </td>
              <td class="px-3 py-2 text-[13px]">
                <div v-if="(node.job.taskCount ?? 0) === 0" class="text-xs text-text-muted">
                  no tasks
                </div>
                <div v-else class="flex items-center gap-1.5">
                  <div class="flex h-2 w-28 rounded-full overflow-hidden bg-surface-sunken">
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
              </td>
              <td class="px-3 py-2 text-xs text-text-muted max-w-xs truncate" :title="node.job.pendingReason ?? ''">
                {{ node.job.pendingReason || '—' }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Tasks table -->
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-semibold uppercase tracking-wider text-text-secondary">
          Tasks
        </h3>
        <div v-if="tasks.length > 0" class="flex items-center gap-2">
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
            class="px-3 py-1.5 text-sm rounded-md border border-surface-border bg-surface-primary text-text-primary placeholder-text-muted focus:outline-none focus:ring-1 focus:ring-accent w-64"
          />
        </div>
      </div>

      <EmptyState v-if="tasks.length === 0" message="No tasks" />
      <EmptyState v-else-if="filteredTasks.length === 0" message="No matching tasks" />

      <div v-else>
        <table class="w-full border-collapse table-fixed">
          <colgroup>
            <col class="w-[4%]" />
            <col class="w-[9%]" />
            <col />
            <col class="w-[6%]" />
            <col class="w-[5%]" />
            <col class="w-[13%]" />
            <col class="w-[8%]" />
            <col class="w-[4%]" />
            <col class="w-[10%]" />
            <col class="w-[11%]" />
          </colgroup>
          <thead>
            <tr class="border-b border-surface-border">
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleSort('task')">
                Task <span v-if="sortColumn === 'task'" class="ml-0.5">{{ sortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleSort('state')">
                State <span v-if="sortColumn === 'state'" class="ml-0.5">{{ sortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Worker</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleSort('mem')">
                Mem <span v-if="sortColumn === 'mem'" class="ml-0.5">{{ sortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleSort('peakMem')">
                Peak Mem <span v-if="sortColumn === 'peakMem'" class="ml-0.5">{{ sortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleSort('cpu')">
                CPU <span v-if="sortColumn === 'cpu'" class="ml-0.5">{{ sortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Started</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary cursor-pointer select-none hover:text-text-primary" @click="toggleSort('duration')">
                Duration <span v-if="sortColumn === 'duration'" class="ml-0.5">{{ sortDir === 'asc' ? '▲' : '▼' }}</span>
              </th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Exit</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Error</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Profiling</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="task in filteredTasks"
              :key="task.taskId"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td class="px-3 py-2 text-[13px] font-mono">
                <RouterLink
                  :to="`/job/${encodeURIComponent(props.jobId)}/task/${encodeURIComponent(task.taskId)}`"
                  class="text-accent hover:underline"
                >
                  {{ taskIndex(task.taskId) }}
                </RouterLink>
              </td>
              <td class="px-3 py-2 text-[13px]">
                <StatusBadge :status="task.state" size="sm" />
                <div v-if="task.pendingReason" class="text-xs text-status-warning mt-0.5 max-w-xs truncate" :title="task.pendingReason">
                  {{ task.pendingReason }}
                </div>
              </td>
              <td class="px-3 py-2 text-[13px] truncate" :title="task.workerId ?? ''">
                <RouterLink
                  v-if="task.workerId"
                  :to="'/worker/' + encodeURIComponent(task.workerId)"
                  class="text-accent hover:underline font-mono text-xs"
                >
                  {{ task.workerId }}
                </RouterLink>
                <span v-else class="text-text-muted">&mdash;</span>
              </td>
              <td class="px-3 py-2 text-[13px] font-mono">
                {{ formatMemMb(task.resourceUsage) }}
              </td>
              <td class="px-3 py-2 text-[13px] font-mono">
                {{ formatPeakMemMb(task.resourceUsage) }}
              </td>
              <td class="px-3 py-2 text-[13px] font-mono">
                {{ formatCpu(task.resourceUsage) }}
              </td>
              <td class="px-3 py-2 text-[13px] font-mono text-text-secondary">
                {{ formatTimestamp(task.startedAt) }}
              </td>
              <td class="px-3 py-2 text-[13px] font-mono text-text-secondary">
                {{ taskDuration(task) }}
              </td>
              <td class="px-3 py-2 text-[13px] font-mono">
                {{ TERMINAL_STATES.has(stateToName(task.state)) && task.exitCode !== undefined ? task.exitCode : '-' }}
              </td>
              <td class="px-3 py-2 text-xs text-text-muted max-w-xs truncate" :title="task.error ?? ''">
                {{ task.error || '-' }}
              </td>
              <td class="px-3 py-2 text-[13px]">
                <div v-if="stateToName(task.state) === 'running'" class="flex gap-1">
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
                <span v-else class="text-text-muted">&mdash;</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Job logs -->
      <div class="mt-6 mb-6">
        <h3 class="text-sm font-semibold uppercase tracking-wider text-text-secondary mb-3">
          Job Logs
        </h3>
        <LogViewer :task-id="jobId" />
      </div>
    </template>

  </PageShell>
</template>
