<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import { useControllerRpc, useLogServerStatsRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { stateToName } from '@/types/status'
import { useBackends } from '@/composables/useBackends'
import {
  isLocal,
  LOCAL_CLUSTER,
  type TaskStatus,
  type GetTaskStatusResponse,
  type EndpointInfo,
  type ListEndpointsResponse,
} from '@/types/rpc'
import { timestampMs, formatBytes, formatCpuMillicores, formatDuration, formatRelativeTime } from '@/utils/formatting'
import { decodeArrowIpc } from '@/utils/arrow'
import { detailSql } from '@/utils/taskStatus'

import { controllerRpcCall } from '@/composables/useRpc'
import { useProfileAction } from '@/composables/useProfileAction'
import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import ResourceGauge from '@/components/shared/ResourceGauge.vue'
import Sparkline from '@/components/shared/Sparkline.vue'
import ProfileButtons from '@/components/shared/ProfileButtons.vue'
import ProfileLink from '@/components/shared/ProfileLink.vue'
import LogViewer from '@/components/shared/LogViewer.vue'
import TaskEventTimeline from '@/components/shared/TaskEventTimeline.vue'
import MarkdownRenderer from '@/components/shared/MarkdownRenderer.vue'
import CopyButton from '@/components/shared/CopyButton.vue'
import EndpointLink from '@/components/shared/EndpointLink.vue'
import ClusterLink from '@/components/shared/ClusterLink.vue'

const props = defineProps<{
  jobId: string
  taskId: string
}>()

const { multiBackend } = useBackends()

const {
  data: taskResponse,
  loading,
  error,
  refresh: fetchTask,
} = useControllerRpc<GetTaskStatusResponse>('GetTaskStatus', () => ({ taskId: props.taskId }))

const task = computed(() => taskResponse.value?.task ?? null)

const jobResources = computed(() => taskResponse.value?.jobResources ?? null)

// Root-cause log lines the controller distilled from a failed task's logs.
const rootCauseHighlights = computed(() => taskResponse.value?.rootCauseHighlights ?? [])

// Endpoints this task registered with the controller. Each is reachable
// through the controller's reverse proxy, so we render a link to jump to an
// attached dashboard/server without looking up its address.
const {
  data: endpointsResponse,
  refresh: fetchEndpoints,
} = useControllerRpc<ListEndpointsResponse>('ListEndpoints', () => ({ taskIds: [props.taskId] }))

const endpoints = computed<EndpointInfo[]>(() => endpointsResponse.value?.endpoints ?? [])

const normalizedState = computed(() => (task.value ? stateToName(task.value.state) : ''))

const isActive = computed(() => {
  const s = normalizedState.value
  return s === 'running' || s === 'building' || s === 'assigned'
})

// The attempt the page is focused on. Defaults to the task's current attempt;
// clicking a row in the Attempts table (selectAttempt) points it elsewhere.
// Drives the Events panel so it follows the same attempt as the log viewer.
const selectedAttemptId = ref<number | undefined>(undefined)
const effectiveAttemptId = computed(() => selectedAttemptId.value ?? task.value?.currentAttemptId)

const startedMs = computed(() => timestampMs(task.value?.startedAt))
const finishedMs = computed(() => timestampMs(task.value?.finishedAt))

const duration = computed(() => {
  if (!startedMs.value) return '-'
  return formatDuration(startedMs.value, finishedMs.value || undefined)
})

const startedDisplay = computed(() =>
  startedMs.value ? formatRelativeTime(startedMs.value) : '-'
)

// --- Per-task resource history sourced from finelog stats (iris.task) ---
//
// One row per attempt-resource update emitted by the worker. The namespace
// is registered eagerly by every worker at startup, so any task we can
// navigate to has a registered table. Rows come back ts DESC; reverse for
// the sparkline (oldest -> newest). Filtered by attempt_id so retried tasks
// do not mix samples from prior attempts.
interface TaskStatRow {
  ts?: string
  cpu_millicores?: number
  memory_mb?: number
  memory_peak_mb?: number
  disk_mb?: number
}

interface QueryResponse {
  arrowIpc?: string
}

function buildTaskStatsSql(taskId: string, attemptId: number | undefined): string {
  // QueryRequest has no param binding; manual DuckDB single-quote escape.
  const escaped = taskId.replace(/'/g, "''")
  const attemptPredicate =
    attemptId !== undefined && attemptId !== null
      ? `AND attempt_id = ${Number(attemptId)}`
      : ''
  return `
SELECT ts, cpu_millicores, memory_mb, memory_peak_mb, disk_mb
FROM "iris.task"
WHERE task_id = '${escaped}'
${attemptPredicate}
ORDER BY ts DESC
LIMIT 200
`.trim()
}

const { data: taskStatsData, refresh: fetchTaskStats } = useLogServerStatsRpc<QueryResponse>(
  'Query',
  () => ({ sql: buildTaskStatsSql(props.taskId, task.value?.currentAttemptId) }),
)

const taskStatsRows = computed<TaskStatRow[]>(() => {
  const ipc = taskStatsData.value?.arrowIpc
  if (!ipc) return []
  return decodeArrowIpc(ipc).rows as TaskStatRow[]
})

// Latest status text row for this task (see detailSql in utils/taskStatus).
interface StatusTextRow {
  status_text_detail_md?: string
  status_text_summary_md?: string
}

const { data: statusTextData, refresh: fetchStatusText } = useLogServerStatsRpc<QueryResponse>(
  'Query',
  () => ({ sql: detailSql(props.taskId) }),
)

const statusTextDetail = computed<string>(() => {
  const ipc = statusTextData.value?.arrowIpc
  if (!ipc) return ''
  const rows = decodeArrowIpc(ipc).rows as StatusTextRow[]
  return rows[0]?.status_text_detail_md ?? ''
})

// --- Scheduling / lifecycle events from finelog stats (iris.task_event) ---
//
// One row per Kubernetes event the worker relayed for this task (pod, kueue,
// container). Newest-first and filtered by attempt so a retry does not show the
// prior attempt's events. This is the primary signal for a task wedged in
// BUILDING/pending: it surfaces the k8s reasons (SchedulingGated,
// ImagePullBackOff, quota/topology denials) behind the wait.
const TASK_EVENT_NAMESPACE = 'iris.task_event'

interface TaskEventRow {
  ts?: number
  type?: string
  reason?: string
  message?: string
  source?: string
  count?: number
}

function buildTaskEventsSql(taskId: string, attemptId: number | undefined): string {
  // QueryRequest has no param binding; manual DuckDB single-quote escape.
  const escaped = taskId.replace(/'/g, "''")
  const attemptPredicate =
    attemptId !== undefined && attemptId !== null
      ? `AND attempt_id = ${Number(attemptId)}`
      : ''
  return `
SELECT ts, type, reason, message, source, count
FROM "${TASK_EVENT_NAMESPACE}"
WHERE task_id = '${escaped}'
${attemptPredicate}
ORDER BY ts DESC
LIMIT 200
`.trim()
}

const { data: taskEventsData, refresh: fetchTaskEvents } = useLogServerStatsRpc<QueryResponse>(
  'Query',
  () => ({ sql: buildTaskEventsSql(props.taskId, effectiveAttemptId.value) }),
)

const taskEvents = computed<TaskEventRow[]>(() => {
  const ipc = taskEventsData.value?.arrowIpc
  if (!ipc) return []
  return decodeArrowIpc(ipc).rows as TaskEventRow[]
})

const orderedTaskStats = computed(() => taskStatsRows.value.slice().reverse())

// Latest sample drives the current-value gauges and labels. resource_usage
// is no longer populated on TaskStatus — the iris.task stats namespace is
// the single source of truth for per-attempt usage.
const latestStat = computed<TaskStatRow | null>(() => taskStatsRows.value[0] ?? null)
const cpuUsed = computed(() => Number(latestStat.value?.cpu_millicores ?? 0) / 1000)
const memUsedMb = computed(() => Number(latestStat.value?.memory_mb ?? 0))
const memPeakMb = computed(() => Number(latestStat.value?.memory_peak_mb ?? 0))
const diskUsedMb = computed(() => Number(latestStat.value?.disk_mb ?? 0))

const cpuHistory = computed(() =>
  orderedTaskStats.value.map((r) => Number(r.cpu_millicores ?? 0) / 1000)
)
const memHistory = computed(() =>
  orderedTaskStats.value.map((r) => Number(r.memory_mb ?? 0))
)

// Use job-level resource limits for gauge totals when available.
const cpuTotal = computed(() => {
  const jobCpu = (jobResources.value?.cpuMillicores ?? 0) / 1000
  if (jobCpu > 0) return jobCpu
  const maxObserved = Math.max(...cpuHistory.value, cpuUsed.value, 0)
  if (maxObserved <= 0) return 1
  return Math.max(1, maxObserved * 1.5)
})

const memTotalMb = computed(() => {
  const jobMemBytes = jobResources.value?.memoryBytes
  if (jobMemBytes) return parseFloat(jobMemBytes) / (1024 * 1024)
  const historyMax = memHistory.value.length > 0 ? Math.max(...memHistory.value) : 0
  const best = Math.max(historyMax, memPeakMb.value, memUsedMb.value)
  if (best <= 0) return 1
  return best * 1.2
})

const diskTotalMb = computed(() => {
  const jobDiskBytes = jobResources.value?.diskBytes
  if (jobDiskBytes) return parseFloat(jobDiskBytes) / (1024 * 1024)
  if (diskUsedMb.value <= 0) return 1
  return diskUsedMb.value * 2
})

// Build metrics
const buildDuration = computed(() => {
  const bm = task.value?.buildMetrics
  if (!bm?.buildStarted || !bm?.buildFinished) return null
  return formatDuration(timestampMs(bm.buildStarted), timestampMs(bm.buildFinished))
})

// Auto-refresh while task is active
const { active: autoRefreshActive, start: startRefresh, stop: stopRefresh } = useAutoRefresh(
  fetchTask,
  5_000,
  false,
)
const { start: startStatsRefresh, stop: stopStatsRefresh } = useAutoRefresh(
  fetchTaskStats,
  5_000,
  false,
)
const { start: startStatusTextRefresh, stop: stopStatusTextRefresh } = useAutoRefresh(
  fetchStatusText,
  5_000,
  false,
)
const { start: startEventsRefresh, stop: stopEventsRefresh } = useAutoRefresh(
  fetchTaskEvents,
  5_000,
  false,
)
const { start: startEndpointsRefresh, stop: stopEndpointsRefresh } = useAutoRefresh(
  fetchEndpoints,
  5_000,
  false,
)

// Re-query events as soon as the focused attempt changes so a manual attempt
// pick (or a new attempt appearing) reflects immediately, not on the next poll.
watch(effectiveAttemptId, () => {
  fetchTaskEvents()
})

watch(isActive, (active) => {
  if (active) {
    startRefresh()
    startStatsRefresh()
    startStatusTextRefresh()
    startEventsRefresh()
    startEndpointsRefresh()
  } else {
    stopRefresh()
    stopStatsRefresh()
    stopStatusTextRefresh()
    stopEventsRefresh()
    stopEndpointsRefresh()
  }
})

onMounted(async () => {
  await fetchTask()
  fetchTaskStats()
  fetchStatusText()
  fetchTaskEvents()
  fetchEndpoints()
  if (isActive.value) {
    startRefresh()
    startStatsRefresh()
    startStatusTextRefresh()
    startEventsRefresh()
    startEndpointsRefresh()
  }
})

const router = useRouter()
const { profiling, profile } = useProfileAction(controllerRpcCall, () => props.taskId)

function handleProfile(type: 'cpu' | 'memory' | 'threads') {
  if (type === 'threads') {
    router.push(`/job/${encodeURIComponent(props.jobId)}/task/${encodeURIComponent(props.taskId)}/threads`)
  } else {
    profile(type)
  }
}

const logViewerRef = ref<{ selectedAttemptId: number } | null>(null)

function selectAttempt(attemptId: number) {
  selectedAttemptId.value = attemptId
  if (logViewerRef.value) {
    logViewerRef.value.selectedAttemptId = attemptId
  }
  const logsEl = document.getElementById('task-logs-section')
  if (logsEl) logsEl.scrollIntoView({ behavior: 'smooth' })
}

// Re-fetch when navigating between tasks (Vue Router reuses the component).
// Clear stale data first so loading/error states render correctly if the fetch fails.
watch(() => props.taskId, async () => {
  taskResponse.value = null
  taskStatsData.value = null
  taskEventsData.value = null
  endpointsResponse.value = null
  // Re-follow the new task's current attempt rather than the previous pick.
  selectedAttemptId.value = undefined
  stopRefresh()
  stopStatsRefresh()
  stopEventsRefresh()
  stopEndpointsRefresh()
  await fetchTask()
  fetchTaskStats()
  fetchTaskEvents()
  fetchEndpoints()
  if (isActive.value) {
    startRefresh()
    startStatsRefresh()
    startEventsRefresh()
    startEndpointsRefresh()
  }
})
</script>

<template>
  <PageShell
    :title="`Task ${taskId}`"
    :back-to="`/job/${jobId}`"
    back-label="Back to Job"
  >
    <!-- Loading -->
    <div
      v-if="loading && !task"
      class="flex items-center justify-center py-16 text-text-muted text-sm"
    >
      Loading task...
    </div>

    <!-- Error -->
    <div
      v-else-if="error && !task"
      class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <template v-else-if="task">
      <!-- Status header cards -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        <!-- Status card -->
        <InfoCard title="Status">
          <InfoRow label="State">
            <StatusBadge :status="task.state" size="sm" />
          </InfoRow>
          <!-- Human-readable status for a waiting/building task (e.g. the Kueue
               admission detail). Often long and multi-line, so it renders as a
               full-width callout under a neutral label rather than a row. -->
          <div v-if="task.statusMessage" class="pt-0.5">
            <div class="text-sm text-text-secondary mb-1">Status</div>
            <div
              class="rounded border border-surface-border bg-surface-sunken px-2.5 py-2
                     font-mono text-xs leading-relaxed text-text-secondary
                     whitespace-pre-wrap break-words"
            >
              {{ task.statusMessage }}
            </div>
          </div>
          <InfoRow v-if="task.workerId" label="Worker">
            <!-- A federated task's worker is an opaque peer-side id with no local
                 worker row, so /worker/<id> would 404 — render it as plain text. -->
            <RouterLink
              v-if="isLocal(task.cluster)"
              :to="`/worker/${task.workerId}`"
              class="font-mono text-accent hover:underline"
            >
              {{ task.workerId }}
            </RouterLink>
            <span v-else class="font-mono text-text-secondary">{{ task.workerId }}</span>
          </InfoRow>
          <InfoRow label="Started">
            <span class="font-mono">{{ startedDisplay }}</span>
          </InfoRow>
          <InfoRow label="Duration">
            <span class="font-mono">{{ duration }}</span>
          </InfoRow>
          <InfoRow v-if="task.currentAttemptId" label="Attempt">
            <span class="font-mono">{{ task.currentAttemptId }}</span>
          </InfoRow>
          <InfoRow v-if="task.exitCode !== undefined" label="Exit Code">
            <span
              class="font-mono"
              :class="task.exitCode === 0 ? 'text-status-success' : 'text-status-danger'"
            >
              {{ task.exitCode }}
            </span>
          </InfoRow>
          <InfoRow v-if="task.pendingReason" label="Pending Reason">
            <span class="text-status-warning">{{ task.pendingReason }}</span>
          </InfoRow>
          <InfoRow v-if="multiBackend && task.backendId" label="Backend">
            <span class="font-mono">{{ task.backendId }}</span>
          </InfoRow>
          <!-- Cluster: every task carries a cluster coordinate (`'local'` by
               default); links inward to the parent's jobs list filtered to it. -->
          <InfoRow label="Cluster">
            <ClusterLink :cluster="task.cluster ?? LOCAL_CLUSTER" />
          </InfoRow>
          <div v-if="isActive" class="mt-3 pt-3 border-t border-surface-border">
            <ProfileButtons :profiling="profiling" @profile="handleProfile" />
          </div>
        </InfoCard>

        <!-- Resources card. Driven entirely by the latest iris.task sample. -->
        <InfoCard title="Resources">
          <template v-if="latestStat">
            <div class="space-y-3">
              <ResourceGauge label="CPU" :used="cpuUsed" :total="cpuTotal" unit="cores" />
              <ResourceGauge
                label="Memory"
                :used="memUsedMb * 1024 * 1024"
                :total="memTotalMb * 1024 * 1024"
                unit="bytes"
              />
              <ResourceGauge
                v-if="diskUsedMb > 0"
                label="Disk"
                :used="diskUsedMb * 1024 * 1024"
                :total="diskTotalMb * 1024 * 1024"
                unit="bytes"
              />
            </div>
            <div class="mt-2 text-xs text-text-muted space-y-0.5">
              <div v-if="latestStat.cpu_millicores">
                CPU: {{ formatCpuMillicores(Number(latestStat.cpu_millicores)) }}
              </div>
              <div v-if="memPeakMb > 0">
                Peak Memory: {{ memPeakMb.toFixed(0) }} MB
              </div>
            </div>
          </template>
          <div v-else class="text-sm text-text-muted py-2">No resource data</div>
        </InfoCard>

        <!-- Build info card -->
        <InfoCard title="Build Info">
          <template v-if="task.buildMetrics">
            <InfoRow label="Image Tag">
              <span class="font-mono text-xs break-all">
                {{ task.buildMetrics.imageTag ?? '-' }}
              </span>
            </InfoRow>
            <InfoRow v-if="buildDuration" label="Build Time">
              <span class="font-mono">{{ buildDuration }}</span>
            </InfoRow>
            <InfoRow label="From Cache">
              <span :class="task.buildMetrics.fromCache ? 'text-status-success' : 'text-text-muted'">
                {{ task.buildMetrics.fromCache ? 'Yes' : 'No' }}
              </span>
            </InfoRow>
          </template>
          <div v-else class="text-sm text-text-muted py-2">No build data</div>
        </InfoCard>
      </div>

      <!-- Scheduling / lifecycle events (finelog iris.task_event). Surfaces why a
           task is wedged in BUILDING; follows the page's selected attempt. -->
      <InfoCard v-if="isActive || taskEvents.length > 0" title="Events" class="mb-6">
        <TaskEventTimeline :events="taskEvents" />
      </InfoCard>

      <!-- Endpoints registered by this task, linked through the controller proxy -->
      <InfoCard v-if="endpoints.length > 0" title="Endpoints" class="mb-6">
        <ul class="divide-y divide-surface-border-subtle">
          <li
            v-for="ep in endpoints"
            :key="ep.endpointId ?? ep.name"
            class="flex flex-wrap items-center gap-x-3 gap-y-1 py-1.5 first:pt-0 last:pb-0"
          >
            <EndpointLink :name="ep.name" class="font-mono text-[13px]" />
            <span v-if="ep.address" class="group/addr inline-flex items-center gap-1 text-xs font-mono text-text-muted">
              {{ ep.address }}
              <CopyButton :value="ep.address" />
            </span>
          </li>
        </ul>
      </InfoCard>

      <!-- Resource sparklines -->
      <div v-if="cpuHistory.length > 1" class="grid grid-cols-2 gap-4 mb-6">
        <div class="rounded-lg border border-surface-border bg-surface p-3">
          <div class="text-xs text-text-secondary mb-2">CPU %</div>
          <Sparkline :data="cpuHistory" :height="40" color="var(--color-accent, #2563eb)" />
          <div class="text-xs font-mono text-text-muted mt-1">{{ cpuUsed.toFixed(0) }}%</div>
        </div>
        <div class="rounded-lg border border-surface-border bg-surface p-3">
          <div class="text-xs text-text-secondary mb-2">Memory (MB)</div>
          <Sparkline :data="memHistory" :height="40" color="var(--color-status-purple, #8b5cf6)" />
          <div class="text-xs font-mono text-text-muted mt-1">{{ memUsedMb.toFixed(0) }} MB</div>
        </div>
      </div>

      <!-- Status text -->
      <InfoCard v-if="statusTextDetail" title="Status Text" class="mb-6">
        <div class="text-sm text-text">
          <MarkdownRenderer :content="statusTextDetail" />
        </div>
      </InfoCard>

      <!-- Likely root cause: log highlights distilled from the failed task's own logs -->
      <div
        v-if="rootCauseHighlights.length"
        class="mb-6 rounded-lg border border-status-danger-border bg-status-danger-bg p-4"
      >
        <h3 class="text-sm font-semibold text-status-danger mb-2">Likely Root Cause</h3>
        <pre class="text-xs font-mono text-status-danger whitespace-pre-wrap break-all">{{ rootCauseHighlights.join('\n') }}</pre>
      </div>

      <!-- Error display -->
      <div
        v-if="task.error"
        class="mb-6 rounded-lg border border-status-danger-border bg-status-danger-bg p-4"
      >
        <h3 class="text-sm font-semibold text-status-danger mb-2">Error</h3>
        <pre class="text-xs font-mono text-status-danger whitespace-pre-wrap break-all">{{ task.error }}</pre>
      </div>

      <!-- Attempts table -->
      <div v-if="task.attempts && task.attempts.length > 0" class="mb-6">
        <h3 class="text-sm font-semibold text-text mb-3">
          Attempts
          <span class="text-text-muted font-normal ml-1">({{ task.attempts.length }})</span>
        </h3>
        <div class="overflow-x-auto rounded-lg border border-surface-border">
          <table class="w-full border-collapse">
            <thead>
              <tr class="border-b border-surface-border">
                <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">
                  Attempt
                </th>
                <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">
                  UID
                </th>
                <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">
                  State
                </th>
                <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">
                  Worker
                </th>
                <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">
                  Exit Code
                </th>
                <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">
                  Duration
                </th>
                <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">
                  Error
                </th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="attempt in task.attempts"
                :key="attempt.attemptId"
                class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors cursor-pointer"
                :class="attempt.attemptId === task.currentAttemptId ? 'bg-accent-subtle border-l-2 border-l-accent' : ''"
                @click="selectAttempt(attempt.attemptId)"
              >
                <td class="px-3 py-2 text-[13px] font-mono">
                  {{ attempt.attemptId }}
                  <span v-if="attempt.attemptId === task.currentAttemptId" class="ml-1 text-xs text-accent font-semibold">current</span>
                </td>
                <td class="px-3 py-2 text-[13px] font-mono text-text-muted">
                  {{ attempt.attemptUid ? attempt.attemptUid.slice(0, 8) : '-' }}
                </td>
                <td class="px-3 py-2 text-[13px]">
                  <StatusBadge :status="attempt.state" size="sm" />
                </td>
                <td class="px-3 py-2 text-[13px]">
                  <RouterLink
                    v-if="attempt.workerId"
                    :to="`/worker/${attempt.workerId}`"
                    class="font-mono text-accent hover:underline"
                    @click.stop
                  >
                    {{ attempt.workerId }}
                  </RouterLink>
                  <span v-else class="text-text-muted">-</span>
                </td>
                <td class="px-3 py-2 text-[13px] font-mono">
                  {{ attempt.exitCode ?? '-' }}
                </td>
                <td class="px-3 py-2 text-[13px] font-mono">
                  {{ formatDuration(timestampMs(attempt.startedAt), timestampMs(attempt.finishedAt) || undefined) }}
                </td>
                <td class="px-3 py-2 text-[13px] text-status-danger truncate max-w-xs">
                  {{ attempt.error ?? '-' }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Task logs -->
      <div id="task-logs-section" class="mb-6">
        <h3 class="text-sm font-semibold text-text mb-3">Logs</h3>
        <LogViewer ref="logViewerRef" :task-id="taskId" :cluster="task.cluster" :attempts="task.attempts" :current-attempt-id="task.currentAttemptId" />
      </div>

      <!-- Latest captured profile for this task; self-hides when none exist -->
      <ProfileLink :source="taskId" class="mb-6" />
    </template>
  </PageShell>
</template>
