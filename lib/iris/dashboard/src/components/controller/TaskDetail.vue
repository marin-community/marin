<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { stateToName } from '@/types/status'
import type {
  TaskStatus,
  TaskStatsSnapshot,
  GetTaskStatusResponse,
} from '@/types/rpc'
import { timestampMs, formatBytes, formatRate, formatCpuMillicores, formatDuration, formatRelativeTime } from '@/utils/formatting'

import { controllerRpcCall } from '@/composables/useRpc'
import { useProfileAction } from '@/composables/useProfileAction'
import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import ResourceGauge from '@/components/shared/ResourceGauge.vue'
import Sparkline from '@/components/shared/Sparkline.vue'
import ProfileButtons from '@/components/shared/ProfileButtons.vue'
import LogViewer from '@/components/shared/LogViewer.vue'

const props = defineProps<{
  jobId: string
  taskId: string
}>()

const {
  data: taskResponse,
  loading,
  error,
  refresh: fetchTask,
} = useControllerRpc<GetTaskStatusResponse>('GetTaskStatus', () => ({ taskId: props.taskId }))

const task = computed(() => taskResponse.value?.task ?? null)
const jobResources = computed(() => taskResponse.value?.jobResources ?? null)

const normalizedState = computed(() => (task.value ? stateToName(task.value.state) : ''))

const isActive = computed(() => {
  const s = normalizedState.value
  return s === 'running' || s === 'building' || s === 'assigned'
})

const startedMs = computed(() => timestampMs(task.value?.startedAt))
const finishedMs = computed(() => timestampMs(task.value?.finishedAt))

const duration = computed(() => {
  if (!startedMs.value) return '-'
  return formatDuration(startedMs.value, finishedMs.value || undefined)
})

const startedDisplay = computed(() =>
  startedMs.value ? formatRelativeTime(startedMs.value) : '-'
)

// Resource gauge values from resourceUsage (MB -> bytes for the gauge)
const cpuUsed = computed(() => (task.value?.resourceUsage?.cpuMillicores ?? 0) / 1000)
const memUsedMb = computed(() => {
  const raw = task.value?.resourceUsage?.memoryMb
  return raw ? parseFloat(raw) : 0
})
const memPeakMb = computed(() => {
  const raw = task.value?.resourceUsage?.memoryPeakMb
  return raw ? parseFloat(raw) : 0
})
const diskUsedMb = computed(() => {
  const raw = task.value?.resourceUsage?.diskMb
  return raw ? parseFloat(raw) : 0
})

const cpuHistory = computed(() =>
  (task.value?.resourceHistory ?? []).map(r => (r.cpuMillicores ?? 0) / 1000)
)
const memHistory = computed(() =>
  (task.value?.resourceHistory ?? []).map(r => r.memoryMb ? parseFloat(r.memoryMb) : 0)
)

const statsHistory = computed(() => task.value?.taskStatsHistory ?? [])

const latestStats = computed(() => statsHistory.value[statsHistory.value.length - 1] ?? null)

// Compute per-second throughput from cumulative deltas, then smooth with a
// 5-point centred moving average to reduce reporting-interval spikes.
function throughputSeries(field: 'itemsProcessed' | 'bytesProcessed'): number[] {
  const snaps = statsHistory.value
  if (snaps.length < 2) return []
  const raw: number[] = []
  for (let i = 1; i < snaps.length; i++) {
    const prev = snaps[i - 1]
    const curr = snaps[i]
    const dtMs = parseFloat(curr.timestampMs ?? '0') - parseFloat(prev.timestampMs ?? '0')
    if (dtMs <= 0) continue
    const dv = parseFloat(curr[field] ?? '0') - parseFloat(prev[field] ?? '0')
    raw.push(Math.max(0, dv / (dtMs / 1000)))
  }
  const window = 5
  return raw.map((_, i) => {
    const lo = Math.max(0, i - Math.floor(window / 2))
    const hi = Math.min(raw.length, lo + window)
    const slice = raw.slice(lo, hi)
    return slice.reduce((a, b) => a + b, 0) / slice.length
  })
}

const itemsThroughput = computed(() => throughputSeries('itemsProcessed'))
const bytesThroughput = computed(() => throughputSeries('bytesProcessed'))

const itemsThroughputMax = computed(() => {
  const m = Math.max(...itemsThroughput.value, 0)
  return m > 0 ? m.toFixed(1) + '/s' : '0/s'
})
const bytesThroughputMax = computed(() => {
  const m = Math.max(...bytesThroughput.value, 0)
  return m > 0 ? formatRate(m) : '0 B/s'
})

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

watch(isActive, (active) => {
  if (active) startRefresh()
  else stopRefresh()
})

onMounted(async () => {
  await fetchTask()
  if (isActive.value) startRefresh()
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
  stopRefresh()
  await fetchTask()
  if (isActive.value) startRefresh()
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
          <InfoRow v-if="task.workerId" label="Worker">
            <RouterLink
              :to="`/worker/${task.workerId}`"
              class="font-mono text-accent hover:underline"
            >
              {{ task.workerId }}
            </RouterLink>
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
          <div v-if="isActive" class="mt-3 pt-3 border-t border-surface-border">
            <ProfileButtons :profiling="profiling" @profile="handleProfile" />
          </div>
        </InfoCard>

        <!-- Resources card -->
        <InfoCard title="Resources">
          <template v-if="task.resourceUsage">
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
              <div v-if="task.resourceUsage.processCount">
                Processes: {{ task.resourceUsage.processCount }}
              </div>
              <div v-if="task.resourceUsage.cpuMillicores">
                CPU: {{ formatCpuMillicores(task.resourceUsage.cpuMillicores) }}
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

      <!-- Resource sparklines -->
      <div v-if="cpuHistory.length > 1" class="grid grid-cols-2 gap-4 mb-6">
        <div class="rounded-lg border border-surface-border bg-surface p-3">
          <div class="text-xs text-text-secondary mb-2">CPU %</div>
          <Sparkline :data="cpuHistory" :width="200" :height="40" fill color="var(--color-accent, #2563eb)" />
          <div class="text-xs font-mono text-text-muted mt-1">{{ cpuUsed.toFixed(0) }}%</div>
        </div>
        <div class="rounded-lg border border-surface-border bg-surface p-3">
          <div class="text-xs text-text-secondary mb-2">Memory (MB)</div>
          <Sparkline :data="memHistory" :width="200" :height="40" fill color="var(--color-status-purple, #8b5cf6)" />
          <div class="text-xs font-mono text-text-muted mt-1">{{ memUsedMb.toFixed(0) }} MB</div>
        </div>
      </div>

      <!-- Task stats -->
      <div v-if="statsHistory.length > 0" class="grid grid-cols-2 gap-4 mb-6">
        <!-- Left: status text + latest values -->
        <div class="rounded-lg border border-surface-border bg-surface p-3 space-y-2">
          <div v-if="task.statusMessage" class="text-xs text-text font-mono whitespace-pre-wrap break-all">{{ task.statusMessage }}</div>
          <div class="text-xs text-text-muted space-y-1 mt-2">
            <div>Items processed: <span class="font-mono text-text">{{ latestStats ? parseInt(latestStats.itemsProcessed ?? '0').toLocaleString() : '—' }}</span></div>
            <div>Bytes processed: <span class="font-mono text-text">{{ latestStats ? formatBytes(parseFloat(latestStats.bytesProcessed ?? '0')) : '—' }}</span></div>
          </div>
        </div>

        <!-- Right: throughput charts -->
        <div class="space-y-3">
          <div class="rounded-lg border border-surface-border bg-surface p-3">
            <div class="text-xs text-text-secondary mb-2">Items / sec</div>
            <Sparkline
              v-if="itemsThroughput.length > 1"
              :data="itemsThroughput"
              :width="200"
              :height="40"
              fill
              color="var(--color-accent, #2563eb)"
              show-y-axis
              :y-axis-top-label="itemsThroughputMax"
            />
            <div class="text-xs font-mono text-text-muted mt-1">
              {{ itemsThroughput.length ? itemsThroughput[itemsThroughput.length - 1].toFixed(1) + ' items/s' : '—' }}
            </div>
          </div>
          <div class="rounded-lg border border-surface-border bg-surface p-3">
            <div class="text-xs text-text-secondary mb-2">Bytes / sec</div>
            <Sparkline
              v-if="bytesThroughput.length > 1"
              :data="bytesThroughput"
              :width="200"
              :height="40"
              fill
              color="var(--color-status-purple, #8b5cf6)"
              show-y-axis
              :y-axis-top-label="bytesThroughputMax"
            />
            <div class="text-xs font-mono text-text-muted mt-1">
              {{ bytesThroughput.length ? formatRate(bytesThroughput[bytesThroughput.length - 1]) : '—' }}
            </div>
          </div>
        </div>
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
        <LogViewer ref="logViewerRef" :task-id="taskId" :attempts="task.attempts" :current-attempt-id="task.currentAttemptId" />
      </div>
    </template>
  </PageShell>
</template>
