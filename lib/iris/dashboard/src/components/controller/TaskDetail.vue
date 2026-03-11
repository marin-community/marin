<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { stateToName } from '@/types/status'
import type {
  ProtoTimestamp,
  TaskStatus,
  GetTaskStatusResponse,
  GetTaskLogsResponse,
} from '@/types/rpc'

import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import ResourceGauge from '@/components/shared/ResourceGauge.vue'
import Sparkline from '@/components/shared/Sparkline.vue'
import LogViewer from '@/components/shared/LogViewer.vue'

const props = defineProps<{
  jobId: string
  taskId: string
}>()

const task = ref<TaskStatus | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

function timestampMs(ts?: ProtoTimestamp): number {
  if (!ts?.epochMs) return 0
  return parseInt(ts.epochMs, 10) || 0
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

function formatDuration(startMs: number, endMs?: number): string {
  if (!startMs) return '-'
  const end = endMs || Date.now()
  const seconds = Math.floor((end - startMs) / 1000)
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

function formatRelativeTime(timestampMs: number): string {
  if (!timestampMs) return '-'
  const seconds = Math.floor((Date.now() - timestampMs) / 1000)
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

async function fetchTask() {
  loading.value = true
  error.value = null
  try {
    const resp = await fetch('/iris.cluster.ControllerService/GetTaskStatus', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ taskId: props.taskId }),
    })
    if (!resp.ok) throw new Error(`GetTaskStatus: ${resp.status}`)
    const data = (await resp.json()) as GetTaskStatusResponse
    task.value = data.task
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

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
const cpuUsed = computed(() => task.value?.resourceUsage?.cpuPercent ?? 0)
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
        </InfoCard>

        <!-- Resources card -->
        <InfoCard title="Resources">
          <template v-if="task.resourceUsage">
            <div class="space-y-3">
              <ResourceGauge label="CPU" :used="cpuUsed" :total="100" unit="%" />
              <ResourceGauge
                label="Memory"
                :used="memUsedMb * 1024 * 1024"
                :total="(memPeakMb || memUsedMb * 1.5) * 1024 * 1024"
                unit="bytes"
              />
              <ResourceGauge
                v-if="diskUsedMb > 0"
                label="Disk"
                :used="diskUsedMb * 1024 * 1024"
                :total="diskUsedMb * 2 * 1024 * 1024"
                unit="bytes"
              />
            </div>
            <div class="mt-2 text-xs text-text-muted space-y-0.5">
              <div v-if="task.resourceUsage.processCount">
                Processes: {{ task.resourceUsage.processCount }}
              </div>
              <div v-if="task.resourceUsage.cpuMillicores">
                CPU millicores: {{ task.resourceUsage.cpuMillicores }}
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

      <!-- Error display -->
      <div
        v-if="task.error"
        class="mb-6 rounded-lg border border-status-danger-border bg-status-danger-bg p-4"
      >
        <h3 class="text-sm font-semibold text-status-danger mb-2">Error</h3>
        <pre class="text-xs font-mono text-status-danger whitespace-pre-wrap break-all">{{ task.error }}</pre>
      </div>

      <!-- Attempts table -->
      <div v-if="task.attempts && task.attempts.length > 1" class="mb-6">
        <h3 class="text-sm font-semibold text-text mb-3">Attempts</h3>
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
                class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
              >
                <td class="px-3 py-2 text-[13px] font-mono">{{ attempt.attemptId }}</td>
                <td class="px-3 py-2 text-[13px]">
                  <StatusBadge :status="attempt.state" size="sm" />
                </td>
                <td class="px-3 py-2 text-[13px]">
                  <RouterLink
                    v-if="attempt.workerId"
                    :to="`/worker/${attempt.workerId}`"
                    class="font-mono text-accent hover:underline"
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
      <div class="mb-6">
        <h3 class="text-sm font-semibold text-text mb-3">Logs</h3>
        <LogViewer :task-id="taskId" />
      </div>
    </template>
  </PageShell>
</template>
