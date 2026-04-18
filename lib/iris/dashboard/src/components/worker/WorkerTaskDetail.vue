<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useWorkerRpc, workerRpcCall } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { useProfileAction } from '@/composables/useProfileAction'
import { stateToName } from '@/types/status'
import type { TaskStatus } from '@/types/rpc'
import { timestampMs, formatBytes, formatCpuMillicores, formatDuration, formatRelativeTime, formatTimestamp } from '@/utils/formatting'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import ResourceGauge from '@/components/shared/ResourceGauge.vue'
import Sparkline from '@/components/shared/Sparkline.vue'
import LogViewer from '@/components/shared/LogViewer.vue'
import ProfileButtons from '@/components/shared/ProfileButtons.vue'

const MAX_HISTORY_SAMPLES = 60

const props = defineProps<{
  taskId: string
}>()

const {
  data: task,
  loading,
  error,
  refresh: fetchTask,
} = useWorkerRpc<TaskStatus>('GetTaskStatus', () => ({ taskId: props.taskId }))

// Track resource history for sparklines (up to MAX_HISTORY_SAMPLES samples)
const cpuHistory = ref<number[]>([])
const memHistory = ref<number[]>([])

function pushSample(history: number[], value: number) {
  history.push(value)
  if (history.length > MAX_HISTORY_SAMPLES) {
    history.splice(0, history.length - MAX_HISTORY_SAMPLES)
  }
}

// Record resource snapshot for sparklines on each successful fetch
watch(task, (t) => {
  const ru = t?.resourceUsage
  if (ru) {
    pushSample(cpuHistory.value, (ru.cpuMillicores ?? 0) / 1000)
    pushSample(memHistory.value, ru.memoryMb ? parseFloat(ru.memoryMb) : 0)
  }
})

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

const cpuUsed = computed(() => (task.value?.resourceUsage?.cpuMillicores ?? 0) / 1000)
const cpuTotal = computed(() => {
  const maxObserved = Math.max(...cpuHistory.value, cpuUsed.value, 0)
  if (maxObserved <= 0) return 1
  return Math.max(1, maxObserved * 1.5)
})

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

const buildDuration = computed(() => {
  const bm = task.value?.buildMetrics
  if (!bm?.buildStarted || !bm?.buildFinished) return null
  return formatDuration(timestampMs(bm.buildStarted), timestampMs(bm.buildFinished))
})

const ports = computed<[string, number][]>(() => {
  const p = task.value?.ports
  if (!p) return []
  return Object.entries(p)
})

const { profiling, profile: handleProfile } = useProfileAction(workerRpcCall, () => props.taskId)

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
  <div class="space-y-6">
    <!-- Back link + header -->
    <div class="flex items-center gap-2 text-sm">
      <a href="/" class="text-accent hover:underline">&larr; Worker Dashboard</a>
    </div>

    <div class="flex items-center justify-between">
      <h2 class="text-xl font-semibold text-text font-mono">Task {{ taskId }}</h2>
      <div class="flex items-center gap-2">
        <span
          v-if="autoRefreshActive"
          class="text-xs text-accent bg-accent-subtle border border-accent-border px-2 py-0.5 rounded-full"
        >
          Auto-refresh
        </span>
        <button
          class="px-3 py-1.5 text-xs border border-surface-border rounded hover:bg-surface-raised text-text-secondary"
          @click="fetchTask"
        >
          Refresh
        </button>
      </div>
    </div>

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
      <!-- Status + timing cards -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <!-- Status -->
        <InfoCard title="Status">
          <InfoRow label="State">
            <StatusBadge :status="task.state" size="sm" />
          </InfoRow>
          <InfoRow v-if="task.workerId" label="Worker ID">
            <span class="font-mono text-xs">{{ task.workerId }}</span>
          </InfoRow>
          <InfoRow v-if="task.currentAttemptId !== undefined" label="Attempt">
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
            <span class="text-status-warning text-xs">{{ task.pendingReason }}</span>
          </InfoRow>
        </InfoCard>

        <!-- Timing -->
        <InfoCard title="Timing">
          <InfoRow label="Started">
            <span class="font-mono text-xs">{{ startedMs ? formatRelativeTime(startedMs) : '-' }}</span>
          </InfoRow>
          <InfoRow label="Started At">
            <span class="font-mono text-xs">{{ formatTimestamp(task.startedAt) }}</span>
          </InfoRow>
          <InfoRow label="Elapsed">
            <span class="font-mono">{{ duration }}</span>
          </InfoRow>
          <InfoRow v-if="finishedMs" label="Finished">
            <span class="font-mono text-xs">{{ formatTimestamp(task.finishedAt) }}</span>
          </InfoRow>
          <template v-if="ports.length > 0">
            <InfoRow v-for="[name, port] in ports" :key="name" :label="`Port: ${name}`">
              <span class="font-mono">{{ port }}</span>
            </InfoRow>
          </template>
        </InfoCard>

        <!-- Resources -->
        <InfoCard title="Resources">
          <template v-if="task.resourceUsage">
            <div class="space-y-3 mb-3">
              <div class="flex items-center gap-2">
                <div class="flex-1">
                  <ResourceGauge label="CPU" :used="cpuUsed" :total="cpuTotal" unit="cores" />
                </div>
                <Sparkline :data="cpuHistory" :width="64" :height="20" />
              </div>
              <div class="flex items-center gap-2">
                <div class="flex-1">
                  <ResourceGauge
                    label="Memory"
                    :used="memUsedMb * 1024 * 1024"
                    :total="(memPeakMb || memUsedMb * 1.5) * 1024 * 1024"
                    unit="bytes"
                  />
                </div>
                <Sparkline :data="memHistory" :width="64" :height="20" />
              </div>
              <ResourceGauge
                v-if="diskUsedMb > 0"
                label="Disk"
                :used="diskUsedMb * 1024 * 1024"
                :total="diskUsedMb * 2 * 1024 * 1024"
                unit="bytes"
              />
            </div>
            <div class="text-xs text-text-muted space-y-0.5">
              <div v-if="task.resourceUsage.processCount">
                Processes: {{ task.resourceUsage.processCount }}
              </div>
              <div v-if="task.resourceUsage.cpuMillicores">
                CPU: {{ formatCpuMillicores(task.resourceUsage.cpuMillicores) }}
              </div>
              <div v-if="memPeakMb">
                Peak memory: {{ formatBytes(memPeakMb * 1024 * 1024) }}
              </div>
            </div>
          </template>
          <div v-else class="text-sm text-text-muted py-2">No resource data</div>
        </InfoCard>
      </div>

      <!-- Build info (when present) -->
      <InfoCard v-if="task.buildMetrics" title="Build Info">
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <InfoRow label="Image Tag">
            <span class="font-mono text-xs break-all">{{ task.buildMetrics.imageTag ?? '-' }}</span>
          </InfoRow>
          <InfoRow v-if="buildDuration" label="Build Time">
            <span class="font-mono">{{ buildDuration }}</span>
          </InfoRow>
          <InfoRow label="From Cache">
            <span :class="task.buildMetrics.fromCache ? 'text-status-success' : 'text-text-muted'">
              {{ task.buildMetrics.fromCache ? 'Yes' : 'No' }}
            </span>
          </InfoRow>
        </div>
      </InfoCard>

      <!-- Profiling buttons (running tasks only) -->
      <ProfileButtons v-if="isActive" :profiling="profiling" @profile="handleProfile" />

      <!-- Error display -->
      <div
        v-if="task.error"
        class="rounded-lg border border-status-danger-border bg-status-danger-bg p-4"
      >
        <h3 class="text-sm font-semibold text-status-danger mb-2">Error</h3>
        <pre class="text-xs font-mono text-status-danger whitespace-pre-wrap break-all">{{ task.error }}</pre>
      </div>

      <!-- Task logs -->
      <div>
        <h3 class="text-sm font-semibold text-text mb-3">Logs</h3>
        <LogViewer :task-id="taskId" source="worker" />
      </div>
    </template>

  </div>
</template>
