<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { stateToName } from '@/types/status'
import type {
  GetWorkerStatusResponse,
  TaskStatus,
  WorkerResourceSnapshot,
  LogEntry,
} from '@/types/rpc'
import { timestampMs, formatBytes, formatCpuMillicores, formatDuration, formatRelativeTime, formatRate, logLevelClass, formatLogTime, formatWorkerDevice } from '@/utils/formatting'

import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import MetricCard from '@/components/shared/MetricCard.vue'
import Sparkline from '@/components/shared/Sparkline.vue'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import CopyButton from '@/components/shared/CopyButton.vue'

const props = defineProps<{
  workerId: string
}>()

const {
  data,
  loading,
  error,
  refresh: fetchWorker,
} = useControllerRpc<GetWorkerStatusResponse>('GetWorkerStatus', () => ({ id: props.workerId }))

const worker = computed(() => data.value?.worker)
const vm = computed(() => data.value?.vm)
const currentResources = computed(() => data.value?.currentResources)
const resourceHistory = computed(() => data.value?.resourceHistory ?? [])
const recentTasks = computed(() => data.value?.recentTasks ?? [])
const workerLogEntries = computed(() => data.value?.workerLogEntries ?? [])
const attributes = computed(() => worker.value?.metadata?.attributes ?? {})

// Sparkline data from resource history
const cpuHistory = computed(() => resourceHistory.value.map((s) => s.hostCpuPercent ?? 0))
const memoryHistory = computed(() =>
  resourceHistory.value.map((s) => parseInt(s.memoryUsedBytes ?? '0', 10))
)

const runningTaskCount = computed(() => worker.value?.runningJobIds?.length ?? 0)

const cpuDisplay = computed(() => {
  const cr = currentResources.value
  if (!cr?.hostCpuPercent) return '-'
  return `${Math.round(cr.hostCpuPercent)}%`
})

const memoryDisplay = computed(() => {
  const cr = currentResources.value
  if (!cr?.memoryUsedBytes) return '-'
  const used = parseInt(cr.memoryUsedBytes, 10)
  const total = parseInt(cr.memoryTotalBytes ?? '0', 10)
  if (total) return `${formatBytes(used)} / ${formatBytes(total)}`
  return formatBytes(used)
})

const taskColumns: Column[] = [
  { key: 'taskId', label: 'Task ID', mono: true },
  { key: 'state', label: 'State' },
  { key: 'memory', label: 'Memory', align: 'right' },
  { key: 'cpu', label: 'CPU', align: 'right' },
  { key: 'duration', label: 'Duration', align: 'right' },
]

useAutoRefresh(fetchWorker, 5_000)
onMounted(fetchWorker)

// Re-fetch when navigating between workers (Vue Router reuses the component).
// Clear stale data first so loading/error states render correctly if the fetch fails.
watch(() => props.workerId, () => {
  data.value = null
  fetchWorker()
})

function attributeDisplay(val: { stringValue?: string; intValue?: string; floatValue?: string }): string {
  if (val.stringValue !== undefined) return val.stringValue
  if (val.intValue !== undefined) return val.intValue
  if (val.floatValue !== undefined) return val.floatValue
  return '-'
}
</script>

<template>
  <PageShell
    :title="`Worker ${workerId}`"
    back-to="/fleet"
    back-label="Back to Fleet"
  >
    <!-- Loading -->
    <div
      v-if="loading && !data"
      class="flex items-center justify-center py-16 text-text-muted text-sm"
    >
      Loading worker...
    </div>

    <!-- Error -->
    <div
      v-else-if="error && !data"
      class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <template v-else-if="data">
      <!-- Header with health badge -->
      <div class="flex items-center gap-3 mb-6">
        <span
          class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold"
          :class="worker?.healthy
            ? 'bg-status-success-bg text-status-success border border-status-success-border'
            : 'bg-status-danger-bg text-status-danger border border-status-danger-border'"
        >
          <span
            class="w-1.5 h-1.5 rounded-full"
            :class="worker?.healthy ? 'bg-status-success' : 'bg-status-danger'"
          />
          {{ worker?.healthy ? 'Healthy' : 'Unhealthy' }}
        </span>
        <span v-if="worker?.address" class="group/addr text-sm text-text-muted font-mono inline-flex items-center gap-1">
          {{ worker.address }}
          <CopyButton :value="worker.address" />
        </span>
        <button
          class="ml-auto px-3 py-1.5 text-xs border border-surface-border rounded hover:bg-surface-sunken"
          @click="fetchWorker"
        >
          Refresh
        </button>
      </div>

      <!-- Metric cards -->
      <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <MetricCard
          :value="runningTaskCount"
          label="Running Tasks"
          :variant="runningTaskCount > 0 ? 'accent' : 'default'"
        />
        <MetricCard :value="cpuDisplay" label="CPU Usage" />
        <MetricCard :value="memoryDisplay" label="Memory" />
        <MetricCard :value="formatWorkerDevice(worker?.metadata)" label="Accelerator" />
      </div>

      <!-- Identity + Health section -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        <InfoCard title="Identity">
          <InfoRow label="Worker ID">
            <span class="font-mono">{{ worker?.workerId }}</span>
          </InfoRow>
          <InfoRow label="Address">
            <span v-if="worker?.address" class="group/addr inline-flex items-center gap-1">
              <CopyButton :value="worker.address" />
              <span class="font-mono">{{ worker.address }}</span>
            </span>
            <span v-else class="font-mono">-</span>
          </InfoRow>
          <InfoRow v-if="worker?.metadata?.attributes?.zone" label="Zone">
            <span class="font-mono">{{ worker.metadata.attributes.zone.stringValue }}</span>
          </InfoRow>
          <InfoRow v-if="worker?.metadata?.gceInstanceName" label="Instance">
            <span class="font-mono">{{ worker.metadata.gceInstanceName }}</span>
          </InfoRow>
          <InfoRow v-if="worker?.metadata?.tpuName" label="TPU Name">
            <span class="font-mono">{{ worker.metadata.tpuName }}</span>
          </InfoRow>
          <InfoRow v-if="worker?.metadata?.tpuWorkerId" label="TPU Worker ID">
            <span class="font-mono">{{ worker.metadata.tpuWorkerId }}</span>
          </InfoRow>
          <InfoRow v-if="data.scaleGroup" label="Scale Group">
            <span class="font-mono">{{ data.scaleGroup }}</span>
          </InfoRow>
          <InfoRow v-if="worker?.metadata?.gitHash" label="Git Hash">
            <span class="font-mono text-xs">{{ worker.metadata.gitHash }}</span>
          </InfoRow>
        </InfoCard>

        <InfoCard title="Health & Resources">
          <InfoRow label="Status">
            <span :class="worker?.healthy ? 'text-status-success' : 'text-status-danger'">
              {{ worker?.healthy ? 'Healthy' : 'Unhealthy' }}
            </span>
          </InfoRow>
          <InfoRow v-if="worker?.statusMessage" label="Message">
            <span class="text-xs">{{ worker.statusMessage }}</span>
          </InfoRow>
          <InfoRow label="Last Heartbeat">
            <span class="font-mono">
              {{ formatRelativeTime(timestampMs(worker?.lastHeartbeat)) }}
            </span>
          </InfoRow>
          <InfoRow v-if="worker?.metadata?.cpuCount" label="CPU Cores">
            <span class="font-mono">{{ worker.metadata.cpuCount }}</span>
          </InfoRow>
          <InfoRow v-if="worker?.metadata?.memoryBytes" label="Total Memory">
            <span class="font-mono">
              {{ formatBytes(parseInt(worker.metadata.memoryBytes, 10)) }}
            </span>
          </InfoRow>
          <InfoRow label="Accelerator">
            {{ formatWorkerDevice(worker?.metadata) }}
          </InfoRow>
          <InfoRow v-if="worker?.consecutiveFailures" label="Consecutive Failures">
            <span class="text-status-danger font-mono">{{ worker.consecutiveFailures }}</span>
          </InfoRow>
        </InfoCard>
      </div>

      <!-- Attributes -->
      <div v-if="Object.keys(attributes).length > 0" class="mb-6">
        <InfoCard title="Attributes">
          <InfoRow v-for="(val, key) in attributes" :key="key" :label="String(key)">
            <span class="font-mono text-xs">{{ attributeDisplay(val) }}</span>
          </InfoRow>
        </InfoCard>
      </div>

      <!-- Live utilization sparklines -->
      <div v-if="resourceHistory.length > 1" class="mb-6">
        <h3 class="text-sm font-semibold text-text mb-3">Live Utilization</h3>
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div class="rounded-lg border border-surface-border bg-surface p-3">
            <div class="text-xs text-text-secondary mb-2">CPU %</div>
            <Sparkline :data="cpuHistory" :width="200" :height="40" color="var(--color-accent, #2563eb)" />
            <div class="text-xs font-mono text-text-muted mt-1">
              {{ cpuDisplay }}
            </div>
          </div>
          <div class="rounded-lg border border-surface-border bg-surface p-3">
            <div class="text-xs text-text-secondary mb-2">Memory</div>
            <Sparkline :data="memoryHistory" :width="200" :height="40" color="var(--color-status-purple, #8b5cf6)" />
            <div class="text-xs font-mono text-text-muted mt-1">
              {{ memoryDisplay }}
            </div>
          </div>
          <div
            v-if="resourceHistory.some((s) => s.netRecvBps)"
            class="rounded-lg border border-surface-border bg-surface p-3"
          >
            <div class="text-xs text-text-secondary mb-2">Network Recv</div>
            <Sparkline
              :data="resourceHistory.map((s) => parseInt(s.netRecvBps ?? '0', 10))"
              :width="200"
              :height="40"
              color="var(--color-status-success, #22c55e)"
            />
            <div class="text-xs font-mono text-text-muted mt-1">
              {{ formatRate(parseInt(currentResources?.netRecvBps ?? '0', 10)) }}
            </div>
          </div>
          <div
            v-if="resourceHistory.some((s) => s.netSentBps)"
            class="rounded-lg border border-surface-border bg-surface p-3"
          >
            <div class="text-xs text-text-secondary mb-2">Network Sent</div>
            <Sparkline
              :data="resourceHistory.map((s) => parseInt(s.netSentBps ?? '0', 10))"
              :width="200"
              :height="40"
              color="var(--color-status-orange, #f97316)"
            />
            <div class="text-xs font-mono text-text-muted mt-1">
              {{ formatRate(parseInt(currentResources?.netSentBps ?? '0', 10)) }}
            </div>
          </div>
        </div>
      </div>

      <!-- Task history -->
      <div v-if="recentTasks.length > 0" class="mb-6">
        <h3 class="text-sm font-semibold text-text mb-3">Task History</h3>
        <div class="rounded-lg border border-surface-border bg-surface overflow-hidden">
          <DataTable
            :columns="taskColumns"
            :rows="recentTasks"
            :page-size="25"
            empty-message="No recent tasks"
          >
            <template #cell-taskId="{ row }">
              <span class="font-mono text-xs">{{ (row as TaskStatus).taskId }}</span>
            </template>
            <template #cell-state="{ row }">
              <StatusBadge :status="(row as TaskStatus).state" size="sm" />
            </template>
            <template #cell-memory="{ row }">
              <span class="font-mono text-xs">
                {{ (row as TaskStatus).resourceUsage?.memoryMb
                  ? parseFloat((row as TaskStatus).resourceUsage!.memoryMb!) + ' MB'
                  : '-' }}
              </span>
            </template>
            <template #cell-cpu="{ row }">
              <span class="font-mono text-xs">
                {{ formatCpuMillicores((row as TaskStatus).resourceUsage?.cpuMillicores) }}
              </span>
            </template>
            <template #cell-duration="{ row }">
              <span class="font-mono text-xs">
                {{ formatDuration(
                  timestampMs((row as TaskStatus).startedAt),
                  timestampMs((row as TaskStatus).finishedAt) || undefined,
                ) }}
              </span>
            </template>
          </DataTable>
        </div>
      </div>

      <!-- Worker daemon logs -->
      <div v-if="workerLogEntries.length > 0" class="mb-6">
        <h3 class="text-sm font-semibold text-text mb-3">Worker Daemon Logs</h3>
        <div
          class="overflow-y-auto rounded-lg border border-surface-border bg-surface"
          style="max-height: 40vh"
        >
          <div
            v-for="(entry, i) in workerLogEntries"
            :key="i"
            :class="[
              'px-3 py-0.5 font-mono text-xs leading-relaxed hover:bg-surface-sunken',
              logLevelClass(entry.level),
            ]"
          >
            <span class="text-text-muted mr-2">{{ formatLogTime(timestampMs(entry.timestamp)) }}</span>
            <span class="whitespace-pre-wrap break-all">{{ entry.data }}</span>
          </div>
        </div>
      </div>

      <!-- Bootstrap logs (raw text) -->
      <div v-if="data.bootstrapLogs" class="mb-6">
        <h3 class="text-sm font-semibold text-text mb-3">Bootstrap Logs</h3>
        <pre
          class="overflow-auto rounded-lg border border-surface-border bg-surface p-4 font-mono text-xs text-text leading-relaxed"
          style="max-height: 40vh"
        >{{ data.bootstrapLogs }}</pre>
      </div>
    </template>
  </PageShell>
</template>
