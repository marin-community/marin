<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { stateToName } from '@/types/status'
import type {
  ProtoTimestamp,
  GetWorkerStatusResponse,
  TaskStatus,
  WorkerResourceSnapshot,
  LogEntry,
} from '@/types/rpc'

import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import MetricCard from '@/components/shared/MetricCard.vue'
import Sparkline from '@/components/shared/Sparkline.vue'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'

const props = defineProps<{
  workerId: string
}>()

const data = ref<GetWorkerStatusResponse | null>(null)
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

function formatRelativeTime(ms: number): string {
  if (!ms) return '-'
  const seconds = Math.floor((Date.now() - ms) / 1000)
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

function formatRate(bytesPerSec: number): string {
  if (!bytesPerSec) return '0 B/s'
  const units = ['B/s', 'KB/s', 'MB/s', 'GB/s']
  const i = Math.min(Math.floor(Math.log(bytesPerSec) / Math.log(1024)), units.length - 1)
  const val = bytesPerSec / Math.pow(1024, i)
  return (val >= 100 ? Math.round(val) : val.toFixed(1)) + ' ' + units[i]
}

function formatDevice(): string {
  const md = data.value?.worker?.metadata
  if (!md) return 'CPU'
  if (md.gpuCount && md.gpuCount > 0) {
    const name = md.gpuName || 'GPU'
    const mem = md.gpuMemoryMb ? ` (${Math.round(md.gpuMemoryMb / 1024)}GB)` : ''
    return `GPU: ${md.gpuCount}x ${name}${mem}`
  }
  if (md.device?.tpu) return `TPU: ${md.device.tpu.variant || 'unknown'}`
  if (md.device?.gpu) return `GPU: ${md.device.gpu.count || 1}x ${md.device.gpu.variant || 'unknown'}`
  return 'CPU'
}

async function fetchWorker() {
  loading.value = true
  error.value = null
  try {
    const resp = await fetch('/iris.cluster.ControllerService/GetWorkerStatus', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: props.workerId }),
    })
    if (!resp.ok) throw new Error(`GetWorkerStatus: ${resp.status}`)
    data.value = (await resp.json()) as GetWorkerStatusResponse
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

const worker = computed(() => data.value?.worker)
const vm = computed(() => data.value?.vm)
const currentResources = computed(() => data.value?.currentResources)
const resourceHistory = computed(() => data.value?.resourceHistory ?? [])
const recentTasks = computed(() => data.value?.recentTasks ?? [])
const workerLogEntries = computed(() => data.value?.workerLogEntries ?? [])
const attributes = computed(() => worker.value?.metadata?.attributes ?? {})

// Sparkline data from resource history
const cpuHistory = computed(() => resourceHistory.value.map((s) => s.cpuPercent ?? 0))
const memoryHistory = computed(() =>
  resourceHistory.value.map((s) => parseInt(s.memoryUsedBytes ?? '0', 10))
)

const runningTaskCount = computed(() => worker.value?.runningJobIds?.length ?? 0)

const cpuDisplay = computed(() => {
  const cr = currentResources.value
  if (!cr?.cpuPercent) return '-'
  return `${Math.round(cr.cpuPercent)}%`
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

function logLevelClass(level: string | undefined): string {
  const lvl = (level ?? 'info').toLowerCase()
  switch (lvl) {
    case 'warning':
      return 'text-status-warning'
    case 'error':
    case 'critical':
      return 'text-status-danger'
    default:
      return 'text-text'
  }
}

function formatLogTime(ts?: ProtoTimestamp): string {
  const ms = timestampMs(ts)
  if (!ms) return ''
  return new Date(ms).toLocaleTimeString()
}

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
        <span v-if="worker?.address" class="text-sm text-text-muted font-mono">
          {{ worker.address }}
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
        <MetricCard :value="formatDevice()" label="Accelerator" />
      </div>

      <!-- Identity + Health section -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        <InfoCard title="Identity">
          <InfoRow label="Worker ID">
            <span class="font-mono">{{ worker?.workerId }}</span>
          </InfoRow>
          <InfoRow label="Address">
            <span class="font-mono">{{ worker?.address ?? '-' }}</span>
          </InfoRow>
          <InfoRow v-if="worker?.metadata?.gceZone" label="Zone">
            <span class="font-mono">{{ worker.metadata.gceZone }}</span>
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
            {{ formatDevice() }}
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
          <div class="rounded-lg border border-surface-border bg-white p-3">
            <div class="text-xs text-text-secondary mb-2">CPU %</div>
            <Sparkline :data="cpuHistory" :width="200" :height="40" color="var(--color-accent, #2563eb)" />
            <div class="text-xs font-mono text-text-muted mt-1">
              {{ cpuDisplay }}
            </div>
          </div>
          <div class="rounded-lg border border-surface-border bg-white p-3">
            <div class="text-xs text-text-secondary mb-2">Memory</div>
            <Sparkline :data="memoryHistory" :width="200" :height="40" color="var(--color-status-purple, #8b5cf6)" />
            <div class="text-xs font-mono text-text-muted mt-1">
              {{ memoryDisplay }}
            </div>
          </div>
          <div
            v-if="resourceHistory.some((s) => s.netRecvBps)"
            class="rounded-lg border border-surface-border bg-white p-3"
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
            class="rounded-lg border border-surface-border bg-white p-3"
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
        <div class="rounded-lg border border-surface-border bg-white overflow-hidden">
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
                {{ (row as TaskStatus).resourceUsage?.cpuPercent != null
                  ? Math.round((row as TaskStatus).resourceUsage!.cpuPercent!) + '%'
                  : '-' }}
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
          class="overflow-y-auto rounded-lg border border-surface-border bg-white"
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
            <span class="text-text-muted mr-2">{{ formatLogTime(entry.timestamp) }}</span>
            <span class="whitespace-pre-wrap break-all">{{ entry.data }}</span>
          </div>
        </div>
      </div>

      <!-- Bootstrap logs (raw text) -->
      <div v-if="data.bootstrapLogs" class="mb-6">
        <h3 class="text-sm font-semibold text-text mb-3">Bootstrap Logs</h3>
        <pre
          class="overflow-auto rounded-lg border border-surface-border bg-white p-4 font-mono text-xs text-text leading-relaxed"
          style="max-height: 40vh"
        >{{ data.bootstrapLogs }}</pre>
      </div>
    </template>
  </PageShell>
</template>
