<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { useControllerRpc, useLogServerStatsRpc, logServiceRpcCall } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import { stateToName } from '@/types/status'
import type {
  GetWorkerStatusResponse,
  WorkerTaskAttempt,
  LogEntry,
  FetchLogsResponse,
} from '@/types/rpc'
import { timestampMs, formatBytes, formatDuration, formatRelativeTime, formatRate, logLevelClass, formatLogTime, formatWorkerDevice } from '@/utils/formatting'
import { decodeArrowIpc } from '@/utils/arrow'

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
const recentAttempts = computed(() => data.value?.recentAttempts ?? [])
// Worker daemon logs are fetched independently via LogService.FetchLogs so a
// slow/unreachable worker can't stall the worker page. Empty until the
// async fetch completes; failures leave the array empty rather than throw.
const workerLogEntries = ref<LogEntry[]>([])
const attributes = computed(() => worker.value?.metadata?.attributes ?? {})

async function fetchWorkerLogs() {
  try {
    const resp = await logServiceRpcCall<FetchLogsResponse>('FetchLogs', {
      source: `/system/worker/${props.workerId}`,
      maxLines: 200,
      tail: true,
    })
    workerLogEntries.value = resp.entries ?? []
  } catch {
    workerLogEntries.value = []
  }
}

// --- Per-worker resource history sourced from finelog stats (iris.worker) ---
//
// Rows come back ts DESC; we reverse for the sparkline (oldest -> newest)
// and treat the first DESC row as the latest "current resources" snapshot.
// The namespace is registered eagerly by every worker at startup, so any
// worker we can navigate to has a registered table — a missing namespace
// here is a real bug, not a cold-start case.
interface WorkerStatRow {
  ts?: string
  cpu_pct?: number
  mem_bytes?: number
  mem_total_bytes?: number
  disk_used_bytes?: number
  disk_total_bytes?: number
  net_recv_bytes?: number
  net_sent_bytes?: number
  running_task_count?: number
}

interface QueryResponse {
  arrowIpc?: string
}

function buildStatsSql(workerId: string): string {
  // QueryRequest has no param binding; manual DuckDB single-quote escape.
  const escaped = workerId.replace(/'/g, "''")
  return `
SELECT ts, cpu_pct, mem_bytes, mem_total_bytes,
       disk_used_bytes, disk_total_bytes,
       net_recv_bytes, net_sent_bytes, running_task_count
FROM "iris.worker"
WHERE worker_id = '${escaped}'
ORDER BY ts DESC
LIMIT 50
`.trim()
}

const { data: statsData, refresh: fetchWorkerStats } = useLogServerStatsRpc<QueryResponse>(
  'Query',
  () => ({ sql: buildStatsSql(props.workerId) }),
)

const statsRows = computed<WorkerStatRow[]>(() => {
  const ipc = statsData.value?.arrowIpc
  if (!ipc) return []
  return decodeArrowIpc(ipc).rows as WorkerStatRow[]
})

// Reversed copy for the sparkline: queries return ts DESC; charts want oldest -> newest.
const orderedStats = computed(() => statsRows.value.slice().reverse())
const latestStat = computed<WorkerStatRow | null>(() => statsRows.value[0] ?? null)

const cpuHistory = computed(() => orderedStats.value.map((s) => Number(s.cpu_pct ?? 0)))
const memoryHistory = computed(() => orderedStats.value.map((s) => Number(s.mem_bytes ?? 0)))
const diskHistory = computed(() => orderedStats.value.map((s) => Number(s.disk_used_bytes ?? 0)))
// Cumulative byte counters; derive per-second deltas from successive samples.
// First sample contributes no point. Counter resets (worker restart) cap to 0.
function ratesFrom(field: 'net_recv_bytes' | 'net_sent_bytes'): number[] {
  const rows = orderedStats.value
  const out: number[] = []
  for (let i = 1; i < rows.length; i++) {
    const cur = Number(rows[i][field] ?? 0)
    const prev = Number(rows[i - 1][field] ?? 0)
    const tCur = new Date(rows[i].ts ?? 0).getTime()
    const tPrev = new Date(rows[i - 1].ts ?? 0).getTime()
    const dt = (tCur - tPrev) / 1000
    out.push(dt > 0 && cur >= prev ? (cur - prev) / dt : 0)
  }
  return out
}
const netRecvHistory = computed(() => ratesFrom('net_recv_bytes'))
const netSentHistory = computed(() => ratesFrom('net_sent_bytes'))

const runningTaskCount = computed(() => worker.value?.runningJobIds?.length ?? 0)

const cpuDisplay = computed(() => {
  const v = latestStat.value?.cpu_pct
  if (v === undefined || v === null) return '-'
  return `${Math.round(Number(v))}%`
})

const memoryDisplay = computed(() => {
  const cr = latestStat.value
  if (!cr?.mem_bytes) return '-'
  const used = Number(cr.mem_bytes)
  const total = Number(cr.mem_total_bytes ?? 0)
  if (total) return `${formatBytes(used)} / ${formatBytes(total)}`
  return formatBytes(used)
})

const diskDisplay = computed(() => {
  const cr = latestStat.value
  if (!cr?.disk_used_bytes) return '-'
  const used = Number(cr.disk_used_bytes)
  const total = Number(cr.disk_total_bytes ?? 0)
  if (total) return `${formatBytes(used)} / ${formatBytes(total)}`
  return formatBytes(used)
})

const diskFreePercent = computed(() => {
  const cr = latestStat.value
  if (!cr?.disk_used_bytes || !cr?.disk_total_bytes) return null
  const used = Number(cr.disk_used_bytes)
  const total = Number(cr.disk_total_bytes)
  if (!total) return null
  return Math.round((1 - used / total) * 100)
})

const taskColumns: Column[] = [
  { key: 'taskId', label: 'Task ID', mono: true },
  { key: 'attempt', label: 'Attempt', align: 'right' },
  { key: 'state', label: 'State' },
  { key: 'duration', label: 'Duration', align: 'right' },
]

useAutoRefresh(fetchWorker, DEFAULT_REFRESH_MS)
useAutoRefresh(fetchWorkerLogs, DEFAULT_REFRESH_MS)
useAutoRefresh(fetchWorkerStats, DEFAULT_REFRESH_MS)
onMounted(() => {
  fetchWorker()
  fetchWorkerLogs()
  fetchWorkerStats()
})

// Re-fetch when navigating between workers (Vue Router reuses the component).
// Clear stale data first so loading/error states render correctly if the fetch fails.
watch(() => props.workerId, () => {
  data.value = null
  workerLogEntries.value = []
  statsData.value = null
  fetchWorker()
  fetchWorkerLogs()
  fetchWorkerStats()
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
        <MetricCard
          :value="diskDisplay"
          label="Disk"
          :variant="diskFreePercent !== null && diskFreePercent < 10 ? 'warning' : 'default'"
        />
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
          <InfoRow v-if="diskDisplay !== '-'" label="Disk Usage">
            <span class="font-mono" :class="diskFreePercent !== null && diskFreePercent < 10 ? 'text-status-danger' : ''">
              {{ diskDisplay }}
              <span v-if="diskFreePercent !== null" class="text-text-muted ml-1">({{ diskFreePercent }}% free)</span>
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

      <!-- Live utilization sparklines (sourced from finelog iris.worker stats) -->
      <div v-if="orderedStats.length > 1" class="mb-6">
        <h3 class="text-sm font-semibold text-text mb-3">Live Utilization</h3>
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div class="rounded-lg border border-surface-border bg-surface p-3">
            <div class="text-xs text-text-secondary mb-2">CPU %</div>
            <Sparkline :data="cpuHistory" :height="40" color="var(--color-accent, #2563eb)" />
            <div class="text-xs font-mono text-text-muted mt-1">
              {{ cpuDisplay }}
            </div>
          </div>
          <div class="rounded-lg border border-surface-border bg-surface p-3">
            <div class="text-xs text-text-secondary mb-2">Memory</div>
            <Sparkline :data="memoryHistory" :height="40" color="var(--color-status-purple, #8b5cf6)" />
            <div class="text-xs font-mono text-text-muted mt-1">
              {{ memoryDisplay }}
            </div>
          </div>
          <div
            v-if="diskHistory.some((v) => v > 0)"
            class="rounded-lg border border-surface-border bg-surface p-3"
          >
            <div class="text-xs text-text-secondary mb-2">Disk</div>
            <Sparkline :data="diskHistory" :height="40" color="var(--color-status-orange, #f97316)" />
            <div class="text-xs font-mono mt-1" :class="diskFreePercent !== null && diskFreePercent < 10 ? 'text-status-danger' : 'text-text-muted'">
              {{ diskDisplay }}
              <span v-if="diskFreePercent !== null" class="ml-1">({{ diskFreePercent }}% free)</span>
            </div>
          </div>
          <div
            v-if="netRecvHistory.some((v) => v > 0)"
            class="rounded-lg border border-surface-border bg-surface p-3"
          >
            <div class="text-xs text-text-secondary mb-2">Network Recv</div>
            <Sparkline
              :data="netRecvHistory"
              :height="40"
              color="var(--color-status-success, #22c55e)"
            />
            <div class="text-xs font-mono text-text-muted mt-1">
              {{ formatRate(netRecvHistory[netRecvHistory.length - 1] ?? 0) }}
            </div>
          </div>
          <div
            v-if="netSentHistory.some((v) => v > 0)"
            class="rounded-lg border border-surface-border bg-surface p-3"
          >
            <div class="text-xs text-text-secondary mb-2">Network Sent</div>
            <Sparkline
              :data="netSentHistory"
              :height="40"
              color="var(--color-status-orange, #f97316)"
            />
            <div class="text-xs font-mono text-text-muted mt-1">
              {{ formatRate(netSentHistory[netSentHistory.length - 1] ?? 0) }}
            </div>
          </div>
        </div>
      </div>

      <!-- Task history (per-attempt: each retry on this worker is its own row) -->
      <div v-if="recentAttempts.length > 0" class="mb-6">
        <h3 class="text-sm font-semibold text-text mb-3">Task History</h3>
        <div class="rounded-lg border border-surface-border bg-surface overflow-hidden">
          <DataTable
            :columns="taskColumns"
            :rows="recentAttempts"
            :page-size="25"
            empty-message="No recent tasks"
          >
            <template #cell-taskId="{ row }">
              <span class="font-mono text-xs">{{ (row as WorkerTaskAttempt).taskId }}</span>
            </template>
            <template #cell-attempt="{ row }">
              <span class="font-mono text-xs">{{ (row as WorkerTaskAttempt).attempt?.attemptId ?? '-' }}</span>
            </template>
            <template #cell-state="{ row }">
              <StatusBadge :status="(row as WorkerTaskAttempt).attempt?.state ?? 0" size="sm" />
            </template>
            <template #cell-duration="{ row }">
              <span class="font-mono text-xs">
                {{ formatDuration(
                  timestampMs((row as WorkerTaskAttempt).attempt?.startedAt),
                  timestampMs((row as WorkerTaskAttempt).attempt?.finishedAt) || undefined,
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
