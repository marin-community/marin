<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type { ListWorkersResponse, WorkerHealthStatus } from '@/types/rpc'
import { timestampMs, formatRelativeTime, formatBytes, formatWorkerDevice } from '@/utils/formatting'

import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import CopyButton from '@/components/shared/CopyButton.vue'

const { data, loading, error, refresh } = useControllerRpc<ListWorkersResponse>('ListWorkers')

useAutoRefresh(refresh, 10_000)
onMounted(refresh)

const workers = computed<WorkerHealthStatus[]>(() => data.value?.workers ?? [])

/** Device type label used for grouping (e.g. "TPU v5p", "GPU A100", "CPU"). */
function deviceType(w: WorkerHealthStatus): string {
  const m = w.metadata
  if (!m) return 'Unknown'
  if (m.device?.tpu?.variant) return `TPU ${m.device.tpu.variant}`
  if (m.device?.gpu?.variant) return `GPU ${m.device.gpu.variant}`
  if (m.gpuCount && m.gpuCount > 0) return `GPU ${m.gpuName ?? 'unknown'}`
  return 'CPU'
}

interface DeviceSummary {
  type: string
  total: number
  healthy: number
  unhealthy: number
  inUse: number
}

const deviceSummary = computed<DeviceSummary[]>(() => {
  const counts = new Map<string, { total: number; healthy: number; unhealthy: number; inUse: number }>()
  for (const w of workers.value) {
    const t = deviceType(w)
    const entry = counts.get(t) ?? { total: 0, healthy: 0, unhealthy: 0, inUse: 0 }
    entry.total++
    if (w.healthy) entry.healthy++
    else entry.unhealthy++
    if ((w.runningJobIds?.length ?? 0) > 0) entry.inUse++
    counts.set(t, entry)
  }
  return Array.from(counts.entries())
    .map(([type, c]) => ({ type, ...c }))
    .sort((a, b) => b.total - a.total)
})

const columns: Column[] = [
  { key: 'workerId', label: 'Worker ID', mono: true },
  { key: 'address', label: 'Address', mono: true },
  { key: 'device', label: 'Accelerator' },
  { key: 'zone', label: 'Zone' },
  { key: 'tpuName', label: 'TPU Name', mono: true },
  { key: 'healthy', label: 'Health', align: 'center' },
  { key: 'cpuCount', label: 'CPU', align: 'right' },
  { key: 'memory', label: 'Memory', align: 'right' },
  { key: 'tasks', label: 'Tasks', align: 'right' },
  { key: 'lastHeartbeat', label: 'Last Heartbeat' },
  { key: 'error', label: 'Error' },
]
</script>

<template>
  <div class="max-w-7xl mx-auto px-6 py-6">
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-xl font-semibold text-text">Fleet</h2>
      <span class="text-xs text-text-muted font-mono">
        {{ workers.length }} worker{{ workers.length !== 1 ? 's' : '' }}
      </span>
    </div>

    <!-- Device type summary -->
    <div v-if="deviceSummary.length > 0" class="grid gap-3 mb-6" :style="{ gridTemplateColumns: `repeat(${Math.min(deviceSummary.length, 6)}, minmax(0, 1fr))` }">
      <div
        v-for="d in deviceSummary"
        :key="d.type"
        class="rounded-lg border border-surface-border bg-surface px-4 py-3"
      >
        <div class="text-2xl font-semibold font-mono tabular-nums text-text">
          {{ d.total }}
        </div>
        <div class="text-xs font-medium text-text-secondary mt-1 uppercase tracking-wider">
          {{ d.type }}
        </div>
        <div class="text-xs mt-0.5" :class="d.total > 0 && d.inUse === d.total ? 'text-status-warning' : 'text-text-muted'">
          {{ d.total > 0 ? Math.round(d.inUse / d.total * 100) : 0 }}% in use ({{ d.inUse }}/{{ d.total }})
        </div>
        <div v-if="d.unhealthy > 0" class="text-xs text-status-danger mt-0.5">
          {{ d.unhealthy }} unhealthy
        </div>
      </div>
    </div>

    <div
      v-if="error"
      class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <EmptyState
      v-if="!loading && workers.length === 0"
      message="No workers registered"
    />

    <div v-else class="rounded-lg border border-surface-border bg-surface overflow-hidden">
      <DataTable
        :columns="columns"
        :rows="workers"
        :loading="loading && workers.length === 0"
        :page-size="50"
        empty-message="No workers"
      >
        <template #cell-workerId="{ row }">
          <RouterLink
            :to="`/worker/${(row as WorkerHealthStatus).workerId}`"
            class="text-accent hover:underline font-mono"
          >
            {{ (row as WorkerHealthStatus).workerId }}
          </RouterLink>
        </template>

        <template #cell-address="{ row }">
          <span v-if="(row as WorkerHealthStatus).address" class="group/addr inline-flex items-center gap-1">
            {{ (row as WorkerHealthStatus).address }}
            <CopyButton :value="(row as WorkerHealthStatus).address!" />
          </span>
          <span v-else>-</span>
        </template>

        <template #cell-device="{ row }">
          <span class="text-xs">{{ formatWorkerDevice((row as WorkerHealthStatus).metadata) }}</span>
        </template>

        <template #cell-zone="{ row }">
          <span class="text-xs font-mono">
            {{ (row as WorkerHealthStatus).metadata?.attributes?.zone?.stringValue ?? '-' }}
          </span>
        </template>

        <template #cell-tpuName="{ row }">
          {{ (row as WorkerHealthStatus).metadata?.tpuName ?? '-' }}
        </template>

        <template #cell-healthy="{ row }">
          <span class="inline-flex items-center gap-1.5">
            <span
              class="w-2 h-2 rounded-full"
              :class="(row as WorkerHealthStatus).healthy ? 'bg-status-success' : 'bg-status-danger'"
            />
            <span
              class="text-xs"
              :class="(row as WorkerHealthStatus).healthy ? 'text-status-success' : 'text-status-danger'"
            >
              {{ (row as WorkerHealthStatus).healthy ? 'Healthy' : 'Unhealthy' }}
            </span>
          </span>
        </template>

        <template #cell-cpuCount="{ row }">
          <span class="font-mono">
            {{ (row as WorkerHealthStatus).metadata?.cpuCount ?? '-' }}
          </span>
        </template>

        <template #cell-memory="{ row }">
          <span class="font-mono text-xs">
            {{ (row as WorkerHealthStatus).metadata?.memoryBytes
              ? formatBytes(parseInt((row as WorkerHealthStatus).metadata!.memoryBytes!, 10))
              : '-' }}
          </span>
        </template>

        <template #cell-tasks="{ row }">
          <span class="font-mono">
            {{ (row as WorkerHealthStatus).runningJobIds?.length ?? 0 }}
          </span>
        </template>

        <template #cell-lastHeartbeat="{ row }">
          <span class="text-xs font-mono">
            {{ formatRelativeTime(timestampMs((row as WorkerHealthStatus).lastHeartbeat)) }}
          </span>
        </template>

        <template #cell-error="{ row }">
          <span
            v-if="(row as WorkerHealthStatus).statusMessage"
            class="text-xs text-status-danger truncate max-w-xs inline-block"
          >
            {{ (row as WorkerHealthStatus).statusMessage }}
          </span>
          <span v-else class="text-text-muted">-</span>
        </template>
      </DataTable>
    </div>
  </div>
</template>
