<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type { ListWorkersResponse, WorkerHealthStatus, ProtoTimestamp } from '@/types/rpc'

import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import EmptyState from '@/components/shared/EmptyState.vue'

const { data, loading, error, refresh } = useControllerRpc<ListWorkersResponse>('ListWorkers')

useAutoRefresh(refresh, 10_000)
onMounted(refresh)

function timestampMs(ts?: ProtoTimestamp): number {
  if (!ts?.epochMs) return 0
  return parseInt(ts.epochMs, 10) || 0
}

function formatRelativeTime(ms: number): string {
  if (!ms) return '-'
  const seconds = Math.floor((Date.now() - ms) / 1000)
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

function formatDevice(w: WorkerHealthStatus): string {
  const md = w.metadata
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

const workers = computed<WorkerHealthStatus[]>(() => data.value?.workers ?? [])

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

    <div v-else class="rounded-lg border border-surface-border bg-white overflow-hidden">
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
          {{ (row as WorkerHealthStatus).address ?? '-' }}
        </template>

        <template #cell-device="{ row }">
          <span class="text-xs">{{ formatDevice(row as WorkerHealthStatus) }}</span>
        </template>

        <template #cell-zone="{ row }">
          <span class="text-xs font-mono">
            {{ (row as WorkerHealthStatus).metadata?.gceZone ?? '-' }}
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
