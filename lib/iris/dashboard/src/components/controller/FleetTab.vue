<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type { ListWorkersResponse, WorkerHealthStatus } from '@/types/rpc'
import { timestampMs, formatRelativeTime, formatBytes, formatWorkerDevice } from '@/utils/formatting'

import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import EmptyState from '@/components/shared/EmptyState.vue'

const { data, loading, error, refresh } = useControllerRpc<ListWorkersResponse>('ListWorkers')

useAutoRefresh(refresh, 10_000)
onMounted(refresh)

const workers = computed<WorkerHealthStatus[]>(() => data.value?.workers ?? [])

const copiedAddress = ref<string | null>(null)

async function copyAddress(addr: string) {
  const ip = addr.replace(/^https?:\/\//, '').replace(/:\d+$/, '')
  await navigator.clipboard.writeText(ip)
  copiedAddress.value = addr
  setTimeout(() => { copiedAddress.value = null }, 1500)
}

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
            <button
              class="text-text-muted hover:text-text opacity-0 group-hover/addr:opacity-100 transition-opacity"
              title="Copy IP"
              @click="copyAddress((row as WorkerHealthStatus).address!)"
            >
              <svg v-if="copiedAddress === (row as WorkerHealthStatus).address" class="w-3.5 h-3.5 text-status-success" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
              </svg>
              <svg v-else class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
              </svg>
            </button>
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
