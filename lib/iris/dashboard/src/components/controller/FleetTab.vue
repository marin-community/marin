<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type { ListWorkersResponse, WorkerHealthStatus, WorkerQuery } from '@/types/rpc'
import { timestampMs, formatRelativeTime, formatBytes, formatWorkerDevice } from '@/utils/formatting'

import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import CopyButton from '@/components/shared/CopyButton.vue'

// Server-side pagination — the controller used to enumerate the entire
// roster on every refresh, which scales linearly with cluster size and
// dominates dashboard latency on large fleets. We page in 50-worker chunks
// and keep the filter inputs in the URL so back-button + sharing work.
const PAGE_SIZE = 50

const SORT_FIELD_MAP: Record<string, string> = {
  workerId: 'WORKER_SORT_FIELD_WORKER_ID',
  lastHeartbeat: 'WORKER_SORT_FIELD_LAST_HEARTBEAT',
  device: 'WORKER_SORT_FIELD_DEVICE_TYPE',
}

type SortField = 'workerId' | 'lastHeartbeat' | 'device'
type SortDir = 'asc' | 'desc'
type DeviceFilter = '' | 'cpu' | 'gpu' | 'tpu'

const SORT_FIELDS: SortField[] = ['workerId', 'lastHeartbeat', 'device']
const SORT_DIRS: SortDir[] = ['asc', 'desc']
const DEVICE_FILTERS: DeviceFilter[] = ['', 'cpu', 'gpu', 'tpu']

const route = useRoute()
const router = useRouter()

function queryStr(v: string | string[] | null | undefined): string {
  if (Array.isArray(v)) return v[0] ?? ''
  return v ?? ''
}

function parseSort(v: string): SortField {
  return SORT_FIELDS.includes(v as SortField) ? (v as SortField) : 'workerId'
}
function parseDir(v: string): SortDir {
  return SORT_DIRS.includes(v as SortDir) ? (v as SortDir) : 'asc'
}
function parsePage(v: string): number {
  const n = Number(v)
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : 0
}
function parseDevice(v: string): DeviceFilter {
  return DEVICE_FILTERS.includes(v as DeviceFilter) ? (v as DeviceFilter) : ''
}

const page = ref(parsePage(queryStr(route.query.page)))
const sortField = ref<SortField>(parseSort(queryStr(route.query.sort)))
const sortDir = ref<SortDir>(parseDir(queryStr(route.query.dir)))
const prefixFilter = ref(queryStr(route.query.prefix))
const localPrefix = ref(queryStr(route.query.prefix))
const deviceFilter = ref<DeviceFilter>(parseDevice(queryStr(route.query.device)))

const { data, loading, error, refresh } = useControllerRpc<ListWorkersResponse>('ListWorkers', () => ({
  query: {
    prefix: prefixFilter.value || undefined,
    type: deviceFilter.value || undefined,
    sortField: SORT_FIELD_MAP[sortField.value],
    sortDirection: sortDir.value === 'asc' ? 'SORT_DIRECTION_ASC' : 'SORT_DIRECTION_DESC',
    offset: page.value * PAGE_SIZE,
    limit: PAGE_SIZE,
  } satisfies WorkerQuery,
}))

useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
onMounted(refresh)

const workers = computed<WorkerHealthStatus[]>(() => data.value?.workers ?? [])
const totalCount = computed(() => data.value?.totalCount ?? 0)
const hasMore = computed(() => data.value?.hasMore ?? false)
const totalPages = computed(() => Math.max(1, Math.ceil(totalCount.value / PAGE_SIZE)))

watch([page, sortField, sortDir, prefixFilter, deviceFilter], () => {
  refresh()
})

watch([prefixFilter, deviceFilter], () => {
  page.value = 0
})

watch([page, sortField, sortDir, prefixFilter, deviceFilter], () => {
  router.replace({
    query: {
      ...route.query,
      sort: sortField.value !== 'workerId' ? sortField.value : undefined,
      dir: sortDir.value !== 'asc' ? sortDir.value : undefined,
      page: page.value !== 0 ? String(page.value) : undefined,
      prefix: prefixFilter.value || undefined,
      device: deviceFilter.value || undefined,
    },
  })
})

function handleFilterSubmit() {
  prefixFilter.value = localPrefix.value
}

function handleFilterClear() {
  localPrefix.value = ''
  prefixFilter.value = ''
  deviceFilter.value = ''
  page.value = 0
}

const hasActiveFilter = computed(() => !!prefixFilter.value || !!deviceFilter.value)

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
        {{ totalCount }} worker{{ totalCount !== 1 ? 's' : '' }}
      </span>
    </div>

    <!-- Filter bar -->
    <div class="mb-4 flex flex-wrap items-center gap-2 sm:gap-3">
      <form class="flex flex-wrap flex-1 sm:flex-initial gap-2" @submit.prevent="handleFilterSubmit">
        <select
          v-model="deviceFilter"
          class="px-3 py-1.5 text-sm border border-surface-border rounded
                 bg-surface text-text
                 focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
        >
          <option value="">All devices</option>
          <option value="cpu">CPU</option>
          <option value="gpu">GPU</option>
          <option value="tpu">TPU</option>
        </select>
        <input
          v-model="localPrefix"
          type="text"
          placeholder="Filter by worker ID or address..."
          class="flex-1 sm:flex-initial sm:w-64 px-3 py-1.5 text-sm border border-surface-border rounded
                 bg-surface placeholder:text-text-muted
                 focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
        />
        <button
          type="submit"
          class="px-3 py-1.5 text-sm border border-surface-border rounded hover:bg-surface-raised"
        >
          Filter
        </button>
        <button
          v-if="hasActiveFilter"
          type="button"
          class="px-3 py-1.5 text-sm border border-surface-border rounded hover:bg-surface-raised text-status-danger"
          @click="handleFilterClear"
        >
          Reset
        </button>
      </form>
    </div>

    <div
      v-if="error"
      class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <EmptyState
      v-if="!loading && workers.length === 0"
      :message="hasActiveFilter ? 'No workers matching filter' : 'No workers registered'"
    />

    <div v-else class="rounded-lg border border-surface-border bg-surface overflow-hidden">
      <DataTable
        :columns="columns"
        :rows="workers"
        :loading="loading && workers.length === 0"
        :page-size="PAGE_SIZE"
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

      <!-- Pagination -->
      <div
        v-if="totalPages > 1"
        class="flex items-center justify-between px-3 py-2 text-xs text-text-secondary border-t border-surface-border"
      >
        <span>
          {{ page * PAGE_SIZE + 1 }}&ndash;{{ Math.min((page + 1) * PAGE_SIZE, totalCount) }}
          of {{ totalCount }}
        </span>
        <div class="flex items-center gap-1">
          <button
            :disabled="page === 0"
            class="px-2 py-1 rounded hover:bg-surface-raised disabled:opacity-30 disabled:cursor-not-allowed"
            @click="page = Math.max(0, page - 1)"
          >
            &larr; Prev
          </button>
          <span class="px-2 font-mono">{{ page + 1 }} / {{ totalPages }}</span>
          <button
            :disabled="!hasMore"
            class="px-2 py-1 rounded hover:bg-surface-raised disabled:opacity-30 disabled:cursor-not-allowed"
            @click="page++"
          >
            Next &rarr;
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
