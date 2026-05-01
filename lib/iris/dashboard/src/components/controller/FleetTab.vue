<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { useLogServerStatsRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type { WorkerHealth, WorkerMetadata } from '@/types/rpc'
import { formatRelativeTime, formatBytes, formatWorkerDevice } from '@/utils/formatting'
import { decodeArrowIpc } from '@/utils/arrow'

import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import CopyButton from '@/components/shared/CopyButton.vue'

// How far back the latest-row-per-worker query looks. Workers heartbeat at
// ~5s; 5 minutes gives plenty of room for one missed heartbeat without
// losing the worker from the pane, while still bounding the SQL.
//
// DuckDB's now() returns a TIMESTAMPTZ; the stored ts column is tz-naive
// TIMESTAMP populated from a UTC-normalized datetime. Pin the comparison to
// UTC by casting now() explicitly so a non-UTC server doesn't filter with
// the wrong window.
const LOOKBACK_MINUTES = 5
const WORKER_STATS_SQL = `
SELECT *
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY worker_id ORDER BY ts DESC) AS _rn
    FROM "iris.worker"
    WHERE ts > (now() AT TIME ZONE 'UTC')::TIMESTAMP - INTERVAL '${LOOKBACK_MINUTES} minutes'
) ranked
WHERE _rn = 1
ORDER BY worker_id
`.trim()

interface QueryResponse {
  arrowIpc?: string
}

const { data, loading, error, refresh } = useLogServerStatsRpc<QueryResponse>('Query', { sql: WORKER_STATS_SQL })

useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
onMounted(refresh)

// Map raw stat rows to WorkerHealth for display.
function statsRowToWorkerHealth(row: Record<string, unknown>): WorkerHealth {
  const healthy = Boolean(row['healthy'])
  const ts = row['ts']
  let statusMessage = ''
  if (!healthy && ts) {
    // ts is an ISO string after Arrow normalization (Date → toISOString())
    const ageS = Math.floor((Date.now() - new Date(ts as string).getTime()) / 1000)
    statusMessage = `Unhealthy (last seen ${ageS}s ago)`
  }

  const metadata: WorkerMetadata = {
    cpuCount: Number(row['cpu_count']) || undefined,
    memoryBytes: row['memory_bytes'] != null ? String(row['memory_bytes']) : undefined,
    tpuName: (row['tpu_name'] as string) || undefined,
    gceInstanceName: (row['gce_instance_name'] as string) || undefined,
    gceZone: (row['zone'] as string) || undefined,
    attributes: {},
  }
  const deviceType = (row['device_type'] as string) || ''
  const deviceVariant = (row['device_variant'] as string) || ''
  const zone = (row['zone'] as string) || ''
  if (deviceType && metadata.attributes) metadata.attributes['device-type'] = { stringValue: deviceType }
  if (deviceVariant && metadata.attributes) metadata.attributes['device-variant'] = { stringValue: deviceVariant }
  if (zone && metadata.attributes) metadata.attributes['zone'] = { stringValue: zone }

  const runningCount = Number(row['running_task_count']) || 0
  // Placeholder list — the cell renders the length, not the values.
  // When we have per-task identity in the stats namespace this becomes the
  // real list; until then we preserve the count.
  const runningJobIds = new Array<string>(runningCount).fill('')

  const tsMs = ts ? new Date(ts as string).getTime() : undefined
  const lastHeartbeat = tsMs ? { epochMs: String(tsMs) } : undefined

  return {
    workerId: String(row['worker_id'] || ''),
    healthy,
    consecutiveFailures: 0,  // Not carried in the stats namespace.
    lastHeartbeat,
    runningJobIds,
    address: (row['address'] as string) || undefined,
    metadata,
    statusMessage,
  }
}

const workers = computed<WorkerHealth[]>(() => {
  if (!data.value?.arrowIpc) return []
  const { rows } = decodeArrowIpc(data.value.arrowIpc)
  return rows.map(statsRowToWorkerHealth)
})

// Expose a ref so the timestamp formatter can use epoch ms directly.
function timestampMsFromEpochMs(epochMs: string | undefined): number {
  return epochMs ? parseInt(epochMs, 10) : 0
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
            :to="`/worker/${(row as WorkerHealth).workerId}`"
            class="text-accent hover:underline font-mono"
          >
            {{ (row as WorkerHealth).workerId }}
          </RouterLink>
        </template>

        <template #cell-address="{ row }">
          <span v-if="(row as WorkerHealth).address" class="group/addr inline-flex items-center gap-1">
            {{ (row as WorkerHealth).address }}
            <CopyButton :value="(row as WorkerHealth).address!" />
          </span>
          <span v-else>-</span>
        </template>

        <template #cell-device="{ row }">
          <span class="text-xs">{{ formatWorkerDevice((row as WorkerHealth).metadata) }}</span>
        </template>

        <template #cell-zone="{ row }">
          <span class="text-xs font-mono">
            {{ (row as WorkerHealth).metadata?.attributes?.['zone']?.stringValue ?? '-' }}
          </span>
        </template>

        <template #cell-tpuName="{ row }">
          {{ (row as WorkerHealth).metadata?.tpuName ?? '-' }}
        </template>

        <template #cell-healthy="{ row }">
          <span class="inline-flex items-center gap-1.5">
            <span
              class="w-2 h-2 rounded-full"
              :class="(row as WorkerHealth).healthy ? 'bg-status-success' : 'bg-status-danger'"
            />
            <span
              class="text-xs"
              :class="(row as WorkerHealth).healthy ? 'text-status-success' : 'text-status-danger'"
            >
              {{ (row as WorkerHealth).healthy ? 'Healthy' : 'Unhealthy' }}
            </span>
          </span>
        </template>

        <template #cell-cpuCount="{ row }">
          <span class="font-mono">
            {{ (row as WorkerHealth).metadata?.cpuCount ?? '-' }}
          </span>
        </template>

        <template #cell-memory="{ row }">
          <span class="font-mono text-xs">
            {{ (row as WorkerHealth).metadata?.memoryBytes
              ? formatBytes(parseInt((row as WorkerHealth).metadata!.memoryBytes!, 10))
              : '-' }}
          </span>
        </template>

        <template #cell-tasks="{ row }">
          <span class="font-mono">
            {{ (row as WorkerHealth).runningJobIds?.length ?? 0 }}
          </span>
        </template>

        <template #cell-lastHeartbeat="{ row }">
          <span class="text-xs font-mono">
            {{ formatRelativeTime(timestampMsFromEpochMs((row as WorkerHealth).lastHeartbeat?.epochMs)) }}
          </span>
        </template>

        <template #cell-error="{ row }">
          <span
            v-if="(row as WorkerHealth).statusMessage"
            class="text-xs text-status-danger truncate max-w-xs inline-block"
          >
            {{ (row as WorkerHealth).statusMessage }}
          </span>
          <span v-else class="text-text-muted">-</span>
        </template>
      </DataTable>
    </div>
  </div>
</template>
