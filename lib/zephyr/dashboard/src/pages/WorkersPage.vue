<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import ErrorBanner from '@/components/ErrorBanner.vue'
import StateBadge from '@/components/StateBadge.vue'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { useDashboardRpc } from '@/composables/useRpc'
import type { ListWorkersResponse } from '@/types/dashboard'
import { formatBytes, formatDuration, formatNumber, irisTaskHref } from '@/utils/formatting'

const search = ref('')
const sortField = ref('worker_id')
const sortDescending = ref(false)
const offset = ref(0)
const limit = 50
const rpc = useDashboardRpc<ListWorkersResponse>('ListWorkers', () => ({
  search: search.value,
  sortField: sortField.value,
  sortDescending: sortDescending.value,
  offset: offset.value,
  limit,
}))
const workers = computed(() => rpc.data.value?.workers ?? [])
const total = computed(() => rpc.data.value?.total ?? 0)

function sort(field: string) {
  if (sortField.value === field) sortDescending.value = !sortDescending.value
  else {
    sortField.value = field
    sortDescending.value = false
  }
  void rpc.refresh()
}

watch(search, () => {
  offset.value = 0
  void rpc.refresh()
})
useAutoRefresh(rpc.refresh)
onMounted(rpc.refresh)
</script>

<template>
  <div class="space-y-4">
    <ErrorBanner :message="rpc.error.value" />
    <section class="card overflow-hidden">
      <div class="flex items-center gap-3 border-b border-surface-border p-4">
        <div>
          <h2 class="text-base font-semibold">Workers</h2>
          <p class="text-xs text-text-secondary">{{ total }} coordinator-wide registered workers</p>
        </div>
        <input
          v-model="search"
          class="ml-auto w-64 rounded-lg border border-surface-border bg-surface-raised px-3 py-2 text-sm outline-none focus:border-accent"
          placeholder="Search workers"
        />
      </div>
      <div class="overflow-x-auto">
        <table class="w-full">
          <thead class="bg-surface-sunken">
            <tr>
              <th class="table-header"><button @click="sort('worker_id')">Worker ↕</button></th>
              <th class="table-header"><button @click="sort('state')">State ↕</button></th>
              <th class="table-header text-right"><button @click="sort('active_shards')">Active shards ↕</button></th>
              <th class="table-header text-right"><button @click="sort('cpu')">CPU ↕</button></th>
              <th class="table-header text-right"><button @click="sort('memory')">Memory ↕</button></th>
              <th class="table-header text-right"><button @click="sort('last_seen')">Heartbeat age ↕</button></th>
              <th class="table-header">Profile</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="worker in workers" :key="worker.workerId" class="hover:bg-surface-sunken/60">
              <td class="table-cell font-mono text-xs font-medium">{{ worker.workerId }}</td>
              <td class="table-cell"><StateBadge :value="worker.state" /></td>
              <td class="table-cell text-right font-mono text-xs tabular-nums">
                {{ worker.assignments?.map((assignment) => `${assignment.executionId.slice(0, 8)}:${assignment.shard}`).join(', ') || '—' }}
              </td>
              <td class="table-cell text-right tabular-nums">{{ formatNumber(worker.cpuPercent) }}%</td>
              <td class="table-cell text-right tabular-nums">{{ formatBytes(worker.memoryBytes) }}</td>
              <td class="table-cell text-right tabular-nums text-text-secondary">{{ formatDuration((worker.lastSeenAgeSeconds ?? 0) * 1000) }}</td>
              <td class="table-cell">
                <a v-if="worker.taskId" :href="irisTaskHref(worker.taskId)" target="_top" class="font-medium text-accent hover:text-accent-hover">Open task</a>
                <span v-else class="text-text-muted">—</span>
              </td>
            </tr>
            <tr v-if="!workers.length">
              <td colspan="7" class="p-12 text-center text-sm text-text-muted">No workers match this view.</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="flex items-center justify-between border-t border-surface-border px-4 py-3 text-xs text-text-secondary">
        <span>{{ offset + 1 }}–{{ Math.min(offset + workers.length, total) }} of {{ total }}</span>
        <div class="flex gap-2">
          <button class="rounded border border-surface-border px-3 py-1.5 disabled:opacity-40" :disabled="offset === 0" @click="offset = Math.max(0, offset - limit); rpc.refresh()">Previous</button>
          <button class="rounded border border-surface-border px-3 py-1.5 disabled:opacity-40" :disabled="offset + limit >= total" @click="offset += limit; rpc.refresh()">Next</button>
        </div>
      </div>
    </section>
  </div>
</template>
