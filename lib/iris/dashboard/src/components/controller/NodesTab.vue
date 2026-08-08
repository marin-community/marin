<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import { useResourceRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type { ResourceListNodesResponse, ResourceNodeSummary } from '@/types/rpc'
import { formatBytes, formatRelativeTime, timestampMs } from '@/utils/formatting'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import SourceWarnings from '@/components/shared/SourceWarnings.vue'

const PAGE_SIZE = 100
const route = useRoute()
const contains = ref(typeof route.query.contains === 'string' ? route.query.contains : '')
const pageToken = ref<string | undefined>()
const previousTokens = ref<(string | undefined)[]>([])
const { data, loading, error, refresh } = useResourceRpc<ResourceListNodesResponse>('ListNodes', () => ({
  query: {
    backendId: typeof route.query.backend === 'string' ? route.query.backend : undefined,
    contains: contains.value || undefined,
    page: { pageSize: PAGE_SIZE, pageToken: pageToken.value },
  },
}))
const nodes = computed(() => data.value?.nodes ?? [])
const columns: Column[] = [
  { key: 'node', label: 'Node' },
  { key: 'health', label: 'Health' },
  { key: 'backend', label: 'Backend' },
  { key: 'capacity', label: 'Capacity' },
  { key: 'tasks', label: 'Tasks', align: 'right' },
  { key: 'observed', label: 'Observed' },
]
function nodeRoute(node: ResourceNodeSummary) {
  return {
    name: 'node-detail',
    params: {
      clusterId: node.identity.key.clusterId,
      backendId: node.identity.backendId,
      nodeUid: node.identity.nodeUid,
      nodeId: node.identity.key.resourceId,
    },
  }
}
function capacity(node: ResourceNodeSummary): string {
  const c = node.capacity
  if (!c) return '—'
  const accelerator = c.acceleratorCount ? `${c.acceleratorCount}× ${c.acceleratorVariant || c.acceleratorKind}` : ''
  const memory = c.memoryBytes ? formatBytes(Number(c.memoryBytes)) : ''
  return [accelerator, memory].filter(Boolean).join(' · ') || '—'
}
function applyFilter() { pageToken.value = undefined; previousTokens.value = []; void refresh() }
function nextPage() {
  const next = data.value?.page?.nextPageToken
  if (!next) return
  previousTokens.value.push(pageToken.value); pageToken.value = next; void refresh()
}
function previousPage() { pageToken.value = previousTokens.value.pop(); void refresh() }
onMounted(refresh)
useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
</script>

<template>
  <section class="space-y-4">
    <div class="flex items-center justify-between gap-3">
      <h2 class="text-xl font-semibold">Nodes</h2>
      <form class="flex gap-2" @submit.prevent="applyFilter">
        <input v-model="contains" class="px-3 py-1.5 text-sm border rounded bg-surface" placeholder="Node ID or address" />
        <button class="px-3 py-1.5 text-sm border rounded">Filter</button>
      </form>
    </div>
    <div v-if="error" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">{{ error }}</div>
    <SourceWarnings :statuses="data?.page?.sourceStatuses" />
    <EmptyState v-if="!loading && nodes.length === 0" message="No nodes found" />
    <DataTable v-else :columns="columns" :rows="nodes" :loading="loading && nodes.length === 0">
      <template #cell-node="{ row }">
        <RouterLink :to="nodeRoute(row as ResourceNodeSummary)" class="font-mono text-accent hover:underline">
          {{ (row as ResourceNodeSummary).identity.key.resourceId }}
        </RouterLink>
      </template>
      <template #cell-health="{ row }"><StatusBadge :status="(row as ResourceNodeSummary).health" size="sm" /></template>
      <template #cell-backend="{ row }"><span class="font-mono text-xs">{{ (row as ResourceNodeSummary).identity.backendId }}</span></template>
      <template #cell-capacity="{ row }">{{ capacity(row as ResourceNodeSummary) }}</template>
      <template #cell-tasks="{ row }">{{ (row as ResourceNodeSummary).runningTaskCount }}</template>
      <template #cell-observed="{ row }">{{ formatRelativeTime(timestampMs((row as ResourceNodeSummary).observedAt)) }}</template>
    </DataTable>
    <div class="flex justify-end gap-2">
      <button :disabled="previousTokens.length === 0" class="px-3 py-1 text-sm border rounded disabled:opacity-40" @click="previousPage">Previous</button>
      <button :disabled="!data?.page?.nextPageToken" class="px-3 py-1 text-sm border rounded disabled:opacity-40" @click="nextPage">Next</button>
    </div>
  </section>
</template>
