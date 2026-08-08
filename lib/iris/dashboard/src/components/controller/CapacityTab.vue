<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { useResourceRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type { ResourceListSlicesResponse, ResourceSliceSummary } from '@/types/rpc'
import { formatRelativeTime, timestampMs } from '@/utils/formatting'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import SourceWarnings from '@/components/shared/SourceWarnings.vue'

const PAGE_SIZE = 100
const route = useRoute()
const scalingGroup = ref(typeof route.query.group === 'string' ? route.query.group : '')
const pageToken = ref<string | undefined>()
const previousTokens = ref<(string | undefined)[]>([])
const { data, loading, error, refresh } = useResourceRpc<ResourceListSlicesResponse>('ListSlices', () => ({
  query: {
    backendId: typeof route.query.backend === 'string' ? route.query.backend : undefined,
    scalingGroupId: scalingGroup.value || undefined,
    page: { pageSize: PAGE_SIZE, pageToken: pageToken.value },
  },
}))
const slices = computed(() => data.value?.slices ?? [])
const columns: Column[] = [
  { key: 'slice', label: 'Slice' },
  { key: 'lifecycle', label: 'Lifecycle' },
  { key: 'backend', label: 'Backend' },
  { key: 'group', label: 'Scaling group' },
  { key: 'members', label: 'Members', align: 'right' },
  { key: 'observed', label: 'Observed' },
]
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
      <div><h2 class="text-xl font-semibold">Capacity</h2><p class="text-sm text-text-muted">Backend-owned slices and observed membership.</p></div>
      <form class="flex gap-2" @submit.prevent="applyFilter">
        <input v-model="scalingGroup" class="px-3 py-1.5 text-sm border rounded bg-surface" placeholder="Scaling group" />
        <button class="px-3 py-1.5 text-sm border rounded">Filter</button>
      </form>
    </div>
    <div v-if="error" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">{{ error }}</div>
    <SourceWarnings :statuses="data?.page?.sourceStatuses" />
    <EmptyState v-if="!loading && slices.length === 0" message="No slices reported" />
    <DataTable v-else :columns="columns" :rows="slices" :loading="loading && slices.length === 0">
      <template #cell-slice="{ row }"><span class="font-mono">{{ (row as ResourceSliceSummary).identity.key.resourceId }}</span></template>
      <template #cell-lifecycle="{ row }"><StatusBadge :status="(row as ResourceSliceSummary).lifecycle" size="sm" /></template>
      <template #cell-backend="{ row }"><span class="font-mono text-xs">{{ (row as ResourceSliceSummary).identity.backendId }}</span></template>
      <template #cell-group="{ row }"><span class="font-mono text-xs">{{ (row as ResourceSliceSummary).scalingGroupId }}</span></template>
      <template #cell-members="{ row }">{{ (row as ResourceSliceSummary).membershipState.includes('UNKNOWN') ? 'unknown' : (row as ResourceSliceSummary).observedMemberCount }}</template>
      <template #cell-observed="{ row }">{{ formatRelativeTime(timestampMs((row as ResourceSliceSummary).observedAt)) }}</template>
    </DataTable>
    <div class="flex justify-end gap-2">
      <button :disabled="previousTokens.length === 0" class="px-3 py-1 text-sm border rounded disabled:opacity-40" @click="previousPage">Previous</button>
      <button :disabled="!data?.page?.nextPageToken" class="px-3 py-1 text-sm border rounded disabled:opacity-40" @click="nextPage">Next</button>
    </div>
  </section>
</template>
