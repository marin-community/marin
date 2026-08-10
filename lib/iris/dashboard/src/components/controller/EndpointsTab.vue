<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { resourceRpcCall, useResourceRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type {
  ResourceDescribeEndpointResponse,
  ResourceEndpointDetail,
  ResourceEndpointSummary,
  ResourceListEndpointsResponse,
} from '@/types/rpc'
import { formatRelativeTime, timestampMs } from '@/utils/formatting'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import CopyButton from '@/components/shared/CopyButton.vue'
import EndpointLink from '@/components/shared/EndpointLink.vue'
import SourceWarnings from '@/components/shared/SourceWarnings.vue'

const PAGE_SIZE = 100
const prefix = ref('')
const pageToken = ref<string | undefined>()
const previousTokens = ref<(string | undefined)[]>([])
const selected = ref<ResourceEndpointDetail | null>(null)
const detailError = ref<string | null>(null)
const { data, loading, error, refresh } = useResourceRpc<ResourceListEndpointsResponse>('ListEndpoints', () => ({
  query: { namePrefix: prefix.value || undefined, page: { pageSize: PAGE_SIZE, pageToken: pageToken.value } },
}))
const endpoints = computed(() => data.value?.endpoints ?? [])
const columns: Column[] = [
  { key: 'name', label: 'Name' },
  { key: 'task', label: 'Task' },
  { key: 'cluster', label: 'Execution cluster' },
  { key: 'access', label: 'Access' },
  { key: 'lease', label: 'Lease' },
]
async function describe(endpoint: ResourceEndpointSummary) {
  detailError.value = null
  try {
    selected.value = (await resourceRpcCall<ResourceDescribeEndpointResponse>('DescribeEndpoint', { endpoint: endpoint.key })).endpoint ?? null
  } catch (e) { detailError.value = e instanceof Error ? e.message : String(e) }
}
function jobIdForTask(taskId: string): string {
  return taskId.slice(0, taskId.lastIndexOf('/'))
}
function applyFilter() { pageToken.value = undefined; previousTokens.value = []; selected.value = null; void refresh() }
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
      <h2 class="text-xl font-semibold">Endpoints</h2>
      <form class="flex gap-2" @submit.prevent="applyFilter">
        <input v-model="prefix" class="px-3 py-1.5 text-sm border rounded bg-surface" placeholder="Name prefix" />
        <button class="px-3 py-1.5 text-sm border rounded">Filter</button>
      </form>
    </div>
    <div v-if="error" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">{{ error }}</div>
    <SourceWarnings :statuses="data?.page?.sourceStatuses" />
    <EmptyState v-if="!loading && endpoints.length === 0" message="No endpoints found" />
    <DataTable v-else :columns="columns" :rows="endpoints" :loading="loading && endpoints.length === 0">
      <template #cell-name="{ row }">
        <div class="flex items-center gap-3">
          <EndpointLink :name="(row as ResourceEndpointSummary).name" class="font-mono" />
          <button class="text-xs text-text-muted hover:text-accent" @click="describe(row as ResourceEndpointSummary)">Details</button>
        </div>
      </template>
      <template #cell-task="{ row }">
        <div v-if="(row as ResourceEndpointSummary).task" class="space-y-0.5">
          <RouterLink
            :to="{ name: 'task-detail', params: { clusterId: (row as ResourceEndpointSummary).task!.clusterId, taskId: (row as ResourceEndpointSummary).task!.resourceId } }"
            class="block font-mono text-xs text-accent hover:underline"
          >{{ (row as ResourceEndpointSummary).task!.resourceId }}</RouterLink>
          <RouterLink
            :to="{ name: 'job-detail', params: { clusterId: (row as ResourceEndpointSummary).task!.clusterId, jobId: jobIdForTask((row as ResourceEndpointSummary).task!.resourceId) } }"
            class="block text-xs text-text-muted hover:text-accent"
          >Job</RouterLink>
        </div>
        <span v-else>—</span>
      </template>
      <template #cell-cluster="{ row }">{{ (row as ResourceEndpointSummary).executionClusterId }}</template>
      <template #cell-access="{ row }">{{ (row as ResourceEndpointSummary).access }}</template>
      <template #cell-lease="{ row }">{{ formatRelativeTime(timestampMs((row as ResourceEndpointSummary).leaseDeadline)) }}</template>
    </DataTable>
    <div v-if="detailError" class="text-sm text-status-danger">{{ detailError }}</div>
    <div v-if="selected" class="p-4 border rounded space-y-2 text-sm">
      <div class="flex justify-between">
        <h3 class="font-semibold font-mono">{{ selected.summary.name }}</h3>
        <button class="text-text-muted" @click="selected = null">Close</button>
      </div>
      <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div><div class="text-xs text-text-muted">Address</div><div class="flex items-center gap-1 font-mono break-all">{{ selected.address }} <CopyButton :value="selected.address" /></div></div>
        <div><div class="text-xs text-text-muted">Access</div><div>{{ selected.summary.access }}</div></div>
        <div><div class="text-xs text-text-muted">Execution cluster</div><div class="font-mono">{{ selected.summary.executionClusterId }}</div></div>
        <div><div class="text-xs text-text-muted">Lease</div><div>{{ formatRelativeTime(timestampMs(selected.summary.leaseDeadline)) }}</div></div>
      </div>
      <EndpointLink :name="selected.summary.name" class="font-mono text-sm" />
      <dl v-if="Object.keys(selected.metadata ?? {}).length" class="grid gap-2 border-t pt-3 sm:grid-cols-2">
        <template v-for="(value, key) in selected.metadata ?? {}" :key="key">
          <dt class="font-mono text-text-muted">{{ key }}</dt><dd>{{ value }}</dd>
        </template>
      </dl>
    </div>
    <div class="flex justify-end gap-2">
      <button :disabled="previousTokens.length === 0" class="px-3 py-1 text-sm border rounded disabled:opacity-40" @click="previousPage">Previous</button>
      <button :disabled="!data?.page?.nextPageToken" class="px-3 py-1 text-sm border rounded disabled:opacity-40" @click="nextPage">Next</button>
    </div>
  </section>
</template>
