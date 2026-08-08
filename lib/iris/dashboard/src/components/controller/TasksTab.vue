<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import { useResourceRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type { ResourceListTasksResponse, ResourceTaskSummary } from '@/types/rpc'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import SourceWarnings from '@/components/shared/SourceWarnings.vue'

const PAGE_SIZE = 100
const route = useRoute()
const pageToken = ref<string | undefined>()
const previousTokens = ref<(string | undefined)[]>([])
const prefix = ref(typeof route.query.prefix === 'string' ? route.query.prefix : '')

const { data, loading, error, refresh } = useResourceRpc<ResourceListTasksResponse>('ListTasks', () => ({
  query: {
    jobIdPrefix: prefix.value || undefined,
    backendId: typeof route.query.backend === 'string' ? route.query.backend : undefined,
    executionClusterId: typeof route.query.cluster === 'string' ? route.query.cluster : undefined,
    page: { pageSize: PAGE_SIZE, pageToken: pageToken.value },
  },
}))

const tasks = computed(() => data.value?.tasks ?? [])
const columns: Column[] = [
  { key: 'task', label: 'Task' },
  { key: 'state', label: 'State' },
  { key: 'backend', label: 'Backend' },
  { key: 'attempt', label: 'Attempt', align: 'right' },
  { key: 'failures', label: 'Failures', align: 'right' },
  { key: 'status', label: 'Status' },
]

function taskRoute(task: ResourceTaskSummary) {
  return { name: 'task-detail', params: { clusterId: task.identity.key.clusterId, taskId: task.identity.key.resourceId } }
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
      <h2 class="text-xl font-semibold">Tasks</h2>
      <form class="flex gap-2" @submit.prevent="applyFilter">
        <input v-model="prefix" class="px-3 py-1.5 text-sm border rounded bg-surface" placeholder="Job ID prefix" />
        <button class="px-3 py-1.5 text-sm border rounded">Filter</button>
      </form>
    </div>
    <div v-if="error" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">{{ error }}</div>
    <SourceWarnings :statuses="data?.page?.sourceStatuses" />
    <EmptyState v-if="!loading && tasks.length === 0" message="No tasks found" />
    <DataTable v-else :columns="columns" :rows="tasks" :loading="loading && tasks.length === 0">
      <template #cell-task="{ row }">
        <RouterLink :to="taskRoute(row as ResourceTaskSummary)" class="font-mono text-accent hover:underline">
          {{ (row as ResourceTaskSummary).identity.key.resourceId }}
        </RouterLink>
      </template>
      <template #cell-state="{ row }"><StatusBadge :status="(row as ResourceTaskSummary).state" size="sm" /></template>
      <template #cell-backend="{ row }"><span class="font-mono text-xs">{{ (row as ResourceTaskSummary).backendId || (row as ResourceTaskSummary).executionClusterId }}</span></template>
      <template #cell-attempt="{ row }">{{ (row as ResourceTaskSummary).currentAttempt?.attemptNumber ?? '—' }}</template>
      <template #cell-failures="{ row }">{{ (row as ResourceTaskSummary).failureCount }}</template>
      <template #cell-status="{ row }"><span class="text-xs">{{ (row as ResourceTaskSummary).statusMessage || (row as ResourceTaskSummary).errorMessage || '—' }}</span></template>
    </DataTable>
    <div class="flex justify-end gap-2">
      <button :disabled="previousTokens.length === 0" class="px-3 py-1 text-sm border rounded disabled:opacity-40" @click="previousPage">Previous</button>
      <button :disabled="!data?.page?.nextPageToken" class="px-3 py-1 text-sm border rounded disabled:opacity-40" @click="nextPage">Next</button>
    </div>
  </section>
</template>
