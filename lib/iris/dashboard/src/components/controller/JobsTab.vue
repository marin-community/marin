<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { useResourceRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type { ResourceJobSummary, ResourceListJobsResponse } from '@/types/rpc'
import { formatRelativeTime, timestampMs } from '@/utils/formatting'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import SourceWarnings from '@/components/shared/SourceWarnings.vue'

const PAGE_SIZE = 50
const route = useRoute()
const router = useRouter()
const owner = ref(typeof route.query.owner === 'string' ? route.query.owner : '')
const jobPrefix = ref(typeof route.query.prefix === 'string' ? route.query.prefix : '')
const pageToken = ref<string | undefined>(undefined)
const previousTokens = ref<(string | undefined)[]>([])

const { data, loading, error, refresh } = useResourceRpc<ResourceListJobsResponse>('ListJobs', () => ({
  query: {
    ownerId: owner.value || undefined,
    jobIdPrefix: jobPrefix.value || undefined,
    backendId: typeof route.query.backend === 'string' ? route.query.backend : undefined,
    executionClusterId: typeof route.query.cluster === 'string' ? route.query.cluster : undefined,
    page: { pageSize: PAGE_SIZE, pageToken: pageToken.value },
  },
}))

const jobs = computed(() => data.value?.jobs ?? [])
const columns: Column[] = [
  { key: 'job', label: 'Job' },
  { key: 'state', label: 'State' },
  { key: 'owner', label: 'Owner' },
  { key: 'backend', label: 'Backend' },
  { key: 'tasks', label: 'Tasks', align: 'right' },
  { key: 'submitted', label: 'Submitted' },
]

function applyFilters() {
  previousTokens.value = []
  pageToken.value = undefined
  void router.replace({ query: { ...route.query, owner: owner.value || undefined, prefix: jobPrefix.value || undefined } })
  void refresh()
}

function nextPage() {
  const next = data.value?.page?.nextPageToken
  if (!next) return
  previousTokens.value.push(pageToken.value)
  pageToken.value = next
  void refresh()
}

function previousPage() {
  pageToken.value = previousTokens.value.pop()
  void refresh()
}

function jobRoute(job: ResourceJobSummary) {
  return {
    name: 'job-detail',
    params: { clusterId: job.identity.key.clusterId, jobId: job.identity.key.resourceId },
  }
}

watch(() => [route.query.backend, route.query.cluster], applyFilters)
onMounted(refresh)
useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
</script>

<template>
  <section class="space-y-4">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <h2 class="text-xl font-semibold">Jobs</h2>
      <form class="flex flex-wrap gap-2" @submit.prevent="applyFilters">
        <input v-model="owner" class="px-3 py-1.5 text-sm border rounded bg-surface" placeholder="Owner" />
        <input v-model="jobPrefix" class="px-3 py-1.5 text-sm border rounded bg-surface" placeholder="Job ID prefix" />
        <button class="px-3 py-1.5 text-sm border rounded hover:bg-surface-raised">Filter</button>
      </form>
    </div>

    <div v-if="error" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">
      {{ error }}
    </div>
    <SourceWarnings :statuses="data?.page?.sourceStatuses" />
    <EmptyState v-if="!loading && jobs.length === 0" message="No jobs found" />
    <DataTable v-else :columns="columns" :rows="jobs" :loading="loading && jobs.length === 0">
      <template #cell-job="{ row }">
        <RouterLink :to="jobRoute(row as ResourceJobSummary)" class="font-mono text-accent hover:underline">
          {{ (row as ResourceJobSummary).identity.key.resourceId }}
        </RouterLink>
      </template>
      <template #cell-state="{ row }"><StatusBadge :status="(row as ResourceJobSummary).state" size="sm" /></template>
      <template #cell-owner="{ row }">{{ (row as ResourceJobSummary).ownerId }}</template>
      <template #cell-backend="{ row }">
        <span class="font-mono text-xs">{{ (row as ResourceJobSummary).backendId || (row as ResourceJobSummary).executionClusterId }}</span>
      </template>
      <template #cell-tasks="{ row }">{{ (row as ResourceJobSummary).numTasks }}</template>
      <template #cell-submitted="{ row }">{{ formatRelativeTime(timestampMs((row as ResourceJobSummary).submittedAt)) }}</template>
    </DataTable>

    <div class="flex justify-end gap-2">
      <button :disabled="previousTokens.length === 0" class="px-3 py-1 text-sm border rounded disabled:opacity-40" @click="previousPage">Previous</button>
      <button :disabled="!data?.page?.nextPageToken" class="px-3 py-1 text-sm border rounded disabled:opacity-40" @click="nextPage">Next</button>
    </div>
  </section>
</template>
