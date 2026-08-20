<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { listResources, RESOURCE_MESSAGES, RESOURCE_TYPES, useListResources } from '@/composables/useResources'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type { ResourceJobSummary } from '@/types/rpc'
import { formatRelativeTime, timestampMs } from '@/utils/formatting'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import SourceWarnings from '@/components/shared/SourceWarnings.vue'
import UsersOverview from '@/components/controller/UsersOverview.vue'

const PAGE_SIZE = 50
const route = useRoute()
const router = useRouter()
const selectedUser = computed(() => typeof route.query.user === 'string' ? route.query.user : '')
const showAll = computed(() => route.query.all === '1')
const backendId = computed(() => typeof route.query.backend === 'string' ? route.query.backend : '')
const clusterId = computed(() => typeof route.query.cluster === 'string' ? route.query.cluster : '')
const inJobList = computed(() => Boolean(selectedUser.value || showAll.value || backendId.value || clusterId.value))
const owner = ref(typeof route.query.owner === 'string' ? route.query.owner : '')
const jobPrefix = ref(typeof route.query.prefix === 'string' ? route.query.prefix : '')
const stateFilter = ref(typeof route.query.state === 'string' ? route.query.state : '')
const pageToken = ref<string | undefined>(undefined)
const previousTokens = ref<(string | undefined)[]>([])
const expandedJobs = ref(new Set<string>())
const childJobs = ref(new Map<string, ResourceJobSummary[]>())
const loadingChildren = ref(new Set<string>())

const { data, loading, error, refresh } = useListResources<ResourceJobSummary>(
  RESOURCE_TYPES.job,
  RESOURCE_MESSAGES.jobQuery,
  () => ({
    ownerId: selectedUser.value || owner.value || undefined,
    jobIdPrefix: jobPrefix.value || undefined,
    backendId: backendId.value || undefined,
    executionClusterId: clusterId.value || undefined,
    states: stateFilter.value ? [stateFilter.value] : undefined,
    topLevelOnly: true,
    page: { pageSize: PAGE_SIZE, pageToken: pageToken.value },
  }),
  'BASIC',
)

const jobs = computed(() => data.value?.items ?? [])
type JobTreeRow = ResourceJobSummary & { depth: number }

const visibleJobs = computed<JobTreeRow[]>(() => {
  const rows: JobTreeRow[] = []
  function append(items: ResourceJobSummary[], depth: number) {
    for (const job of items) {
      rows.push({ ...job, depth })
      const jobId = job.identity.key.resourceId
      if (expandedJobs.value.has(jobId)) append(childJobs.value.get(jobId) ?? [], depth + 1)
    }
  }
  append(jobs.value, 0)
  return rows
})
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
  expandedJobs.value = new Set()
  childJobs.value = new Map()
  void router.replace({
    query: {
      ...route.query,
      owner: selectedUser.value ? undefined : owner.value || undefined,
      prefix: jobPrefix.value || undefined,
      state: stateFilter.value || undefined,
    },
  })
  void refreshJobs()
}

async function fetchChildren(job: ResourceJobSummary): Promise<ResourceJobSummary[]> {
  const items: ResourceJobSummary[] = []
  let nextPageToken: string | undefined
  do {
    const response = await listResources<ResourceJobSummary>(
      RESOURCE_TYPES.job,
      RESOURCE_MESSAGES.jobQuery,
      {
        parent: job.identity.key,
        page: { pageSize: 500, pageToken: nextPageToken },
      },
      'BASIC',
    )
    items.push(...response.items)
    nextPageToken = response.page?.nextPageToken || undefined
  } while (nextPageToken)
  return items
}

async function toggleChildren(job: ResourceJobSummary) {
  const jobId = job.identity.key.resourceId
  if (expandedJobs.value.has(jobId)) {
    const next = new Set(expandedJobs.value)
    next.delete(jobId)
    expandedJobs.value = next
    return
  }
  loadingChildren.value = new Set(loadingChildren.value).add(jobId)
  try {
    const children = await fetchChildren(job)
    const nextChildren = new Map(childJobs.value)
    nextChildren.set(jobId, children)
    childJobs.value = nextChildren
    if (children.length > 0) expandedJobs.value = new Set(expandedJobs.value).add(jobId)
  } finally {
    const nextLoading = new Set(loadingChildren.value)
    nextLoading.delete(jobId)
    loadingChildren.value = nextLoading
  }
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

async function refreshJobs() {
  if (!inJobList.value) return
  await refresh()
  if (expandedJobs.value.size === 0) return
  const loaded = [...jobs.value, ...[...childJobs.value.values()].flat()]
  const refreshed = await Promise.all(
    [...expandedJobs.value].map(async (jobId) => {
      const job = loaded.find(candidate => candidate.identity.key.resourceId === jobId)
      return job ? [jobId, await fetchChildren(job)] as const : null
    }),
  )
  childJobs.value = new Map(refreshed.filter(entry => entry !== null))
}

watch([selectedUser, showAll, backendId, clusterId], () => {
  previousTokens.value = []
  pageToken.value = undefined
  expandedJobs.value = new Set()
  childJobs.value = new Map()
  void refreshJobs()
})
onMounted(refreshJobs)
useAutoRefresh(refreshJobs, DEFAULT_REFRESH_MS)
</script>

<template>
  <UsersOverview v-if="!inJobList" />
  <section v-else class="space-y-4">
    <div v-if="selectedUser" class="flex items-center gap-2 text-sm text-text-secondary">
      <RouterLink to="/" class="text-accent hover:underline">Users</RouterLink>
      <span>/</span>
      <span class="font-mono text-text">{{ selectedUser }}</span>
    </div>
    <div class="flex flex-wrap items-center justify-between gap-3">
      <h2 class="text-xl font-semibold">{{ selectedUser ? `${selectedUser}'s jobs` : 'Jobs' }}</h2>
      <form class="flex flex-wrap gap-2" @submit.prevent="applyFilters">
        <input v-if="!selectedUser" v-model="owner" class="px-3 py-1.5 text-sm border rounded bg-surface" placeholder="Owner" />
        <input v-model="jobPrefix" class="px-3 py-1.5 text-sm border rounded bg-surface" placeholder="Job ID prefix" />
        <select v-model="stateFilter" class="px-3 py-1.5 text-sm border rounded bg-surface">
          <option value="">All states</option>
          <option value="JOB_STATE_PENDING">Pending</option>
          <option value="JOB_STATE_BUILDING">Building</option>
          <option value="JOB_STATE_RUNNING">Running</option>
          <option value="JOB_STATE_SUCCEEDED">Succeeded</option>
          <option value="JOB_STATE_FAILED">Failed</option>
          <option value="JOB_STATE_KILLED">Killed</option>
          <option value="JOB_STATE_UNSCHEDULABLE">Unschedulable</option>
        </select>
        <button class="px-3 py-1.5 text-sm border rounded hover:bg-surface-raised">Filter</button>
      </form>
    </div>

    <div v-if="error" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">
      {{ error }}
    </div>
    <SourceWarnings :statuses="data?.page?.sourceStatuses" />
    <EmptyState v-if="!loading && jobs.length === 0" message="No jobs found" />
    <DataTable v-else :columns="columns" :rows="visibleJobs" :loading="loading && jobs.length === 0">
      <template #cell-job="{ row }">
        <div class="flex items-center gap-1" :style="{ paddingLeft: `${(row as JobTreeRow).depth * 1.25}rem` }">
          <button
            class="w-5 font-mono text-xs text-text-muted hover:text-accent disabled:opacity-40"
            :disabled="loadingChildren.has((row as JobTreeRow).identity.key.resourceId)"
            :aria-label="`Toggle children of ${(row as JobTreeRow).identity.key.resourceId}`"
            @click="toggleChildren(row as JobTreeRow)"
          >
            {{ expandedJobs.has((row as JobTreeRow).identity.key.resourceId) ? '▾' : '▸' }}
          </button>
          <RouterLink :to="jobRoute(row as JobTreeRow)" class="font-mono text-accent hover:underline">
            {{ (row as JobTreeRow).identity.key.resourceId }}
          </RouterLink>
        </div>
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
