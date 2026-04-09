<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { controllerRpcCall, useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { stateToName, stateDisplayName } from '@/types/status'
import type { JobState } from '@/types/status'
import type { JobStatus, JobQuery, ListJobsResponse } from '@/types/rpc'
import { timestampMs, formatDuration, formatRelativeTime } from '@/utils/formatting'
import { flattenLoadedJobTree, getLeafJobName } from '@/utils/jobTree'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import EmptyState from '@/components/shared/EmptyState.vue'

const PAGE_SIZE = 50

const SORT_FIELD_MAP: Record<string, string> = {
  date: 'JOB_SORT_FIELD_DATE',
  name: 'JOB_SORT_FIELD_NAME',
  state: 'JOB_SORT_FIELD_STATE',
  failures: 'JOB_SORT_FIELD_FAILURES',
  preemptions: 'JOB_SORT_FIELD_PREEMPTIONS',
}

type SortField = 'date' | 'name' | 'state' | 'failures' | 'preemptions'
type SortDir = 'asc' | 'desc'

const copiedJob = ref<string | null>(null)

async function copyJobName(name: string) {
  await navigator.clipboard.writeText(name)
  copiedJob.value = name
  setTimeout(() => { copiedJob.value = null }, 1500)
}

const EXPANDED_JOBS_KEY = 'iris.controller.expandedJobs'

// -- State --

const page = ref(0)
const sortField = ref<SortField>('date')
const sortDir = ref<SortDir>('desc')
const nameFilter = ref('')
const localFilter = ref('')
const stateFilter = ref('')
const expandedJobs = ref<Set<string>>(loadExpandedJobs())
const childJobsByParent = ref<Map<string, JobStatus[]>>(new Map())
const loadingChildJobs = ref<Set<string>>(new Set())

const JOB_STATES: JobState[] = [
  'pending', 'building', 'running', 'succeeded', 'failed', 'killed', 'worker_failed', 'unschedulable',
]

const {
  data: listResponse,
  loading,
  error,
  refresh: fetchJobs,
} = useControllerRpc<ListJobsResponse>('ListJobs', () => ({
  query: {
    scope: 'JOB_QUERY_SCOPE_ROOTS',
    offset: page.value * PAGE_SIZE,
    limit: PAGE_SIZE,
    sortField: SORT_FIELD_MAP[sortField.value],
    sortDirection: sortDir.value === 'asc' ? 'SORT_DIRECTION_ASC' : 'SORT_DIRECTION_DESC',
    nameFilter: nameFilter.value || undefined,
    stateFilter: stateFilter.value || undefined,
  } satisfies JobQuery,
}))

const jobs = computed(() => listResponse.value?.jobs ?? [])
const totalCount = computed(() => listResponse.value?.totalCount ?? 0)
const hasMore = computed(() => listResponse.value?.hasMore ?? false)

// -- Session storage for expanded state --

function loadExpandedJobs(): Set<string> {
  try {
    const stored = sessionStorage.getItem(EXPANDED_JOBS_KEY)
    return stored ? new Set(JSON.parse(stored) as string[]) : new Set()
  } catch {
    return new Set()
  }
}

function saveExpandedJobs() {
  try {
    sessionStorage.setItem(EXPANDED_JOBS_KEY, JSON.stringify([...expandedJobs.value]))
  } catch {
    // ignore
  }
}

async function refreshExpandedChildren() {
  const expandedIds = [...expandedJobs.value]
  for (const parentJobId of expandedIds) {
    const payload = await controllerRpcCall<ListJobsResponse>('ListJobs', {
      query: {
        scope: 'JOB_QUERY_SCOPE_CHILDREN',
        parentJobId,
        sortField: SORT_FIELD_MAP[sortField.value],
        sortDirection: sortDir.value === 'asc' ? 'SORT_DIRECTION_ASC' : 'SORT_DIRECTION_DESC',
        stateFilter: stateFilter.value || undefined,
      } satisfies JobQuery,
    })
    const nextChildren = new Map(childJobsByParent.value)
    nextChildren.set(parentJobId, payload.jobs ?? [])
    childJobsByParent.value = nextChildren
  }
}

async function fetchAll() {
  await fetchJobs()
  await refreshExpandedChildren()
}

onMounted(fetchAll)
useAutoRefresh(fetchAll, 30_000)

watch([page, sortField, sortDir, nameFilter, stateFilter], () => {
  childJobsByParent.value = new Map()
  expandedJobs.value = new Set()
  saveExpandedJobs()
  fetchJobs()
})

watch(stateFilter, () => {
  page.value = 0
})

// -- Job tree (lazy-loaded children) --

const flattenedJobs = computed(() => flattenLoadedJobTree(jobs.value, childJobsByParent.value, expandedJobs.value))

// -- Interactions --
async function loadChildJobs(parentJobId: string) {
  const nextLoading = new Set(loadingChildJobs.value)
  nextLoading.add(parentJobId)
  loadingChildJobs.value = nextLoading
  try {
    const payload = await controllerRpcCall<ListJobsResponse>('ListJobs', {
      query: {
        scope: 'JOB_QUERY_SCOPE_CHILDREN',
        parentJobId,
        sortField: SORT_FIELD_MAP[sortField.value],
        sortDirection: sortDir.value === 'asc' ? 'SORT_DIRECTION_ASC' : 'SORT_DIRECTION_DESC',
        stateFilter: stateFilter.value || undefined,
      } satisfies JobQuery,
    })
    const nextChildren = new Map(childJobsByParent.value)
    nextChildren.set(parentJobId, payload.jobs ?? [])
    childJobsByParent.value = nextChildren
  } finally {
    const doneLoading = new Set(loadingChildJobs.value)
    doneLoading.delete(parentJobId)
    loadingChildJobs.value = doneLoading
  }
}

function toggleExpanded(job: JobStatus) {
  const next = new Set(expandedJobs.value)
  if (next.has(job.jobId)) {
    next.delete(job.jobId)
  } else {
    next.add(job.jobId)
    if (!childJobsByParent.value.has(job.jobId)) {
      void loadChildJobs(job.jobId)
    }
  }
  expandedJobs.value = next
  saveExpandedJobs()
}

function handleSort(field: SortField) {
  if (sortField.value === field) {
    sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortField.value = field
    sortDir.value = field === 'date' ? 'desc' : 'asc'
  }
  page.value = 0
}

function handleFilterSubmit() {
  nameFilter.value = localFilter.value
  page.value = 0
}

function handleFilterClear() {
  localFilter.value = ''
  nameFilter.value = ''
  stateFilter.value = ''
  page.value = 0
}

const hasActiveFilter = computed(() => !!nameFilter.value || !!stateFilter.value)

// -- Formatting --

function jobDuration(job: JobStatus): string {
  const started = timestampMs(job.startedAt)
  if (started) {
    const ended = timestampMs(job.finishedAt) || Date.now()
    return formatDuration(started, ended)
  }
  const submitted = timestampMs(job.submittedAt)
  if (submitted) {
    return 'queued ' + formatRelativeTime(submitted)
  }
  return '-'
}

// -- Progress bar --

interface ProgressSegment {
  count: number
  colorClass: string
  label: string
}

const SEGMENT_COLORS: Record<string, string> = {
  succeeded: 'bg-status-success',
  running: 'bg-accent',
  building: 'bg-status-purple',
  assigned: 'bg-status-orange',
  failed: 'bg-status-danger',
  worker_failed: 'bg-status-danger',
  preempted: 'bg-status-warning',
  killed: 'bg-text-muted',
  pending: 'bg-surface-border',
}

function progressSegments(job: JobStatus): ProgressSegment[] {
  const counts = job.taskStateCounts ?? {}
  const total = job.taskCount ?? 0
  if (total === 0) return []

  const succeeded = counts['succeeded'] ?? 0
  const running = counts['running'] ?? 0
  const building = counts['building'] ?? 0
  const assigned = counts['assigned'] ?? 0
  const failed = counts['failed'] ?? 0
  const workerFailed = counts['worker_failed'] ?? 0
  const preempted = counts['preempted'] ?? 0
  const killed = counts['killed'] ?? 0
  const pending = total - succeeded - running - building - assigned - failed - workerFailed - preempted - killed

  return [
    { count: succeeded, colorClass: SEGMENT_COLORS['succeeded'], label: 'succeeded' },
    { count: running, colorClass: SEGMENT_COLORS['running'], label: 'running' },
    { count: building, colorClass: SEGMENT_COLORS['building'], label: 'building' },
    { count: assigned, colorClass: SEGMENT_COLORS['assigned'], label: 'assigned' },
    { count: failed, colorClass: SEGMENT_COLORS['failed'], label: 'failed' },
    { count: workerFailed, colorClass: SEGMENT_COLORS['worker_failed'], label: 'worker_failed' },
    { count: preempted, colorClass: SEGMENT_COLORS['preempted'], label: 'preempted' },
    { count: killed, colorClass: SEGMENT_COLORS['killed'], label: 'killed' },
    { count: Math.max(0, pending), colorClass: SEGMENT_COLORS['pending'], label: 'pending' },
  ].filter(s => s.count > 0)
}

function progressSummary(job: JobStatus): string {
  const counts = job.taskStateCounts ?? {}
  const running = counts['running'] ?? 0
  const total = job.taskCount ?? 0
  const succeeded = counts['succeeded'] ?? 0
  if (running > 0) return `${running} running`
  return `${succeeded}/${total}`
}

// -- Pagination --

const totalPages = computed(() => Math.max(1, Math.ceil(totalCount.value / PAGE_SIZE)))

// -- Sortable columns --

interface SortableCol {
  field: SortField
  label: string
}

const SORTABLE_COLS: SortableCol[] = [
  { field: 'name', label: 'Name' },
  { field: 'state', label: 'State' },
  { field: 'date', label: 'Date' },
  { field: 'failures', label: 'Failures' },
  { field: 'preemptions', label: 'Preemptions' },
]

function sortIndicator(field: SortField): string {
  if (sortField.value !== field) return '↕'
  return sortDir.value === 'asc' ? '↑' : '↓'
}
</script>

<template>
  <!-- Filter bar -->
  <div class="mb-4 flex items-center gap-3">
    <form class="flex gap-2" @submit.prevent="handleFilterSubmit">
      <select
        v-model="stateFilter"
        class="px-3 py-1.5 text-sm border border-surface-border rounded
               bg-surface text-text
               focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
      >
        <option value="">All states</option>
        <option v-for="s in JOB_STATES" :key="s" :value="s">{{ stateDisplayName(s) }}</option>
      </select>
      <input
        v-model="localFilter"
        type="text"
        placeholder="Filter by name..."
        class="w-52 px-3 py-1.5 text-sm border border-surface-border rounded
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
    <span class="text-[13px] text-text-secondary">
      {{ totalCount }} job{{ totalCount !== 1 ? 's' : '' }}
    </span>
  </div>

  <!-- Error -->
  <div
    v-if="error"
    class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
  >
    {{ error }}
  </div>

  <!-- Loading -->
  <div v-if="loading && jobs.length === 0" class="flex items-center justify-center py-12 text-text-muted text-sm">
    <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
    Loading...
  </div>

  <!-- Empty state -->
  <EmptyState
    v-else-if="!loading && jobs.length === 0"
    :message="hasActiveFilter ? 'No jobs matching filter' : 'No jobs'"
  />

  <!-- Jobs table -->
  <div v-else class="overflow-x-auto">
    <table class="w-full border-collapse">
      <thead>
        <tr class="border-b border-surface-border">
          <th
            v-for="col in SORTABLE_COLS"
            :key="col.field"
            class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary
                   cursor-pointer select-none hover:text-text"
            @click="handleSort(col.field)"
          >
            <span class="inline-flex items-center gap-1">
              {{ col.label }}
              <span :class="sortField === col.field ? 'text-accent' : 'text-text-muted/40'">
                {{ sortIndicator(col.field) }}
              </span>
            </span>
          </th>
          <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Tasks
          </th>
          <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Diagnostic
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="node in flattenedJobs"
          :key="node.job.jobId"
          class="group/row border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
        >
          <!-- Name -->
          <td
            class="px-3 py-2 text-[13px]"
            :style="{ paddingLeft: (node.depth * 20 + 12) + 'px' }"
          >
            <span class="inline-flex items-center gap-1">
              <button
                v-if="node.job.hasChildren"
                class="text-text-muted hover:text-text select-none w-4 text-center text-xs"
                @click.stop="toggleExpanded(node.job)"
              >
                {{ loadingChildJobs.has(node.job.jobId) ? '…' : (expandedJobs.has(node.job.jobId) ? '▼' : '▶') }}
              </button>
              <span v-else class="w-4" />
              <RouterLink
                :to="'/job/' + encodeURIComponent(node.job.jobId)"
                class="text-accent hover:underline font-mono"
              >
                {{ node.depth > 0 ? getLeafJobName(node.job.name) : (node.job.name || 'unnamed') }}
              </RouterLink>
              <button
                v-if="node.job.name"
                class="ml-1 text-text-muted hover:text-text opacity-0 group-hover/row:opacity-100 transition-opacity"
                title="Copy job name"
                @click.stop="copyJobName(node.job.name)"
              >
                <svg v-if="copiedJob === node.job.name" class="w-3.5 h-3.5 text-status-success" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
                <svg v-else class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                  <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
                </svg>
              </button>
            </span>
          </td>

          <!-- State -->
          <td class="px-3 py-2 text-[13px]">
            <StatusBadge :status="node.job.state" size="sm" />
          </td>

          <!-- Date -->
          <td class="px-3 py-2 text-[13px] text-text-secondary font-mono">
            {{ jobDuration(node.job) }}
          </td>

          <!-- Failures -->
          <td class="px-3 py-2 text-[13px] text-right tabular-nums">
            {{ node.job.failureCount ?? 0 }}
          </td>

          <!-- Preemptions -->
          <td class="px-3 py-2 text-[13px] text-right tabular-nums">
            {{ node.job.preemptionCount ?? 0 }}
          </td>

          <!-- Tasks progress bar -->
          <td class="px-3 py-2 text-[13px]">
            <div v-if="(node.job.taskCount ?? 0) === 0" class="text-xs text-text-muted">
              no tasks
            </div>
            <div v-else class="flex items-center gap-1.5">
              <div class="flex h-2 w-28 rounded-full overflow-hidden bg-surface-sunken">
                <div
                  v-for="(seg, i) in progressSegments(node.job)"
                  :key="i"
                  :class="seg.colorClass"
                  :style="{ width: (seg.count / (node.job.taskCount ?? 1) * 100).toFixed(1) + '%' }"
                  :title="seg.label + ': ' + seg.count"
                />
              </div>
              <span class="text-xs text-text-secondary whitespace-nowrap">
                {{ progressSummary(node.job) }}
              </span>
            </div>
          </td>

          <!-- Diagnostic -->
          <td
            class="px-3 py-2 text-xs text-text-muted max-w-xs truncate"
            :title="node.job.pendingReason ?? ''"
          >
            {{ node.job.pendingReason || '—' }}
          </td>
        </tr>
      </tbody>
    </table>

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
</template>
