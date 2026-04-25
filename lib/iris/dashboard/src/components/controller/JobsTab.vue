<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { controllerRpcCall, useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import { SEGMENT_COLORS, stateToName, stateDisplayName } from '@/types/status'
import type { JobState } from '@/types/status'
import type { JobStatus, JobQuery, ListJobsResponse, GetJobStatusResponse } from '@/types/rpc'
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

const SORT_FIELDS: SortField[] = ['date', 'name', 'state', 'failures', 'preemptions']
const SORT_DIRS: SortDir[] = ['asc', 'desc']

const copiedJob = ref<string | null>(null)

async function copyJobName(name: string) {
  await navigator.clipboard.writeText(name)
  copiedJob.value = name
  setTimeout(() => { copiedJob.value = null }, 1500)
}

const route = useRoute()
const router = useRouter()

const EXPANDED_JOBS_KEY = 'iris.controller.expandedJobs'
const STARRED_JOBS_KEY = 'iris.controller.starredJobs'
const MAX_STARRED_JOBS = 10

// -- State (hydrated from URL query params) --

/** Safely extract a single string from a Vue Router query value (string | string[] | null). */
function queryStr(v: string | string[] | null | undefined): string {
  if (Array.isArray(v)) return v[0] ?? ''
  return v ?? ''
}

function parseSort(v: string): SortField {
  return SORT_FIELDS.includes(v as SortField) ? (v as SortField) : 'date'
}
function parseDir(v: string): SortDir {
  return SORT_DIRS.includes(v as SortDir) ? (v as SortDir) : 'desc'
}
function parsePage(v: string): number {
  const n = Number(v)
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : 0
}

const page = ref(parsePage(queryStr(route.query.page)))
const sortField = ref<SortField>(parseSort(queryStr(route.query.sort)))
const sortDir = ref<SortDir>(parseDir(queryStr(route.query.dir)))
const nameFilter = ref(queryStr(route.query.name))
const localFilter = ref(queryStr(route.query.name))
const stateFilter = ref(queryStr(route.query.state))
const expandedJobs = ref<Set<string>>(loadExpandedJobs())
const childJobsByParent = ref<Map<string, JobStatus[]>>(new Map())
const loadingChildJobs = ref<Set<string>>(new Set())
const starredJobIds = ref<Set<string>>(loadStarredJobs())
const showStarredOnly = ref(queryStr(route.query.starred) === '1')
const starredJobsData = ref<JobStatus[]>([])
const starredLoading = ref(false)
const starredError = ref<string | null>(null)
const starLimitNotice = ref<string | null>(null)

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

// -- Local storage for starred jobs (persists across sessions) --

function loadStarredJobs(): Set<string> {
  try {
    const stored = localStorage.getItem(STARRED_JOBS_KEY)
    return stored ? new Set(JSON.parse(stored) as string[]) : new Set()
  } catch {
    return new Set()
  }
}

function saveStarredJobs() {
  try {
    localStorage.setItem(STARRED_JOBS_KEY, JSON.stringify([...starredJobIds.value]))
  } catch {
    // ignore
  }
}

function toggleStar(job: JobStatus) {
  const next = new Set(starredJobIds.value)
  if (next.has(job.jobId)) {
    next.delete(job.jobId)
  } else {
    if (next.size >= MAX_STARRED_JOBS) {
      starLimitNotice.value = `You can star at most ${MAX_STARRED_JOBS} jobs — unstar one first.`
      setTimeout(() => { starLimitNotice.value = null }, 4000)
      return
    }
    next.add(job.jobId)
  }
  starredJobIds.value = next
  saveStarredJobs()
  if (showStarredOnly.value) {
    void fetchStarredJobs()
  }
}

// Fetch each starred job individually — the ListJobs RPC does not support
// filtering by a set of job IDs, so this is the simplest correct way to
// show only starred jobs without losing any due to pagination.
async function fetchStarredJobs() {
  const ids = [...starredJobIds.value]
  if (ids.length === 0) {
    starredJobsData.value = []
    starredError.value = null
    return
  }
  starredLoading.value = true
  starredError.value = null
  try {
    const results = await Promise.allSettled(
      ids.map(id => controllerRpcCall<GetJobStatusResponse>('GetJobStatus', { jobId: id })),
    )
    starredJobsData.value = results
      .filter((r): r is PromiseFulfilledResult<GetJobStatusResponse> => r.status === 'fulfilled' && !!r.value?.job)
      .map(r => r.value.job)
    const failures = results.filter(r => r.status === 'rejected').length
    if (failures > 0 && starredJobsData.value.length === 0) {
      starredError.value = `Failed to load ${failures} starred job${failures !== 1 ? 's' : ''}`
    }
  } finally {
    starredLoading.value = false
  }
}

async function loadChildJobs(parentJobId: string) {
  if (loadingChildJobs.value.has(parentJobId)) return
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

async function refreshExpandedChildren() {
  await Promise.all([...expandedJobs.value].map(loadChildJobs))
}

async function fetchAll() {
  if (showStarredOnly.value) {
    await fetchStarredJobs()
    await refreshExpandedChildren()
    return
  }
  await fetchJobs()
  await refreshExpandedChildren()
}

onMounted(fetchAll)
useAutoRefresh(fetchAll, DEFAULT_REFRESH_MS)

watch([page, sortField, sortDir, nameFilter, stateFilter], () => {
  childJobsByParent.value = new Map()
  expandedJobs.value = new Set()
  saveExpandedJobs()
  if (!showStarredOnly.value) fetchJobs()
})

watch(showStarredOnly, (on) => {
  childJobsByParent.value = new Map()
  expandedJobs.value = new Set()
  saveExpandedJobs()
  if (on) void fetchStarredJobs()
  else void fetchJobs()
})

watch(stateFilter, () => {
  page.value = 0
})

// Sync filter/sort/page state into the URL so back-button and link sharing work.
watch([page, sortField, sortDir, nameFilter, stateFilter, showStarredOnly], () => {
  router.replace({
    query: {
      ...route.query,
      sort: sortField.value !== 'date' ? sortField.value : undefined,
      dir: sortDir.value !== 'desc' ? sortDir.value : undefined,
      page: page.value !== 0 ? String(page.value) : undefined,
      name: nameFilter.value || undefined,
      state: stateFilter.value || undefined,
      starred: showStarredOnly.value ? '1' : undefined,
    },
  })
})

// -- Starred-only client-side filter + sort --

function jobSortKey(job: JobStatus, field: SortField): number | string {
  switch (field) {
    case 'date': return timestampMs(job.submittedAt) || 0
    case 'name': return job.name ?? ''
    case 'state': return stateToName(job.state)
    case 'failures': return job.failureCount ?? 0
    case 'preemptions': return job.preemptionCount ?? 0
  }
}

function compareJobs(a: JobStatus, b: JobStatus): number {
  const av = jobSortKey(a, sortField.value)
  const bv = jobSortKey(b, sortField.value)
  const sign = sortDir.value === 'asc' ? 1 : -1
  if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * sign
  return String(av).localeCompare(String(bv)) * sign
}

const filteredStarredJobs = computed(() => {
  const ids = starredJobIds.value
  const nameF = nameFilter.value.toLowerCase()
  const stateF = stateFilter.value
  return starredJobsData.value
    .filter(j => ids.has(j.jobId))
    .filter(j => !nameF || (j.name ?? '').toLowerCase().includes(nameF))
    .filter(j => !stateF || stateToName(j.state) === stateF)
    .slice()
    .sort(compareJobs)
})

const effectiveJobs = computed(() => showStarredOnly.value ? filteredStarredJobs.value : jobs.value)
const effectiveLoading = computed(() => showStarredOnly.value ? starredLoading.value : loading.value)
const effectiveError = computed(() => showStarredOnly.value ? starredError.value : error.value)
const effectiveTotalCount = computed(() => showStarredOnly.value ? filteredStarredJobs.value.length : totalCount.value)

// -- Job tree (lazy-loaded children) --

const flattenedJobs = computed(() => flattenLoadedJobTree(effectiveJobs.value, childJobsByParent.value, expandedJobs.value))

// Whether a row should render the expand toggle. In starred-only mode we
// may have fetched the job via GetJobStatus against an older controller
// that doesn't populate `has_children`; show the toggle defensively for
// top-level rows and let `loadChildJobs` reveal whether it actually has
// children.
function showExpandToggle(job: JobStatus, depth: number): boolean {
  if (job.hasChildren) return true
  if (showStarredOnly.value && depth === 0) return true
  return false
}

// -- Interactions --
async function toggleExpanded(job: JobStatus) {
  const next = new Set(expandedJobs.value)
  if (next.has(job.jobId)) {
    next.delete(job.jobId)
    expandedJobs.value = next
    saveExpandedJobs()
    return
  }
  next.add(job.jobId)
  expandedJobs.value = next
  saveExpandedJobs()
  if (!childJobsByParent.value.has(job.jobId)) {
    await loadChildJobs(job.jobId)
    // Defensive: auto-collapse if the load returned no children, so the
    // expanded arrow doesn't dangle over an empty list (matters when the
    // server doesn't populate hasChildren on GetJobStatus responses).
    if ((childJobsByParent.value.get(job.jobId) ?? []).length === 0) {
      const reset = new Set(expandedJobs.value)
      reset.delete(job.jobId)
      expandedJobs.value = reset
      saveExpandedJobs()
    }
  }
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
  showStarredOnly.value = false
  page.value = 0
}

const hasActiveFilter = computed(() => !!nameFilter.value || !!stateFilter.value || showStarredOnly.value)

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

// SEGMENT_COLORS lives in @/types/status so the dashboard legend can stay in
// sync with a single canonical definition.

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
  hide?: string
}

const SORTABLE_COLS: SortableCol[] = [
  { field: 'name', label: 'Name' },
  { field: 'state', label: 'State' },
  { field: 'date', label: 'Date', hide: 'hidden sm:table-cell' },
  { field: 'failures', label: 'Failed Attempts', hide: 'hidden md:table-cell' },
  { field: 'preemptions', label: 'Preemptions', hide: 'hidden lg:table-cell' },
]

function sortIndicator(field: SortField): string {
  if (sortField.value !== field) return '↕'
  return sortDir.value === 'asc' ? '↑' : '↓'
}
</script>

<template>
  <!-- Filter bar -->
  <div class="mb-4 flex flex-wrap items-center gap-2 sm:gap-3">
    <form class="flex flex-wrap flex-1 sm:flex-initial gap-2" @submit.prevent="handleFilterSubmit">
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
        class="flex-1 sm:flex-initial sm:w-52 px-3 py-1.5 text-sm border border-surface-border rounded
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
    <button
      type="button"
      :class="[
        'inline-flex items-center gap-1.5 px-3 py-1.5 text-sm border rounded',
        showStarredOnly
          ? 'border-status-warning-border bg-status-warning-bg text-status-warning'
          : 'border-surface-border hover:bg-surface-raised',
      ]"
      :title="showStarredOnly ? 'Show all jobs' : 'Show only starred jobs'"
      @click="showStarredOnly = !showStarredOnly"
    >
      <svg v-if="showStarredOnly" class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.966a1 1 0 00.95.69h4.17c.969 0 1.371 1.24.588 1.81l-3.37 2.45a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.37-2.45a1 1 0 00-1.176 0l-3.37 2.45c-.784.57-1.838-.196-1.539-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.06 9.393c-.783-.57-.38-1.81.588-1.81h4.17a1 1 0 00.95-.69l1.286-3.966z" />
      </svg>
      <svg v-else class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
      </svg>
      Starred
      <span v-if="starredJobIds.size > 0" class="text-xs tabular-nums opacity-70">
        ({{ starredJobIds.size }})
      </span>
    </button>
    <span class="text-[13px] text-text-secondary">
      {{ effectiveTotalCount }} job{{ effectiveTotalCount !== 1 ? 's' : '' }}
    </span>
  </div>

  <!-- Error -->
  <div
    v-if="effectiveError"
    class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
  >
    {{ effectiveError }}
  </div>

  <!-- Star-limit notice -->
  <div
    v-if="starLimitNotice"
    class="mb-4 px-4 py-2 text-sm text-status-warning bg-status-warning-bg rounded-lg border border-status-warning-border"
  >
    {{ starLimitNotice }}
  </div>

  <!-- Loading -->
  <div v-if="effectiveLoading && effectiveJobs.length === 0" class="flex items-center justify-center py-12 text-text-muted text-sm">
    <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
    Loading...
  </div>

  <!-- Empty state -->
  <EmptyState
    v-else-if="!effectiveLoading && effectiveJobs.length === 0"
    :message="showStarredOnly && starredJobIds.size === 0
      ? 'No starred jobs — click the star next to a top-level job to pin it here'
      : (hasActiveFilter ? 'No jobs matching filter' : 'No jobs')"
  />

  <!-- Mobile/desktop split: cards on xs, table on sm+. Pagination is shared. -->
  <template v-else>
  <!-- Mobile: stacked card-grid (one card per job).
       Tables don't fit on phones once you have a status badge + progress bar +
       a job-name column, so on xs we render a vertical grid of cards instead. -->
  <div class="sm:hidden grid grid-cols-1 gap-2">
    <div
      v-for="node in flattenedJobs"
      :key="'card-' + node.job.jobId"
      class="rounded-lg border border-surface-border bg-surface px-3 py-2"
      :style="node.depth > 0 ? { marginLeft: (Math.min(node.depth, 3) * 12) + 'px' } : undefined"
    >
      <!-- Row 1: expand, name, star -->
      <div class="flex items-start gap-1.5">
        <button
          v-if="showExpandToggle(node.job, node.depth)"
          class="text-text-muted hover:text-text select-none w-4 text-center text-xs shrink-0 mt-0.5"
          @click.stop="toggleExpanded(node.job)"
        >
          {{ loadingChildJobs.has(node.job.jobId) ? '…' : (expandedJobs.has(node.job.jobId) ? '▼' : '▶') }}
        </button>
        <span v-else class="w-4 shrink-0" />
        <RouterLink
          :to="'/job/' + encodeURIComponent(node.job.jobId)"
          class="text-accent hover:underline font-mono text-[13px] flex-1 min-w-0 break-anywhere"
        >
          {{ node.depth > 0 ? getLeafJobName(node.job.name) : (node.job.name || 'unnamed') }}
        </RouterLink>
        <button
          v-if="node.depth === 0"
          :class="[
            'shrink-0 p-1 -m-1',
            starredJobIds.has(node.job.jobId)
              ? 'text-status-warning'
              : 'text-text-muted hover:text-text',
          ]"
          :title="starredJobIds.has(node.job.jobId) ? 'Unstar job' : 'Star job'"
          @click.stop="toggleStar(node.job)"
        >
          <svg v-if="starredJobIds.has(node.job.jobId)" class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.966a1 1 0 00.95.69h4.17c.969 0 1.371 1.24.588 1.81l-3.37 2.45a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.37-2.45a1 1 0 00-1.176 0l-3.37 2.45c-.784.57-1.838-.196-1.539-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.06 9.393c-.783-.57-.38-1.81.588-1.81h4.17a1 1 0 00.95-.69l1.286-3.966z" />
          </svg>
          <svg v-else class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
          </svg>
        </button>
      </div>
      <!-- Row 2: state + counters -->
      <div class="mt-1.5 pl-5 flex items-center gap-2 flex-wrap">
        <StatusBadge :status="node.job.state" size="sm" />
        <span class="text-xs text-text-muted font-mono">
          {{ jobDuration(node.job) }}
          <span v-if="(node.job.failureCount ?? 0) > 0" class="text-status-danger">
            · {{ node.job.failureCount }} failed
          </span>
          <span v-if="(node.job.preemptionCount ?? 0) > 0">
            · {{ node.job.preemptionCount }} preempted
          </span>
        </span>
      </div>
      <!-- Row 3: pending reason (if any) -->
      <div
        v-if="node.job.pendingReason"
        class="mt-1 pl-5 text-xs text-text-muted"
        :title="node.job.pendingReason"
      >
        {{ node.job.pendingReason }}
      </div>
      <!-- Row 4: progress bar (if there are tasks) -->
      <div v-if="(node.job.taskCount ?? 0) > 0" class="mt-2 pl-5 flex items-center gap-2">
        <div class="flex h-2 flex-1 rounded-full overflow-hidden bg-surface-sunken">
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
    </div>
  </div>

  <!-- Desktop: tabular layout (sm+) -->
  <div class="hidden sm:block overflow-x-auto">
    <table class="w-full border-collapse">
      <thead>
        <tr class="border-b border-surface-border">
          <th
            v-for="col in SORTABLE_COLS"
            :key="col.field"
            :class="[
              'px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary',
              'cursor-pointer select-none hover:text-text',
              col.hide,
            ]"
            @click="handleSort(col.field)"
          >
            <span class="inline-flex items-center gap-1">
              {{ col.label }}
              <span :class="sortField === col.field ? 'text-accent' : 'text-text-muted/40'">
                {{ sortIndicator(col.field) }}
              </span>
            </span>
          </th>
          <th class="px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Tasks
          </th>
          <th class="hidden lg:table-cell px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
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
            class="px-2 sm:px-3 py-2 text-[13px]"
            :style="{ paddingLeft: (node.depth * 20 + 12) + 'px' }"
          >
            <span class="inline-flex items-center gap-1 max-w-full">
              <button
                v-if="showExpandToggle(node.job, node.depth)"
                class="text-text-muted hover:text-text select-none w-4 text-center text-xs shrink-0"
                @click.stop="toggleExpanded(node.job)"
              >
                {{ loadingChildJobs.has(node.job.jobId) ? '…' : (expandedJobs.has(node.job.jobId) ? '▼' : '▶') }}
              </button>
              <span v-else class="w-4 shrink-0" />
              <RouterLink
                :to="'/job/' + encodeURIComponent(node.job.jobId)"
                class="text-accent hover:underline font-mono break-anywhere"
              >
                {{ node.depth > 0 ? getLeafJobName(node.job.name) : (node.job.name || 'unnamed') }}
              </RouterLink>
              <button
                v-if="node.job.name"
                class="ml-1 text-text-muted hover:text-text opacity-0 group-hover/row:opacity-100 transition-opacity shrink-0"
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
              <button
                v-if="node.depth === 0"
                :class="[
                  'ml-1 transition-opacity shrink-0',
                  starredJobIds.has(node.job.jobId)
                    ? 'text-status-warning opacity-100'
                    : 'text-text-muted hover:text-text opacity-0 group-hover/row:opacity-100',
                ]"
                :title="starredJobIds.has(node.job.jobId) ? 'Unstar job' : 'Star job'"
                @click.stop="toggleStar(node.job)"
              >
                <svg v-if="starredJobIds.has(node.job.jobId)" class="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.966a1 1 0 00.95.69h4.17c.969 0 1.371 1.24.588 1.81l-3.37 2.45a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.37-2.45a1 1 0 00-1.176 0l-3.37 2.45c-.784.57-1.838-.196-1.539-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.06 9.393c-.783-.57-.38-1.81.588-1.81h4.17a1 1 0 00.95-.69l1.286-3.966z" />
                </svg>
                <svg v-else class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                </svg>
              </button>
            </span>
          </td>

          <!-- State -->
          <td class="px-2 sm:px-3 py-2 text-[13px]">
            <StatusBadge :status="node.job.state" size="sm" />
          </td>

          <!-- Date -->
          <td class="hidden sm:table-cell px-2 sm:px-3 py-2 text-[13px] text-text-secondary font-mono">
            {{ jobDuration(node.job) }}
          </td>

          <!-- Failures -->
          <td class="hidden md:table-cell px-2 sm:px-3 py-2 text-[13px] text-right tabular-nums">
            <span
              v-if="(node.job.failureCount ?? 0) > 0"
              class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium
                     text-status-danger bg-status-danger-bg border border-status-danger-border"
              :title="node.job.failureCount + ' failed task attempt' + ((node.job.failureCount ?? 0) !== 1 ? 's' : '') + ' (including retries)'"
            >
              {{ node.job.failureCount }}
            </span>
            <span v-else class="text-text-muted">0</span>
          </td>

          <!-- Preemptions -->
          <td class="hidden lg:table-cell px-2 sm:px-3 py-2 text-[13px] text-right tabular-nums">
            {{ node.job.preemptionCount ?? 0 }}
          </td>

          <!-- Tasks progress bar -->
          <td class="px-2 sm:px-3 py-2 text-[13px]">
            <div v-if="(node.job.taskCount ?? 0) === 0" class="text-xs text-text-muted">
              no tasks
            </div>
            <div v-else class="flex items-center gap-1.5">
              <div class="flex h-2 w-16 sm:w-28 rounded-full overflow-hidden bg-surface-sunken">
                <div
                  v-for="(seg, i) in progressSegments(node.job)"
                  :key="i"
                  :class="seg.colorClass"
                  :style="{ width: (seg.count / (node.job.taskCount ?? 1) * 100).toFixed(1) + '%' }"
                  :title="seg.label + ': ' + seg.count"
                />
              </div>
              <span class="hidden sm:inline text-xs text-text-secondary whitespace-nowrap">
                {{ progressSummary(node.job) }}
              </span>
            </div>
          </td>

          <!-- Diagnostic -->
          <td
            class="hidden lg:table-cell px-2 sm:px-3 py-2 text-xs text-text-muted max-w-xs truncate"
            :title="node.job.pendingReason ?? ''"
          >
            {{ node.job.pendingReason || '—' }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>

  <!-- Pagination (shared between mobile cards and desktop table) -->
  <div
    v-if="!showStarredOnly && totalPages > 1"
    class="mt-2 flex items-center justify-between px-2 sm:px-3 py-2 text-xs text-text-secondary border-t border-surface-border"
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
  </template>
</template>
