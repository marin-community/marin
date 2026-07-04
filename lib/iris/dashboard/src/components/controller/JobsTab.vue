<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import type { LocationQueryValue } from 'vue-router'
import { controllerRpcCall, useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import { SEGMENT_COLORS, stateDisplayName } from '@/types/status'
import type { JobState } from '@/types/status'
import { LOCAL_CLUSTER, type JobStatus, type JobQuery, type ListJobsResponse } from '@/types/rpc'
import { timestampMs, formatDuration, formatRelativeTime } from '@/utils/formatting'
import { flattenLoadedJobTree, getLeafJobName } from '@/utils/jobTree'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import LoadingSpinner from '@/components/shared/LoadingSpinner.vue'
import UsersOverview from '@/components/controller/UsersOverview.vue'
import { useMediaQuery } from '@/composables/useMediaQuery'
import { useBackends } from '@/composables/useBackends'

// Tailwind's `sm` breakpoint is 640px. Below that we render mobile cards;
// at/above we render the desktop table. Switched via v-if so only one
// variant is in the DOM at a time (otherwise duplicate text trips Playwright
// locator's `.first` matcher in CI).
const isMobile = useMediaQuery('(max-width: 639px)')

const { multiBackend, currentBackend, currentCluster, ensurePeers } = useBackends()

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

// -- Front-page mode --
//
// The landing page (`/`) groups jobs by their top-level owner via
// UsersOverview. Drilling into a user (`/?user=<id>`) renders this jobs table
// scoped to that owner; `/?all=1` shows the flat, cross-user root list for ops.
// `selectedUser` / `showAll` are derived from the URL so back/forward and
// shared links land on the same view.

/** Safely extract a single string from a Vue Router query value. */
function queryStr(v: LocationQueryValue | LocationQueryValue[] | undefined): string {
  if (Array.isArray(v)) return v[0] ?? ''
  return v ?? ''
}

const selectedUser = computed(() => queryStr(route.query.user))
const showAll = computed(() => queryStr(route.query.all) === '1')
const backendId = computed(() => currentBackend(route))
const clusterId = computed(() => currentCluster(route))
// Scoping to one backend or one peer cluster drills straight into the
// (server-side filtered) job list: the cross-fleet UsersOverview is an
// all-targets aggregate with no such filter, so it stays the landing view.
const inJobList = computed(
  () => !!selectedUser.value || showAll.value || !!backendId.value || !!clusterId.value,
)

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

const JOB_STATES: JobState[] = [
  'pending', 'building', 'running', 'succeeded', 'failed', 'killed', 'worker_failed', 'unschedulable',
]

// Anchored prefix that scopes the root list to one owner. Job ids are wire
// names of the form `/<user>/<job>`, so the prefix is `/<user>/`.
const jobIdPrefix = computed(() => (selectedUser.value ? `/${selectedUser.value}/` : undefined))

// Show the Backend column only in "All backends" mode (no scope selected) and
// only when the controller has more than one backend.
const showBackendColumn = computed(() => multiBackend.value && !backendId.value)

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
    jobIdPrefix: jobIdPrefix.value,
    backendId: backendId.value || undefined,
    cluster: clusterId.value || undefined,
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
  if (!inJobList.value) return  // UsersOverview owns its own data
  await fetchJobs()
  await refreshExpandedChildren()
}

onMounted(fetchAll)
// Load the peer roster so `?cluster=` validation and the Cluster column's
// filter target resolve; inert (no roster) on a single-cluster deployment.
onMounted(ensurePeers)
useAutoRefresh(fetchAll, DEFAULT_REFRESH_MS)

// Re-fetch from scratch whenever the scope or any query knob changes.
watch([page, sortField, sortDir, nameFilter, stateFilter, selectedUser, showAll, backendId, clusterId], () => {
  childJobsByParent.value = new Map()
  expandedJobs.value = new Set()
  saveExpandedJobs()
  if (inJobList.value) fetchJobs()
})

// A different owner (or the all-jobs view) is a different result set — start at
// the first page so a stale offset can't land out of range.
watch([stateFilter, selectedUser, showAll, backendId, clusterId], () => {
  page.value = 0
})

// Sync filter/sort/page state into the URL so back-button and link sharing work.
// `user`/`all` are spread through from the current query untouched.
watch([page, sortField, sortDir, nameFilter, stateFilter], () => {
  router.replace({
    query: {
      ...route.query,
      sort: sortField.value !== 'date' ? sortField.value : undefined,
      dir: sortDir.value !== 'desc' ? sortDir.value : undefined,
      page: page.value !== 0 ? String(page.value) : undefined,
      name: nameFilter.value || undefined,
      state: stateFilter.value || undefined,
    },
  })
})

// -- Job tree (lazy-loaded children) --

const flattenedJobs = computed(() => flattenLoadedJobTree(jobs.value, childJobsByParent.value, expandedJobs.value))

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
  const coschedFailed = counts['cosched_failed'] ?? 0
  const preempted = counts['preempted'] ?? 0
  const killed = counts['killed'] ?? 0
  const pending =
    total - succeeded - running - building - assigned - failed - workerFailed - coschedFailed - preempted - killed

  return [
    { count: succeeded, colorClass: SEGMENT_COLORS['succeeded'], label: 'succeeded' },
    { count: running, colorClass: SEGMENT_COLORS['running'], label: 'running' },
    { count: building, colorClass: SEGMENT_COLORS['building'], label: 'building' },
    { count: assigned, colorClass: SEGMENT_COLORS['assigned'], label: 'assigned' },
    { count: failed, colorClass: SEGMENT_COLORS['failed'], label: 'failed' },
    { count: workerFailed, colorClass: SEGMENT_COLORS['worker_failed'], label: 'worker_failed' },
    { count: coschedFailed, colorClass: SEGMENT_COLORS['cosched_failed'], label: 'cosched_failed' },
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
  <!-- Front page: group jobs by owner. The jobs table only renders once the
       user has drilled into an owner or chosen the flat all-jobs view. -->
  <UsersOverview v-if="!inJobList" />

  <template v-else>
  <!-- Breadcrumb back to the user overview -->
  <nav class="mb-4 flex items-center gap-1.5 text-sm" aria-label="Breadcrumb">
    <RouterLink :to="{ path: '/' }" class="text-accent hover:underline">Users</RouterLink>
    <span class="text-text-muted" aria-hidden="true">/</span>
    <span class="font-mono font-semibold text-text">
      {{ selectedUser || 'All jobs' }}
    </span>
  </nav>

  <!-- Filter bar -->
  <div class="mb-4 flex flex-wrap items-center gap-2 sm:gap-3">
    <form class="flex flex-wrap flex-1 sm:flex-initial gap-2" @submit.prevent="handleFilterSubmit">
      <select
        v-model="stateFilter"
        aria-label="Filter by state"
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
        aria-label="Filter by job name"
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
  <LoadingSpinner v-if="loading && jobs.length === 0" />

  <!-- Empty state -->
  <EmptyState
    v-else-if="!loading && jobs.length === 0"
    :message="hasActiveFilter ? 'No jobs matching filter' : 'No jobs'"
  />

  <!-- Mobile/desktop split: cards on xs, table on sm+. Pagination is shared.
       Switched via v-if (not CSS) so only one variant renders, keeping the DOM
       free of duplicate text-content that confuses Playwright `.first` matchers. -->
  <template v-else>
  <!-- Mobile: stacked card-grid (one card per job). -->
  <div v-if="isMobile" class="grid grid-cols-1 gap-2">
    <div
      v-for="node in flattenedJobs"
      :key="'card-' + node.job.jobId"
      class="rounded-lg border border-surface-border bg-surface px-3 py-2"
      :style="node.depth > 0 ? { marginLeft: (Math.min(node.depth, 3) * 12) + 'px' } : undefined"
    >
      <!-- Row 1: expand, name -->
      <div class="flex items-start gap-1.5">
        <button
          v-if="node.job.hasChildren"
          class="text-text-muted hover:text-text select-none w-4 text-center text-xs shrink-0 mt-0.5"
          :aria-label="expandedJobs.has(node.job.jobId) ? 'Collapse children' : 'Expand children'"
          :aria-expanded="expandedJobs.has(node.job.jobId)"
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
      </div>
      <!-- Row 2: state + counters -->
      <div class="mt-1.5 pl-5 flex items-center gap-2 flex-wrap">
        <StatusBadge :status="node.job.state" size="sm" />
        <span
          class="inline-flex items-center rounded bg-surface-sunken px-1.5 py-0.5 font-mono text-[11px] text-text-secondary"
          :title="'Cluster: ' + (node.job.cluster ?? LOCAL_CLUSTER)"
        >
          {{ node.job.cluster ?? LOCAL_CLUSTER }}
        </span>
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
  <div v-else class="overflow-x-auto">
    <table class="w-full border-collapse">
      <thead>
        <tr class="border-b border-surface-border">
          <th
            v-for="col in SORTABLE_COLS"
            :key="col.field"
            scope="col"
            :aria-sort="sortField === col.field ? (sortDir === 'asc' ? 'ascending' : 'descending') : 'none'"
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
          <th
            v-if="showBackendColumn"
            scope="col"
            class="hidden md:table-cell px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary"
          >
            Backend
          </th>
          <!-- Cluster: every job carries a coordinate (`'local'` by default), so
               each row is tagged; a single-cluster deployment reads all `local`. -->
          <th
            scope="col"
            class="hidden md:table-cell px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary"
          >
            Cluster
          </th>
          <th scope="col" class="px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Tasks
          </th>
          <th scope="col" class="hidden lg:table-cell px-2 sm:px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
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
                v-if="node.job.hasChildren"
                class="text-text-muted hover:text-text select-none w-4 text-center text-xs shrink-0"
                :aria-label="expandedJobs.has(node.job.jobId) ? 'Collapse children' : 'Expand children'"
                :aria-expanded="expandedJobs.has(node.job.jobId)"
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
                :aria-label="'Copy job name ' + node.job.name"
                title="Copy job name"
                @click.stop="copyJobName(node.job.name)"
              >
                <svg v-if="copiedJob === node.job.name" class="w-3.5 h-3.5 text-status-success" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
                <svg v-else class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                  <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
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

          <!-- Backend (multi-backend All mode only) -->
          <td
            v-if="showBackendColumn"
            class="hidden md:table-cell px-2 sm:px-3 py-2 text-[13px]"
          >
            <button
              v-if="node.job.backendId"
              class="text-accent hover:underline font-mono text-xs"
              @click="router.replace({ query: { ...route.query, backend: node.job.backendId, cluster: undefined } })"
            >
              {{ node.job.backendId }}
            </button>
            <span v-else class="text-text-muted">—</span>
          </td>

          <!-- Cluster: the row's coordinate (`'local'` or a peer id); clicking
               filters the list to that cluster. -->
          <td class="hidden md:table-cell px-2 sm:px-3 py-2 text-[13px]">
            <button
              class="text-accent hover:underline font-mono text-xs"
              @click="router.replace({ query: { ...route.query, cluster: node.job.cluster ?? LOCAL_CLUSTER, backend: undefined } })"
            >
              {{ node.job.cluster ?? LOCAL_CLUSTER }}
            </button>
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
    v-if="totalPages > 1"
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
</template>
