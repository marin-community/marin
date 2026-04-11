<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { controllerRpcCall, useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type {
  GetSchedulerStateResponse,
  SchedulerUserBudget,
  ListUsersResponse,
  UserSummary,
  ListJobsResponse,
  JobStatus,
  JobQuery,
} from '@/types/rpc'
import { timestampMs, formatRelativeTime } from '@/utils/formatting'
import EmptyState from '@/components/shared/EmptyState.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'

// -- Scheduler State --

const { data: schedulerData, loading: schedulerLoading, error: schedulerError, refresh: refreshScheduler } =
  useControllerRpc<GetSchedulerStateResponse>('GetSchedulerState')

// -- Users --

const { data: usersData, loading: usersLoading, error: usersError, refresh: refreshUsers } =
  useControllerRpc<ListUsersResponse>('ListUsers')

// -- Unscheduled Jobs --

const UNSCHEDULED_PAGE_SIZE = 25
const unscheduledPage = ref(0)
const unscheduledSearch = ref('')
const unscheduledSearchInput = ref('')
const unscheduledJobs = ref<JobStatus[]>([])
const unscheduledTotal = ref(0)
const unscheduledHasMore = ref(false)
const unscheduledLoading = ref(false)
const unscheduledError = ref<string | null>(null)

async function fetchUnscheduledJobs() {
  unscheduledLoading.value = true
  unscheduledError.value = null
  try {
    const query: JobQuery = {
      scope: 'JOB_QUERY_SCOPE_ROOTS',
      stateFilter: 'pending',
      offset: unscheduledPage.value * UNSCHEDULED_PAGE_SIZE,
      limit: UNSCHEDULED_PAGE_SIZE,
      sortField: 'SORT_FIELD_SUBMITTED_AT',
      sortDirection: 'SORT_DIRECTION_DESC',
    }
    if (unscheduledSearch.value.trim()) {
      query.nameFilter = unscheduledSearch.value.trim()
    }
    const resp = await controllerRpcCall<ListJobsResponse>('ListJobs', { query })
    unscheduledJobs.value = resp.jobs ?? []
    unscheduledTotal.value = resp.totalCount ?? 0
    unscheduledHasMore.value = resp.hasMore ?? false
  } catch (e) {
    unscheduledError.value = e instanceof Error ? e.message : String(e)
  } finally {
    unscheduledLoading.value = false
  }
}

const unscheduledTotalPages = computed(() =>
  Math.max(1, Math.ceil(unscheduledTotal.value / UNSCHEDULED_PAGE_SIZE))
)

function applyUnscheduledSearch() {
  unscheduledSearch.value = unscheduledSearchInput.value
  unscheduledPage.value = 0
  fetchUnscheduledJobs()
}

watch(unscheduledPage, () => fetchUnscheduledJobs())

// -- Refresh all --

async function refreshAll() {
  await Promise.all([refreshScheduler(), refreshUsers(), fetchUnscheduledJobs()])
}

useAutoRefresh(refreshAll, 15_000)
onMounted(refreshAll)

// -- Scheduler computed --

const TERMINAL_JOB_STATES = new Set(['succeeded', 'failed', 'killed', 'worker_failed', 'preempted'])

const userBudgets = computed<SchedulerUserBudget[]>(() => schedulerData.value?.userBudgets ?? [])

const users = computed<UserSummary[]>(() => usersData.value?.users ?? [])

// Merge user stats with budget data
interface MergedUser {
  userId: string
  activeJobs: number
  runningJobs: number
  pendingJobs: number
  runningTasks: number
  totalTasks: number
  budgetSpent: string
  budgetLimit: string
  utilizationPercent: number
  maxBand: string
  effectiveBand: string
  hasBudget: boolean
}

const mergedUsers = computed<MergedUser[]>(() => {
  const budgetMap = new Map<string, SchedulerUserBudget>()
  for (const b of userBudgets.value) {
    budgetMap.set(b.userId, b)
  }

  const userMap = new Map<string, UserSummary>()
  for (const u of users.value) {
    userMap.set(u.user, u)
  }

  // All unique user IDs from both sources
  const allUserIds = new Set([...budgetMap.keys(), ...userMap.keys()])
  const result: MergedUser[] = []

  for (const userId of allUserIds) {
    const user = userMap.get(userId)
    const budget = budgetMap.get(userId)

    const jobCounts = user?.jobStateCounts ?? {}
    const taskCounts = user?.taskStateCounts ?? {}

    const activeJobs = Object.entries(jobCounts)
      .filter(([state]) => !TERMINAL_JOB_STATES.has(state))
      .reduce((acc, [, count]) => acc + count, 0)

    result.push({
      userId,
      activeJobs,
      runningJobs: jobCounts['running'] ?? 0,
      pendingJobs: (jobCounts['pending'] ?? 0) + (jobCounts['unschedulable'] ?? 0),
      runningTasks: taskCounts['running'] ?? 0,
      totalTasks: Object.values(taskCounts).reduce((a, b) => a + b, 0),
      budgetSpent: budget?.budgetSpent ?? '-',
      budgetLimit: budget?.budgetLimit ?? '-',
      utilizationPercent: budget?.utilizationPercent ?? 0,
      maxBand: budget?.maxBand ?? '',
      effectiveBand: budget?.effectiveBand ?? '',
      hasBudget: !!budget,
    })
  }

  // Sort: most active users first
  result.sort((a, b) => b.activeJobs - a.activeJobs || b.runningTasks - a.runningTasks || a.userId.localeCompare(b.userId))
  return result
})

// -- Helpers --

function bandDisplayName(band: string | undefined): string {
  if (!band) return 'Unknown'
  const name = band.replace(/^PRIORITY_BAND_/, '')
  return name.charAt(0) + name.slice(1).toLowerCase()
}

function bandColor(band: string | undefined): string {
  if (!band) return 'text-text-muted'
  const name = band.replace(/^PRIORITY_BAND_/, '')
  if (name === 'PRODUCTION') return 'text-status-danger'
  if (name === 'INTERACTIVE') return 'text-accent'
  if (name === 'BATCH') return 'text-text-muted'
  return 'text-text-muted'
}

function utilizationColor(pct: number): string {
  if (pct >= 100) return 'text-status-danger font-semibold'
  if (pct >= 75) return 'text-status-warning'
  return 'text-text-primary'
}

const loading = computed(() => (schedulerLoading.value || usersLoading.value) && !schedulerData.value && !usersData.value)
const error = computed(() => schedulerError.value || usersError.value)
const hasData = computed(() => schedulerData.value || usersData.value)
</script>

<template>
  <!-- Error -->
  <div
    v-if="error"
    class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
  >
    {{ error }}
  </div>

  <!-- Loading -->
  <div v-if="loading" class="flex items-center justify-center py-12 text-text-muted text-sm">
    <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
    Loading...
  </div>

  <div v-else-if="hasData" class="space-y-8">
    <!-- User Overview (merged Users tab + Scheduler budgets) -->
    <section>
      <h2 class="text-lg font-semibold mb-3">Users &amp; Quotas</h2>
      <EmptyState v-if="mergedUsers.length === 0" message="No users" />
      <div v-else class="overflow-x-auto">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border">
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">User</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Active Jobs</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Running</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Pending</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Running Tasks</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Total Tasks</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Spent</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Limit</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Utilization</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Band</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="user in mergedUsers"
              :key="user.userId"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td class="px-3 py-2 text-[13px] font-mono">{{ user.userId || '(unknown)' }}</td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">{{ user.activeJobs }}</td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">
                <span :class="user.runningJobs > 0 ? 'text-accent font-semibold' : ''">{{ user.runningJobs }}</span>
              </td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">
                <span :class="user.pendingJobs > 0 ? 'text-status-warning' : ''">{{ user.pendingJobs }}</span>
              </td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">
                <span :class="user.runningTasks > 0 ? 'text-accent font-semibold' : ''">{{ user.runningTasks }}</span>
              </td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">{{ user.totalTasks }}</td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">
                {{ user.hasBudget ? user.budgetSpent : '-' }}
              </td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">
                {{ !user.hasBudget ? '-' : user.budgetLimit === '0' ? 'Unlimited' : user.budgetLimit }}
              </td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums" :class="user.hasBudget ? utilizationColor(user.utilizationPercent) : ''">
                {{ !user.hasBudget ? '-' : user.budgetLimit === '0' ? '-' : user.utilizationPercent.toFixed(1) + '%' }}
              </td>
              <td class="px-3 py-2 text-[13px]">
                <template v-if="user.hasBudget">
                  <span :class="bandColor(user.effectiveBand)">{{ bandDisplayName(user.effectiveBand) }}</span>
                  <span
                    v-if="user.maxBand !== user.effectiveBand"
                    class="ml-1 text-xs text-status-warning"
                  >
                    (max: {{ bandDisplayName(user.maxBand) }})
                  </span>
                </template>
                <span v-else class="text-text-muted">-</span>
              </td>
            </tr>
          </tbody>
        </table>
        <div class="px-3 py-2 text-xs text-text-secondary border-t border-surface-border">
          {{ mergedUsers.length }} user{{ mergedUsers.length !== 1 ? 's' : '' }}
        </div>
      </div>
    </section>

    <!-- Pending Jobs -->
    <section>
      <h2 class="text-lg font-semibold mb-3">Pending Jobs ({{ unscheduledTotal }})</h2>
      <div class="flex items-center gap-3 mb-3">
        <form class="flex items-center gap-2" @submit.prevent="applyUnscheduledSearch">
          <input
            v-model="unscheduledSearchInput"
            type="text"
            placeholder="Search by job name..."
            class="w-64 px-3 py-1.5 bg-surface border border-surface-border rounded
                   text-sm font-mono placeholder:text-text-muted
                   focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
          />
          <button
            type="submit"
            class="px-3 py-1.5 text-sm border border-surface-border rounded hover:bg-surface-raised text-text-secondary"
          >
            Search
          </button>
          <button
            v-if="unscheduledSearch"
            type="button"
            class="px-3 py-1.5 text-sm border border-surface-border rounded hover:bg-surface-raised text-text-muted"
            @click="unscheduledSearchInput = ''; applyUnscheduledSearch()"
          >
            Clear
          </button>
        </form>
      </div>

      <div
        v-if="unscheduledError"
        class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
      >
        {{ unscheduledError }}
      </div>

      <div v-if="unscheduledLoading && unscheduledJobs.length === 0" class="flex items-center justify-center py-8 text-text-muted text-sm">
        <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        Loading...
      </div>

      <EmptyState v-else-if="unscheduledJobs.length === 0" message="No pending jobs" />

      <div v-else class="overflow-x-auto">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border">
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Job</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">User</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">State</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Pending Reason</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Submitted</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="job in unscheduledJobs"
              :key="job.jobId"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td class="px-3 py-2 text-[13px] font-mono">
                <RouterLink :to="`/job/${encodeURIComponent(job.jobId)}`" class="text-accent hover:underline">
                  {{ job.name || job.jobId }}
                </RouterLink>
              </td>
              <td class="px-3 py-2 text-[13px]">{{ job.jobId.split('/')[0] }}</td>
              <td class="px-3 py-2 text-[13px]">
                <StatusBadge :status="job.state" size="sm" />
              </td>
              <td class="px-3 py-2 text-[13px] text-status-warning max-w-md truncate" :title="job.pendingReason ?? ''">
                {{ job.pendingReason || '-' }}
              </td>
              <td class="px-3 py-2 text-[13px] font-mono text-text-secondary">
                {{ job.submittedAt ? formatRelativeTime(timestampMs(job.submittedAt)) : '-' }}
              </td>
            </tr>
          </tbody>
        </table>
        <!-- Pagination -->
        <div v-if="unscheduledTotalPages > 1" class="flex items-center justify-between px-3 py-2 text-xs text-text-secondary border-t border-surface-border">
          <span>
            {{ unscheduledPage * UNSCHEDULED_PAGE_SIZE + 1 }}&ndash;{{ Math.min((unscheduledPage + 1) * UNSCHEDULED_PAGE_SIZE, unscheduledTotal) }}
            of {{ unscheduledTotal }} jobs
          </span>
          <div class="flex items-center gap-1">
            <button
              :disabled="unscheduledPage === 0"
              class="px-2 py-1 rounded hover:bg-surface-raised disabled:opacity-30 disabled:cursor-not-allowed"
              @click="unscheduledPage--"
            >
              &larr; Prev
            </button>
            <span class="px-2 font-mono">{{ unscheduledPage + 1 }} / {{ unscheduledTotalPages }}</span>
            <button
              :disabled="unscheduledPage >= unscheduledTotalPages - 1"
              class="px-2 py-1 rounded hover:bg-surface-raised disabled:opacity-30 disabled:cursor-not-allowed"
              @click="unscheduledPage++"
            >
              Next &rarr;
            </button>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>
