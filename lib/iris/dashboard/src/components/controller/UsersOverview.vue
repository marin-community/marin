<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type { ListUsersResponse, UserSummary } from '@/types/rpc'
import EmptyState from '@/components/shared/EmptyState.vue'
import LoadingSpinner from '@/components/shared/LoadingSpinner.vue'

// The landing page groups jobs by their top-level owner (the user id, which is
// the first path component of every job name). Each user drills into the jobs
// table scoped to that owner (`/?user=<id>`). Stars pin a user to the top so
// people can keep their own username one glance away.

const STARRED_USERS_KEY = 'iris.controller.starredUsers'
const MAX_STARRED_USERS = 10

const TERMINAL_JOB_STATES = new Set(['succeeded', 'failed', 'killed', 'worker_failed', 'preempted'])

const { data, loading, error, refresh } = useControllerRpc<ListUsersResponse>('ListUsers')

onMounted(refresh)
useAutoRefresh(refresh, DEFAULT_REFRESH_MS)

const search = ref('')
const starLimitNotice = ref<string | null>(null)
const starredUsers = ref<Set<string>>(loadStarredUsers())

function loadStarredUsers(): Set<string> {
  try {
    const stored = localStorage.getItem(STARRED_USERS_KEY)
    return stored ? new Set(JSON.parse(stored) as string[]) : new Set()
  } catch {
    return new Set()
  }
}

function saveStarredUsers() {
  try {
    localStorage.setItem(STARRED_USERS_KEY, JSON.stringify([...starredUsers.value]))
  } catch {
    // ignore
  }
}

function toggleStar(user: string) {
  const next = new Set(starredUsers.value)
  if (next.has(user)) {
    next.delete(user)
  } else {
    if (next.size >= MAX_STARRED_USERS) {
      starLimitNotice.value = `You can star at most ${MAX_STARRED_USERS} users — unstar one first.`
      setTimeout(() => { starLimitNotice.value = null }, 4000)
      return
    }
    next.add(user)
  }
  starredUsers.value = next
  saveStarredUsers()
}

interface UserRow {
  user: string
  role: string
  activeJobs: number
  runningJobs: number
  pendingJobs: number
  runningTasks: number
  totalTasks: number
  starred: boolean
}

function toRow(summary: UserSummary): UserRow {
  const jobCounts = summary.jobStateCounts ?? {}
  const taskCounts = summary.taskStateCounts ?? {}
  const activeJobs = Object.entries(jobCounts)
    .filter(([state]) => !TERMINAL_JOB_STATES.has(state))
    .reduce((acc, [, count]) => acc + count, 0)
  return {
    user: summary.user,
    role: summary.role ?? '',
    activeJobs,
    runningJobs: jobCounts['running'] ?? 0,
    pendingJobs: (jobCounts['pending'] ?? 0) + (jobCounts['unschedulable'] ?? 0),
    runningTasks: taskCounts['running'] ?? 0,
    totalTasks: Object.values(taskCounts).reduce((a, b) => a + b, 0),
    starred: starredUsers.value.has(summary.user),
  }
}

// Alphabetical by user id, with starred users pinned to the top.
const rows = computed<UserRow[]>(() => {
  const term = search.value.trim().toLowerCase()
  return (data.value?.users ?? [])
    .map(toRow)
    .filter(r => !term || r.user.toLowerCase().includes(term))
    .sort((a, b) =>
      Number(b.starred) - Number(a.starred) ||
      a.user.localeCompare(b.user),
    )
})
</script>

<template>
  <div>
    <!-- Toolbar: search + escape hatch to the flat all-jobs view -->
    <div class="mb-4 flex flex-wrap items-center gap-2 sm:gap-3">
      <input
        v-model="search"
        type="text"
        placeholder="Filter users…"
        aria-label="Filter users by id"
        class="flex-1 sm:flex-initial sm:w-64 px-3 py-1.5 text-sm border border-surface-border rounded
               bg-surface placeholder:text-text-muted
               focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
      />
      <span class="text-[13px] text-text-secondary">
        {{ rows.length }} user{{ rows.length !== 1 ? 's' : '' }}
      </span>
      <RouterLink
        :to="{ path: '/', query: { all: '1' } }"
        class="ml-auto px-3 py-1.5 text-sm border border-surface-border rounded hover:bg-surface-raised text-text-secondary"
      >
        View all jobs &rarr;
      </RouterLink>
    </div>

    <!-- Error -->
    <div
      v-if="error"
      class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <!-- Star-limit notice -->
    <div
      v-if="starLimitNotice"
      class="mb-4 px-4 py-2 text-sm text-status-warning bg-status-warning-bg rounded-lg border border-status-warning-border"
    >
      {{ starLimitNotice }}
    </div>

    <LoadingSpinner v-if="loading && rows.length === 0" />

    <EmptyState
      v-else-if="rows.length === 0"
      :message="search.trim() ? 'No users matching filter' : 'No users'"
    />

    <!-- Owner list: starred first, then alphabetical. -->
    <div v-else class="overflow-x-auto">
      <table class="w-full border-collapse">
        <thead>
          <tr class="border-b border-surface-border">
            <th scope="col" class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">User</th>
            <th scope="col" class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Role</th>
            <th scope="col" class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Running</th>
            <th scope="col" class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Pending</th>
            <th scope="col" class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Active Jobs</th>
            <th scope="col" class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Running Tasks</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in rows"
            :key="row.user"
            class="group/row border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
          >
            <td class="px-3 py-2 text-[13px]">
              <span class="inline-flex items-center gap-1.5 max-w-full">
                <button
                  type="button"
                  :aria-label="row.starred ? `Unstar ${row.user}` : `Star ${row.user}`"
                  :aria-pressed="row.starred"
                  :title="row.starred ? 'Unstar user' : 'Star user to pin it to the top'"
                  class="shrink-0 transition-opacity"
                  :class="row.starred
                    ? 'text-status-warning opacity-100'
                    : 'text-text-muted hover:text-text opacity-0 group-hover/row:opacity-100 focus:opacity-100'"
                  @click="toggleStar(row.user)"
                >
                  <svg v-if="row.starred" class="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.966a1 1 0 00.95.69h4.17c.969 0 1.371 1.24.588 1.81l-3.37 2.45a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.37-2.45a1 1 0 00-1.176 0l-3.37 2.45c-.784.57-1.838-.196-1.539-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.06 9.393c-.783-.57-.38-1.81.588-1.81h4.17a1 1 0 00.95-.69l1.286-3.966z" />
                  </svg>
                  <svg v-else class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                  </svg>
                </button>
                <RouterLink
                  v-if="row.user"
                  :to="{ path: '/', query: { user: row.user } }"
                  class="text-accent hover:underline font-mono break-anywhere"
                >{{ row.user }}</RouterLink>
                <span v-else class="text-text-muted font-mono">(unknown)</span>
              </span>
            </td>
            <td class="px-3 py-2 text-[13px]">
              <span
                v-if="row.role"
                class="inline-block px-1.5 py-0.5 rounded text-xs font-medium"
                :class="row.role === 'admin'
                  ? 'bg-status-warning-bg text-status-warning'
                  : 'bg-surface-raised text-text-secondary'"
              >{{ row.role }}</span>
              <span v-else class="text-text-muted">—</span>
            </td>
            <td class="px-3 py-2 text-[13px] text-right tabular-nums">
              <span :class="row.runningJobs > 0 ? 'text-accent font-semibold' : 'text-text-muted'">{{ row.runningJobs }}</span>
            </td>
            <td class="px-3 py-2 text-[13px] text-right tabular-nums">
              <span :class="row.pendingJobs > 0 ? 'text-status-warning font-semibold' : 'text-text-muted'">{{ row.pendingJobs }}</span>
            </td>
            <td class="px-3 py-2 text-[13px] text-right tabular-nums text-text">{{ row.activeJobs }}</td>
            <td class="px-3 py-2 text-[13px] text-right tabular-nums">
              <span :class="row.runningTasks > 0 ? 'text-accent font-semibold' : 'text-text-muted'">{{ row.runningTasks }}</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
