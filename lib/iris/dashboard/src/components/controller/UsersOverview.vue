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
    activeJobs,
    runningJobs: jobCounts['running'] ?? 0,
    pendingJobs: (jobCounts['pending'] ?? 0) + (jobCounts['unschedulable'] ?? 0),
    runningTasks: taskCounts['running'] ?? 0,
    totalTasks: Object.values(taskCounts).reduce((a, b) => a + b, 0),
    starred: starredUsers.value.has(summary.user),
  }
}

const rows = computed<UserRow[]>(() => {
  const term = search.value.trim().toLowerCase()
  return (data.value?.users ?? [])
    .map(toRow)
    .filter(r => !term || r.user.toLowerCase().includes(term))
    .sort((a, b) =>
      Number(b.starred) - Number(a.starred) ||
      b.activeJobs - a.activeJobs ||
      b.runningTasks - a.runningTasks ||
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

    <!-- User cards: starred first, then by activity. -->
    <div v-else class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <div
        v-for="row in rows"
        :key="row.user"
        class="group/card relative rounded-lg border bg-surface transition-colors hover:bg-surface-raised hover:border-accent/40"
        :class="row.starred ? 'border-status-warning-border' : 'border-surface-border'"
      >
        <RouterLink
          :to="{ path: '/', query: { user: row.user } }"
          class="block px-4 py-3 focus:outline-none focus:ring-2 focus:ring-accent/30 rounded-lg"
          :aria-label="`View jobs for ${row.user || 'unknown user'}`"
        >
          <div class="flex items-center gap-2">
            <span class="font-mono text-sm font-semibold text-accent truncate min-w-0 flex-1">
              {{ row.user || '(unknown)' }}
            </span>
            <span class="text-text-muted text-xs shrink-0 opacity-0 group-hover/card:opacity-100 transition-opacity" aria-hidden="true">
              &rarr;
            </span>
          </div>
          <!-- Aggregate counts. Each metric is labelled in text, not encoded by
               colour alone, so the card reads without relying on hue. -->
          <dl class="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[13px]">
            <div class="flex items-center justify-between">
              <dt class="text-text-secondary">Running</dt>
              <dd class="tabular-nums" :class="row.runningJobs > 0 ? 'text-accent font-semibold' : 'text-text-muted'">
                {{ row.runningJobs }}
              </dd>
            </div>
            <div class="flex items-center justify-between">
              <dt class="text-text-secondary">Pending</dt>
              <dd class="tabular-nums" :class="row.pendingJobs > 0 ? 'text-status-warning font-semibold' : 'text-text-muted'">
                {{ row.pendingJobs }}
              </dd>
            </div>
            <div class="flex items-center justify-between">
              <dt class="text-text-secondary">Active jobs</dt>
              <dd class="tabular-nums text-text">{{ row.activeJobs }}</dd>
            </div>
            <div class="flex items-center justify-between">
              <dt class="text-text-secondary">Running tasks</dt>
              <dd class="tabular-nums" :class="row.runningTasks > 0 ? 'text-accent font-semibold' : 'text-text-muted'">
                {{ row.runningTasks }}
              </dd>
            </div>
          </dl>
        </RouterLink>
        <!-- Star toggle, overlaid top-right so it stays clear of the card link. -->
        <button
          type="button"
          :aria-label="row.starred ? `Unstar ${row.user}` : `Star ${row.user}`"
          :aria-pressed="row.starred"
          :title="row.starred ? 'Unstar user' : 'Star user to pin it to the top'"
          class="absolute top-2 right-2 p-1 rounded transition-colors"
          :class="row.starred
            ? 'text-status-warning'
            : 'text-text-muted opacity-0 group-hover/card:opacity-100 focus:opacity-100 hover:text-text'"
          @click="toggleStar(row.user)"
        >
          <svg v-if="row.starred" class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.966a1 1 0 00.95.69h4.17c.969 0 1.371 1.24.588 1.81l-3.37 2.45a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.37-2.45a1 1 0 00-1.176 0l-3.37 2.45c-.784.57-1.838-.196-1.539-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.06 9.393c-.783-.57-.38-1.81.588-1.81h4.17a1 1 0 00.95-.69l1.286-3.966z" />
          </svg>
          <svg v-else class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>
