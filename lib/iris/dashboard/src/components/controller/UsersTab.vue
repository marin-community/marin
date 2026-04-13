<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type { ListUsersResponse, UserSummary } from '@/types/rpc'
import EmptyState from '@/components/shared/EmptyState.vue'

const TERMINAL_JOB_STATES = new Set(['succeeded', 'failed', 'killed', 'worker_failed', 'preempted'])

const { data, loading, error, refresh } = useControllerRpc<ListUsersResponse>('ListUsers')

useAutoRefresh(refresh, 30_000)
onMounted(refresh)

const users = computed<UserSummary[]>(() => data.value?.users ?? [])

function countByStates(counts?: Record<string, number>, states?: string[]): number {
  if (!counts) return 0
  if (!states) return Object.values(counts).reduce((a, b) => a + b, 0)
  return states.reduce((acc, s) => acc + (counts[s] ?? 0), 0)
}

function activeJobCount(user: UserSummary): number {
  if (!user.jobStateCounts) return 0
  return Object.entries(user.jobStateCounts)
    .filter(([state]) => !TERMINAL_JOB_STATES.has(state))
    .reduce((acc, [, count]) => acc + count, 0)
}

function runningJobCount(user: UserSummary): number {
  return user.jobStateCounts?.['running'] ?? 0
}

function pendingJobCount(user: UserSummary): number {
  return (user.jobStateCounts?.['pending'] ?? 0) + (user.jobStateCounts?.['unschedulable'] ?? 0)
}

function totalTaskCount(user: UserSummary): number {
  return countByStates(user.taskStateCounts)
}

function runningTaskCount(user: UserSummary): number {
  return user.taskStateCounts?.['running'] ?? 0
}

function succeededTaskCount(user: UserSummary): number {
  return user.taskStateCounts?.['succeeded'] ?? 0
}
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
  <div v-if="loading && !data" class="flex items-center justify-center py-12 text-text-muted text-sm">
    <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
    Loading...
  </div>

  <!-- Empty state -->
  <EmptyState
    v-else-if="!loading && users.length === 0"
    message="No users"
  />

  <!-- Users table -->
  <div v-else class="overflow-x-auto">
    <table class="w-full border-collapse">
      <thead>
        <tr class="border-b border-surface-border">
          <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
            User
          </th>
          <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Active Jobs
          </th>
          <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Running Jobs
          </th>
          <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Pending Jobs
          </th>
          <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Total Tasks
          </th>
          <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Running Tasks
          </th>
          <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Succeeded Tasks
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="user in users"
          :key="user.user"
          class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
        >
          <td class="px-3 py-2 text-[13px] font-mono">{{ user.user || '(unknown)' }}</td>
          <td class="px-3 py-2 text-[13px] text-right tabular-nums">{{ activeJobCount(user) }}</td>
          <td class="px-3 py-2 text-[13px] text-right tabular-nums">
            <span :class="runningJobCount(user) > 0 ? 'text-accent font-semibold' : ''">
              {{ runningJobCount(user) }}
            </span>
          </td>
          <td class="px-3 py-2 text-[13px] text-right tabular-nums">
            <span :class="pendingJobCount(user) > 0 ? 'text-status-warning' : ''">
              {{ pendingJobCount(user) }}
            </span>
          </td>
          <td class="px-3 py-2 text-[13px] text-right tabular-nums">{{ totalTaskCount(user) }}</td>
          <td class="px-3 py-2 text-[13px] text-right tabular-nums">
            <span :class="runningTaskCount(user) > 0 ? 'text-accent font-semibold' : ''">
              {{ runningTaskCount(user) }}
            </span>
          </td>
          <td class="px-3 py-2 text-[13px] text-right tabular-nums text-status-success">
            {{ succeededTaskCount(user) }}
          </td>
        </tr>
      </tbody>
    </table>
    <div class="px-3 py-2 text-xs text-text-secondary border-t border-surface-border">
      {{ users.length }} user{{ users.length !== 1 ? 's' : '' }}
    </div>
  </div>
</template>
