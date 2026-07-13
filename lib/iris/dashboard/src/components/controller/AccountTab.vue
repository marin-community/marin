<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useControllerRpc } from '@/composables/useRpc'
import type { GetCurrentUserResponse, ListUsersResponse, UserSummary } from '@/types/rpc'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'

const { data: currentUser, refresh: refreshUser } = useControllerRpc<GetCurrentUserResponse>('GetCurrentUser')
const { data: usersData, refresh: refreshUsers } = useControllerRpc<ListUsersResponse>('ListUsers')

const userSummary = computed<UserSummary | null>(() => {
  if (!usersData.value?.users || !currentUser.value) return null
  return usersData.value.users.find(u => u.user === currentUser.value!.userId) ?? null
})

const TERMINAL_JOB_STATES = new Set(['succeeded', 'failed', 'killed', 'worker_failed', 'preempted'])

function activeJobCount(summary: UserSummary): number {
  if (!summary.jobStateCounts) return 0
  return Object.entries(summary.jobStateCounts)
    .filter(([state]) => !TERMINAL_JOB_STATES.has(state))
    .reduce((acc, [, count]) => acc + count, 0)
}

function countByStates(counts?: Record<string, number>): number {
  if (!counts) return 0
  return Object.values(counts).reduce((a, b) => a + b, 0)
}

onMounted(async () => {
  await refreshUser()
  await refreshUsers()
})
</script>

<template>
  <!-- Loading -->
  <div v-if="!currentUser" class="flex items-center justify-center py-12 text-text-muted text-sm">
    <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
    Loading...
  </div>

  <div v-else class="space-y-6">
    <!-- Identity & job summary -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <InfoCard title="Identity">
        <InfoRow label="User ID">
          <span class="font-mono">{{ currentUser.userId }}</span>
        </InfoRow>
        <InfoRow label="Role">
          <span
            class="inline-block px-2 py-0.5 text-xs font-medium rounded-full"
            :class="currentUser.role === 'admin'
              ? 'bg-accent/10 text-accent'
              : 'bg-surface-raised text-text-secondary'"
          >
            {{ currentUser.role }}
          </span>
        </InfoRow>
        <InfoRow v-if="currentUser.displayName" label="Display Name">
          {{ currentUser.displayName }}
        </InfoRow>
      </InfoCard>

      <InfoCard title="Jobs & Tasks">
        <template v-if="userSummary">
          <InfoRow label="Active Jobs">
            <span class="font-mono tabular-nums" :class="activeJobCount(userSummary) > 0 ? 'text-accent font-semibold' : ''">
              {{ activeJobCount(userSummary) }}
            </span>
          </InfoRow>
          <InfoRow label="Running Jobs">
            <span class="font-mono tabular-nums" :class="(userSummary.jobStateCounts?.['running'] ?? 0) > 0 ? 'text-accent font-semibold' : ''">
              {{ userSummary.jobStateCounts?.['running'] ?? 0 }}
            </span>
          </InfoRow>
          <InfoRow label="Total Tasks">
            <span class="font-mono tabular-nums">{{ countByStates(userSummary.taskStateCounts) }}</span>
          </InfoRow>
          <InfoRow label="Running Tasks">
            <span class="font-mono tabular-nums" :class="(userSummary.taskStateCounts?.['running'] ?? 0) > 0 ? 'text-accent font-semibold' : ''">
              {{ userSummary.taskStateCounts?.['running'] ?? 0 }}
            </span>
          </InfoRow>
          <InfoRow label="Succeeded Tasks">
            <span class="font-mono tabular-nums text-status-success">
              {{ userSummary.taskStateCounts?.['succeeded'] ?? 0 }}
            </span>
          </InfoRow>
        </template>
        <div v-else class="text-sm text-text-muted">No job data available</div>
      </InfoCard>
    </div>
  </div>
</template>
