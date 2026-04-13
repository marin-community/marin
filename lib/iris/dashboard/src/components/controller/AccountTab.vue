<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useControllerRpc, controllerRpcCall } from '@/composables/useRpc'
import type { GetCurrentUserResponse, ListApiKeysResponse, ApiKeyInfo, ListUsersResponse, UserSummary } from '@/types/rpc'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'

const authEnabled = ref(false)

const { data: currentUser, refresh: refreshUser } = useControllerRpc<GetCurrentUserResponse>('GetCurrentUser')
const { data: usersData, refresh: refreshUsers } = useControllerRpc<ListUsersResponse>('ListUsers')
const { data: keysData, refresh: refreshKeys } = useControllerRpc<ListApiKeysResponse>('ListApiKeys', () => ({
  userId: currentUser.value?.userId ?? '',
}))

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

async function revokeKey(keyId: string) {
  await controllerRpcCall('RevokeApiKey', { keyId })
  await refreshKeys()
}

function formatTimestamp(ms: string): string {
  const n = parseInt(ms, 10)
  if (!n) return '\u2014'
  return new Date(n).toLocaleString()
}

onMounted(async () => {
  try {
    const resp = await fetch('/auth/config')
    if (resp.ok) {
      const config = await resp.json()
      authEnabled.value = config.auth_enabled ?? false
    }
  } catch { /* auth config endpoint unavailable */ }

  await refreshUser()
  await refreshUsers()
})

watch(currentUser, (user) => {
  if (user && authEnabled.value) {
    refreshKeys()
  }
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

    <!-- API Keys (only when auth is enabled) -->
    <div v-if="authEnabled">
      <h3 class="text-sm font-semibold text-text mb-3">API Keys</h3>

      <div v-if="!keysData" class="text-sm text-text-muted">Loading keys...</div>
      <div v-else-if="keysData.keys.length === 0" class="text-sm text-text-muted">No API keys</div>
      <div v-else class="overflow-x-auto rounded-lg border border-surface-border bg-surface">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border">
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Name</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Key ID</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Prefix</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Created</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Last Used</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Status</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary"></th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="key in keysData.keys"
              :key="key.keyId"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td class="px-3 py-2 text-[13px]">{{ key.name }}</td>
              <td class="px-3 py-2 text-[13px] font-mono text-text-secondary">{{ key.keyId }}</td>
              <td class="px-3 py-2 text-[13px] font-mono">{{ key.keyPrefix }}...</td>
              <td class="px-3 py-2 text-[13px] text-text-secondary">{{ formatTimestamp(key.createdAtMs) }}</td>
              <td class="px-3 py-2 text-[13px] text-text-secondary">{{ formatTimestamp(key.lastUsedAtMs) }}</td>
              <td class="px-3 py-2 text-[13px]">
                <span
                  class="inline-block px-2 py-0.5 text-xs font-medium rounded-full"
                  :class="key.revoked
                    ? 'bg-status-danger-bg text-status-danger'
                    : 'bg-status-success-bg text-status-success'"
                >
                  {{ key.revoked ? 'Revoked' : 'Active' }}
                </span>
              </td>
              <td class="px-3 py-2 text-[13px]">
                <button
                  v-if="!key.revoked"
                  class="text-xs text-status-danger hover:text-status-danger/80 transition-colors"
                  @click="revokeKey(key.keyId)"
                >
                  Revoke
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
