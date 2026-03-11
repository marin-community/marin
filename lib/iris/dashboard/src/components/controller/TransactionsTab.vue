<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type { GetTransactionsResponse, TransactionAction, ProtoTimestamp } from '@/types/rpc'
import EmptyState from '@/components/shared/EmptyState.vue'

const { data, loading, error, refresh } = useControllerRpc<GetTransactionsResponse>('GetTransactions')

useAutoRefresh(refresh, 5_000)
onMounted(refresh)

const actions = computed<TransactionAction[]>(() => {
  const raw = data.value?.actions ?? []
  return [...raw].reverse()
})

function formatTime(ts?: ProtoTimestamp): string {
  if (!ts?.epochMs) return '-'
  const ms = parseInt(ts.epochMs, 10)
  if (!ms) return '-'
  const d = new Date(ms)
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  const ss = String(d.getSeconds()).padStart(2, '0')
  const mmm = String(d.getMilliseconds()).padStart(3, '0')
  return `${hh}:${mm}:${ss}.${mmm}`
}

function parseDetails(details?: string): string {
  if (!details) return '-'
  try {
    const obj = JSON.parse(details) as Record<string, unknown>
    return Object.entries(obj)
      .map(([k, v]) => `${k}=${String(v)}`)
      .join(', ')
  } catch {
    return details
  }
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
    v-else-if="!loading && actions.length === 0"
    message="No transactions recorded"
  />

  <!-- Transactions table -->
  <div v-else class="overflow-x-auto">
    <table class="w-full border-collapse">
      <thead>
        <tr class="border-b border-surface-border">
          <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary w-32">
            Time
          </th>
          <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary w-36">
            Action
          </th>
          <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Entity
          </th>
          <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Details
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="(action, i) in actions"
          :key="i"
          class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
        >
          <td class="px-3 py-2 text-[13px] font-mono text-text-secondary whitespace-nowrap">
            {{ formatTime(action.timestamp) }}
          </td>
          <td class="px-3 py-2 text-[13px] font-mono">{{ action.action ?? '-' }}</td>
          <td class="px-3 py-2 text-[13px] font-mono text-text-secondary truncate max-w-xs" :title="action.entityId">
            {{ action.entityId ?? '-' }}
          </td>
          <td class="px-3 py-2 text-xs text-text-muted truncate max-w-sm" :title="parseDetails(action.details)">
            {{ parseDetails(action.details) }}
          </td>
        </tr>
      </tbody>
    </table>
    <div class="px-3 py-2 text-xs text-text-secondary border-t border-surface-border">
      {{ actions.length }} transaction{{ actions.length !== 1 ? 's' : '' }} (newest first)
    </div>
  </div>
</template>
