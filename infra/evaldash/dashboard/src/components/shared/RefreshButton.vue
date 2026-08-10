<script setup lang="ts">
import { useServerRefresh } from '@/composables/useRefresh'
import { formatClock } from '@/utils/formatting'

// Both the header and the status page mount this; the composable's shared state keeps them
// in lockstep — one spinner, one "updated" stamp, one error.
const { refreshing, lastRefreshAt, refreshError, refreshNow } = useServerRefresh()
</script>

<template>
  <div class="inline-flex items-center gap-2">
    <button
      class="text-xs px-2 py-1 rounded border border-surface-border hover:bg-surface-raised inline-flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
      :disabled="refreshing"
      title="Ingest now, then reload this view"
      @click="refreshNow"
    >
      <span :class="['inline-block', { 'animate-spin': refreshing }]" aria-hidden="true">⟳</span>
      {{ refreshing ? 'Refreshing' : 'Refresh' }}
    </button>
    <span
      v-if="refreshError"
      class="text-xs text-status-danger inline-flex items-center gap-1"
      :title="refreshError"
    >
      <span aria-hidden="true">⚠</span> refresh failed
    </span>
    <span
      v-else-if="lastRefreshAt"
      class="text-xs text-text-muted tabular-nums"
    >updated {{ formatClock(lastRefreshAt) }}</span>
  </div>
</template>
