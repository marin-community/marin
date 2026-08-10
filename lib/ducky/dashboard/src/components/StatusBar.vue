<script setup lang="ts">
import { computed } from 'vue'
import type { QueryResult } from '../composables/useQuery'
import { fmtBytes, fmtDuration } from '../utils/formatting'

const props = defineProps<{ result: QueryResult; ttlDays: number | null }>()

const rowsLabel = computed(() => {
  const shown = props.result.rows.length
  return props.result.truncated
    ? `showing ${shown.toLocaleString()} of ${props.result.total_rows.toLocaleString()} rows`
    : `${shown.toLocaleString()} row${shown === 1 ? '' : 's'}`
})
</script>

<template>
  <div class="flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px] text-text-secondary">
    <span>{{ rowsLabel }}</span>
    <span class="text-text-muted">·</span>
    <span>{{ fmtDuration(result.elapsed_ms) }}</span>
    <span class="text-text-muted">·</span>
    <span>{{ fmtBytes(result.result_bytes) }} result</span>
    <span class="text-text-muted">·</span>
    <span
      v-if="result.cached"
      class="rounded bg-status-success/10 px-1.5 py-0.5 font-semibold text-status-success"
      >cached ✓</span
    >
    <span v-else class="rounded bg-surface-sunken px-1.5 py-0.5 text-text-muted">computed</span>
    <span class="text-text-muted">·</span>
    <span class="text-text-muted">output:</span>
    <code class="rounded bg-surface-sunken px-1.5 py-0.5 font-mono text-text">{{ result.result_path }}</code>
    <span v-if="ttlDays != null" class="text-text-muted">(expires in {{ ttlDays }}d)</span>
  </div>
</template>
