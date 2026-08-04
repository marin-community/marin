<script setup lang="ts">
import { computed } from 'vue'

import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import TimeSeriesChart from '@/components/dashboard/TimeSeriesChart.vue'
import type { DashboardPanel } from '@/types/dashboard'
import type { ArrowResult } from '@/utils/arrow'

const props = defineProps<{
  panel: DashboardPanel
  result: ArrowResult | null
  loading: boolean
  error: string | null
  rowCount: number
  elapsedMs: number | null
}>()

defineEmits<{ run: [] }>()

const columns = computed<Column[]>(() =>
  (props.result?.columns ?? []).map((column) => ({ key: column, label: column, mono: true })),
)

const stats = computed(() => {
  const row = props.result?.rows[0]
  if (!row) return []
  return Object.entries(row).filter((entry): entry is [string, number] =>
    typeof entry[1] === 'number' && Number.isFinite(entry[1]),
  )
})

function formatStat(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(value)
}
</script>

<template>
  <section class="bg-surface border border-surface-border rounded-lg overflow-hidden min-w-0">
    <header class="flex items-start justify-between gap-3 px-4 py-3 border-b border-surface-border-subtle">
      <div class="min-w-0">
        <h3 class="font-semibold text-text">{{ panel.title }}</h3>
        <p v-if="panel.description" class="text-xs text-text-secondary mt-0.5">{{ panel.description }}</p>
      </div>
      <button
        class="shrink-0 px-2.5 py-1 text-xs rounded border border-surface-border hover:bg-surface-raised disabled:opacity-50"
        :disabled="loading"
        @click="$emit('run')"
      >{{ loading ? 'Running…' : 'Run' }}</button>
    </header>

    <div
      v-if="error"
      class="m-4 px-3 py-2 text-sm font-mono whitespace-pre-wrap text-status-danger bg-status-danger-bg border border-status-danger-border rounded"
    >{{ error }}</div>
    <div v-else-if="loading && !result" class="py-16 text-center text-sm text-text-muted">Running query…</div>
    <div v-else-if="panel.kind === 'timeseries' && result" class="p-2">
      <TimeSeriesChart :rows="result.rows" :types="result.types" />
    </div>
    <div v-else-if="panel.kind === 'stat' && result" class="p-5">
      <div v-if="stats.length" class="grid grid-cols-2 gap-4">
        <div v-for="[name, value] in stats" :key="name" class="bg-surface-raised rounded p-4">
          <div class="text-xs uppercase tracking-wide text-text-muted">{{ name }}</div>
          <div class="mt-1 text-2xl font-semibold tabular-nums">{{ formatStat(value) }}</div>
        </div>
      </div>
      <div v-else class="py-10 text-center text-sm text-text-muted">No numeric result.</div>
    </div>
    <DataTable
      v-else-if="panel.kind === 'table' && result"
      :columns="columns"
      :rows="result.rows"
      :page-size="25"
      empty-message="No rows in this time range."
    />
    <div v-else class="py-16 text-center text-sm text-text-muted">Run this panel to load data.</div>

    <footer v-if="result && !loading" class="flex items-center justify-between gap-3 px-4 py-2 border-t border-surface-border-subtle text-[11px] text-text-muted">
      <span>{{ rowCount.toLocaleString() }} row{{ rowCount === 1 ? '' : 's' }}</span>
      <span v-if="elapsedMs !== null">{{ (elapsedMs / 1000).toFixed(2) }} s end-to-end</span>
    </footer>
  </section>
</template>
