<script setup lang="ts">
import { computed, ref } from 'vue'
import type { Overview, SourceSummaryRow } from '../types'
import { fmt } from '../utils/formatting'

const props = defineProps<{ ov: Overview }>()

type SortKey = keyof SourceSummaryRow

const COLUMNS: { key: SortKey; label: string }[] = [
  { key: 'source', label: 'source' },
  { key: 'docs_est', label: 'docs' },
  { key: 'q_avg', label: 'quality avg' },
  { key: 'q_sd', label: 'q sd' },
  { key: 'q_zero', label: 'frac score≈0' },
  { key: 'drop_rate', label: 'dedup drop' },
  { key: 'dup_largest', label: 'largest dup' },
  { key: 'decon_pct', label: 'contam %' },
]

const sort = ref<{ key: SortKey; dir: 1 | -1 }>({ key: 'drop_rate', dir: -1 })

function setSort(key: SortKey) {
  sort.value =
    sort.value.key === key
      ? { key, dir: sort.value.dir === 1 ? -1 : 1 }
      : { key, dir: key === 'source' ? 1 : -1 }
}

const rows = computed(() => {
  const { key, dir } = sort.value
  return [...(props.ov.source_summary || [])].sort((a, b) => {
    const x = a[key]
    const y = b[key]
    if (x == null) return 1
    if (y == null) return -1
    return (x > y ? 1 : x < y ? -1 : 0) * dir
  })
})

// While the background computation runs, count sources that already have their
// full stats (quality present) so the banner shows real progress.
const withStats = computed(() => (props.ov.source_summary || []).filter((r) => r.q_avg != null).length)

// Same thresholds as the legacy dashboard: red for pathological values, orange for suspect ones.
function warnClass(key: SortKey, value: unknown): string {
  if (value == null || typeof value !== 'number') return ''
  if (key === 'drop_rate' && value > 0.5) return 'bg-status-danger/10'
  if (key === 'q_avg' && value < 0.05) return 'bg-status-danger/10'
  if (key === 'q_zero' && value > 0.3) return 'bg-status-warning/10'
  if (key === 'q_sd' && value < 0.05) return 'bg-status-warning/10'
  if (key === 'decon_pct' && value > 1) return 'bg-status-danger/10'
  return ''
}

function display(key: SortKey, value: unknown): string {
  if (key === 'docs_est') return fmt(value as number | null)
  return value == null ? '—' : String(value)
}
</script>

<template>
  <div class="rounded-lg border border-surface-border bg-surface p-4">
    <p v-if="!ov.source_summary_ready" class="mb-2 text-sm text-accent">
      computing per-source stats in the background… {{ withStats }}/{{ ov.source_summary_total }}
    </p>
    <template v-if="rows.length">
      <h3 class="mb-1 font-semibold">
        Per-source pipeline summary
        <span class="font-normal text-text-muted">(sampled; click a header to sort)</span>
      </h3>
      <p class="mb-2 text-sm text-text-muted">
        red = dedup drop &gt;50% or quality avg &lt;0.05 or contamination &gt;1%; orange = frac≈0 &gt;0.3 or quality sd
        &lt;0.05 (no discrimination)
      </p>
      <table class="w-full border-collapse text-[13px]">
        <thead>
          <tr>
            <th
              v-for="col in COLUMNS"
              :key="col.key"
              class="cursor-pointer border-b border-surface-border px-2 py-1 text-left font-semibold text-text-muted"
              @click="setSort(col.key)"
            >
              {{ col.label }}{{ sort.key === col.key ? (sort.dir > 0 ? ' ▲' : ' ▼') : '' }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in rows" :key="row.source">
            <td
              v-for="col in COLUMNS"
              :key="col.key"
              class="border-b border-surface-border px-2 py-1"
              :class="warnClass(col.key, row[col.key])"
            >
              {{ display(col.key, row[col.key]) }}
            </td>
          </tr>
        </tbody>
      </table>
    </template>
    <p v-else-if="ov.source_summary_ready" class="text-sm text-text-muted">no source summary available</p>
  </div>
</template>
