<script setup lang="ts">
/**
 * Score-over-time modal for one (model, task): a line of every run's primary metric with its 95%
 * interval as a band and a point per run (tooltip = run id + git sha). Fetched from /api/history.
 *
 * The band is the interval the panel uses, so a run that lost items shows a visibly wider band than
 * one that graded everything, and a step between two runs can be read against how well each is
 * determined.
 */
import { computed, onMounted, onUnmounted, watch } from 'vue'
import * as Plot from '@observablehq/plot'
import { RouterLink } from 'vue-router'
import { formatCoverage, formatInterval, formatScore, formatTimestamp, shortSha } from '@/utils/formatting'
import { useApi } from '@/composables/useApi'
import { isPartialCoverage } from '@/utils/panel'
import { INTERVAL_KIND, type HistoryPoint, type HistoryResponse } from '@/types/api'
import PlotFigure from '@/components/charts/PlotFigure.vue'
import StatusChip from '@/components/shared/StatusChip.vue'

const props = defineProps<{ model: string; task: string }>()
const emit = defineEmits<{ close: [] }>()

const { data, loading, error, refresh } = useApi<HistoryResponse>(
  () => `api/history?model=${encodeURIComponent(props.model)}&task=${encodeURIComponent(props.task)}`,
)

onMounted(refresh)
watch(() => [props.model, props.task], refresh)

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape') emit('close')
}
onMounted(() => window.addEventListener('keydown', onKey))
onUnmounted(() => window.removeEventListener('keydown', onKey))

interface Point {
  t: Date
  value: number
  low: number
  high: number
  run_id: string
  git_sha: string
}

const points = computed<Point[]>(() =>
  (data.value?.points ?? [])
    .filter((p) => p.created_at)
    .map((p) => ({
      t: new Date(p.created_at),
      value: p.value,
      low: p.low,
      high: p.high,
      run_id: p.run_id,
      git_sha: p.git_sha,
    })),
)

const options = computed<Record<string, unknown>>(() => ({
  height: 300,
  marginLeft: 48,
  marginBottom: 34,
  style: { color: 'currentColor', background: 'transparent' },
  x: { type: 'utc', label: null, grid: false },
  y: { label: 'primary metric', grid: true },
  marks: [
    Plot.areaY(points.value, {
      x: 't',
      y1: 'low',
      y2: 'high',
      fill: 'currentColor',
      fillOpacity: 0.12,
      curve: 'monotone-x',
    }),
    Plot.lineY(points.value, { x: 't', y: 'value', stroke: 'currentColor', strokeWidth: 1.5, curve: 'monotone-x' }),
    Plot.dot(points.value, {
      x: 't',
      y: 'value',
      fill: 'currentColor',
      r: 3.5,
      title: (d: Point) => `${d.run_id}\n${shortSha(d.git_sha)}\n${formatScore(d.value)} (${formatInterval(d.low, d.high)})`,
    }),
    Plot.ruleY([0], { stroke: 'currentColor', strokeOpacity: 0.15 }),
  ],
}))

// What a point's interval covers: the graded share when the run reports one, otherwise a note that
// completeness is unknown rather than a silent claim that nothing was lost.
function coverageNote(point: HistoryPoint): string {
  if (point.interval_kind !== INTERVAL_KIND.IDENTIFIED) return 'coverage unreported'
  return isPartialCoverage(point) ? formatCoverage(point.coverage) : 'complete'
}
</script>

<template>
  <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" @click.self="emit('close')">
    <div
      class="w-full max-w-3xl max-h-[85vh] overflow-auto rounded-lg border border-surface-border bg-surface p-5 shadow-xl"
    >
      <div class="flex items-start justify-between gap-3 mb-3">
        <div>
          <h3 class="text-sm font-semibold">Score over time</h3>
          <p class="text-xs text-text-muted mt-0.5">
            <span class="font-mono">{{ model }}</span> · {{ task }} · band = 95% interval
          </p>
        </div>
        <button
          class="text-xs px-2 py-1 rounded border border-surface-border hover:bg-surface-raised"
          @click="emit('close')"
        >
          Close
        </button>
      </div>

      <div
        v-if="error"
        class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2 mb-3"
      >
        {{ error }}
      </div>
      <div v-if="loading && !data" class="text-sm text-text-muted py-12 text-center">Loading…</div>

      <template v-else-if="data">
        <p v-if="points.length === 0" class="text-sm text-text-muted py-8 text-center">No scored runs for this cell.</p>
        <template v-else>
          <PlotFigure :options="options" />
          <div class="mt-4 overflow-x-auto rounded border border-surface-border">
            <table class="w-full border-collapse text-xs">
              <thead>
                <tr class="border-b border-surface-border bg-surface-raised text-text-secondary">
                  <th class="px-2 py-1.5 text-left">Run</th>
                  <th class="px-2 py-1.5 text-left">When</th>
                  <th class="px-2 py-1.5 text-left">Status</th>
                  <th class="px-2 py-1.5 text-left">git</th>
                  <th class="px-2 py-1.5 text-left">Graded</th>
                  <th class="px-2 py-1.5 text-right">Score</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="p in data.points"
                  :key="p.run_id"
                  class="border-b border-surface-border-subtle hover:bg-surface-raised"
                >
                  <td class="px-2 py-1.5">
                    <RouterLink
                      :to="`/runs/${p.run_id}`"
                      class="font-mono text-accent hover:text-accent-hover hover:underline"
                      @click="emit('close')"
                      >{{ p.run_id }}</RouterLink
                    >
                  </td>
                  <td class="px-2 py-1.5 whitespace-nowrap text-text-secondary">{{ formatTimestamp(p.created_at) }}</td>
                  <td class="px-2 py-1.5"><StatusChip :status="p.status" /></td>
                  <td class="px-2 py-1.5 font-mono text-text-secondary">{{ shortSha(p.git_sha) }}</td>
                  <td
                    class="px-2 py-1.5 font-mono whitespace-nowrap"
                    :class="p.coverage !== null && p.coverage < 1 ? 'text-status-warning' : 'text-text-muted'"
                  >
                    {{ coverageNote(p) }}
                  </td>
                  <td class="px-2 py-1.5 text-right tabular-nums font-mono whitespace-nowrap">
                    {{ formatScore(p.value) }}
                    <span class="block text-[10px] text-text-muted leading-none">{{ formatInterval(p.low, p.high) }}</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </template>
      </template>
    </div>
  </div>
</template>
