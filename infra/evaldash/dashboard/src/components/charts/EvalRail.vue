<script setup lang="ts">
/**
 * The eval rail: one gauge per benchmark, in a fixed column order shared across the app.
 * Height is the score, the whisker is ±stderr, the warm caret is the fleet best, and a
 * benchmark the model never ran is an explicit dashed slot (not a missing dot). Failed and
 * infra runs render as a coloured status glyph. Two sizes: `sm` is the strip inside a
 * leaderboard row; `lg` is the axed panel on the model page, with a value label, a benchmark
 * name, and — when history is supplied — a score-over-runs sparkline under each gauge.
 */
import { computed } from 'vue'
import type { MatrixCell } from '@/types/api'
import { formatScore, formatStderr } from '@/utils/formatting'
import { scoreColor } from '@/utils/score'
import { type BestCell } from '@/utils/matrix'
import { tip, type TipContent } from '@/composables/tooltip'

interface HistoryPoint {
  created_at: string | null
  value: number
}

// Sparkline geometry, shared by the polyline and its dots so they never drift apart.
const SPARK = { width: 90, height: 26, pad: 3 }

const props = withDefaults(
  defineProps<{
    tasks: string[]
    cells: Record<string, MatrixCell>
    best: Record<string, BestCell>
    model: string
    size?: 'sm' | 'lg'
    history?: Record<string, HistoryPoint[]>
  }>(),
  { size: 'sm' },
)

const emit = defineEmits<{ (e: 'pick', task: string): void }>()

const H = computed(() => (props.size === 'lg' ? 150 : 40))
const W = computed(() => (props.size === 'lg' ? 46 : 15))

interface Gauge {
  task: string
  cell: MatrixCell | null
  kind: 'score' | 'missing' | 'failed' | 'infra'
  fillH: number
  whiskerBottom: number
  whiskerHeight: number
  bestY: number
  isBest: boolean
  color: string
  content: TipContent
}

const gauges = computed<Gauge[]>(() =>
  props.tasks.map((task) => {
    const cell = props.cells[task] ?? null
    const best = props.best[task]
    const bestLine = best
      ? { label: 'fleet best', value: `${formatScore(best.value)}·${best.model}`, tone: 'best' as const }
      : { label: 'fleet best', value: '—', tone: 'muted' as const }

    if (!cell) {
      return {
        task, cell, kind: 'missing', fillH: 0, whiskerBottom: 0, whiskerHeight: 0, bestY: 0, isBest: false,
        color: '', content: { title: task, lines: [{ label: props.model, value: 'not run', tone: 'muted' }, bestLine] },
      }
    }
    if (cell.value === null) {
      const kind = cell.status === 'infra_failed' ? 'infra' : 'failed'
      return {
        task, cell, kind, fillH: 0, whiskerBottom: 0, whiskerHeight: 0, bestY: 0, isBest: false, color: '',
        content: { title: task, lines: [{ label: props.model, value: cell.status.replace('_', ' '), tone: 'muted' }] },
      }
    }
    const v = Math.max(0, Math.min(1, cell.value))
    const fillH = Math.max(3, v * H.value)
    const stderrHeight = cell.stderr ? cell.stderr * H.value : 0
    const bestY = best ? Math.max(0, Math.min(1, best.value)) * H.value : 0
    const isBest = best?.model === props.model
    return {
      task, cell, kind: 'score', fillH, whiskerBottom: fillH - stderrHeight, whiskerHeight: stderrHeight * 2, bestY, isBest,
      color: scoreColor(v),
      content: {
        title: task,
        lines: [
          { label: props.model, value: `${formatScore(cell.value)} ${formatStderr(cell.value, cell.stderr)}`.trim() },
          { label: 'metric', value: cell.metric ?? '—', tone: 'muted' },
          isBest
            ? { label: 'fleet best', value: 'this model', tone: 'best' }
            : bestLine,
        ],
      },
    }
  }),
)

function sparkPoints(points: HistoryPoint[]): { x: number; y: number }[] {
  const { width, height, pad } = SPARK
  const innerW = width - 2 * pad
  const innerH = height - 2 * pad
  return points.map((p, i) => ({
    x: pad + (points.length > 1 ? (i * innerW) / (points.length - 1) : 0),
    y: height - pad - Math.max(0, Math.min(1, p.value)) * innerH,
  }))
}
function sparkLine(points: HistoryPoint[]): string {
  return sparkPoints(points)
    .map((pt) => `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`)
    .join(' ')
}
</script>

<template>
  <div :class="size === 'lg' ? 'flex items-end gap-5' : 'inline-flex items-end gap-[3px]'">
    <div
      v-for="g in gauges"
      :key="g.task"
      :class="size === 'lg' ? 'flex flex-col items-center gap-2.5 w-[104px]' : ''"
    >
      <!-- gauge -->
      <div
        class="relative rounded"
        :style="{ width: `${W}px`, height: `${H}px` }"
        :class="[
          g.kind === 'failed' ? 'bg-status-danger-bg' : g.kind === 'infra' ? 'bg-status-warning-bg' : 'bg-surface-sunken',
          g.kind === 'score' ? 'cursor-pointer hover:outline hover:outline-2 hover:outline-offset-1 hover:outline-accent-border' : '',
          g.kind === 'missing' ? 'border border-dashed border-surface-border bg-transparent' : '',
        ]"
        v-on="g.kind !== 'missing' ? tip(g.content) : {}"
        @click="g.kind === 'score' && g.cell && emit('pick', g.task)"
      >
        <!-- score fill -->
        <div
          v-if="g.kind === 'score'"
          class="absolute inset-x-0 bottom-0 rounded-b"
          :style="{ height: `${g.fillH}px`, background: g.color }"
        />
        <!-- stderr whisker -->
        <div
          v-if="g.kind === 'score' && g.whiskerHeight > 0"
          class="absolute left-1/2 -translate-x-1/2 w-[1.5px] opacity-50"
          style="background: var(--c-text)"
          :style="{ bottom: `${g.whiskerBottom}px`, height: `${g.whiskerHeight}px` }"
        />
        <!-- fleet-best caret -->
        <div
          v-if="g.kind === 'score' && g.bestY > 0"
          class="absolute -inset-x-px h-[2px] rounded-full"
          style="background: var(--c-best)"
          :style="{ bottom: `${g.bestY - 1}px` }"
        />
        <!-- status glyph -->
        <div
          v-if="g.kind === 'failed' || g.kind === 'infra'"
          class="absolute inset-0 flex items-center justify-center font-mono font-bold"
          :class="g.kind === 'failed' ? 'text-status-danger' : 'text-status-warning'"
          :style="{ fontSize: size === 'lg' ? '14px' : '11px' }"
        >{{ g.kind === 'failed' ? '✕' : '!' }}</div>
        <!-- missing marker (sm) -->
        <div
          v-if="g.kind === 'missing' && size === 'sm'"
          class="absolute left-1/2 bottom-1 -translate-x-1/2 w-[5px] h-[5px] rounded-full border border-dashed border-text-muted opacity-60"
        />
        <div
          v-if="g.kind === 'missing' && size === 'lg'"
          class="absolute inset-0 flex items-center justify-center text-center text-text-muted font-mono text-[10px] px-1.5"
        >not run</div>
      </div>

      <!-- large-mode value + sparkline + name -->
      <template v-if="size === 'lg'">
        <div class="font-mono text-lg font-semibold" :class="{ 'text-text-muted': g.kind !== 'score' }">
          <template v-if="g.kind === 'score' && g.cell">
            {{ formatScore(g.cell.value) }}<span class="text-[11px] font-normal text-text-muted"> {{ formatStderr(g.cell.value, g.cell.stderr) }}</span>
          </template>
          <template v-else>—</template>
        </div>
        <div class="flex items-center" :style="{ height: `${SPARK.height}px` }">
          <svg v-if="history && history[g.task] && history[g.task].length >= 2" :width="SPARK.width" :height="SPARK.height">
            <polyline :points="sparkLine(history[g.task])" fill="none" stroke="var(--c-accent)" stroke-width="1.5" />
            <circle
              v-for="(pt, i) in sparkPoints(history[g.task])"
              :key="i"
              :cx="pt.x.toFixed(1)"
              :cy="pt.y.toFixed(1)"
              :r="i === history[g.task].length - 1 ? 2.5 : 1.8"
              fill="var(--c-accent)"
            />
          </svg>
          <span v-else-if="g.kind === 'score'" class="font-mono text-[10px] text-text-muted">1 run</span>
        </div>
        <div class="font-mono text-xs text-text-secondary text-center">{{ g.task }}</div>
      </template>
    </div>
  </div>
</template>
