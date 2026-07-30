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
import { tip, type TipContent } from '@/composables/tooltip'

export interface BestCell {
  value: number
  model: string
}

interface HistoryPoint {
  created_at: string | null
  value: number
}

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

const emit = defineEmits<{ (e: 'pick', task: string, cell: MatrixCell): void }>()

const H = computed(() => (props.size === 'lg' ? 150 : 40))
const W = computed(() => (props.size === 'lg' ? 46 : 15))

interface Gauge {
  task: string
  cell: MatrixCell | null
  kind: 'score' | 'missing' | 'failed' | 'infra'
  fillH: number
  whiskBottom: number
  whiskH: number
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
        task, cell, kind: 'missing', fillH: 0, whiskBottom: 0, whiskH: 0, bestY: 0, isBest: false,
        color: '', content: { title: task, lines: [{ label: props.model, value: 'not run', tone: 'muted' }, bestLine] },
      }
    }
    if (cell.value === null) {
      const kind = cell.status === 'infra_failed' ? 'infra' : 'failed'
      return {
        task, cell, kind, fillH: 0, whiskBottom: 0, whiskH: 0, bestY: 0, isBest: false, color: '',
        content: { title: task, lines: [{ label: props.model, value: cell.status.replace('_', ' '), tone: 'muted' }] },
      }
    }
    const v = Math.max(0, Math.min(1, cell.value))
    const fillH = Math.max(3, v * H.value)
    const seH = cell.stderr ? cell.stderr * H.value : 0
    const bestY = best ? Math.max(0, Math.min(1, best.value)) * H.value : 0
    const isBest = best?.model === props.model
    return {
      task, cell, kind: 'score', fillH, whiskBottom: fillH - seH, whiskH: seH * 2, bestY, isBest,
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

function spark(points: HistoryPoint[]): string {
  if (points.length < 2) return ''
  const w = 90, h = 26, pad = 3
  const xs = (i: number) => pad + (i * (w - 2 * pad)) / (points.length - 1)
  const ys = (v: number) => h - pad - Math.max(0, Math.min(1, v)) * (h - 2 * pad)
  return points.map((p, i) => `${xs(i).toFixed(1)},${ys(p.value).toFixed(1)}`).join(' ')
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
        @click="g.kind === 'score' && g.cell && emit('pick', g.task, g.cell)"
      >
        <!-- score fill -->
        <div
          v-if="g.kind === 'score'"
          class="absolute inset-x-0 bottom-0 rounded-b"
          :style="{ height: `${g.fillH}px`, background: g.color }"
        />
        <!-- stderr whisker -->
        <div
          v-if="g.kind === 'score' && g.whiskH > 0"
          class="absolute left-1/2 -translate-x-1/2 w-[1.5px] opacity-50"
          style="background: var(--c-text)"
          :style="{ bottom: `${g.whiskBottom}px`, height: `${g.whiskH}px` }"
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
        <div class="h-[26px] flex items-center">
          <svg v-if="history && history[g.task] && history[g.task].length >= 2" width="90" height="26">
            <polyline :points="spark(history[g.task])" fill="none" stroke="var(--c-accent)" stroke-width="1.5" />
            <circle
              v-for="(p, i) in history[g.task]"
              :key="i"
              :cx="(3 + (i * 84) / (history[g.task].length - 1)).toFixed(1)"
              :cy="(26 - 3 - Math.max(0, Math.min(1, p.value)) * 20).toFixed(1)"
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
