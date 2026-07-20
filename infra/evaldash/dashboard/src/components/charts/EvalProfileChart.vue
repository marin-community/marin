<script setup lang="ts">
/**
 * Dot-strip overview of every model's score on every task: one row per task (ordered as in
 * the matrix), one dot per model, x = primary-metric score. All models render muted except
 * the reference model ("snowball"), which is emphasized and gets a thin rule to the
 * fleet-best dot on each row so the gap is visible at a glance without per-point labels.
 */
import { computed } from 'vue'
import * as Plot from '@observablehq/plot'
import type { Matrix } from '@/types/api'
import PlotFigure from '@/components/charts/PlotFigure.vue'

const props = defineProps<{ matrix: Matrix }>()

const REFERENCE_MODEL = 'snowball'
const HEIGHT_PER_TASK = 26
const MARGIN_TOP = 8
const MARGIN_BOTTOM = 28

interface Dot {
  task: string
  model: string
  value: number
  isReference: boolean
}

const dots = computed<Dot[]>(() => {
  const rows: Dot[] = []
  for (const row of props.matrix.rows) {
    const isReference = row.model === REFERENCE_MODEL
    for (const task of props.matrix.tasks) {
      const value = row.cells[task]?.value
      if (value === null || value === undefined) continue
      rows.push({ task, model: row.model, value, isReference })
    }
  }
  return rows
})

const hasReference = computed(() => props.matrix.rows.some((r) => r.model === REFERENCE_MODEL))

interface GapLine {
  task: string
  x1: number
  x2: number
}

// One segment per task from the reference model's score to the fleet-best score, skipped
// where the reference model has no score or is itself the fleet best.
const gapLines = computed<GapLine[]>(() => {
  if (!hasReference.value) return []
  const lines: GapLine[] = []
  for (const task of props.matrix.tasks) {
    let referenceValue: number | null = null
    let best: number | null = null
    for (const row of props.matrix.rows) {
      const value = row.cells[task]?.value
      if (value === null || value === undefined) continue
      if (row.model === REFERENCE_MODEL) referenceValue = value
      if (best === null || value > best) best = value
    }
    if (referenceValue !== null && best !== null && best !== referenceValue) {
      lines.push({ task, x1: referenceValue, x2: best })
    }
  }
  return lines
})

const options = computed<Record<string, unknown>>(() => ({
  height: props.matrix.tasks.length * HEIGHT_PER_TASK + MARGIN_TOP + MARGIN_BOTTOM,
  marginTop: MARGIN_TOP,
  marginBottom: MARGIN_BOTTOM,
  marginLeft: 150,
  marginRight: 16,
  style: { color: 'currentColor', background: 'transparent', fontSize: '11px' },
  x: { domain: [0, 1], label: 'primary metric', grid: true },
  y: { domain: props.matrix.tasks, label: null },
  marks: [
    Plot.ruleY(gapLines.value, {
      y: 'task',
      x1: 'x1',
      x2: 'x2',
      stroke: 'var(--c-accent)',
      strokeOpacity: 0.4,
      strokeWidth: 1.5,
    }),
    Plot.dot(dots.value, {
      x: 'value',
      y: 'task',
      r: (d: Dot) => (d.isReference ? 5 : 3),
      fill: (d: Dot) => (d.isReference ? 'var(--c-accent)' : 'var(--c-text-muted)'),
      fillOpacity: (d: Dot) => (d.isReference ? 1 : 0.55),
      title: (d: Dot) => `${d.model} · ${d.value.toFixed(3)}`,
    }),
  ],
}))
</script>

<template>
  <div class="rounded-lg border border-surface-border bg-surface p-4">
    <PlotFigure v-if="dots.length" :options="options" />
    <p v-else class="text-sm text-text-muted py-8 text-center">No scored tasks yet.</p>
  </div>
</template>
