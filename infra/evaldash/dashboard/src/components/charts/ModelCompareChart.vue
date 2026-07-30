<script setup lang="ts">
/**
 * Grouped bar chart comparing 2–4 selected models across tasks, one facet per task, with a
 * stderr whisker (Plot.ruleX) over each bar. Reads the already-loaded matrix cells, so it
 * needs no extra fetch.
 */
import { computed } from 'vue'
import * as Plot from '@observablehq/plot'
import type { Matrix } from '@/types/api'
import PlotFigure from '@/components/charts/PlotFigure.vue'

const props = defineProps<{ matrix: Matrix; models: string[] }>()

interface Bar {
  task: string
  model: string
  value: number
  lo: number
  hi: number
}

const bars = computed<Bar[]>(() => {
  const rows: Bar[] = []
  for (const model of props.models) {
    const row = props.matrix.rows.find((r) => r.model === model)
    if (!row) continue
    for (const task of props.matrix.tasks) {
      const cell = row.cells[task]
      if (!cell || cell.value === null) continue
      const se = cell.stderr ?? 0
      rows.push({ task, model, value: cell.value, lo: cell.value - se, hi: cell.value + se })
    }
  }
  return rows
})

const options = computed<Record<string, unknown>>(() => ({
  height: 340,
  marginBottom: 76,
  marginLeft: 44,
  style: { color: 'currentColor', background: 'transparent' },
  x: { axis: null },
  fx: { label: null, tickRotate: -30 },
  y: { label: 'primary metric', grid: true },
  color: { legend: true },
  marks: [
    Plot.barY(bars.value, { fx: 'task', x: 'model', y: 'value', fill: 'model' }),
    Plot.ruleX(bars.value, { fx: 'task', x: 'model', y1: 'lo', y2: 'hi', stroke: 'currentColor', strokeOpacity: 0.55 }),
    Plot.ruleY([0], { stroke: 'currentColor', strokeOpacity: 0.2 }),
  ],
}))
</script>

<template>
  <div class="rounded-lg border border-surface-border bg-surface p-4">
    <PlotFigure v-if="bars.length" :options="options" />
    <p v-else class="text-sm text-text-muted py-8 text-center">No shared scored tasks for the selected models.</p>
  </div>
</template>
