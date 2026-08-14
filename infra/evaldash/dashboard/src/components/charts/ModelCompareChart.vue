<script setup lang="ts">
/**
 * Grouped bar chart comparing 2–4 selected models across benchmarks, one facet per benchmark, with
 * the cell's 95% interval drawn over each bar (Plot.ruleX). Reads the already-loaded panel cells, so
 * it needs no extra fetch. The whisker is the same interval the tables show: it widens for items a
 * run attempted but never graded, so a partly-graded bar is visibly less determined than a full one.
 */
import { computed } from 'vue'
import * as Plot from '@observablehq/plot'
import type { PanelCell } from '@/types/api'
import PlotFigure from '@/components/charts/PlotFigure.vue'

// Only what the chart reads: the benchmark order and each model's cells. Both the panel view and the
// compare view can supply that without reshaping their own payload.
interface ChartSeries {
  model: string
  cells: Record<string, PanelCell>
}

const props = defineProps<{ benchmarks: string[]; series: ChartSeries[]; models: string[] }>()

interface Bar {
  task: string
  model: string
  value: number
  low: number
  high: number
}

const bars = computed<Bar[]>(() => {
  const rows: Bar[] = []
  for (const model of props.models) {
    const series = props.series.find((s) => s.model === model)
    if (!series) continue
    for (const task of props.benchmarks) {
      const cell = series.cells[task]
      if (!cell) continue
      rows.push({ task, model, value: cell.value, low: cell.low, high: cell.high })
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
  // Fixed categorical order (dataviz-validated slots 1–4); a legend keeps identity off colour alone.
  color: { legend: true, range: ['#2a78d6', '#1baf7a', '#eda100', '#008300'] },
  marks: [
    Plot.barY(bars.value, { fx: 'task', x: 'model', y: 'value', fill: 'model' }),
    Plot.ruleX(bars.value, {
      fx: 'task',
      x: 'model',
      y1: 'low',
      y2: 'high',
      stroke: 'currentColor',
      strokeOpacity: 0.55,
    }),
    Plot.ruleY([0], { stroke: 'currentColor', strokeOpacity: 0.2 }),
  ],
}))
</script>

<template>
  <div class="rounded-lg border border-surface-border bg-surface p-4">
    <PlotFigure v-if="bars.length" :options="options" />
    <p v-else class="text-sm text-text-muted py-8 text-center">No scored benchmarks for the selected models.</p>
  </div>
</template>
