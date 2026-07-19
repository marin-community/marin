<script setup lang="ts">
/**
 * Hand-rolled inline-SVG spark strip: one bar per task, showing a model's standing on that
 * task relative to the rest of the fleet. Bar height is (value - min) / (max - min) across
 * the models scored on that task (a single-scored task renders full height); fill marks the
 * fleet-best bar with the success color and the fleet-worst with danger, leaving the rest in
 * the ambient text color. An unscored task renders as a faint 2px baseline stub. No Plot
 * dependency — this is small enough, and dense enough, to draw as raw <rect> elements.
 */
import { computed } from 'vue'
import type { Matrix, MatrixCell } from '@/types/api'

const props = defineProps<{
  model: string
  tasks: string[]
  matrix: Matrix
}>()

const WIDTH = 140
const HEIGHT = 24
const GAP = 1
const STUB_HEIGHT = 2

interface TaskStat {
  min: number
  max: number
  bestModel: string
  worstModel: string
}

const cells = computed<Record<string, MatrixCell>>(
  () => props.matrix.rows.find((r) => r.model === props.model)?.cells ?? {},
)

// Per-task fleet min/max/best/worst, computed once per strip over the models that have a
// score for that task.
const taskStats = computed<Record<string, TaskStat>>(() => {
  const out: Record<string, TaskStat> = {}
  for (const task of props.tasks) {
    let min = Infinity
    let max = -Infinity
    let bestModel = ''
    let worstModel = ''
    for (const row of props.matrix.rows) {
      const value = row.cells[task]?.value
      if (value === null || value === undefined) continue
      if (value > max) {
        max = value
        bestModel = row.model
      }
      if (value < min) {
        min = value
        worstModel = row.model
      }
    }
    if (bestModel) out[task] = { min, max, bestModel, worstModel }
  }
  return out
})

interface Bar {
  task: string
  x: number
  width: number
  y: number
  height: number
  fill: string
  opacity: number
  title: string
}

const bars = computed<Bar[]>(() => {
  const tasks = props.tasks
  const n = tasks.length
  if (n === 0) return []
  const width = Math.max(1, (WIDTH - GAP * (n - 1)) / n)
  return tasks.map((task, i) => {
    const x = i * (width + GAP)
    const stat = taskStats.value[task]
    const value = cells.value[task]?.value ?? null

    if (value === null || !stat) {
      const against = stat ? ` (best ${stat.max.toFixed(3)} ${stat.bestModel})` : ''
      return {
        task,
        x,
        width,
        y: HEIGHT - STUB_HEIGHT,
        height: STUB_HEIGHT,
        fill: 'currentColor',
        opacity: 0.15,
        title: `${task} · missing${against}`,
      }
    }

    const span = stat.max - stat.min
    const frac = span < 1e-9 ? 1 : (value - stat.min) / span
    const height = Math.max(STUB_HEIGHT, frac * HEIGHT)
    const isBest = props.model === stat.bestModel
    const isWorst = props.model === stat.worstModel
    const fill = isBest ? 'var(--c-status-success)' : isWorst ? 'var(--c-status-danger)' : 'currentColor'
    const opacity = isBest ? 1 : isWorst ? 0.6 : 0.35

    return {
      task,
      x,
      width,
      y: HEIGHT - height,
      height,
      fill,
      opacity,
      title: `${task} · ${value.toFixed(3)} (best ${stat.max.toFixed(3)} ${stat.bestModel})`,
    }
  })
})
</script>

<template>
  <svg
    v-if="bars.length"
    :width="WIDTH"
    :height="HEIGHT"
    :viewBox="`0 0 ${WIDTH} ${HEIGHT}`"
    class="inline-block align-middle"
    role="img"
    :aria-label="`per-task score profile for ${model}`"
  >
    <rect
      v-for="bar in bars"
      :key="bar.task"
      :x="bar.x"
      :y="bar.y"
      :width="bar.width"
      :height="bar.height"
      :fill="bar.fill"
      :fill-opacity="bar.opacity"
      rx="0.5"
    >
      <title>{{ bar.title }}</title>
    </rect>
  </svg>
</template>
