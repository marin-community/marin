<script setup lang="ts">
import * as Plot from '@observablehq/plot'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { MetricPoint } from '@/types/dashboard'
import { numeric } from '@/utils/formatting'

const props = defineProps<{
  title: string
  points: MetricPoint[]
  field: 'itemRate' | 'byteRate' | 'cpuCores' | 'memoryBytes'
  unit: string
}>()

const root = ref<HTMLElement | null>(null)
let observer: ResizeObserver | null = null

function value(point: MetricPoint): number {
  if (props.field === 'memoryBytes') return numeric(point.memoryBytes)
  return point[props.field] ?? 0
}

function render() {
  if (!root.value) return
  const width = Math.max(root.value.clientWidth, 320)
  const rows = props.points.map((point) => ({
    time: new Date(numeric(point.timestampMs)),
    stage: point.stage || 'pipeline',
    value: value(point),
  }))
  const chart = Plot.plot({
    width,
    height: 230,
    marginLeft: 56,
    marginBottom: 36,
    style: { background: 'transparent', color: 'var(--c-text-secondary)', fontSize: '11px' },
    x: { type: 'utc', label: null, grid: false },
    y: { label: props.unit, grid: true, nice: true },
    color: { legend: rows.some((row) => row.stage !== rows[0]?.stage) },
    marks: [
      Plot.ruleY([0], { stroke: 'var(--c-border)' }),
      Plot.lineY(rows, { x: 'time', y: 'value', stroke: 'stage', strokeWidth: 2, tip: true }),
      Plot.dot(rows, { x: 'time', y: 'value', fill: 'stage', r: 2 }),
    ],
  })
  root.value.replaceChildren(chart)
}

watch(() => [props.points, props.field], render, { deep: true })
onMounted(() => {
  observer = new ResizeObserver(render)
  observer.observe(root.value as HTMLElement)
  render()
})
onBeforeUnmount(() => observer?.disconnect())
</script>

<template>
  <section class="card overflow-hidden">
    <div class="border-b border-surface-border px-5 py-3">
      <h2 class="text-sm font-semibold">{{ title }}</h2>
    </div>
    <div v-if="points.length" ref="root" class="plot min-h-[230px] w-full px-2 py-3" />
    <div v-else class="grid h-[230px] place-items-center text-sm text-text-muted">No samples yet</div>
  </section>
</template>
