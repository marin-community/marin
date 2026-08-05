<script setup lang="ts">
import { computed } from 'vue'
import { formatAxisTime } from '@/utils/formatting'
import { timeZoneMode } from '@/composables/useDisplayPrefs'

const props = defineProps<{
  rows: Record<string, unknown>[]
  types: Record<string, string>
}>()

const WIDTH = 900
const HEIGHT = 280
const LEFT = 64
const RIGHT = 20
const TOP = 18
const BOTTOM = 42
const COLORS = Array.from({ length: 8 }, (_, i) => `var(--c-series-${i + 1})`)

interface Point {
  time: number
  value: number
}

interface SeriesPath {
  name: string
  color: string
  path: string
}

interface ChartModel {
  error: string | null
  paths: SeriesPath[]
  minTime: number
  maxTime: number
  minValue: number
  maxValue: number
}

function timestamp(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value !== 'string') return null
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : null
}

function formatValue(value: number): string {
  if (Math.abs(value) >= 1_000_000) return value.toExponential(2)
  if (Math.abs(value) >= 100) return value.toFixed(0)
  if (Math.abs(value) >= 1) return value.toFixed(2)
  return value.toPrecision(3)
}

function schemaContractError(types: Record<string, string>): string | null {
  const timeType = types.t ?? ''
  const seriesType = types.series ?? ''
  const valueType = types.value ?? ''
  if (!/Timestamp|Date|Int|UInt|Float/.test(timeType)) {
    return `t must be an Arrow timestamp, date, or number (received ${timeType || 'unknown'})`
  }
  if (!/Utf8/.test(seriesType)) {
    return `series must be an Arrow string (received ${seriesType || 'unknown'})`
  }
  if (!/Int|UInt|Float|Decimal/.test(valueType)) {
    return `value must be an Arrow number (received ${valueType || 'unknown'})`
  }
  return null
}

const model = computed<ChartModel>(() => {
  if (props.rows.length === 0) {
    return { error: null, paths: [], minTime: 0, maxTime: 1, minValue: 0, maxValue: 1 }
  }
  for (const column of ['t', 'series', 'value']) {
    if (!(column in props.rows[0])) {
      return {
        error: `time-series query must return t, series, value (received ${Object.keys(props.rows[0]).join(', ')})`,
        paths: [], minTime: 0, maxTime: 1, minValue: 0, maxValue: 1,
      }
    }
  }
  const schemaError = schemaContractError(props.types)
  if (schemaError) {
    return { error: schemaError, paths: [], minTime: 0, maxTime: 1, minValue: 0, maxValue: 1 }
  }

  const grouped = new Map<string, Point[]>()
  for (const [index, row] of props.rows.entries()) {
    const time = timestamp(row.t)
    if (time === null || typeof row.series !== 'string' || typeof row.value !== 'number' || !Number.isFinite(row.value)) {
      const schema = ['t', 'series', 'value'].map((name) => `${name}: ${props.types[name] ?? 'unknown'}`).join(', ')
      return {
        error: `row ${index + 1} violates the time-series contract (${schema})`,
        paths: [], minTime: 0, maxTime: 1, minValue: 0, maxValue: 1,
      }
    }
    const points = grouped.get(row.series) ?? []
    points.push({ time, value: row.value })
    grouped.set(row.series, points)
  }

  const all = [...grouped.values()].flat()
  const minTime = Math.min(...all.map((point) => point.time))
  const maxTime = Math.max(...all.map((point) => point.time))
  const rawMinValue = Math.min(...all.map((point) => point.value))
  const rawMaxValue = Math.max(...all.map((point) => point.value))
  const minValue = rawMinValue === rawMaxValue ? rawMinValue - Math.max(1, Math.abs(rawMinValue) * 0.05) : rawMinValue
  const maxValue = rawMinValue === rawMaxValue ? rawMaxValue + Math.max(1, Math.abs(rawMaxValue) * 0.05) : rawMaxValue
  const timeSpan = Math.max(1, maxTime - minTime)
  const valueSpan = Math.max(Number.EPSILON, maxValue - minValue)
  const x = (time: number) => LEFT + ((time - minTime) / timeSpan) * (WIDTH - LEFT - RIGHT)
  const y = (value: number) => TOP + (1 - (value - minValue) / valueSpan) * (HEIGHT - TOP - BOTTOM)
  const paths = [...grouped.entries()].map(([name, points], index) => ({
    name,
    color: COLORS[index % COLORS.length],
    path: points
      .sort((a, b) => a.time - b.time)
      .map((point, pointIndex) => `${pointIndex === 0 ? 'M' : 'L'} ${x(point.time).toFixed(2)} ${y(point.value).toFixed(2)}`)
      .join(' '),
  }))
  return { error: null, paths, minTime, maxTime, minValue, maxValue }
})

const yTicks = computed(() => Array.from({ length: 5 }, (_, index) => {
  const ratio = index / 4
  const value = model.value.maxValue - ratio * (model.value.maxValue - model.value.minValue)
  return { y: TOP + ratio * (HEIGHT - TOP - BOTTOM), label: formatValue(value) }
}))

const xTicks = computed(() => Array.from({ length: 5 }, (_, index) => {
  const ratio = index / 4
  const time = model.value.minTime + ratio * (model.value.maxTime - model.value.minTime)
  return {
    x: LEFT + ratio * (WIDTH - LEFT - RIGHT),
    label: formatAxisTime(time, model.value.maxTime - model.value.minTime, timeZoneMode.value),
  }
}))
</script>

<template>
  <div
    v-if="model.error"
    class="px-3 py-2 text-sm font-mono text-status-danger bg-status-danger-bg border border-status-danger-border rounded"
  >{{ model.error }}</div>
  <div v-else-if="model.paths.length === 0" class="py-12 text-center text-sm text-text-muted">
    No points in this time range.
  </div>
  <div v-else>
    <svg class="w-full h-auto" :viewBox="`0 0 ${WIDTH} ${HEIGHT}`" role="img" aria-label="Time-series chart">
      <g v-for="tick in yTicks" :key="tick.y">
        <line :x1="LEFT" :x2="WIDTH - RIGHT" :y1="tick.y" :y2="tick.y" stroke="var(--c-surface-border-subtle)" />
        <text :x="LEFT - 9" :y="tick.y + 4" text-anchor="end" fill="var(--c-text-muted)" font-size="11">{{ tick.label }}</text>
      </g>
      <g v-for="tick in xTicks" :key="tick.x">
        <text :x="tick.x" :y="HEIGHT - 14" text-anchor="middle" fill="var(--c-text-muted)" font-size="11">{{ tick.label }}</text>
      </g>
      <path
        v-for="path in model.paths"
        :key="path.name"
        :d="path.path"
        fill="none"
        :stroke="path.color"
        stroke-width="2"
        stroke-linejoin="round"
        stroke-linecap="round"
      />
    </svg>
    <div class="flex flex-wrap gap-x-4 gap-y-1 px-4 pb-2">
      <span v-for="path in model.paths" :key="path.name" class="inline-flex items-center gap-1.5 text-xs text-text-secondary">
        <span class="w-2.5 h-2.5 rounded-full" :style="{ backgroundColor: path.color }" />
        {{ path.name }}
      </span>
    </div>
  </div>
</template>
