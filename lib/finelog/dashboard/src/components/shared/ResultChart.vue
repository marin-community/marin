<script setup lang="ts">
/**
 * Chart any query result.
 *
 * The caller names an x column, a y column, and an optional series column; this
 * component owns scales, marks, axes, and the hover readout. Nothing about the
 * query shape is assumed — x may be an instant, a number, or a category, and the
 * series column may be absent, in which case the result is one line.
 */
import { computed, ref } from 'vue'
import type { ColumnKind } from '@/utils/columnKind'
import { formatAxisTime, formatMetric, formatTimestampMs } from '@/utils/formatting'
import { timeZoneMode } from '@/composables/useDisplayPrefs'

export type ChartMark = 'line' | 'bar' | 'scatter'

const props = defineProps<{
  rows: Record<string, unknown>[]
  kinds: Record<string, ColumnKind>
  x: string
  y: string
  series?: string
  mark: ChartMark
}>()

const WIDTH = 960
const HEIGHT = 320
const PAD = { left: 68, right: 16, top: 16, bottom: 40 }
const SERIES_COLORS = Array.from({ length: 8 }, (_, i) => `var(--c-series-${i + 1})`)
/** Above this, a legend stops being readable and the colours stop being distinct. */
const MAX_SERIES = 12

interface Point {
  xv: number
  yv: number
  label: string
  row: Record<string, unknown>
}
interface Series {
  name: string
  color: string
  points: Point[]
}

const hover = ref<{ px: number; py: number; point: Point; series: string } | null>(null)

const xIsCategorical = computed(() => {
  const kind = props.kinds[props.x]
  return kind !== 'timestamp' && kind !== 'number'
})

/** Distinct x categories in first-seen order; empty when x is continuous. */
const categories = computed<string[]>(() => {
  if (!xIsCategorical.value) return []
  const seen: string[] = []
  for (const row of props.rows) {
    const key = String(row[props.x] ?? '')
    if (!seen.includes(key)) seen.push(key)
  }
  return seen
})

const model = computed<{ series: Series[]; error: string | null }>(() => {
  if (!props.x || !props.y) return { series: [], error: null }
  const grouped = new Map<string, Point[]>()
  for (const row of props.rows) {
    const yRaw = row[props.y]
    if (typeof yRaw !== 'number' || !Number.isFinite(yRaw)) continue
    const xRaw = row[props.x]
    const xv = xIsCategorical.value
      ? categories.value.indexOf(String(xRaw ?? ''))
      : typeof xRaw === 'number'
        ? xRaw
        : Number(xRaw)
    if (!Number.isFinite(xv)) continue
    const name = props.series ? String(row[props.series] ?? '—') : props.y
    const points = grouped.get(name) ?? []
    points.push({ xv, yv: yRaw, label: String(xRaw ?? ''), row })
    grouped.set(name, points)
  }
  if (grouped.size === 0) {
    return { series: [], error: `No numeric points. Check that ${props.y} is a number column.` }
  }
  if (grouped.size > MAX_SERIES) {
    return {
      series: [],
      error: `${grouped.size} series is too many to read. Add a filter, or GROUP BY fewer values of ${props.series}.`,
    }
  }
  const series = [...grouped.entries()].map(([name, points], i) => ({
    name,
    color: SERIES_COLORS[i % SERIES_COLORS.length],
    points: points.sort((a, b) => a.xv - b.xv),
  }))
  return { series, error: null }
})

const bounds = computed(() => {
  const all = model.value.series.flatMap((s) => s.points)
  if (all.length === 0) return { x0: 0, x1: 1, y0: 0, y1: 1 }
  const xs = all.map((p) => p.xv)
  const ys = all.map((p) => p.yv)
  let y0 = Math.min(...ys)
  let y1 = Math.max(...ys)
  // Bars read as a proportion, so their baseline belongs at zero.
  if (props.mark === 'bar') y0 = Math.min(0, y0)
  if (y0 === y1) {
    const pad = Math.max(1, Math.abs(y0) * 0.05)
    y0 -= pad
    y1 += pad
  } else {
    // Headroom to the next gridline, so the tallest mark does not run into the
    // top edge and the axis ends on a labelled value.
    y1 += (y1 - y0) * 0.04
  }
  return { x0: Math.min(...xs), x1: Math.max(...xs), y0, y1 }
})

function sx(v: number): number {
  const { x0, x1 } = bounds.value
  const span = x1 - x0 || 1
  const inner = WIDTH - PAD.left - PAD.right
  // Categorical x sits at band centres so the first and last bar stay inside the plot.
  if (xIsCategorical.value) {
    const n = Math.max(1, categories.value.length)
    return PAD.left + ((v + 0.5) / n) * inner
  }
  return PAD.left + ((v - x0) / span) * inner
}

function sy(v: number): number {
  const { y0, y1 } = bounds.value
  const span = y1 - y0 || 1
  return PAD.top + (1 - (v - y0) / span) * (HEIGHT - PAD.top - PAD.bottom)
}

const barWidth = computed(() => {
  const inner = WIDTH - PAD.left - PAD.right
  const slots = xIsCategorical.value
    ? Math.max(1, categories.value.length)
    : Math.max(1, model.value.series[0]?.points.length ?? 1)
  const groups = Math.max(1, model.value.series.length)
  return Math.max(1, (inner / slots) * 0.7 / groups)
})

function barX(point: Point, seriesIndex: number): number {
  const groups = Math.max(1, model.value.series.length)
  return sx(point.xv) - (barWidth.value * groups) / 2 + barWidth.value * seriesIndex
}

function linePath(points: Point[]): string {
  return points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.xv).toFixed(2)} ${sy(p.yv).toFixed(2)}`).join(' ')
}

/**
 * Shorten `text` from the middle. Category labels here are metric and job names
 * that share a long prefix, so cutting the tail makes neighbouring bars
 * indistinguishable; keeping both ends preserves what tells them apart.
 */
function elideMiddle(text: string, max: number): string {
  if (text.length <= max) return text
  const head = Math.ceil((max - 1) / 2)
  return `${text.slice(0, head)}…${text.slice(text.length - (max - 1 - head))}`
}

/**
 * Round gridline values covering `[lo, hi]` — steps of 1, 2, or 5 times a power
 * of ten. Interpolating the exact bounds instead gives labels like 16.289 and
 * 12.2169, which are precise and unreadable.
 */
function niceTicks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo
  if (!(span > 0)) return [lo]
  const rough = span / count
  const magnitude = 10 ** Math.floor(Math.log10(rough))
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= rough) ?? 10 * magnitude
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v)
  return out.length >= 2 ? out : [lo, hi]
}

const yTicks = computed(() => {
  const { y0, y1 } = bounds.value
  return niceTicks(y0, y1, 4).map((v) => ({ y: sy(v), label: formatMetric(v) }))
})

const xTicks = computed(() => {
  if (xIsCategorical.value) {
    const cats = categories.value
    const step = Math.max(1, Math.ceil(cats.length / 8))
    const shown = Math.ceil(cats.length / step)
    // Characters that fit one label's share of the axis, at the ~6px advance of
    // the 11px tick face. Hovering a mark still shows the untruncated value.
    const budget = Math.max(8, Math.floor((WIDTH - PAD.left - PAD.right) / shown / 6))
    return cats
      .map((label, i) => ({ i, label }))
      .filter(({ i }) => i % step === 0)
      .map(({ i, label }) => ({ x: sx(i), label: elideMiddle(label, budget) }))
  }
  const { x0, x1 } = bounds.value
  const isTime = props.kinds[props.x] === 'timestamp'
  return Array.from({ length: 5 }, (_, i) => {
    const v = x0 + (i / 4) * (x1 - x0)
    return {
      x: sx(v),
      label: isTime ? formatAxisTime(v, x1 - x0, timeZoneMode.value) : formatMetric(v),
    }
  })
})

function formatX(point: Point): string {
  if (props.kinds[props.x] === 'timestamp') return formatTimestampMs(point.xv, timeZoneMode.value)
  if (xIsCategorical.value) return point.label
  return formatMetric(point.xv)
}

/** Nearest point to the pointer, in screen space, across every series. */
function onMove(e: MouseEvent) {
  const svg = e.currentTarget as SVGSVGElement
  const box = svg.getBoundingClientRect()
  const px = ((e.clientX - box.left) / box.width) * WIDTH
  const py = ((e.clientY - box.top) / box.height) * HEIGHT
  let best: typeof hover.value = null
  let bestDist = Infinity
  for (const s of model.value.series) {
    for (const p of s.points) {
      const dx = sx(p.xv) - px
      const dy = sy(p.yv) - py
      const dist = dx * dx + dy * dy
      if (dist < bestDist) {
        bestDist = dist
        best = { px: sx(p.xv), py: sy(p.yv), point: p, series: s.name }
      }
    }
  }
  hover.value = best
}

const tooltipAnchor = computed(() => {
  if (!hover.value) return { x: 0, y: 0, flip: false }
  const flip = hover.value.px > WIDTH * 0.6
  return { x: hover.value.px, y: hover.value.py, flip }
})
</script>

<template>
  <div
    v-if="model.error"
    class="mx-4 my-3 px-3 py-2 text-sm text-status-danger bg-status-danger-bg border border-status-danger-border rounded"
  >{{ model.error }}</div>
  <div v-else-if="model.series.length === 0" class="py-12 text-center text-sm text-text-muted">
    Pick an x and y column to plot.
  </div>
  <div v-else class="relative">
    <svg
      class="w-full h-auto"
      :viewBox="`0 0 ${WIDTH} ${HEIGHT}`"
      role="img"
      :aria-label="`${mark} chart of ${y} against ${x}`"
      @mousemove="onMove"
      @mouseleave="hover = null"
    >
      <g v-for="tick in yTicks" :key="`y${tick.y}`">
        <line
          :x1="PAD.left" :x2="WIDTH - PAD.right" :y1="tick.y" :y2="tick.y"
          stroke="var(--c-surface-border-subtle)"
        />
        <text
          :x="PAD.left - 9" :y="tick.y + 4" text-anchor="end"
          fill="var(--c-text-muted)" font-size="11" font-variant-numeric="tabular-nums"
        >{{ tick.label }}</text>
      </g>
      <text
        v-for="tick in xTicks" :key="`x${tick.x}${tick.label}`"
        :x="tick.x" :y="HEIGHT - 14" text-anchor="middle"
        fill="var(--c-text-muted)" font-size="11"
      >{{ tick.label }}</text>

      <template v-if="mark === 'bar'">
        <g v-for="(s, si) in model.series" :key="s.name">
          <rect
            v-for="(p, pi) in s.points" :key="pi"
            :x="barX(p, si)" :width="barWidth"
            :y="Math.min(sy(p.yv), sy(Math.max(0, bounds.y0)))"
            :height="Math.max(1, Math.abs(sy(p.yv) - sy(Math.max(0, bounds.y0))))"
            :fill="s.color"
          />
        </g>
      </template>
      <template v-else-if="mark === 'scatter'">
        <g v-for="s in model.series" :key="s.name">
          <circle
            v-for="(p, pi) in s.points" :key="pi"
            :cx="sx(p.xv)" :cy="sy(p.yv)" r="2.5" :fill="s.color" fill-opacity="0.75"
          />
        </g>
      </template>
      <template v-else>
        <path
          v-for="s in model.series" :key="s.name"
          :d="linePath(s.points)" fill="none" :stroke="s.color"
          stroke-width="1.75" stroke-linejoin="round" stroke-linecap="round"
        />
      </template>

      <g v-if="hover">
        <line
          :x1="hover.px" :x2="hover.px" :y1="PAD.top" :y2="HEIGHT - PAD.bottom"
          stroke="var(--c-text-muted)" stroke-dasharray="3 3" stroke-opacity="0.6"
        />
        <circle :cx="hover.px" :cy="hover.py" r="4" fill="var(--c-surface)" stroke="var(--c-text)" stroke-width="1.5" />
      </g>
    </svg>

    <div
      v-if="hover"
      class="pointer-events-none absolute z-10 px-2.5 py-1.5 rounded border border-surface-border bg-surface shadow-lg text-xs font-mono whitespace-nowrap"
      :style="{
        left: `${(tooltipAnchor.x / WIDTH) * 100}%`,
        top: `${(tooltipAnchor.y / HEIGHT) * 100}%`,
        transform: tooltipAnchor.flip ? 'translate(-108%, -50%)' : 'translate(8%, -50%)',
      }"
    >
      <div v-if="series" class="text-text-secondary">{{ hover.series }}</div>
      <div>{{ formatX(hover.point) }}</div>
      <div class="font-semibold">{{ formatMetric(hover.point.yv) }}</div>
    </div>

    <div v-if="series" class="flex flex-wrap gap-x-4 gap-y-1 px-4 pb-3">
      <span
        v-for="s in model.series" :key="s.name"
        class="inline-flex items-center gap-1.5 text-xs text-text-secondary"
      >
        <span class="w-2.5 h-2.5 rounded-full shrink-0" :style="{ backgroundColor: s.color }" />
        {{ s.name }}
      </span>
    </div>
  </div>
</template>
