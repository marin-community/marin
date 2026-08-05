<script setup lang="ts">
/**
 * Chart any query result.
 *
 * The caller names an x column, a y column, and an optional series column, and
 * chooses the scale and smoothing; this component owns scales, marks, axes, and
 * every interaction — drag to zoom the x range, click a legend entry to mute a
 * series, hover for every series' value at that instant. Nothing about the query
 * shape is assumed: x may be an instant, a number, or a category, and the series
 * column may be absent, in which case the result is one line.
 */
import { computed, ref, watch } from 'vue'
import type { ColumnKind } from '@/utils/columnKind'
import {
  decimate, logTicks, niceTicks, smooth, MAX_POINTS_PER_SERIES, MAX_SERIES, type ChartMark,
} from '@/utils/chart'
import { formatAxisTime, formatMetric, formatTimestampMs } from '@/utils/formatting'
import { timeZoneMode } from '@/composables/useDisplayPrefs'

const props = withDefaults(defineProps<{
  rows: Record<string, unknown>[]
  kinds: Record<string, ColumnKind>
  x: string
  y: string
  series?: string
  mark: ChartMark
  /** EMA weight in [0, 1). 0 draws the samples as they are. */
  smoothing?: number
  yScale?: 'linear' | 'log'
}>(), { smoothing: 0, yScale: 'linear' })

const WIDTH = 960
const HEIGHT = 320
const PAD = { left: 68, right: 16, top: 16, bottom: 40 }
const SERIES_COLORS = Array.from({ length: 8 }, (_, i) => `var(--c-series-${i + 1})`)
/** Drag shorter than this is a click, not a zoom. */
const MIN_ZOOM_PX = 6

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
  raw: Point[]
}

const hoverX = ref<number | null>(null)
const muted = ref(new Set<string>())
const zoom = ref<{ x0: number; x1: number } | null>(null)
const drag = ref<{ from: number; to: number } | null>(null)

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

const model = computed<{ series: Series[]; error: string | null; decimated: boolean }>(() => {
  if (!props.x || !props.y) return { series: [], error: null, decimated: false }
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
    return { series: [], error: `No numeric points. Check that ${props.y} is a number column.`, decimated: false }
  }
  if (grouped.size > MAX_SERIES) {
    return {
      series: [],
      error: `${grouped.size} series is too many to read. Add a filter, or GROUP BY fewer values of ${props.series}.`,
      decimated: false,
    }
  }
  let decimated = false
  const series = [...grouped.entries()].map(([name, points], i) => {
    const sorted = points.sort((a, b) => a.xv - b.xv)
    if (sorted.length > MAX_POINTS_PER_SERIES) decimated = true
    // Smooth the full run before thinning it: an EMA over already-thinned points
    // would average the decimation's min/max pairs rather than the samples.
    const raw = decimate(sorted)
    return {
      name,
      color: SERIES_COLORS[i % SERIES_COLORS.length],
      points: props.smoothing > 0 ? decimate(smooth(sorted, props.smoothing)) : raw,
      raw,
    }
  })
  return { series, error: null, decimated }
})

const visible = computed(() => model.value.series.filter((s) => !muted.value.has(s.name)))

/** Points inside the current zoom; the full run when the chart is unzoomed. */
function inView(points: Point[]): Point[] {
  const z = zoom.value
  if (!z) return points
  return points.filter((p) => p.xv >= z.x0 && p.xv <= z.x1)
}

/** Log needs strictly positive values, so a series that crosses zero stays linear. */
const logUsable = computed(
  () => props.yScale === 'log' && visible.value.every((s) => inView(s.points).every((p) => p.yv > 0)),
)

const bounds = computed(() => {
  const all = visible.value.flatMap((s) => inView(s.points))
  if (all.length === 0) return { x0: 0, x1: 1, y0: 0, y1: 1 }
  const xs = all.map((p) => p.xv)
  const ys = all.map((p) => p.yv)
  let y0 = Math.min(...ys)
  let y1 = Math.max(...ys)
  // Bars are read against a zero baseline, so the range has to contain zero on
  // whichever side it falls — including all-negative values, where zero is the
  // top of the axis and a range that stopped at the largest value would draw
  // every bar from a baseline off the top of the plot.
  if (props.mark === 'bar' && !logUsable.value) {
    y0 = Math.min(0, y0)
    y1 = Math.max(0, y1)
  }
  if (y0 === y1) {
    const pad = Math.max(1, Math.abs(y0) * 0.05)
    y0 -= pad
    y1 += pad
  } else if (!logUsable.value) {
    // Headroom to the next gridline, so the tallest mark does not run into the
    // top edge and the axis ends on a labelled value.
    y1 += (y1 - y0) * 0.04
  }
  const z = zoom.value
  return { x0: z ? z.x0 : Math.min(...xs), x1: z ? z.x1 : Math.max(...xs), y0, y1 }
})

function sx(v: number): number {
  const { x0, x1 } = bounds.value
  const span = x1 - x0 || 1
  const inner = WIDTH - PAD.left - PAD.right
  // Categorical x sits at band centres so the first and last bar stay inside the plot.
  if (xIsCategorical.value && !zoom.value) {
    const n = Math.max(1, categories.value.length)
    return PAD.left + ((v + 0.5) / n) * inner
  }
  return PAD.left + ((v - x0) / span) * inner
}

/** Screen x back to a data value, for reading the pointer. */
function invertX(px: number): number {
  const { x0, x1 } = bounds.value
  const inner = WIDTH - PAD.left - PAD.right
  if (xIsCategorical.value && !zoom.value) {
    const n = Math.max(1, categories.value.length)
    return ((px - PAD.left) / inner) * n - 0.5
  }
  return x0 + ((px - PAD.left) / inner) * (x1 - x0 || 1)
}

function sy(v: number): number {
  const { y0, y1 } = bounds.value
  const plot = HEIGHT - PAD.top - PAD.bottom
  if (logUsable.value) {
    const l0 = Math.log10(y0)
    const l1 = Math.log10(y1)
    return PAD.top + (1 - (Math.log10(v) - l0) / (l1 - l0 || 1)) * plot
  }
  return PAD.top + (1 - (v - y0) / (y1 - y0 || 1)) * plot
}

const barWidth = computed(() => {
  const inner = WIDTH - PAD.left - PAD.right
  const slots = xIsCategorical.value
    ? Math.max(1, categories.value.length)
    : Math.max(1, inView(visible.value[0]?.points ?? []).length)
  const groups = Math.max(1, visible.value.length)
  return Math.max(1, (inner / slots) * 0.7 / groups)
})

function barX(point: Point, seriesIndex: number): number {
  const groups = Math.max(1, visible.value.length)
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

const yTicks = computed(() => {
  const { y0, y1 } = bounds.value
  const values = logUsable.value ? logTicks(y0, y1) : niceTicks(y0, y1, 4)
  return values.map((v) => ({ y: sy(v), label: formatMetric(v) }))
})

const xTicks = computed(() => {
  if (xIsCategorical.value && !zoom.value) {
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
      label: isTime
        ? formatAxisTime(v, x1 - x0, timeZoneMode.value)
        : xIsCategorical.value
          ? elideMiddle(categories.value[Math.round(v)] ?? '', 14)
          : formatMetric(v),
    }
  })
})

function formatXValue(xv: number, label: string): string {
  if (props.kinds[props.x] === 'timestamp') return formatTimestampMs(xv, timeZoneMode.value)
  if (xIsCategorical.value) return label
  return formatMetric(xv)
}

/**
 * Every visible series' value at the hovered x.
 *
 * Reading one series at a time is the wrong unit: the question a multi-series
 * chart is asked is how the lines compare at an instant, which a nearest-point
 * readout can only answer one line per hover.
 */
const readout = computed(() => {
  const at = hoverX.value
  if (at === null) return null
  const entries: { name: string; color: string; value: number; point: Point }[] = []
  for (const s of visible.value) {
    const points = inView(s.points)
    if (!points.length) continue
    let best = points[0]
    for (const p of points) {
      if (Math.abs(p.xv - at) < Math.abs(best.xv - at)) best = p
    }
    entries.push({ name: s.name, color: s.color, value: best.yv, point: best })
  }
  if (!entries.length) return null
  const anchor = entries.reduce((a, b) => (Math.abs(a.point.xv - at) <= Math.abs(b.point.xv - at) ? a : b))
  return { entries, anchor, px: sx(anchor.point.xv) }
})

function pointerX(e: MouseEvent): number {
  const svg = e.currentTarget as SVGSVGElement
  const box = svg.getBoundingClientRect()
  return ((e.clientX - box.left) / box.width) * WIDTH
}

function onMove(e: MouseEvent) {
  const px = pointerX(e)
  hoverX.value = invertX(px)
  if (drag.value) drag.value = { ...drag.value, to: px }
}

function onDown(e: MouseEvent) {
  drag.value = { from: pointerX(e), to: pointerX(e) }
}

function onUp() {
  const d = drag.value
  drag.value = null
  if (!d || Math.abs(d.to - d.from) < MIN_ZOOM_PX) return
  const a = invertX(Math.min(d.from, d.to))
  const b = invertX(Math.max(d.from, d.to))
  if (b > a) zoom.value = { x0: a, x1: b }
}

function toggleSeries(name: string) {
  const next = new Set(muted.value)
  if (next.has(name)) next.delete(name)
  else if (next.size + 1 < model.value.series.length) next.add(name)
  muted.value = next
}

// A new result is a new chart; a zoom or a muted series from the last one would
// silently hide rows the reader just asked for.
watch(() => [props.rows, props.x, props.y, props.series], () => {
  zoom.value = null
  muted.value = new Set()
})

const dragRect = computed(() => {
  const d = drag.value
  if (!d || Math.abs(d.to - d.from) < MIN_ZOOM_PX) return null
  return { x: Math.min(d.from, d.to), width: Math.abs(d.to - d.from) }
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
      class="w-full h-auto select-none"
      :class="drag ? 'cursor-col-resize' : 'cursor-crosshair'"
      :viewBox="`0 0 ${WIDTH} ${HEIGHT}`"
      role="img"
      :aria-label="`${mark} chart of ${y} against ${x}`"
      @mousemove="onMove"
      @mousedown.prevent="onDown"
      @mouseup="onUp"
      @mouseleave="hoverX = null; drag = null"
      @dblclick="zoom = null"
    >
      <defs>
        <clipPath id="finelog-plot-clip">
          <rect
            :x="PAD.left" :y="PAD.top"
            :width="WIDTH - PAD.left - PAD.right" :height="HEIGHT - PAD.top - PAD.bottom"
          />
        </clipPath>
      </defs>

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

      <g clip-path="url(#finelog-plot-clip)">
        <template v-if="mark === 'bar'">
          <g v-for="(s, si) in visible" :key="s.name">
            <rect
              v-for="(p, pi) in inView(s.points)" :key="pi"
              :x="barX(p, si)" :width="barWidth"
              :y="Math.min(sy(p.yv), sy(logUsable ? bounds.y0 : Math.max(0, bounds.y0)))"
              :height="Math.max(1, Math.abs(sy(p.yv) - sy(logUsable ? bounds.y0 : Math.max(0, bounds.y0))))"
              :fill="s.color"
            />
          </g>
        </template>
        <template v-else-if="mark === 'scatter'">
          <g v-for="s in visible" :key="s.name">
            <circle
              v-for="(p, pi) in inView(s.points)" :key="pi"
              :cx="sx(p.xv)" :cy="sy(p.yv)" r="2.5" :fill="s.color" fill-opacity="0.75"
            />
          </g>
        </template>
        <template v-else>
          <!-- Smoothing hides where the samples actually fell, so the raw run
               stays on the plot behind the smoothed one. -->
          <path
            v-for="s in (smoothing > 0 ? visible : [])" :key="`raw-${s.name}`"
            :d="linePath(inView(s.raw))" fill="none" :stroke="s.color"
            stroke-width="1" stroke-opacity="0.22"
          />
          <path
            v-for="s in visible" :key="s.name"
            :d="linePath(inView(s.points))" fill="none" :stroke="s.color"
            stroke-width="1.75" stroke-linejoin="round" stroke-linecap="round"
          />
        </template>

        <rect
          v-if="dragRect"
          :x="dragRect.x" :y="PAD.top" :width="dragRect.width" :height="HEIGHT - PAD.top - PAD.bottom"
          fill="var(--c-accent)" fill-opacity="0.12" stroke="var(--c-accent)" stroke-opacity="0.4"
        />
        <g v-if="readout && !drag">
          <line
            :x1="readout.px" :x2="readout.px" :y1="PAD.top" :y2="HEIGHT - PAD.bottom"
            stroke="var(--c-text-muted)" stroke-dasharray="3 3" stroke-opacity="0.6"
          />
          <circle
            v-for="e in readout.entries" :key="e.name"
            :cx="sx(e.point.xv)" :cy="sy(e.value)" r="3.5"
            fill="var(--c-surface)" :stroke="e.color" stroke-width="1.75"
          />
        </g>
      </g>
    </svg>

    <div
      v-if="readout && !drag"
      class="pointer-events-none absolute z-10 px-2.5 py-1.5 rounded border border-surface-border bg-surface shadow-lg text-xs font-mono whitespace-nowrap"
      :style="{
        left: `${(readout.px / WIDTH) * 100}%`,
        top: '8px',
        transform: readout.px > WIDTH * 0.6 ? 'translateX(-104%)' : 'translateX(4%)',
      }"
    >
      <div class="text-text-secondary pb-0.5">{{ formatXValue(readout.anchor.point.xv, readout.anchor.point.label) }}</div>
      <div v-for="e in readout.entries" :key="e.name" class="flex items-center gap-1.5">
        <span class="w-2 h-2 rounded-full shrink-0" :style="{ backgroundColor: e.color }" />
        <span v-if="series" class="text-text-secondary">{{ e.name }}</span>
        <span class="font-semibold ml-auto pl-3">{{ formatMetric(e.value) }}</span>
      </div>
    </div>

    <div class="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 pb-3">
      <button
        v-for="s in model.series" :key="s.name"
        class="inline-flex items-center gap-1.5 text-xs"
        :class="muted.has(s.name) ? 'text-text-muted opacity-50' : 'text-text-secondary'"
        :title="muted.has(s.name) ? `Show ${s.name}` : `Hide ${s.name}`"
        @click="toggleSeries(s.name)"
      >
        <span class="w-2.5 h-2.5 rounded-full shrink-0" :style="{ backgroundColor: s.color }" />
        {{ series ? s.name : y }}
      </button>
      <span v-if="model.decimated" class="text-xs text-text-muted ml-auto">
        thinned to {{ MAX_POINTS_PER_SERIES.toLocaleString() }} points per series
      </span>
      <button
        v-if="zoom"
        class="text-xs px-2 py-0.5 rounded border border-surface-border hover:bg-surface-raised"
        :class="model.decimated ? '' : 'ml-auto'"
        @click="zoom = null"
      >Reset zoom</button>
      <span v-else-if="!model.decimated" class="text-xs text-text-muted ml-auto">Drag to zoom</span>
    </div>
  </div>
</template>
