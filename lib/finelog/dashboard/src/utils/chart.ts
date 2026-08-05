/**
 * Vocabulary and maths shared between the chart and the controls that configure
 * it.
 *
 * A Vue `<script setup>` block cannot export runtime values, so a constant the
 * chart and its pickers must agree on lives here rather than in the component —
 * as does anything worth testing without mounting a component.
 */

export type ChartMark = 'line' | 'bar' | 'scatter'

/**
 * Series a chart will draw. Above this a legend stops being readable and the
 * palette starts repeating colours, so a column with more distinct values is
 * not offered as a series split and the chart refuses to plot one.
 */
export const MAX_SERIES = 12

/**
 * Points one series draws before it is decimated.
 *
 * A raw `SELECT` over a metric can return tens of thousands of rows per series,
 * which is more marks than the plot has pixels: the excess costs layout time and
 * draws nothing the reader can see.
 */
export const MAX_POINTS_PER_SERIES = 2_000

/** A plottable point, reduced to what the maths here needs. */
export interface XY {
  xv: number
  yv: number
}

/**
 * Thin `points` to at most `max`, keeping the extremes.
 *
 * Even-stride sampling would drop the one-sample spike that is usually the
 * reason someone is looking at the chart. Splitting the run into buckets and
 * keeping each bucket's minimum and maximum preserves the envelope, so a
 * decimated line still reaches every peak and trough the full one does.
 *
 * `points` must already be sorted by `xv`.
 */
export function decimate<T extends XY>(points: T[], max: number = MAX_POINTS_PER_SERIES): T[] {
  if (points.length <= max) return points
  const buckets = Math.max(1, Math.floor(max / 2))
  const perBucket = points.length / buckets
  const out: T[] = []
  for (let b = 0; b < buckets; b++) {
    const start = Math.floor(b * perBucket)
    const end = Math.min(points.length, Math.floor((b + 1) * perBucket))
    if (start >= end) continue
    let lo = points[start]
    let hi = points[start]
    for (let i = start + 1; i < end; i++) {
      if (points[i].yv < lo.yv) lo = points[i]
      if (points[i].yv > hi.yv) hi = points[i]
    }
    // Emit in x order so the line does not double back on itself.
    if (lo === hi) out.push(lo)
    else if (lo.xv <= hi.xv) out.push(lo, hi)
    else out.push(hi, lo)
  }
  return out
}

/**
 * Exponential moving average of `points`, with `weight` in `[0, 1)`.
 *
 * Debiased, so the curve starts at the first sample instead of being dragged up
 * from zero — an undebiased EMA makes every metric look like it ramped from
 * nothing, which is exactly the artifact a reader would mistake for a warmup.
 * `weight` 0 returns the input unchanged.
 */
export function smooth<T extends XY>(points: T[], weight: number): T[] {
  if (weight <= 0 || points.length === 0) return points
  const w = Math.min(weight, 0.999)
  let last = 0
  let debias = 0
  return points.map((p) => {
    last = last * w + (1 - w) * p.yv
    debias = debias * w + (1 - w)
    return { ...p, yv: debias > 0 ? last / debias : p.yv }
  })
}

/**
 * Round gridline values covering `[lo, hi]` — steps of 1, 2, or 5 times a power
 * of ten. Interpolating the exact bounds instead gives labels like 16.289 and
 * 12.2169, which are precise and unreadable.
 */
export function niceTicks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo
  if (!(span > 0)) return [lo]
  const rough = span / count
  const magnitude = 10 ** Math.floor(Math.log10(rough))
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= rough) ?? 10 * magnitude
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v)
  return out.length >= 2 ? out : [lo, hi]
}

/**
 * Gridline values for a log axis covering `[lo, hi]`: every power of ten in
 * range, subdivided by 2 and 5 when the span is narrow enough that decades
 * alone would leave one or two lines on the plot.
 *
 * A range inside a single decade has no round power of ten to land on, so it
 * borrows the linear axis's steps — still positioned logarithmically, but
 * labelled 4,500 and 5,000 rather than with the raw bounds.
 */
export function logTicks(lo: number, hi: number): number[] {
  if (!(lo > 0) || !(hi > lo)) return niceTicks(lo, hi, 4)
  const first = Math.floor(Math.log10(lo))
  const last = Math.ceil(Math.log10(hi))
  // Decades alone leave a three-decade range with two or three gridlines, which
  // is too few to read a value off. Subdivide until the span is wide enough that
  // powers of ten carry the axis on their own.
  const mantissas = last - first <= 3 ? [1, 2, 5] : [1]
  const out: number[] = []
  for (let decade = first; decade <= last; decade++) {
    for (const m of mantissas) {
      const v = m * 10 ** decade
      if (v >= lo && v <= hi) out.push(v)
    }
  }
  return out.length >= 2 ? out : niceTicks(lo, hi, 4)
}
