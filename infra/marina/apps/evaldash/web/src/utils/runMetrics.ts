// Shaping of a run's metric tables: the declared protocol rows a benchmark reports, and the
// evaluator's raw metric map kept behind a fold.

import type { BenchmarkMetric, EvalRecord, EvalTask } from '@/types/api'

export interface MetricRow {
  task: string
  metric: string
  value: number
}

/** Strip lm-eval's `,<filter>` suffix from a raw metric key. */
function baseMetric(key: string): string {
  return key.split(',', 1)[0]
}

/**
 * Whether one metrics-map key belongs to a declared task.
 *
 * Mirrors the evaluator's naming: a single-task record owns every key, and otherwise a key names the
 * benchmark (or ends in it), or carries the task's own name or directory prefix.
 */
function keyBelongsToTask(record: EvalRecord, key: string, task: EvalTask): boolean {
  if (record.eval.tasks.length <= 1) return true
  const leaf = key.includes('/') ? key.slice(key.lastIndexOf('/') + 1) : key
  const declared = task.benchmark?.task
  if (declared !== undefined && (key === declared || leaf === declared)) return true
  return key === task.name || leaf === task.name || key.startsWith(`${task.name}/`)
}

/** The benchmark's declared value for one metric: canonical copy first, then the raw source metric. */
function declaredValue(record: EvalRecord, task: EvalTask, metric: BenchmarkMetric): number | null {
  for (const [key, values] of Object.entries(record.canonical_metrics ?? {})) {
    if (!keyBelongsToTask(record, key, task)) continue
    if (metric.name in values) return values[metric.name]
  }
  for (const [key, values] of Object.entries(record.metrics ?? {})) {
    if (!keyBelongsToTask(record, key, task)) continue
    for (const [source, value] of Object.entries(values)) {
      if (baseMetric(source) === metric.source_name) return value
    }
  }
  return null
}

/**
 * The declared protocol metrics for every task, primary first, in declaration order.
 *
 * A metric with no resolvable value is left out rather than shown as a guess: the page already
 * explains a missing grade elsewhere, and a fabricated cell would hide the gap.
 */
export function declaredMetricRows(record: EvalRecord): MetricRow[] {
  const rows: MetricRow[] = []
  for (const task of record.eval.tasks) {
    const benchmark = task.benchmark
    if (benchmark === null || benchmark === undefined) continue
    const ordered = [
      ...benchmark.metrics.filter((metric) => metric.name === benchmark.primary_metric),
      ...benchmark.metrics.filter((metric) => metric.name !== benchmark.primary_metric),
    ]
    for (const metric of ordered) {
      const value = declaredValue(record, task, metric)
      if (value !== null) rows.push({ task: task.name, metric: metric.name, value })
    }
  }
  return rows
}

/** Every raw `(task, metric, value)` in the record's evaluator metric map, sorted for stable display. */
export function rawMetricRows(record: EvalRecord): MetricRow[] {
  const rows: MetricRow[] = []
  for (const [task, metrics] of Object.entries(record.metrics ?? {})) {
    for (const [metric, value] of Object.entries(metrics)) {
      rows.push({ task, metric, value })
    }
  }
  return rows.sort((a, b) => a.task.localeCompare(b.task) || a.metric.localeCompare(b.metric))
}

/**
 * The run page's metric tables: the declared protocol rows when a record declares any, plus the
 * evaluator's raw rows for the folded "all raw metrics" section (or as the fallback table).
 */
export function metricView(record: EvalRecord): { declared: MetricRow[]; raw: MetricRow[] } {
  return { declared: declaredMetricRows(record), raw: rawMetricRows(record) }
}
