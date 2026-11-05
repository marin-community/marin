// Shaping of a run's metric tables: the rows the API resolved for the top table, and the
// evaluator's raw metric map kept behind a fold.

import type { EvalRecord } from '@/types/api'

export interface MetricRow {
  task: string
  metric: string
  value: number
}

/** Every raw `(task, metric, value)` in the record's evaluator metric map, sorted for stable display. */
export function rawMetricRows(record: EvalRecord): MetricRow[] {
  const rows: MetricRow[] = []
  for (const [task, metrics] of Object.entries(record.metrics)) {
    for (const [metric, value] of Object.entries(metrics)) {
      rows.push({ task, metric, value })
    }
  }
  return rows.sort((a, b) => a.task.localeCompare(b.task) || a.metric.localeCompare(b.metric))
}

/**
 * The run page's metric tables.
 *
 * ``rows`` is the API's resolved top table: a benchmark's declared protocol metrics, primary first,
 * followed by the raw rows of tasks without a declaration. ``hasDeclared`` says whether any row came
 * from a declaration, which is when the raw metric map also deserves its folded section.
 */
export function metricView(record: EvalRecord): { rows: MetricRow[]; raw: MetricRow[]; hasDeclared: boolean } {
  const raw = rawMetricRows(record)
  const rows = record.metric_rows.map(({ task, metric, value }) => ({ task, metric, value }))
  return { rows, raw, hasDeclared: record.metric_rows.some((row) => row.declared) }
}
