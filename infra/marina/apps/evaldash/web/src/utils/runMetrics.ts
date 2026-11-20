import type { EvalRecord } from '@/types/api'

export interface MetricRow {
  task: string
  metric: string
  value: number
}

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
 * The run page's metric tables: the API's resolved rows (declared protocol metrics first, then any
 * undeclared task's raw rows) and the raw map, which gets a folded section when a row was declared.
 */
export function metricView(record: EvalRecord): { rows: MetricRow[]; raw: MetricRow[]; hasDeclared: boolean } {
  const raw = rawMetricRows(record)
  const rows = record.metric_rows.map(({ task, metric, value }) => ({ task, metric, value }))
  return { rows, raw, hasDeclared: record.metric_rows.some((row) => row.declared) }
}
