import type { PanelCell } from '@/types/api'

/** Return the metric name if all cells agree on name and kind, or null for an empty or mixed list. */
export function sharedMetric(cells: readonly Pick<PanelCell, 'metric' | 'metric_kind'>[]): string | null {
  const first = cells[0]
  if (!first || cells.some((cell) => cell.metric !== first.metric || cell.metric_kind !== first.metric_kind)) return null
  return first.metric
}
