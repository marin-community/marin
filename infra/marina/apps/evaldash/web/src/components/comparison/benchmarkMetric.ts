import type { PanelCell } from '@/types/api'

export function sharedMetric(cells: readonly Pick<PanelCell, 'metric' | 'metric_kind'>[]): string | null {
  const first = cells[0]
  if (!first || cells.some((cell) => cell.metric !== first.metric || cell.metric_kind !== first.metric_kind)) return null
  return first.metric
}
