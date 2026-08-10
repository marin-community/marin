import { INTERVAL_KIND, type PanelCell, type PanelRow } from '@/types/api'

// The fleet best on one benchmark and the model that holds it — the rail caret and the panel column
// marker. Ranked by the interval's lower bound, like every other ordering in the app: a run that
// lost items cannot take the crown on the strength of the items it kept.
export interface BestCell {
  value: number
  model: string
}

// Index panel rows by model name for O(1) cell lookup. Used wherever a page reads a specific model's
// cells out of the fleet panel (panel page, model index, compare).
export function cellsByModel(rows: PanelRow[]): Record<string, Record<string, PanelCell>> {
  const out: Record<string, Record<string, PanelCell>> = {}
  for (const row of rows) out[row.model] = row.cells
  return out
}

// For each benchmark, the model with the highest interval lower bound across the given rows.
export function fleetBest(rows: PanelRow[], benchmarks: string[]): Record<string, BestCell> {
  const out: Record<string, BestCell> = {}
  for (const benchmark of benchmarks) {
    let bestBound = -Infinity
    let bestModel = ''
    let bestValue = 0
    for (const row of rows) {
      const cell = row.cells[benchmark]
      if (cell && cell.low > bestBound) {
        bestBound = cell.low
        bestValue = cell.value
        bestModel = row.model
      }
    }
    if (bestModel) out[benchmark] = { value: bestValue, model: bestModel }
  }
  return out
}

// Whether a cell's run reports it graded less than it attempted. False when no attempted count was
// reported at all: that is unknown coverage, not partial coverage, and the cell's interval kind
// already says so.
export function isPartialCoverage(cell: PanelCell): boolean {
  return cell.interval_kind === INTERVAL_KIND.IDENTIFIED && cell.coverage !== null && cell.coverage < 1
}
