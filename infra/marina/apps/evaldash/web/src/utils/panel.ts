import { INTERVAL_KIND, type IntervalKind, type PanelCell, type PanelRow } from '@/types/api'

// The fleet best on one benchmark and the model that holds it — the rail caret and the panel column
// marker. Ranked by the score, as the panel sorts; Compare is where an ordering claim is made and it
// ranks on the interval instead.
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

// Orders two cells on one benchmark: the higher score first, and on a tie the higher lower bound. The
// panel's column sort and the fleet-best marker both use this rule, so they cannot disagree.
export function compareCells(a: PanelCell, b: PanelCell): number {
  return b.value - a.value || b.low - a.low
}

// For each benchmark, the model whose cell leads under compareCells across the given rows.
export function fleetBest(rows: PanelRow[], benchmarks: string[]): Record<string, BestCell> {
  const out: Record<string, BestCell> = {}
  for (const benchmark of benchmarks) {
    let best: { cell: PanelCell; model: string } | null = null
    for (const row of rows) {
      const cell = row.cells[benchmark]
      if (cell && (best === null || compareCells(cell, best.cell) < 0)) best = { cell, model: row.model }
    }
    if (best) out[benchmark] = { value: best.cell.value, model: best.model }
  }
  return out
}

export function withVariant(selected: Iterable<string>, variants: string[], chosen: string): Set<string> {
  const next = new Set(selected)
  for (const name of variants) next.delete(name)
  next.add(chosen)
  return next
}

// The two fields any coverage question needs. A panel cell, a history point, and a run headline all
// carry them, so the one predicate below serves every view instead of each hand-rolling it.
export interface Covered {
  interval_kind: IntervalKind
  coverage: number | null
}

// Whether a run reports it graded less than it attempted. False when no attempted count was reported
// at all: that is unknown coverage, not partial coverage, and the interval kind already says so.
export function isPartialCoverage(cell: Covered): boolean {
  return cell.interval_kind === INTERVAL_KIND.IDENTIFIED && cell.coverage !== null && cell.coverage < 1
}
