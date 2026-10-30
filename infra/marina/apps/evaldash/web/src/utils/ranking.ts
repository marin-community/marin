import type { PanelCell, PanelRow } from '../types/api'

// Higher scores lead; the lower interval bound breaks score ties. The panel's column sort and
// fleet-best marker use this ordering too.
export function compareCells(a: PanelCell, b: PanelCell): number {
  return b.value - a.value || b.low - a.low
}

/** Mean reciprocal rank over each model's completed results on the selected benchmarks. */
export function meanReciprocalRanks(rows: PanelRow[], benchmarks: string[]): Map<string, number> {
  const totals = new Map<string, { sum: number; count: number }>()
  for (const benchmark of benchmarks) {
    const ranked = rows
      .filter((row) => row.cells[benchmark])
      .sort((a, b) => compareCells(a.cells[benchmark], b.cells[benchmark]))
    let rank = 0
    for (let index = 0; index < ranked.length; index++) {
      if (index === 0 || compareCells(ranked[index - 1].cells[benchmark], ranked[index].cells[benchmark]) !== 0) {
        rank = index + 1
      }
      const model = ranked[index].model
      const total = totals.get(model) ?? { sum: 0, count: 0 }
      total.sum += 1 / rank
      total.count++
      totals.set(model, total)
    }
  }
  return new Map([...totals].map(([model, { sum, count }]) => [model, sum / count]))
}
