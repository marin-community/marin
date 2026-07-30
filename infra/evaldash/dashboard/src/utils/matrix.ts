import type { MatrixCell, MatrixRow } from '@/types/api'

// The fleet best score on one benchmark and the model that holds it — the rail caret and the
// matrix column marker.
export interface BestCell {
  value: number
  model: string
}

// Index matrix rows by model name for O(1) cell lookup. Used wherever a page reads a specific
// model's cells out of the fleet matrix (leaderboard, model index, compare).
export function cellsByModel(rows: MatrixRow[]): Record<string, Record<string, MatrixCell>> {
  const out: Record<string, Record<string, MatrixCell>> = {}
  for (const row of rows) out[row.model] = row.cells
  return out
}

// For each task, the highest-scoring model and its score across the given rows.
export function fleetBest(rows: MatrixRow[], tasks: string[]): Record<string, BestCell> {
  const out: Record<string, BestCell> = {}
  for (const task of tasks) {
    let bestValue = -Infinity
    let bestModel = ''
    for (const row of rows) {
      const cell = row.cells[task]
      if (cell && cell.value !== null && cell.value > bestValue) {
        bestValue = cell.value
        bestModel = row.model
      }
    }
    if (bestModel) out[task] = { value: bestValue, model: bestModel }
  }
  return out
}
