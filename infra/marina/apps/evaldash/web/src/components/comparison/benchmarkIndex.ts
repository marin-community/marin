import type { ComparisonRow } from '@/types/api'

export type BenchmarkScope = 'all' | 'shared'

const PAGE_SIZE = 12

interface BenchmarkQuery {
  scope: string
  search: string
  page: string
  focus: string
}

function modelCoverage(row: ComparisonRow): number {
  return Object.keys(row.cells).length
}

export function orderBenchmarks(rows: readonly ComparisonRow[]): ComparisonRow[] {
  return [...rows].sort((a, b) => modelCoverage(b) - modelCoverage(a) || a.benchmark.localeCompare(b.benchmark))
}

export function benchmarkPage(ordered: readonly ComparisonRow[], query: BenchmarkQuery) {
  const shared = ordered.filter((row) => row.shared)
  const scope: BenchmarkScope = query.scope === 'all' || query.scope === 'shared'
    ? query.scope
    : shared.length ? 'shared' : 'all'
  const terms = query.search.toLowerCase().trim().split(/\s+/).filter(Boolean)
  const matches = (scope === 'shared' ? shared : ordered).filter((row) => {
    const label = row.benchmark.toLowerCase()
    return terms.every((term) => label.includes(term))
  })
  const pageCount = Math.max(1, Math.ceil(matches.length / PAGE_SIZE))
  const page = Math.min(Math.max(1, Number.parseInt(query.page, 10) || 1), pageCount)
  const offset = (page - 1) * PAGE_SIZE
  const rows = matches.slice(offset, offset + PAGE_SIZE)
  const focused = query.focus ? matches.find((row) => row.benchmark === query.focus) : rows[0]

  return { scope, rows, focused, page, pageCount, offset, total: matches.length, allCount: ordered.length, sharedCount: shared.length }
}
