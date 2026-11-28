const numberFormat = new Intl.NumberFormat()

export function formatPageRange(offset: number, count: number, total: number): string {
  const totalLabel = numberFormat.format(total)
  if (count === 0) return totalLabel
  const start = numberFormat.format(offset + 1)
  const end = numberFormat.format(offset + count)
  return `${start}–${end} of ${totalLabel}`
}

export function lastPageIndex(total: number, pageSize: number): number {
  return Math.max(0, Math.ceil(total / pageSize) - 1)
}
