export function fmtBytes(n: number | null | undefined): string {
  if (n == null) return '?'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let size = n
  let i = 0
  while (size >= 1024 && i < units.length - 1) {
    size /= 1024
    i++
  }
  return `${i === 0 ? size : size.toFixed(1)} ${units[i]}`
}

export function fmtDuration(ms: number | null | undefined): string {
  if (ms == null) return '?'
  if (ms < 1000) return `${ms} ms`
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`
}

export function fmtCell(cell: unknown): string {
  return cell === null || cell === undefined ? 'NULL' : String(cell)
}
