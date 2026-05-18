/** Formatters for finelog dashboard cells. */

export function formatRelativeTime(ms: number): string {
  if (!ms) return '—'
  const delta = Date.now() - ms
  const sec = Math.floor(delta / 1000)
  if (sec < 5) return 'just now'
  if (sec < 60) return `${sec}s ago`
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min}m ago`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr}h ago`
  const day = Math.floor(hr / 24)
  return `${day}d ago`
}

export function formatTimestampMs(ms: number): string {
  if (!ms) return '—'
  const d = new Date(ms)
  return d.toISOString().replace('T', ' ').replace(/\..+/, '')
}

export function formatNumber(n: number | undefined | null): string {
  if (n === null || n === undefined) return '—'
  return n.toLocaleString()
}

export function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let i = 0
  let v = bytes
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`
}
