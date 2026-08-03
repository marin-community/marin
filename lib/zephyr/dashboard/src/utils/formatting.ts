export function formatCount(value: number | undefined): string {
  return new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(value ?? 0)
}

export function formatNumber(value: number | undefined, digits = 1): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value ?? 0)
}

export function formatBytes(value: number | undefined): string {
  let amount = value ?? 0
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let unit = 0
  while (Math.abs(amount) >= 1024 && unit < units.length - 1) {
    amount /= 1024
    unit += 1
  }
  return `${formatNumber(amount, amount < 10 ? 1 : 0)} ${units[unit]}`
}

export function formatDuration(milliseconds: number): string {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`
  const hours = Math.floor(minutes / 60)
  return `${hours}h ${minutes % 60}m`
}

export function formatState(value: string | undefined): string {
  return (value || 'unknown').replace(/_/g, ' ')
}

export function irisTaskHref(taskId: string): string {
  const slash = taskId.lastIndexOf('/')
  const jobId = slash > 0 ? taskId.slice(0, slash) : taskId
  return `${window.location.origin}/#/job/${encodeURIComponent(jobId)}/task/${encodeURIComponent(taskId)}`
}
