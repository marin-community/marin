// Small presentational helpers shared across pages.

export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

// Aggregates and undeclared metrics can lack a kind. Their scores stay in raw units.
interface MetricUnits {
  metric_kind?: string
}

export function isPercentMetric(cell: MetricUnits): boolean {
  return cell.metric_kind === 'binary'
}

function formatMetricValue(value: number, cell: MetricUnits): string {
  return isPercentMetric(cell) ? (100 * value).toFixed(1) : value.toFixed(3)
}

export function formatScore(cell: MetricUnits & { value: number }): string {
  return formatMetricValue(cell.value, cell) + (isPercentMetric(cell) ? '%' : '')
}

export function formatInterval(cell: MetricUnits & { low: number; high: number }): string {
  return `${formatMetricValue(cell.low, cell)}–${formatMetricValue(cell.high, cell)}${isPercentMetric(cell) ? '%' : ''}`
}

// Signed differences use percentage points for binary metrics and source units otherwise.
export function formatDelta(value: number, cell: MetricUnits): string {
  return (value > 0 ? '+' : '') + formatMetricValue(value, cell)
}

export function shortSha(sha: string | null | undefined): string {
  if (!sha) return '—'
  return sha.length > 10 ? sha.slice(0, 10) : sha
}

// A browsable Cloud Console URL for a gs:// object path, or null for anything else (local dev paths,
// or s3:// CoreWeave paths whose console is not public). Used to linkify results/weight locations.
export function objectStoreUrl(path: string | null | undefined): string | null {
  if (!path || !path.startsWith('gs://')) return null
  return `https://console.cloud.google.com/storage/browser/${path.slice('gs://'.length)}`
}

// The GitHub commit page for a marin git SHA, or null when absent.
export function githubCommitUrl(sha: string | null | undefined): string | null {
  return sha ? `https://github.com/marin-community/marin/commit/${sha}` : null
}

export function irisJobUrl(path: string): string {
  return `https://iris.oa.dev/#/job/${encodeURIComponent(path)}`
}

export function formatRelativeAge(iso: string | null | undefined): string {
  if (!iso) return 'never'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return iso
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000))
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

export function formatClock(d: Date): string {
  return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

// The share of attempted items a run graded, e.g. '92.5% graded', or '' when it reports no attempted
// count at all (in which case completeness is unknown, not complete).
export function formatCoverage(coverage: number | null | undefined): string {
  if (coverage === null || coverage === undefined || Number.isNaN(coverage)) return ''
  return `${(coverage * 100).toFixed(1)}% graded`
}

// Epoch-milliseconds timestamp (iris/finelog) -> short local time, or '—'.
export function formatMillis(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return '—'
  return formatTimestamp(new Date(ms).toISOString())
}
export function protoTimestampMillis(
  timestamp: { epoch_ms?: string | number } | null | undefined,
): number | null {
  return timestamp?.epoch_ms === undefined ? null : Number(timestamp.epoch_ms)
}

export function formatDuration(startMs: number | null | undefined, endMs: number | null | undefined): string {
  if (!startMs) return '—'
  const end = endMs ?? Date.now()
  const seconds = Math.max(0, Math.round((end - startMs) / 1000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`
  const hours = Math.floor(minutes / 60)
  return `${hours}h ${minutes % 60}m`
}
