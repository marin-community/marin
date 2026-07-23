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

export function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  // Accuracy-style metrics live in [0, 1]; show as a percentage. Anything else
  // (already a percentage, a count) is shown with three significant decimals.
  if (value >= 0 && value <= 1) return (value * 100).toFixed(1)
  return value.toFixed(3)
}

export function shortSha(sha: string | null | undefined): string {
  if (!sha) return '—'
  return sha.length > 10 ? sha.slice(0, 10) : sha
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

// A ±standard-error string in the same percentage units as formatScore, or '' when absent.
export function formatStderr(value: number | null | undefined, stderr: number | null | undefined): string {
  if (stderr === null || stderr === undefined || Number.isNaN(stderr)) return ''
  const scaled = value !== null && value !== undefined && value >= 0 && value <= 1 ? stderr * 100 : stderr
  return `±${scaled.toFixed(1)}`
}

// A signed delta against a reference score (e.g. a leaderboard leader), in the same
// percentage-point scale formatScore displays, e.g. '-3.1'. Empty string when there is
// nothing to compare (no value, or the value is the reference itself).
export function formatDelta(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return ''
  const scaled = value * 100
  return `${scaled > 0 ? '+' : ''}${scaled.toFixed(1)}`
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
