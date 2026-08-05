/** Formatters for finelog dashboard cells. */

import type { TimeZoneMode } from '@/composables/useDisplayPrefs'

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

/** Local-time parts of `ms`, in `zone` — ICU handles the offset and DST. */
function zonedParts(ms: number, zone: string): Record<string, string> {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: zone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).formatToParts(new Date(ms))
  return Object.fromEntries(parts.map((p) => [p.type, p.value]))
}

/**
 * Render epoch milliseconds as `YYYY-MM-DD HH:MM:SS.mmm`, in the viewer's zone
 * or UTC. `raw` returns the integer unchanged, which is what you paste back into
 * a SQL predicate.
 *
 * Millisecond precision is kept because finelog's own timestamps are
 * millisecond-resolution and log ordering within a second is often the question
 * being asked.
 */
export function formatTimestampMs(ms: number | null | undefined, mode: TimeZoneMode = 'utc'): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '—'
  if (mode === 'raw') return String(ms)
  const zone = mode === 'utc' ? 'UTC' : Intl.DateTimeFormat().resolvedOptions().timeZone
  const p = zonedParts(ms, zone)
  const millis = String(Math.abs(Math.trunc(ms)) % 1000).padStart(3, '0')
  return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}:${p.second}.${millis}`
}

/** Short axis label for `ms` — time of day, with the date when the span crosses one. */
export function formatAxisTime(ms: number, spanMs: number, mode: TimeZoneMode): string {
  const zone = mode === 'utc' || mode === 'raw' ? 'UTC' : Intl.DateTimeFormat().resolvedOptions().timeZone
  const p = zonedParts(ms, zone)
  if (spanMs > 24 * 60 * 60 * 1000) return `${p.month}-${p.day} ${p.hour}:${p.minute}`
  return `${p.hour}:${p.minute}`
}

/** The zone abbreviation shown next to a timestamp control. */
export function timeZoneLabel(mode: TimeZoneMode): string {
  if (mode === 'utc') return 'UTC'
  if (mode === 'raw') return 'epoch ms'
  return Intl.DateTimeFormat().resolvedOptions().timeZone
}

/**
 * Render a measurement compactly: thousands separators for integers, and at most
 * four significant decimals so a float64 metric does not print 17 digits.
 */
export function formatMetric(value: number): string {
  if (!Number.isFinite(value)) return String(value)
  if (Number.isInteger(value)) return value.toLocaleString()
  const magnitude = Math.abs(value)
  if (magnitude !== 0 && (magnitude >= 1e9 || magnitude < 1e-4)) return value.toExponential(3)
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 })
}
