import type { ProtoTimestamp } from '@/types/rpc'

/** Parse a ProtoTimestamp to epoch milliseconds. */
export function timestampMs(ts?: ProtoTimestamp): number {
  if (!ts?.epochMs) return 0
  return parseInt(ts.epochMs, 10) || 0
}

/** Format epoch ms as a locale date/time string. */
export function formatTimestamp(ts?: ProtoTimestamp): string {
  const ms = timestampMs(ts)
  if (!ms) return '-'
  return new Date(ms).toLocaleString()
}

/** Format epoch ms as relative time ("5s ago", "3m ago", etc). */
export function formatRelativeTime(ms: number): string {
  if (!ms) return '-'
  const seconds = Math.floor((Date.now() - ms) / 1000)
  if (seconds < 0) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

/** Format a byte count as "1.5 GB", "200 MB", etc. */
export function formatBytes(bytes: number): string {
  if (!bytes || bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const val = bytes / Math.pow(1024, i)
  return (val >= 100 ? Math.round(val) : val.toFixed(1)) + ' ' + units[i]
}

/** Format CPU millicores as "750m", "1.2c", etc. */
export function formatCpuMillicores(millicores?: number): string {
  if (!millicores) return '-'
  if (millicores < 1000) return `${millicores}m`
  const cores = millicores / 1000
  return Number.isInteger(cores) ? `${cores}c` : `${cores.toFixed(1)}c`
}

/** Format a byte rate as "1.5 MB/s", etc. */
export function formatRate(bytesPerSec: number): string {
  if (!bytesPerSec) return '0 B/s'
  const units = ['B/s', 'KB/s', 'MB/s', 'GB/s']
  const i = Math.min(Math.floor(Math.log(bytesPerSec) / Math.log(1024)), units.length - 1)
  const val = bytesPerSec / Math.pow(1024, i)
  return (val >= 100 ? Math.round(val) : val.toFixed(1)) + ' ' + units[i]
}

/** Format duration between two epoch-ms timestamps. endMs defaults to now. */
export function formatDuration(startMs: number, endMs?: number): string {
  if (!startMs) return '-'
  const end = endMs || Date.now()
  const diffSec = Math.floor((end - startMs) / 1000)
  if (diffSec < 0) return '-'
  if (diffSec < 60) return `${diffSec}s`
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ${diffSec % 60}s`
  const hours = Math.floor(diffSec / 3600)
  const mins = Math.floor((diffSec % 3600) / 60)
  return `${hours}h ${mins}m`
}

/**
 * Which clock a log panel renders timestamps on and reads typed instants
 * against. UTC matches the timestamps most processes embed in their raw log
 * lines, so the rendered prefix lines up with the line text.
 */
export type TimeZoneName = 'local' | 'utc'

/** "HH:MM:SS.mmm" for a date already resolved, with no absent-value guard. */
function clockTime(d: Date, zone: TimeZoneName): string {
  const utc = zone === 'utc'
  const hh = String(utc ? d.getUTCHours() : d.getHours()).padStart(2, '0')
  const mm = String(utc ? d.getUTCMinutes() : d.getMinutes()).padStart(2, '0')
  const ss = String(utc ? d.getUTCSeconds() : d.getSeconds()).padStart(2, '0')
  const ms = String(utc ? d.getUTCMilliseconds() : d.getMilliseconds()).padStart(3, '0')
  return `${hh}:${mm}:${ss}.${ms}`
}

/** Format epoch ms as "HH:MM:SS.mmm" on `zone`'s clock; blank when absent. */
export function formatLogTime(epochMs: number, zone: TimeZoneName = 'local'): string {
  if (!epochMs) return ''
  return clockTime(new Date(epochMs), zone)
}

/**
 * Format epoch ms as "YYYY-MM-DD HH:MM:SS.mmm" on `zone`'s clock.
 *
 * Unlike `formatLogTime` this has no absent-value guard: it labels a bound the
 * operator chose, where epoch 0 is a real instant rather than a missing one.
 */
export function formatLogTimestamp(epochMs: number, zone: TimeZoneName): string {
  const utc = zone === 'utc'
  const d = new Date(epochMs)
  const year = String(utc ? d.getUTCFullYear() : d.getFullYear()).padStart(4, '0')
  const month = String((utc ? d.getUTCMonth() : d.getMonth()) + 1).padStart(2, '0')
  const day = String(utc ? d.getUTCDate() : d.getDate()).padStart(2, '0')
  return `${year}-${month}-${day} ${clockTime(d, zone)}`
}

// Every field is optional so that a date alone, a time alone, or both parse. The
// separators are loose because the point is to accept whatever an operator
// copied: a rendered dashboard timestamp, an ISO string, a glog datestamp.
const LOG_TIMESTAMP = /^(?:(\d{4})-?(\d{2})-?(\d{2}))?(?:[ T]*(\d{1,2}):(\d{2})(?::(\d{2}))?(?:[.,](\d{1,3}))?)?\s*(Z|UTC)?$/i

/**
 * Read a typed or pasted instant as epoch ms, or null when it makes no sense.
 *
 * Accepts a date, a time, or both, plus bare epoch seconds and milliseconds. A
 * value with no date is read as today and one with no time as midnight, both on
 * `zone`'s clock; a trailing `Z` overrides `zone` and forces UTC.
 */
export function parseLogTimestamp(value: string, zone: TimeZoneName): number | null {
  const text = value.trim()
  if (!text) return null
  if (/^\d{10}$/.test(text)) return Number(text) * 1000
  if (/^\d{13}$/.test(text)) return Number(text)

  const match = LOG_TIMESTAMP.exec(text)
  if (!match) return null
  // Destructuring defaults stand in for the unmatched groups, which TypeScript
  // types as `string` even though the runtime leaves them undefined.
  const [, year = '', month = '', day = '', hour = '', minute = '', second = '', fraction = '', utcMark = ''] = match
  // Every group is optional, so a string of only separators matches nothing.
  if (!year && !hour) return null

  const utc = zone === 'utc' || utcMark !== ''
  const now = new Date()
  const y = year ? Number(year) : utc ? now.getUTCFullYear() : now.getFullYear()
  const mo = month ? Number(month) : (utc ? now.getUTCMonth() : now.getMonth()) + 1
  const d = day ? Number(day) : utc ? now.getUTCDate() : now.getDate()
  const h = hour ? Number(hour) : 0
  const mi = minute ? Number(minute) : 0
  const s = second ? Number(second) : 0
  const ms = fraction ? Number(fraction.padEnd(3, '0')) : 0
  // Reject out-of-range fields rather than letting Date roll them over, so a
  // typo lands as "unrecognized" instead of silently moving the window.
  if (mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || s > 59) return null
  return utc ? Date.UTC(y, mo - 1, d, h, mi, s, ms) : new Date(y, mo - 1, d, h, mi, s, ms).getTime()
}

/** Format an uptime duration in milliseconds as a human-readable string. */
export function formatUptime(uptimeMs?: string): string {
  if (!uptimeMs) return '-'
  const ms = parseInt(uptimeMs, 10)
  if (!ms) return '-'
  const seconds = Math.floor(ms / 1000)
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  const hours = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  if (hours < 24) return `${hours}h ${mins}m`
  return `${Math.floor(hours / 24)}d ${hours % 24}h`
}

/** The severity names a `LogEntry.level` can normalize to. */
export type LogLevelName = 'unknown' | 'debug' | 'info' | 'warning' | 'error' | 'critical'

/**
 * Normalize a wire `LogLevel` to its bare severity name.
 *
 * proto3 JSON renders an enum as its declared name, so the wire carries
 * `"LOG_LEVEL_ERROR"`, not `"ERROR"`. Levels finelog could not parse from the
 * line arrive as `LOG_LEVEL_UNKNOWN` and normalize to `'unknown'`.
 */
export function logLevelName(level: string | undefined): LogLevelName {
  const name = (level ?? '').toLowerCase().replace(/^log_level_/, '')
  switch (name) {
    case 'debug':
    case 'info':
    case 'warning':
    case 'error':
    case 'critical': return name
    default: return 'unknown'
  }
}

/** Text color class for a log level string. */
export function logLevelClass(level: string | undefined): string {
  switch (logLevelName(level)) {
    case 'debug': return 'text-text-muted'
    case 'warning': return 'text-status-warning'
    case 'error':
    case 'critical': return 'text-status-danger'
    default: return 'text-text'
  }
}

/** Format a worker's device metadata as a human-readable string. */
export function formatWorkerDevice(metadata: { gpuCount?: number; gpuName?: string; gpuMemoryMb?: number; device?: { tpu?: { variant?: string }; gpu?: { count?: number; variant?: string } } } | null | undefined): string {
  if (!metadata) return 'CPU'
  if (metadata.gpuCount && metadata.gpuCount > 0) {
    const name = metadata.gpuName || 'GPU'
    const mem = metadata.gpuMemoryMb ? ` (${Math.round(metadata.gpuMemoryMb / 1024)}GB)` : ''
    return `GPU: ${metadata.gpuCount}x ${name}${mem}`
  }
  if (metadata.device?.tpu) return `TPU: ${metadata.device.tpu.variant || 'unknown'}`
  if (metadata.device?.gpu) return `GPU: ${metadata.device.gpu.count || 1}x ${metadata.device.gpu.variant || 'unknown'}`
  return 'CPU'
}

/** Human-readable name for a `PRIORITY_BAND_*` enum value. */
export function bandDisplayName(band: string | undefined): string {
  if (!band) return 'Unknown'
  const name = band.replace(/^PRIORITY_BAND_/, '')
  return name.charAt(0) + name.slice(1).toLowerCase()
}

/** Tailwind text color class for a `PRIORITY_BAND_*` enum value. */
export function bandColor(band: string | undefined): string {
  if (!band) return 'text-text-muted'
  const name = band.replace(/^PRIORITY_BAND_/, '')
  if (name === 'PRODUCTION') return 'text-status-danger'
  if (name === 'INTERACTIVE') return 'text-accent'
  if (name === 'BATCH') return 'text-text-muted'
  return 'text-text-muted'
}

/** Format a DeviceConfig proto as a human-readable string. */
export function formatDeviceConfig(device: { tpu?: { variant?: string; topology?: string; count?: number }; gpu?: { variant?: string; count?: number; memoryGb?: number } } | null | undefined): string | null {
  if (!device) return null
  if (device.tpu) {
    let s = device.tpu.variant ?? 'tpu'
    if (device.tpu.topology) s += ` (${device.tpu.topology})`
    if (device.tpu.count) s += ` x${device.tpu.count}`
    return s
  }
  if (device.gpu) {
    let s = device.gpu.variant ?? 'gpu'
    if (device.gpu.count) s += ` x${device.gpu.count}`
    if (device.gpu.memoryGb) s += ` (${device.gpu.memoryGb}GB)`
    return s
  }
  return null
}
