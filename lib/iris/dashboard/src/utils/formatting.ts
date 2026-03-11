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

/** Format epoch ms as "HH:MM:SS.mmm". */
export function formatLogTime(epochMs: number): string {
  if (!epochMs) return ''
  const d = new Date(epochMs)
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  const ss = String(d.getSeconds()).padStart(2, '0')
  const ms = String(d.getMilliseconds()).padStart(3, '0')
  return `${hh}:${mm}:${ss}.${ms}`
}
