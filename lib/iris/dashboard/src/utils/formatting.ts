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

/** CSS class for a log level string. */
export function logLevelClass(level: string | undefined): string {
  const lvl = (level ?? 'info').toLowerCase()
  switch (lvl) {
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
