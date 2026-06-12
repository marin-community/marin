<script setup lang="ts">
/**
 * "Latest profile" link for a single source (a task wire ID).
 *
 * Queries the most recent iris.profile capture for `source` and renders one
 * link. Speedscope captures open in the bundled viewer (same-origin via a blob
 * URL); every other format downloads. The viewer is opened synchronously inside
 * the click gesture (see openSpeedscopeWindow) so it isn't suppressed as a popup
 * after the fetch latency. Only metadata is selected — never profile_data — so
 * the lookup stays cheap; the bytes are fetched on click. Self-hides when the
 * source has no captures.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useLogServerStatsRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { decodeArrowIpc } from '@/utils/arrow'
import { openSpeedscopeWindow } from '@/utils/speedscope'

interface Props {
  source: string
  refreshIntervalMs?: number
}
const props = withDefaults(defineProps<Props>(), {
  refreshIntervalMs: 30_000,
})

interface ProfileRow {
  // Epoch milliseconds: the stats server returns captured_at as a TIMESTAMP,
  // which apache-arrow decodes to a number (ms since epoch), not a string.
  captured_at?: number
  source?: string
  type?: string
  format?: string
}

interface QueryResponse {
  arrowIpc?: string
}

function escape(value: string): string {
  return value.replace(/'/g, "''")
}

function formatCaptured(capturedAt: number | undefined): string {
  return capturedAt == null ? '-' : new Date(capturedAt).toLocaleString()
}

const { data, refresh } = useLogServerStatsRpc<QueryResponse>('Query', () => {
  const src = escape(props.source)
  return {
    sql: `SELECT captured_at, source, type, format
FROM "iris.profile"
WHERE source = '${src}'
ORDER BY captured_at DESC
LIMIT 1`,
  }
})

useAutoRefresh(refresh, props.refreshIntervalMs)
onMounted(refresh)
watch(() => props.source, refresh)

const latest = computed<ProfileRow | null>(() => {
  const ipc = data.value?.arrowIpc
  if (!ipc) return null
  const rows = decodeArrowIpc(ipc).rows as ProfileRow[]
  return rows.length ? rows[0] : null
})

const busy = ref(false)

function isSpeedscope(format: string | undefined): boolean {
  return (format ?? '').toLowerCase() === 'speedscope'
}

function profileExtension(format: string | undefined): string {
  switch ((format ?? '').toLowerCase()) {
    case 'flamegraph': return 'svg'
    case 'html': return 'html'
    case 'speedscope': return 'out'
    case 'table':
    case 'stats': return 'txt'
    default: return 'bin'
  }
}

function defaultLabel(source: string): string {
  if (source === '/system/controller') return 'controller'
  if (source.startsWith('/system/worker/')) return 'worker-' + source.slice('/system/worker/'.length)
  return source.replace(/^\//, '').replace(/\//g, '_')
}

async function openLatest() {
  const row = latest.value
  if (busy.value || !row || row.captured_at == null || !row.source) return
  busy.value = true
  // Open the viewer synchronously within the click gesture so it isn't blocked
  // as a popup after the fetch; non-speedscope captures download instead.
  const pending = isSpeedscope(row.format) ? openSpeedscopeWindow() : null
  try {
    // captured_at arrives as epoch ms; match the TIMESTAMP column via epoch_ms()
    // rather than a string literal (the server rejects a raw ms string).
    const sql = `SELECT profile_data, type, format FROM "iris.profile" WHERE source = '${escape(row.source)}' AND epoch_ms(captured_at) = ${row.captured_at} LIMIT 1`
    const resp = await fetch('/proxy/system.log-server/finelog.stats.StatsService/Query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sql }),
    })
    if (!resp.ok) throw new Error(`Query: ${resp.status} ${resp.statusText}`)
    const payload = await resp.json() as QueryResponse
    if (!payload.arrowIpc) {
      pending?.cancel()
      return
    }
    interface FetchRow { profile_data?: Uint8Array; type?: string; format?: string }
    const fetched = decodeArrowIpc(payload.arrowIpc).rows as FetchRow[]
    if (!fetched.length || !fetched[0].profile_data) {
      pending?.cancel()
      return
    }
    const bytes = fetched[0].profile_data
    const label = defaultLabel(row.source)
    if (pending) {
      pending.show(bytes, label)
      return
    }
    const ext = profileExtension(fetched[0].format)
    const ts = new Date(row.captured_at).toISOString().replace(/T/g, '_').replace(/:/g, '-').replace(/\.\d+Z$/, '')
    const blob = new Blob([bytes], { type: 'application/octet-stream' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${ts}_profile-${label}.${ext}`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    pending?.cancel()
    alert(`Failed to open profile: ${e instanceof Error ? e.message : e}`)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div v-if="latest" class="text-sm">
    <a
      href="#"
      class="text-accent hover:underline"
      :class="{ 'pointer-events-none opacity-60': busy }"
      @click.prevent="openLatest"
    >{{ isSpeedscope(latest.format) ? 'View latest profile' : 'Download latest profile' }}<span aria-hidden="true">{{ isSpeedscope(latest.format) ? ' ↗' : ' ↓' }}</span></a>
    <span class="text-text-muted ml-2 text-xs font-mono">{{ [latest.type, latest.format].filter(Boolean).join(' ') }} · {{ formatCaptured(latest.captured_at) }}</span>
  </div>
</template>
