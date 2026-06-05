<script setup lang="ts">
/**
 * Compact "recent profiles" list sourced from the iris.profile finelog namespace.
 *
 * Lists the last 10 captures for `source` and anything nested under it: pass a
 * task wire ID, /system/worker/<id>, or /system/controller to scope to one
 * source, or a job ID to cover every task under that job. The list never reads
 * the profile_data blob — the bytes are fetched on click — so it stays cheap on
 * the log server.
 *
 * Each entry is a link. Speedscope captures open in the bundled viewer; every
 * other format downloads. The viewer is opened synchronously inside the click
 * gesture (see openSpeedscopeWindow) so it isn't suppressed as a popup after the
 * fetch latency.
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

interface ProfileHistoryRow {
  // Epoch milliseconds: the stats server returns captured_at as a TIMESTAMP,
  // which apache-arrow decodes to a number (ms since epoch), not a string.
  captured_at?: number
  source?: string
  type?: string
  format?: string
  trigger?: string
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

// Match the source itself plus anything nested under it (a job → its tasks). The
// '/' delimiter keeps '/task/1' from also matching '/task/10'. Only small
// metadata columns are selected — never profile_data — so the list stays cheap.
const { data, refresh } = useLogServerStatsRpc<QueryResponse>('Query', () => {
  const src = escape(props.source)
  return {
    sql: `SELECT captured_at, source, type, format, trigger
FROM "iris.profile"
WHERE source = '${src}' OR starts_with(source, '${src}/')
ORDER BY captured_at DESC
LIMIT 10`,
  }
})

useAutoRefresh(refresh, props.refreshIntervalMs)
onMounted(refresh)
watch(() => props.source, refresh)

const rows = computed<ProfileHistoryRow[]>(() => {
  const ipc = data.value?.arrowIpc
  if (!ipc) return []
  return decodeArrowIpc(ipc).rows as ProfileHistoryRow[]
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

// Short descriptor for the link text, e.g. "cpu speedscope". Speedscope rows get
// an "open in viewer" affordance; others read as a download.
function profileLabel(row: ProfileHistoryRow): string {
  const parts = [row.type, row.format].filter(Boolean)
  return parts.length ? parts.join(' ') : 'profile'
}

async function openProfile(row: ProfileHistoryRow) {
  if (busy.value || row.captured_at == null || !row.source) return
  busy.value = true
  // Open the viewer synchronously within the click gesture so it isn't blocked
  // as a popup after the fetch; non-speedscope rows download instead.
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

defineExpose({ refresh })
</script>

<template>
  <div v-if="rows.length > 0" class="text-sm">
    <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Recent Profiles</h3>
    <ul class="space-y-0.5">
      <li v-for="row in rows" :key="`${row.source ?? ''}@${row.captured_at ?? ''}`">
        <a
          href="#"
          class="text-accent hover:underline"
          :class="{ 'pointer-events-none opacity-60': busy }"
          @click.prevent="openProfile(row)"
        >{{ profileLabel(row) }}<span aria-hidden="true">{{ isSpeedscope(row.format) ? ' ↗' : ' ↓' }}</span></a>
        <span class="text-text-muted ml-2 text-xs font-mono">{{ formatCaptured(row.captured_at) }}</span>
        <span v-if="row.trigger" class="text-text-muted ml-2 text-xs">· {{ row.trigger }}</span>
      </li>
    </ul>
  </div>
</template>
