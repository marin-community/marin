<script setup lang="ts">
/**
 * Profile history panel sourced from the iris.profile finelog namespace.
 *
 * Lists the last 50 captures for `source` (a task wire ID, /system/worker/<id>,
 * or /system/controller). Click a row to download the captured bytes.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useLogServerStatsRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { decodeArrowIpc } from '@/utils/arrow'
import { formatBytes } from '@/utils/formatting'

interface Props {
  source: string
  downloadLabel?: string
  refreshIntervalMs?: number
}
const props = withDefaults(defineProps<Props>(), {
  downloadLabel: '',
  refreshIntervalMs: 30_000,
})

interface ProfileHistoryRow {
  captured_at?: string
  type?: string
  attempt_id?: number | null
  vm_id?: string
  duration_seconds?: number
  format?: string
  trigger?: string
  size_bytes?: number
}

interface QueryResponse {
  arrowIpc?: string
}

function escape(value: string): string {
  return value.replace(/'/g, "''")
}

const { data, refresh } = useLogServerStatsRpc<QueryResponse>('Query', () => ({
  sql: `SELECT captured_at, type, attempt_id, vm_id, duration_seconds, format, trigger, length(profile_data) AS size_bytes
FROM "iris.profile"
WHERE source = '${escape(props.source)}'
ORDER BY captured_at DESC
LIMIT 50`,
}))

useAutoRefresh(refresh, props.refreshIntervalMs)
onMounted(refresh)
watch(() => props.source, refresh)

const rows = computed<ProfileHistoryRow[]>(() => {
  const ipc = data.value?.arrowIpc
  if (!ipc) return []
  return decodeArrowIpc(ipc).rows as ProfileHistoryRow[]
})

const downloading = ref(false)

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

async function downloadProfile(row: ProfileHistoryRow) {
  if (downloading.value || !row.captured_at) return
  downloading.value = true
  try {
    const sql = `SELECT profile_data, type, format FROM "iris.profile" WHERE source = '${escape(props.source)}' AND captured_at = '${escape(row.captured_at)}' LIMIT 1`
    const resp = await fetch('/proxy/system.log-server/finelog.stats.StatsService/Query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sql }),
    })
    if (!resp.ok) throw new Error(`Query: ${resp.status} ${resp.statusText}`)
    const payload = await resp.json() as QueryResponse
    if (!payload.arrowIpc) return
    interface FetchRow { profile_data?: Uint8Array; type?: string; format?: string }
    const fetched = decodeArrowIpc(payload.arrowIpc).rows as FetchRow[]
    if (!fetched.length || !fetched[0].profile_data) return
    const bytes = fetched[0].profile_data
    const ext = profileExtension(fetched[0].format)
    const ts = row.captured_at.replace(/[T]/g, '_').replace(/:/g, '-').replace(/\.\d+/, '')
    const label = props.downloadLabel || defaultLabel(props.source)
    const blob = new Blob([bytes], { type: 'application/octet-stream' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${ts}_profile-${label}.${ext}`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    alert(`Download failed: ${e instanceof Error ? e.message : e}`)
  } finally {
    downloading.value = false
  }
}

defineExpose({ refresh })
</script>

<template>
  <div v-if="rows.length > 0">
    <h3 class="text-sm font-semibold text-text mb-3">Profile History</h3>
    <div class="overflow-x-auto rounded-lg border border-surface-border">
      <table class="w-full border-collapse">
        <thead>
          <tr class="border-b border-surface-border">
            <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Captured</th>
            <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Type</th>
            <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Format</th>
            <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Trigger</th>
            <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-right">Size</th>
            <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-right">Duration</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in rows"
            :key="row.captured_at ?? ''"
            class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors cursor-pointer"
            @click="downloadProfile(row)"
          >
            <td class="px-3 py-2 text-[13px] font-mono">{{ row.captured_at ?? '-' }}</td>
            <td class="px-3 py-2 text-[13px] font-mono">{{ row.type ?? '-' }}</td>
            <td class="px-3 py-2 text-[13px] font-mono">{{ row.format ?? '-' }}</td>
            <td class="px-3 py-2 text-[13px] font-mono">{{ row.trigger ?? '-' }}</td>
            <td class="px-3 py-2 text-[13px] font-mono text-right">{{ row.size_bytes !== undefined ? formatBytes(Number(row.size_bytes)) : '-' }}</td>
            <td class="px-3 py-2 text-[13px] font-mono text-right">{{ row.duration_seconds !== undefined ? `${row.duration_seconds}s` : '-' }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
