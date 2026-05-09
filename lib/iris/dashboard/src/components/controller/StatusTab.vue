<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useControllerRpc, controllerRpcCall, useLogServerStatsRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import { useProfileAction } from '@/composables/useProfileAction'
import type { GetProcessStatusResponse, ProcessInfo } from '@/types/rpc'
import { formatBytes, formatCpuMillicores, formatUptime } from '@/utils/formatting'
import { decodeArrowIpc } from '@/utils/arrow'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import LogViewer from '@/components/shared/LogViewer.vue'
import ProfileButtons from '@/components/shared/ProfileButtons.vue'
import RpcStatsPanel from '@/components/controller/RpcStatsPanel.vue'

const { data, loading, error, refresh } = useControllerRpc<GetProcessStatusResponse>('GetProcessStatus')
const { profiling, profile } = useProfileAction(controllerRpcCall, '/system/controller')

// --- Profile history sourced from finelog stats (iris.profile) ---
//
// Lists recent profile captures for the controller process, filtered by
// source = '/system/controller'. Rows come back captured_at DESC.
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

const CONTROLLER_PROFILE_SOURCE = '/system/controller'

const profileHistorySql = `
SELECT
  captured_at,
  type,
  attempt_id,
  vm_id,
  duration_seconds,
  format,
  trigger,
  length(profile_data) AS size_bytes
FROM "iris.profile"
WHERE source = '${CONTROLLER_PROFILE_SOURCE}'
ORDER BY captured_at DESC
LIMIT 50
`.trim()

const { data: profileHistoryData, refresh: fetchProfileHistory } = useLogServerStatsRpc<QueryResponse>(
  'Query',
  { sql: profileHistorySql },
)

useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
useAutoRefresh(fetchProfileHistory, 30_000)
onMounted(() => {
  refresh()
  fetchProfileHistory()
})

const profileHistoryRows = computed<ProfileHistoryRow[]>(() => {
  const ipc = profileHistoryData.value?.arrowIpc
  if (!ipc) return []
  return decodeArrowIpc(ipc).rows as ProfileHistoryRow[]
})

const downloadingProfile = ref(false)

/** Extension derived from the profile format value, mirroring useProfileAction.ts. */
function profileExtension(format: string | undefined): string {
  if (!format) return 'bin'
  const f = format.toLowerCase()
  if (f === 'flamegraph') return 'svg'
  if (f === 'html') return 'html'
  if (f === 'speedscope') return 'out'
  if (f === 'raw') return 'bin'
  if (f === 'table' || f === 'stats') return 'txt'
  return 'bin'
}

async function downloadProfile(row: ProfileHistoryRow) {
  if (downloadingProfile.value) return
  downloadingProfile.value = true
  try {
    const escapedTs = (row.captured_at ?? '').replace(/'/g, "''")
    const sql = `SELECT profile_data, type, format FROM "iris.profile" WHERE source = '${CONTROLLER_PROFILE_SOURCE}' AND captured_at = '${escapedTs}' LIMIT 1`
    const resp = await fetch('/proxy/system.log-server/finelog.stats.StatsService/Query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sql }),
    })
    if (!resp.ok) throw new Error(`Query: ${resp.status} ${resp.statusText}`)
    const payload = await resp.json() as QueryResponse
    if (!payload.arrowIpc) return
    interface FetchRow { profile_data?: string; type?: string; format?: string }
    const rows = decodeArrowIpc(payload.arrowIpc).rows as FetchRow[]
    if (!rows.length || !rows[0].profile_data) return
    const base64 = rows[0].profile_data
    const bin = atob(base64)
    const bytes = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
    const ext = profileExtension(rows[0].format)
    const ts = (row.captured_at ?? new Date().toISOString()).replace(/[T]/g, '_').replace(/:/g, '-').replace(/\.\d+/, '')
    const blob = new Blob([bytes], { type: 'application/octet-stream' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${ts}_profile-controller.${ext}`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    alert(`Download failed: ${e instanceof Error ? e.message : e}`)
  } finally {
    downloadingProfile.value = false
  }
}

const info = computed<ProcessInfo | null>(() => data.value?.processInfo ?? null)

const rssBytes = computed(() => {
  const raw = info.value?.memoryRssBytes
  return raw ? parseInt(raw, 10) : 0
})

const vmsBytes = computed(() => {
  const raw = info.value?.memoryVmsBytes
  return raw ? parseInt(raw, 10) : 0
})

const totalBytes = computed(() => {
  const raw = info.value?.memoryTotalBytes
  return raw ? parseInt(raw, 10) : 0
})
</script>

<template>
  <!-- Error -->
  <div
    v-if="error"
    class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
  >
    {{ error }}
  </div>

  <!-- Loading -->
  <div v-if="loading && !data" class="flex items-center justify-center py-12 text-text-muted text-sm">
    <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
    Loading...
  </div>

  <div v-else-if="info" class="space-y-6">
    <!-- Process info cards -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <!-- Identity -->
      <InfoCard title="Process">
        <InfoRow label="Hostname">
          <span class="font-mono">{{ info.hostname ?? '-' }}</span>
        </InfoRow>
        <InfoRow label="PID">
          <span class="font-mono">{{ info.pid ?? '-' }}</span>
        </InfoRow>
        <InfoRow label="Python">
          <span class="font-mono">{{ info.pythonVersion ?? '-' }}</span>
        </InfoRow>
        <InfoRow label="Uptime">
          <span class="font-mono">{{ formatUptime(info.uptimeMs) }}</span>
        </InfoRow>
        <InfoRow v-if="info.gitHash" label="Git Hash">
          <span class="font-mono text-xs">{{ info.gitHash }}</span>
        </InfoRow>
      </InfoCard>

      <!-- Resources -->
      <InfoCard title="Resources">
        <InfoRow label="Memory RSS">
          <span class="font-mono">{{ rssBytes ? formatBytes(rssBytes) : '-' }}</span>
        </InfoRow>
        <InfoRow label="Memory VMS">
          <span class="font-mono">{{ vmsBytes ? formatBytes(vmsBytes) : '-' }}</span>
        </InfoRow>
        <InfoRow v-if="totalBytes" label="System Memory">
          <span class="font-mono">{{ formatBytes(totalBytes) }}</span>
        </InfoRow>
        <InfoRow label="Process CPU">
          <span class="font-mono">{{ info.cpuMillicores !== undefined ? formatCpuMillicores(info.cpuMillicores) : '-' }}</span>
        </InfoRow>
        <InfoRow v-if="info.cpuCount" label="CPU Cores">
          <span class="font-mono">{{ info.cpuCount }}</span>
        </InfoRow>
        <InfoRow label="Threads">
          <span class="font-mono">{{ info.threadCount ?? '-' }}</span>
        </InfoRow>
        <InfoRow label="Open FDs">
          <span class="font-mono">{{ info.openFdCount ?? '-' }}</span>
        </InfoRow>
      </InfoCard>
    </div>

    <!-- Process profiling -->
    <ProfileButtons :profiling="profiling" @profile="profile" />

    <!-- RPC statistics -->
    <div>
      <h3 class="text-sm font-semibold text-text mb-3">RPC Statistics</h3>
      <RpcStatsPanel />
    </div>

    <!-- Profile history (sourced from finelog iris.profile) -->
    <div v-if="profileHistoryRows.length > 0">
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
              v-for="row in profileHistoryRows"
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

    <!-- Process logs -->
    <div>
      <h3 class="text-sm font-semibold text-text mb-3">Controller Logs</h3>
      <LogViewer source="controller" />
    </div>
  </div>
</template>
