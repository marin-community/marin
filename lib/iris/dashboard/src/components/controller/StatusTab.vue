<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type { GetProcessStatusResponse, ProcessInfo } from '@/types/rpc'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import LogViewer from '@/components/shared/LogViewer.vue'

const { data, loading, error, refresh } = useControllerRpc<GetProcessStatusResponse>('GetProcessStatus')

useAutoRefresh(refresh, 10_000)
onMounted(refresh)

const info = computed<ProcessInfo | null>(() => data.value?.processInfo ?? null)

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

function formatUptime(uptimeMs?: string): string {
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
        <InfoRow label="CPU %">
          <span class="font-mono">{{ info.cpuPercent !== undefined ? info.cpuPercent.toFixed(1) + '%' : '-' }}</span>
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

    <!-- Process logs -->
    <div>
      <h3 class="text-sm font-semibold text-text mb-3">Controller Logs</h3>
      <LogViewer source="controller" />
    </div>
  </div>
</template>
