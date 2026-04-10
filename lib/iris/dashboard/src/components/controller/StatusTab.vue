<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useControllerRpc, controllerRpcCall } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { useProfileAction } from '@/composables/useProfileAction'
import type { GetProcessStatusResponse, ProcessInfo } from '@/types/rpc'
import { formatBytes, formatCpuMillicores, formatUptime } from '@/utils/formatting'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import LogViewer from '@/components/shared/LogViewer.vue'
import ProfileButtons from '@/components/shared/ProfileButtons.vue'

const { data, loading, error, refresh } = useControllerRpc<GetProcessStatusResponse>('GetProcessStatus')
const { profiling, profile } = useProfileAction(controllerRpcCall, '/system/process')

useAutoRefresh(refresh, 10_000)
onMounted(refresh)

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

    <!-- Process logs -->
    <div>
      <h3 class="text-sm font-semibold text-text mb-3">Controller Logs</h3>
      <LogViewer source="controller" />
    </div>
  </div>
</template>
