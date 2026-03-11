<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useControllerRpc, useWorkerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type { FetchLogsResponse, GetTaskLogsResponse, LogEntry } from '@/types/rpc'
import { timestampMs } from '@/utils/formatting'

const props = withDefaults(defineProps<{
  taskId?: string
  workerId?: string
  source?: 'controller' | 'worker'
  maxHeight?: string
}>(), {
  maxHeight: '60vh',
})

const filter = ref('')
const level = ref('info')
const tailLines = ref(500)

const LOG_LEVEL_PRIORITY: Record<string, number> = {
  debug: 0,
  info: 1,
  warning: 2,
  error: 3,
  critical: 4,
}

function levelPriority(lvl: string | undefined): number {
  if (!lvl) return 1
  return LOG_LEVEL_PRIORITY[lvl.toLowerCase()] ?? 1
}

// Choose the right RPC based on what we're viewing
const useRpc = props.source === 'worker' ? useWorkerRpc : useControllerRpc

const taskLogState = props.taskId
  ? useRpc<GetTaskLogsResponse>('GetTaskLogs', { id: props.taskId, maxTotalLines: tailLines.value })
  : null

const processLogState = !props.taskId
  ? useRpc<FetchLogsResponse>('FetchLogs', {
      source: props.workerId ? `/worker/${props.workerId}` : '/system/process',
      maxLines: tailLines.value,
      tail: true,
    })
  : null

const rpcState = taskLogState ?? processLogState!

async function doRefresh() {
  await rpcState.refresh()
}

const { active: autoRefreshActive, toggle: toggleAutoRefresh } = useAutoRefresh(doRefresh, 30_000)

onMounted(doRefresh)

function extractEntries(): LogEntry[] {
  if (taskLogState?.data.value) {
    const resp = taskLogState.data.value
    return (resp.taskLogs ?? []).flatMap(batch => batch.logs ?? [])
  }
  if (processLogState?.data.value) {
    return processLogState.data.value.entries ?? []
  }
  return []
}

const filteredLogs = computed(() => {
  const entries = extractEntries()
  const minPriority = levelPriority(level.value)
  const filterText = filter.value.toLowerCase()

  return entries.filter(entry => {
    if (filterText && !(entry.data ?? '').toLowerCase().includes(filterText)) return false
    return levelPriority(entry.level) >= minPriority
  })
})

function logLevelClass(entryLevel: string | undefined): string {
  const lvl = (entryLevel ?? 'info').toLowerCase()
  switch (lvl) {
    case 'debug': return 'text-text-muted'
    case 'warning': return 'text-status-warning'
    case 'error':
    case 'critical': return 'text-status-danger'
    default: return 'text-text'
  }
}

function formatLogTime(timestamp: { epochMs: string } | undefined): string {
  const ms = timestampMs(timestamp)
  if (!ms) return ''
  return new Date(ms).toLocaleTimeString()
}
</script>

<template>
  <div class="space-y-2">
    <div class="flex items-center gap-3 text-sm">
      <input
        v-model="filter"
        type="text"
        placeholder="Filter logs..."
        class="w-64 px-3 py-1.5 bg-surface border border-surface-border rounded
               text-sm font-mono placeholder:text-text-muted
               focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
      />
      <select
        v-model="level"
        class="px-2 py-1.5 border border-surface-border rounded text-sm"
      >
        <option value="debug">Debug</option>
        <option value="info">Info</option>
        <option value="warning">Warning</option>
        <option value="error">Error</option>
      </select>
      <button
        class="px-2 py-1.5 border border-surface-border rounded text-sm hover:bg-surface-sunken"
        :class="autoRefreshActive ? 'text-accent' : 'text-text-muted'"
        @click="toggleAutoRefresh"
      >
        {{ autoRefreshActive ? 'Auto ⟳' : 'Paused' }}
      </button>
      <span class="ml-auto text-xs text-text-muted font-mono">
        {{ filteredLogs.length }} lines
      </span>
    </div>

    <div
      v-if="rpcState.error.value"
      class="px-3 py-2 text-sm text-status-danger bg-status-danger-bg rounded border border-status-danger-border"
    >
      {{ rpcState.error.value }}
    </div>

    <div
      class="overflow-y-auto rounded-lg border border-surface-border bg-white"
      :style="{ maxHeight: maxHeight }"
    >
      <div
        v-if="rpcState.loading.value && filteredLogs.length === 0"
        class="py-12 text-center text-text-muted text-sm"
      >
        Loading logs...
      </div>
      <div
        v-else-if="filteredLogs.length === 0"
        class="py-12 text-center text-text-muted text-sm"
      >
        No log entries
      </div>
      <div
        v-for="(entry, i) in filteredLogs"
        :key="i"
        :class="[
          'px-3 py-0.5 font-mono text-xs leading-relaxed hover:bg-surface-sunken',
          logLevelClass(entry.level),
        ]"
      >
        <span class="text-text-muted mr-2">{{ formatLogTime(entry.timestamp) }}</span>
        <span class="whitespace-pre-wrap break-all">{{ entry.data }}</span>
      </div>
    </div>
  </div>
</template>
