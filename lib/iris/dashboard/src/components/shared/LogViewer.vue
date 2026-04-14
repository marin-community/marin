<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useLogServiceRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type { FetchLogsResponse, LogEntry, TaskAttempt } from '@/types/rpc'
import { timestampMs, logLevelClass, formatLogTime } from '@/utils/formatting'

const props = withDefaults(defineProps<{
  taskId?: string
  workerId?: string
  source?: 'controller' | 'worker'
  maxHeight?: string
  attempts?: TaskAttempt[]
  currentAttemptId?: number
}>(), {
  maxHeight: '60vh',
})

const filter = ref('')
const level = ref('info')
const tailLines = ref(500)
const selectedAttemptId = ref(props.currentAttemptId ?? -1)

// FetchLogs is served by the LogService (co-hosted on the controller)
const useRpc = useLogServiceRpc

// Task IDs end with a numeric segment (e.g. /alice/job/0), job IDs don't.
const isTask = props.taskId ? /\/\d+$/.test(props.taskId) : false

const taskLogState = props.taskId
  ? useRpc<FetchLogsResponse>('FetchLogs', () => ({
      source: selectedAttemptId.value >= 0
        ? `${props.taskId}:${selectedAttemptId.value}`
        : isTask
          ? `${props.taskId}:.*`
          : `${props.taskId}/\\d+:.*`,
      maxLines: tailLines.value || undefined,
      tail: true,
      substring: filter.value || undefined,
      minLevel: level.value ? level.value.toUpperCase() : undefined,
    }))
  : null

const processLogState = !props.taskId
  ? useRpc<FetchLogsResponse>('FetchLogs', () => ({
      source: props.workerId ? `/system/worker/${props.workerId}` : '/system/controller',
      maxLines: tailLines.value || undefined,
      tail: true,
      substring: filter.value || undefined,
      minLevel: level.value ? level.value.toUpperCase() : undefined,
    }))
  : null

const rpcState = taskLogState ?? processLogState!

async function doRefresh() {
  await rpcState.refresh()
}

const { active: autoRefreshActive, toggle: toggleAutoRefresh } = useAutoRefresh(doRefresh, 30_000)

watch(selectedAttemptId, () => doRefresh())
watch(tailLines, () => doRefresh())
watch(level, () => doRefresh())

let filterDebounce: ReturnType<typeof setTimeout> | undefined
watch(filter, () => {
  if (filterDebounce) clearTimeout(filterDebounce)
  filterDebounce = setTimeout(() => doRefresh(), 250)
})
watch(
  () => [props.taskId, props.currentAttemptId] as const,
  ([taskId, currentAttemptId], [previousTaskId, previousCurrentAttemptId]) => {
    if (taskId !== previousTaskId) {
      selectedAttemptId.value = -1
      doRefresh()
      return
    }
    if (taskId === undefined || currentAttemptId === previousCurrentAttemptId) return
    if (selectedAttemptId.value === -1) {
      doRefresh()
      return
    }
    if (selectedAttemptId.value === previousCurrentAttemptId) {
      selectedAttemptId.value = currentAttemptId ?? -1
    }
  },
)
watch(() => props.workerId, () => doRefresh())

onMounted(doRefresh)

function extractEntries(): LogEntry[] {
  if (taskLogState?.data.value) {
    return taskLogState.data.value.entries ?? []
  }
  if (processLogState?.data.value) {
    return processLogState.data.value.entries ?? []
  }
  return []
}

const filteredLogs = computed(() => extractEntries())

defineExpose({ selectedAttemptId })
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
      <select
        v-model.number="tailLines"
        class="px-2 py-1.5 border border-surface-border rounded text-sm"
      >
        <option :value="500">500 lines</option>
        <option :value="1000">1,000 lines</option>
        <option :value="5000">5,000 lines</option>
        <option :value="10000">10,000 lines</option>
      </select>
      <select
        v-if="attempts && attempts.length > 0"
        v-model.number="selectedAttemptId"
        class="px-2 py-1.5 border border-surface-border rounded text-sm"
      >
        <option :value="-1">All attempts</option>
        <option v-for="a in attempts" :key="a.attemptId" :value="a.attemptId">
          Attempt {{ a.attemptId }}
        </option>
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
      class="overflow-y-auto rounded-lg border border-surface-border bg-surface"
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
        <span class="text-text-muted mr-2">{{ formatLogTime(timestampMs(entry.timestamp)) }}</span>
        <span class="whitespace-pre-wrap break-all">{{ entry.data }}</span>
      </div>
    </div>
  </div>
</template>
