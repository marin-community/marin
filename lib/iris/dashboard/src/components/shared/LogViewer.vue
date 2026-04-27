<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { logServiceRpcCall } from '@/composables/useRpc'
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

// Cap per-poll response size for cursor-based incremental polls. If more than
// this many lines arrive between polls we'll catch up over subsequent polls
// rather than asking the server for an unbounded batch.
const AUTO_REFRESH_MAX_LINES = 2000
const POLL_INTERVAL_MS = 30_000
// Retain at most this many rendered lines to keep the DOM bounded.
const MAX_RETAINED_LINES = 20_000

const filter = ref('')
const level = ref('info')
const tailLines = ref(500)
const selectedAttemptId = ref(props.currentAttemptId ?? -1)

const entries = ref<LogEntry[]>([])
const loading = ref(false)
const errorMsg = ref<string | null>(null)
// proto JSON encodes int64 as string; 0/"0" both mean "no cursor".
const cursor = ref<string | number | null>(null)

// Task IDs end with a numeric segment (e.g. /alice/job/0), job IDs don't.
const isTask = props.taskId ? /\/\d+$/.test(props.taskId) : false

function computeSource(): string {
  if (props.taskId) {
    if (selectedAttemptId.value >= 0) return `${props.taskId}:${selectedAttemptId.value}`
    return isTask ? `${props.taskId}:.*` : `${props.taskId}/\\d+:.*`
  }
  return props.workerId ? `/system/worker/${props.workerId}` : '/system/controller'
}

// Monotonic generation to discard responses from superseded requests (e.g.
// when the filter changes while a poll is in flight).
let generation = 0

async function fetchTail() {
  const gen = ++generation
  loading.value = true
  errorMsg.value = null
  try {
    const resp = await logServiceRpcCall<FetchLogsResponse>('FetchLogs', {
      source: computeSource(),
      maxLines: tailLines.value || undefined,
      tail: true,
      substring: filter.value || undefined,
      minLevel: level.value ? level.value.toUpperCase() : undefined,
    })
    if (gen !== generation) return
    entries.value = resp.entries ?? []
    cursor.value = resp.cursor ?? null
  } catch (e) {
    if (gen !== generation) return
    errorMsg.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (gen === generation) loading.value = false
  }
}

async function fetchIncremental() {
  // If we don't yet have a cursor (first load raced, or reset just happened),
  // fall back to a tail fetch so we always show something.
  if (cursor.value === null || cursor.value === undefined) {
    await fetchTail()
    return
  }
  const gen = ++generation
  // Incremental polls don't toggle `loading` so the UI doesn't flash on every
  // poll; the user only sees the spinner on the initial/tail load.
  try {
    const resp = await logServiceRpcCall<FetchLogsResponse>('FetchLogs', {
      source: computeSource(),
      maxLines: AUTO_REFRESH_MAX_LINES,
      tail: false,
      cursor: cursor.value,
      substring: filter.value || undefined,
      minLevel: level.value ? level.value.toUpperCase() : undefined,
    })
    if (gen !== generation) return
    const newEntries = resp.entries ?? []
    if (newEntries.length > 0) {
      const combined = entries.value.concat(newEntries)
      entries.value = combined.length > MAX_RETAINED_LINES
        ? combined.slice(combined.length - MAX_RETAINED_LINES)
        : combined
    }
    if (resp.cursor !== undefined && resp.cursor !== null) {
      cursor.value = resp.cursor
    }
    errorMsg.value = null
  } catch (e) {
    if (gen !== generation) return
    // If the cursor is no longer valid (server restart, store rewind), fall
    // back to a fresh tail fetch on the next poll.
    cursor.value = null
    errorMsg.value = e instanceof Error ? e.message : String(e)
  }
}

async function doPoll() {
  await fetchIncremental()
}

// Reset the cursor and do a full tail fetch. Used whenever the filter set
// changes (substring, minLevel, source key, attempt, tail size) — the cursor
// from the previous filter isn't meaningful for the new criteria.
async function resetAndFetch() {
  cursor.value = null
  entries.value = []
  await fetchTail()
}

const { active: autoRefreshActive, toggle: toggleAutoRefresh } = useAutoRefresh(doPoll, POLL_INTERVAL_MS)

watch(selectedAttemptId, resetAndFetch)
watch(tailLines, resetAndFetch)
watch(level, resetAndFetch)

let filterDebounce: ReturnType<typeof setTimeout> | undefined
watch(filter, () => {
  if (filterDebounce) clearTimeout(filterDebounce)
  filterDebounce = setTimeout(resetAndFetch, 250)
})
watch(
  () => [props.taskId, props.currentAttemptId] as const,
  ([taskId, currentAttemptId], [previousTaskId, previousCurrentAttemptId]) => {
    if (taskId !== previousTaskId) {
      selectedAttemptId.value = -1
      resetAndFetch()
      return
    }
    if (taskId === undefined || currentAttemptId === previousCurrentAttemptId) return
    if (selectedAttemptId.value === -1) {
      resetAndFetch()
      return
    }
    if (selectedAttemptId.value === previousCurrentAttemptId) {
      selectedAttemptId.value = currentAttemptId ?? -1
    }
  },
)
watch(() => props.workerId, resetAndFetch)

onMounted(resetAndFetch)

const filteredLogs = computed<LogEntry[]>(() => entries.value)

defineExpose({ selectedAttemptId })
</script>

<template>
  <div class="space-y-2">
    <div class="flex flex-wrap items-center gap-2 sm:gap-3 text-sm">
      <input
        v-model="filter"
        type="text"
        placeholder="Filter logs..."
        class="w-full sm:w-64 px-3 py-1.5 bg-surface border border-surface-border rounded
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
      v-if="errorMsg"
      class="px-3 py-2 text-sm text-status-danger bg-status-danger-bg rounded border border-status-danger-border"
    >
      {{ errorMsg }}
    </div>

    <div
      class="overflow-y-auto rounded-lg border border-surface-border bg-surface"
      :style="{ maxHeight: maxHeight }"
    >
      <div
        v-if="loading && filteredLogs.length === 0"
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
