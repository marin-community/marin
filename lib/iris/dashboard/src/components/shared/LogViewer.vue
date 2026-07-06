<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import { logServiceRpcCall } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { isFederated, type FetchLogsResponse, type LogEntry, type TaskAttempt } from '@/types/rpc'
import { timestampMs, logLevelClass, formatLogTime } from '@/utils/formatting'
import { parseLogLinks } from '@/utils/logLinks'

const props = withDefaults(defineProps<{
  taskId?: string
  workerId?: string
  maxHeight?: string
  attempts?: TaskAttempt[]
  currentAttemptId?: number
  // Cluster-wide explorer with no fixed context: default the source to a
  // match-everything prefix instead of the local process stream.
  standalone?: boolean
  // The owning job/task's cluster coordinate. A federated row's logs live in the
  // shared finelog under the same `taskId` key but stamped with the peer's id, so
  // we pass `cluster` through as a FetchLogs filter (see baseRequest). A local
  // row (or a context with no cluster) sends no filter and reads its own rows.
  cluster?: string
}>(), {
  maxHeight: '60vh',
})

// The finelog query key is the local task/job id — identity is cluster-invariant,
// so a federated row is served under the same key (disambiguated by `cluster`).
const sourceBase = computed(() => props.taskId)

// Cap per-poll response size for cursor-based incremental polls. If more than
// this many lines arrive between polls we'll catch up over subsequent polls
// rather than asking the server for an unbounded batch.
const AUTO_REFRESH_MAX_LINES = 2000
const POLL_INTERVAL_MS = 30_000
// Retain at most this many rendered lines to keep the DOM bounded.
const MAX_RETAINED_LINES = 20_000

const route = useRoute()

type MatchScope = 'EXACT' | 'PREFIX' | 'REGEX'
type WireMatchScope = 'MATCH_SCOPE_EXACT' | 'MATCH_SCOPE_PREFIX' | 'MATCH_SCOPE_REGEX'
const WIRE_SCOPE: Record<MatchScope, WireMatchScope> = {
  EXACT: 'MATCH_SCOPE_EXACT',
  PREFIX: 'MATCH_SCOPE_PREFIX',
  REGEX: 'MATCH_SCOPE_REGEX',
}

// Relative time windows for the "Since" selector. 0 = no lower bound.
const SINCE_PRESETS: { label: string; ms: number }[] = [
  { label: 'All time', ms: 0 },
  { label: 'Last 15m', ms: 15 * 60_000 },
  { label: 'Last 1h', ms: 60 * 60_000 },
  { label: 'Last 6h', ms: 6 * 3_600_000 },
  { label: 'Last 24h', ms: 24 * 3_600_000 },
  { label: 'Last 7d', ms: 7 * 86_400_000 },
]

const filter = ref('')
const level = ref('info')
const tailLines = ref(500)
const selectedAttemptId = ref(props.currentAttemptId ?? -1)

// Editable query. Pre-filled from props (task/worker/controller context) but the
// user is free to retype the source or widen the match scope — that's how you
// search by job (`/alice/job/` prefix), by user (`/alice/` prefix), or by an
// arbitrary key pattern (regex).
const sourceInput = ref('')
const matchScope = ref<MatchScope>('EXACT')

// Lower time bound. A preset (relative window) and an absolute datetime-local
// are mutually exclusive — setting one clears the other.
const presetMs = ref(0)
const customSince = ref('')

// Timezone for rendering log timestamps and interpreting the date picker.
// Stored timestamps are UTC epoch ms regardless; 'local' shows the browser's
// zone (default), 'utc' lines the prefix up with the UTC timestamps processes
// embed in their raw log lines.
const timeZone = ref<'local' | 'utc'>('local')

const entries = ref<LogEntry[]>([])
const loading = ref(false)
const errorMsg = ref<string | null>(null)
// proto JSON encodes int64 as string; 0/"0" both mean "no cursor".
const cursor = ref<string | number | null>(null)

// Task IDs end with a numeric segment (e.g. /alice/job/0), job IDs don't.
const isTask = computed(() => (sourceBase.value ? /\/\d+$/.test(sourceBase.value) : false))

function attemptFromRoute(): number {
  const raw = route.query.attempt
  const n = typeof raw === 'string' ? Number(raw) : NaN
  return Number.isInteger(n) && n >= 0 ? n : -1
}

// Derive the default (source, scope) for the current context. Standalone use
// (no task/worker/controller context) defaults to a cluster-wide prefix so the
// explorer shows something before the user narrows it down.
function defaultSource(): { source: string; scope: MatchScope } {
  if (sourceBase.value) {
    if (selectedAttemptId.value >= 0) {
      return { source: `${sourceBase.value}:${selectedAttemptId.value}`, scope: 'EXACT' }
    }
    return { source: isTask.value ? `${sourceBase.value}:` : `${sourceBase.value}/`, scope: 'PREFIX' }
  }
  if (props.workerId) return { source: `/system/worker/${props.workerId}`, scope: 'EXACT' }
  if (props.standalone) return { source: '/', scope: 'PREFIX' }
  return { source: '/system/controller', scope: 'EXACT' }
}

// Reset the editable query to the context default, then refetch. Used on mount
// and whenever the surrounding context changes (props, selected attempt).
function applyDefaults() {
  const d = defaultSource()
  sourceInput.value = d.source
  matchScope.value = d.scope
  resetAndFetch()
}

// A `datetime-local` value (e.g. "2026-06-30T02:08") carries no timezone, so
// interpret it in the panel's selected zone. This keeps the lower bound aligned
// with the timestamps the panel renders instead of silently assuming local.
function parseDateTimeLocal(value: string): number {
  const m = value.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?/)
  if (!m) return NaN
  const [year, month, day, hour, minute, second] = m.slice(1).map((p) => Number(p ?? 0))
  return timeZone.value === 'utc'
    ? Date.UTC(year, month - 1, day, hour, minute, second)
    : new Date(year, month - 1, day, hour, minute, second).getTime()
}

// Resolve the lower time bound at request time. A relative preset must be
// recomputed on every fetch (including auto-refresh polls) so the window stays
// anchored to "now", not to when the preset was first selected.
function computeSinceMs(): number | undefined {
  if (customSince.value) {
    const ms = parseDateTimeLocal(customSince.value)
    return Number.isNaN(ms) ? undefined : ms
  }
  return presetMs.value > 0 ? Date.now() - presetMs.value : undefined
}

// Monotonic generation to discard responses from superseded requests (e.g.
// when the filter changes while a poll is in flight).
let generation = 0

function baseRequest() {
  return {
    source: sourceInput.value,
    matchScope: WIRE_SCOPE[matchScope.value],
    substring: filter.value || undefined,
    minLevel: level.value ? level.value.toUpperCase() : undefined,
    sinceMs: computeSinceMs(),
    // Federated rows target the peer's relayed logs in the shared hub store; a
    // local row (or no cluster context) sends no filter and reads its own rows.
    cluster: isFederated(props.cluster) ? props.cluster : undefined,
  }
}

async function fetchTail() {
  if (!sourceInput.value) {
    entries.value = []
    return
  }
  const gen = ++generation
  loading.value = true
  errorMsg.value = null
  try {
    // An explicit start date means "read forward from here": fetch the first
    // maxLines entries at/after sinceMs (oldest-first). Relative presets and the
    // default view stay anchored to now, so they tail the newest maxLines.
    const resp = await logServiceRpcCall<FetchLogsResponse>('FetchLogs', {
      ...baseRequest(),
      maxLines: tailLines.value || undefined,
      tail: !customSince.value,
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
      ...baseRequest(),
      maxLines: AUTO_REFRESH_MAX_LINES,
      tail: false,
      cursor: cursor.value,
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

// Reset the cursor and do a full tail fetch. Used whenever the query changes
// (source, scope, substring, level, since, attempt, tail size) — the cursor
// from the previous query isn't meaningful for the new criteria.
async function resetAndFetch() {
  cursor.value = null
  entries.value = []
  await fetchTail()
}

const { active: autoRefreshActive, toggle: toggleAutoRefresh } = useAutoRefresh(doPoll, POLL_INTERVAL_MS)

// Free-text fields (source key, substring filter) apply on Enter, not on every
// keystroke. The discrete selectors below refetch immediately on change. The
// match-scope select refetches via @change rather than a watch, so the
// reassignment in applyDefaults() doesn't fire a redundant second fetch.
watch(selectedAttemptId, applyDefaults)
watch(tailLines, resetAndFetch)
watch(level, resetAndFetch)
watch([presetMs, customSince], resetAndFetch)
// Changing the zone only alters the query when an absolute date is in effect
// (it reinterprets the picker); otherwise it just re-renders timestamps.
watch(timeZone, () => {
  if (customSince.value) resetAndFetch()
})

function selectPreset(ms: number) {
  // The synthetic "Custom" option (-1) only appears while an absolute time is
  // set; re-selecting it is a no-op so the datetime input stays in effect.
  if (ms < 0) return
  customSince.value = ''
  presetMs.value = ms
}

watch(customSince, (val) => {
  if (val) presetMs.value = 0
})

watch(
  () => [props.taskId, props.currentAttemptId] as const,
  ([taskId, currentAttemptId], [previousTaskId, previousCurrentAttemptId]) => {
    if (taskId !== previousTaskId) {
      selectedAttemptId.value = attemptFromRoute()
      applyDefaults()
      return
    }
    if (taskId === undefined || currentAttemptId === previousCurrentAttemptId) return
    if (selectedAttemptId.value === -1) {
      applyDefaults()
      return
    }
    if (selectedAttemptId.value === previousCurrentAttemptId) {
      selectedAttemptId.value = currentAttemptId ?? -1
    }
  },
)
watch(() => props.workerId, applyDefaults)
// The owning row's cluster arrives async (after GetJob/TaskStatus resolves) and
// only changes the FetchLogs filter, not the source key — re-query in place.
watch(() => props.cluster, resetAndFetch)

// vue-router reuses this instance when only the query changes (e.g. clicking a
// link to a different attempt of the same task), so onMounted alone won't catch
// it — keep selectedAttemptId in sync with ?attempt= on query-only navigation.
watch(() => route.query.attempt, () => {
  if (!props.taskId) return
  const routeAttempt = attemptFromRoute()
  if (routeAttempt >= 0 && routeAttempt !== selectedAttemptId.value) {
    selectedAttemptId.value = routeAttempt
  }
})

onMounted(() => {
  if (props.taskId) {
    const routeAttempt = attemptFromRoute()
    if (routeAttempt >= 0) selectedAttemptId.value = routeAttempt
  }
  applyDefaults()
})

// Job-aggregate mode shows logs from many tasks; render a per-line link to the
// originating task. Single-task mode would link every line to itself, so skip.
// A federated job's tasks are mirrored locally under the same ids, so the links
// resolve for federated rows too.
const showTaskLinks = computed(() => {
  if (!props.taskId) return false
  return !/\/\d+$/.test(props.taskId)
})

interface TaskRef {
  taskId: string
  taskIndex: string
}

function parseTaskFromKey(key: string | undefined): TaskRef | null {
  if (!key) return null
  const colonIdx = key.lastIndexOf(':')
  const taskId = colonIdx > 0 ? key.slice(0, colonIdx) : key
  const lastSlash = taskId.lastIndexOf('/')
  if (lastSlash < 0) return null
  const taskIndex = taskId.slice(lastSlash + 1)
  if (!/^\d+$/.test(taskIndex)) return null
  return { taskId, taskIndex }
}

interface LogRow {
  entry: LogEntry
  taskRef: TaskRef | null
  segments: ReturnType<typeof parseLogLinks>
}

const logRows = computed<LogRow[]>(() =>
  entries.value.map(entry => ({
    entry,
    taskRef: showTaskLinks.value ? parseTaskFromKey(entry.key) : null,
    // proto3-JSON omits default scalars, so an empty log line arrives with
    // `data` absent (undefined); coalesce so matchAll() doesn't throw.
    segments: parseLogLinks(entry.data ?? ''),
  })),
)

defineExpose({ selectedAttemptId })
</script>

<template>
  <div class="space-y-2">
    <!-- Query row: source + match scope (+ attempt picker on task pages) -->
    <div class="flex flex-wrap items-center gap-2 sm:gap-3 text-sm">
      <input
        v-model="sourceInput"
        type="text"
        spellcheck="false"
        placeholder="Source key, e.g. /alice/job/ or /system/worker/… (Enter to apply)"
        class="w-full sm:w-96 px-3 py-1.5 bg-surface border border-surface-border rounded
               text-sm font-mono placeholder:text-text-muted
               focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
        @keyup.enter="resetAndFetch"
      />
      <select
        v-model="matchScope"
        title="How the source is matched against log keys"
        class="px-2 py-1.5 border border-surface-border rounded text-sm"
        @change="resetAndFetch"
      >
        <option value="EXACT">Exact</option>
        <option value="PREFIX">Prefix</option>
        <option value="REGEX">Regex</option>
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
    </div>

    <!-- Filter row: text search + level + since + tail size + auto-refresh -->
    <div class="flex flex-wrap items-center gap-2 sm:gap-3 text-sm">
      <input
        v-model="filter"
        type="text"
        placeholder="Filter text… (Enter to apply)"
        class="w-full sm:w-56 px-3 py-1.5 bg-surface border border-surface-border rounded
               text-sm font-mono placeholder:text-text-muted
               focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
        @keyup.enter="resetAndFetch"
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
        :value="customSince ? -1 : presetMs"
        title="Only show logs newer than this"
        class="px-2 py-1.5 border border-surface-border rounded text-sm"
        @change="selectPreset(Number(($event.target as HTMLSelectElement).value))"
      >
        <option v-if="customSince" :value="-1">Custom</option>
        <option v-for="p in SINCE_PRESETS" :key="p.ms" :value="p.ms">{{ p.label }}</option>
      </select>
      <input
        v-model="customSince"
        type="datetime-local"
        title="Show logs since a specific date/time"
        class="px-2 py-1.5 border border-surface-border rounded text-sm"
      />
      <select
        v-model="timeZone"
        title="Timezone for displayed timestamps and the date picker"
        class="px-2 py-1.5 border border-surface-border rounded text-sm"
      >
        <option value="local">Local</option>
        <option value="utc">UTC</option>
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
      <button
        class="px-2 py-1.5 border border-surface-border rounded text-sm hover:bg-surface-sunken"
        :class="autoRefreshActive ? 'text-accent' : 'text-text-muted'"
        @click="toggleAutoRefresh"
      >
        {{ autoRefreshActive ? 'Auto ⟳' : 'Paused' }}
      </button>
      <span class="ml-auto text-xs text-text-muted font-mono">
        {{ logRows.length }} lines
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
        v-if="loading && logRows.length === 0"
        class="py-12 text-center text-text-muted text-sm"
      >
        Loading logs...
      </div>
      <div
        v-else-if="logRows.length === 0"
        class="py-12 text-center text-text-muted text-sm"
      >
        No log entries
      </div>
      <div
        v-for="(row, i) in logRows"
        :key="i"
        :class="[
          'px-3 py-0.5 font-mono text-xs leading-relaxed hover:bg-surface-sunken',
          logLevelClass(row.entry.level),
        ]"
      >
        <RouterLink
          v-if="row.taskRef && props.taskId"
          :to="`/job/${encodeURIComponent(props.taskId)}/task/${encodeURIComponent(row.taskRef.taskId)}`"
          class="text-accent hover:underline mr-2"
          :title="row.taskRef.taskId"
        >
          T{{ row.taskRef.taskIndex }}
        </RouterLink>
        <span class="text-text-muted mr-2">{{ formatLogTime(timestampMs(row.entry.timestamp), timeZone === 'utc') }}</span>
        <span class="whitespace-pre-wrap break-all"><template
          v-for="(seg, j) in row.segments"
          :key="j"
        ><RouterLink
          v-if="seg.to"
          :to="seg.to"
          class="text-accent hover:underline"
        >{{ seg.text }}</RouterLink><template v-else>{{ seg.text }}</template></template></span>
      </div>
    </div>
  </div>
</template>
