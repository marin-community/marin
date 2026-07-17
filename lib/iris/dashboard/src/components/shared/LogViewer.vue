<script setup lang="ts">
import { ref, computed, nextTick, onMounted, onUnmounted, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { logServiceRpcCall } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { useIndexCursor } from '@/composables/useIndexCursor'
import { useLogSearch } from '@/composables/useLogSearch'
import { useCopyToClipboard } from '@/composables/useCopyToClipboard'
import { CUSTOM_PRESET, SINCE_PRESETS, type TimeZoneName, useTimeWindow } from '@/composables/useTimeWindow'
import { isFederated, type FetchLogsResponse, type LogEntry, type TaskAttempt } from '@/types/rpc'
import { timestampMs, logLevelName, formatLogTime } from '@/utils/formatting'
import { parseLogLinks } from '@/utils/logLinks'
import { groupNearbyIndices, highlightSegments, isExceptionEntry, type HighlightedSegment } from '@/utils/logSearch'

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
// Lines fetched either side of a row when expanding its context.
const CONTEXT_LINES = 25
// Bound the spliced-in context set; the oldest expansions are evicted first.
const MAX_CONTEXT_ROWS = 5_000
// Rows within this distance of each other count as one failure when stepping
// through exceptions — a traceback trips several patterns a few lines apart.
const EXCEPTION_GROUP_GAP = 5
// Re-running the matcher over every loaded line on each keystroke is wasteful.
const SEARCH_DEBOUNCE_MS = 120
// Treat the viewport as "at the bottom" within this many pixels of the end.
const FOLLOW_TAIL_SLACK_PX = 40

const route = useRoute()
const router = useRouter()

type MatchScope = 'EXACT' | 'PREFIX' | 'REGEX'
type WireMatchScope = 'MATCH_SCOPE_EXACT' | 'MATCH_SCOPE_PREFIX' | 'MATCH_SCOPE_REGEX'
const WIRE_SCOPE: Record<MatchScope, WireMatchScope> = {
  EXACT: 'MATCH_SCOPE_EXACT',
  PREFIX: 'MATCH_SCOPE_PREFIX',
  REGEX: 'MATCH_SCOPE_REGEX',
}

// Server-side narrowing: `filter` becomes FetchLogs.substring, so non-matching
// lines never arrive. Distinct from `search` below, which only marks lines.
const filter = ref('')
const level = ref('info')
const tailLines = ref(500)
const selectedAttemptId = ref(props.currentAttemptId ?? -1)

// Client-side find-in-loaded-lines: marks matches and steps between them without
// removing their surrounding context.
const search = useLogSearch(SEARCH_DEBOUNCE_MS)
const searchInput = ref<HTMLInputElement | null>(null)

// Editable query. Pre-filled from props (task/worker/controller context) but the
// user is free to retype the source or widen the match scope — that's how you
// search by job (`/alice/job/` prefix), by user (`/alice/` prefix), or by an
// arbitrary key pattern (regex).
const sourceInput = ref('')
const matchScope = ref<MatchScope>('EXACT')

// Timezone for rendering log timestamps and interpreting the date picker.
// Stored timestamps are UTC epoch ms regardless; 'local' shows the browser's
// zone (default), 'utc' lines the prefix up with the UTC timestamps processes
// embed in their raw log lines.
const timeZone = ref<TimeZoneName>('local')

const { presetMs, customSince, absolute: absoluteSince, sinceMs, selectPreset } = useTimeWindow(timeZone)

const entries = ref<LogEntry[]>([])
// Lines pulled in by expanding a row's context, keyed by seq so they dedupe
// against `entries` and splice into the right place. Iteration follows first
// insertion, so `loadContext` evicts the earliest-expanded window first.
const contextEntries = ref<Map<number, LogEntry>>(new Map())
const contextPending = ref(false)
const loading = ref(false)
const errorMsg = ref<string | null>(null)
// proto JSON encodes int64 as string; 0/"0" both mean "no cursor".
const cursor = ref<string | number | null>(null)

const wrap = ref(true)
const followTail = ref(true)
const selectedSeq = ref<number | null>(null)
const scrollBox = ref<HTMLElement | null>(null)

// Task IDs end with a numeric segment (e.g. /alice/job/0), job IDs don't.
const isTask = computed(() => (sourceBase.value ? /\/\d+$/.test(sourceBase.value) : false))

/** The store row id, as a number. proto JSON renders int64 as a string. */
function seqOf(entry: LogEntry): number {
  return Number(entry.seq ?? 0)
}

function attemptFromRoute(): number {
  const raw = route.query.attempt
  const n = typeof raw === 'string' ? Number(raw) : NaN
  return Number.isInteger(n) && n >= 0 ? n : -1
}

function seqFromRoute(): number | null {
  const raw = route.query.logSeq
  const n = typeof raw === 'string' ? Number(raw) : NaN
  return Number.isInteger(n) && n > 0 ? n : null
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
function applyDefaults(): Promise<void> {
  const d = defaultSource()
  sourceInput.value = d.source
  matchScope.value = d.scope
  return resetAndFetch()
}

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
  seq: number
  isContext: boolean
  isException: boolean
  taskRef: TaskRef | null
  segments: HighlightedSegment[]
  hasMatch: boolean
}

// Context rows merge into the primary result set by seq. A row that is both a
// filter hit and part of an expanded window renders as a hit.
const mergedEntries = computed<{ entry: LogEntry; isContext: boolean }[]>(() => {
  if (contextEntries.value.size === 0) {
    return entries.value.map((entry) => ({ entry, isContext: false }))
  }
  const bySeq = new Map<number, { entry: LogEntry; isContext: boolean }>()
  for (const [seq, entry] of contextEntries.value) bySeq.set(seq, { entry, isContext: true })
  for (const entry of entries.value) bySeq.set(seqOf(entry), { entry, isContext: false })
  return [...bySeq.entries()].sort((a, b) => a[0] - b[0]).map(([, row]) => row)
})

// Link parsing is independent of the search query, so it stays out of the
// per-keystroke path below.
const linkedRows = computed(() =>
  mergedEntries.value.map(({ entry, isContext }) => ({
    entry,
    isContext,
    seq: seqOf(entry),
    isException: isExceptionEntry(entry),
    taskRef: showTaskLinks.value ? parseTaskFromKey(entry.key) : null,
    // proto3-JSON omits default scalars, so an empty log line arrives with
    // `data` absent (undefined); coalesce so matchAll() doesn't throw.
    segments: parseLogLinks(entry.data ?? ''),
  })),
)

const logRows = computed<LogRow[]>(() => {
  const matcher = search.matcher.value
  if (!matcher) return linkedRows.value.map((row) => ({ ...row, hasMatch: false }))
  return linkedRows.value.map((row) => {
    const ranges = matcher.find(row.entry.data ?? '')
    return { ...row, segments: highlightSegments(row.segments, ranges), hasMatch: ranges.length > 0 }
  })
})

const matchIndices = computed(() => logRows.value.flatMap((row, i) => (row.hasMatch ? [i] : [])))
const exceptionIndices = computed(() =>
  groupNearbyIndices(logRows.value.flatMap((row, i) => (row.isException ? [i] : [])), EXCEPTION_GROUP_GAP),
)

const matchCursor = useIndexCursor(matchIndices)
const exceptionCursor = useIndexCursor(exceptionIndices)
const filterActive = computed(() => filter.value.length > 0)

const exceptionLabel = computed(() => {
  const total = exceptionIndices.value.length
  if (total === 0) return 'No exceptions'
  if (exceptionCursor.position.value < 0) {
    return total === 1 ? 'Jump to exception' : `Jump to exception (${total})`
  }
  return `Exception ${exceptionCursor.position.value + 1} / ${total}`
})
// Monotonic generation to discard responses from superseded requests (e.g.
// when the filter changes while a poll is in flight).
let generation = 0

/** The stream selector: which keys this panel reads. Shared by every request. */
function sourceRequest() {
  return {
    source: sourceInput.value,
    matchScope: WIRE_SCOPE[matchScope.value],
    // Federated rows target the peer's relayed logs in the shared hub store; a
    // local row (or no cluster context) sends no filter and reads its own rows.
    cluster: isFederated(props.cluster) ? props.cluster : undefined,
  }
}

function baseRequest() {
  return {
    ...sourceRequest(),
    substring: filter.value || undefined,
    minLevel: level.value ? level.value.toUpperCase() : undefined,
    sinceMs: sinceMs(),
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
      tail: !absoluteSince.value,
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

/** Drop the pinned row and everything expanded around it. */
function clearAnchor() {
  selectedSeq.value = null
  contextEntries.value = new Map()
  if (route.query.logSeq !== undefined) {
    const { logSeq: _dropped, ...rest } = route.query
    router.replace({ query: rest })
  }
}

// Reset the cursor and do a full tail fetch. Used whenever the query changes
// (source, scope, substring, level, since, attempt, tail size) — the cursor
// from the previous query isn't meaningful for the new criteria.
async function resetAndFetch() {
  cursor.value = null
  entries.value = []
  clearAnchor()
  matchCursor.reset()
  exceptionCursor.reset()
  await fetchTail()
}

/**
 * Splice the neighbourhood of row `seq` into the view.
 *
 * The neighbourhood is unfiltered: only the stream selector is sent, so the
 * spliced rows include the ones the active filter hides.
 */
async function loadContext(seq: number) {
  if (seq <= 0 || contextPending.value) return
  contextPending.value = true
  try {
    const [before, fromAnchor] = await Promise.all([
      logServiceRpcCall<FetchLogsResponse>('FetchLogs', {
        ...sourceRequest(),
        // Exclusive, so tailing it yields the rows ending just before the anchor.
        untilCursor: seq,
        tail: true,
        maxLines: CONTEXT_LINES,
      }),
      logServiceRpcCall<FetchLogsResponse>('FetchLogs', {
        ...sourceRequest(),
        // Also exclusive: seq - 1 is the tightest bound that still returns the anchor.
        cursor: seq - 1,
        tail: false,
        maxLines: CONTEXT_LINES + 1,
      }),
    ])
    const next = new Map(contextEntries.value)
    for (const entry of [...(before.entries ?? []), ...(fromAnchor.entries ?? [])]) {
      next.set(seqOf(entry), entry)
    }
    while (next.size > MAX_CONTEXT_ROWS) {
      const oldest = next.keys().next().value
      if (oldest === undefined) break
      next.delete(oldest)
    }
    contextEntries.value = next
    errorMsg.value = null
  } catch (e) {
    errorMsg.value = e instanceof Error ? e.message : String(e)
  } finally {
    contextPending.value = false
  }
}

/** Pin a row, expand its context, and scroll it into view. */
async function revealSeq(seq: number) {
  followTail.value = false
  await loadContext(seq)
  selectRow(seq)
  const index = logRows.value.findIndex((row) => row.seq === seq)
  if (index >= 0) scrollToRow(index)
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

// A different query means the old match position is meaningless.
watch([search.applied, search.caseSensitive, search.useRegex], matchCursor.reset)


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

onMounted(async () => {
  window.addEventListener('keydown', onKeydown)
  // Capture the permalink before applyDefaults' reset strips it from the query.
  const anchor = seqFromRoute()
  if (props.taskId) {
    const routeAttempt = attemptFromRoute()
    if (routeAttempt >= 0) selectedAttemptId.value = routeAttempt
  }
  await applyDefaults()
  if (anchor !== null) await revealSeq(anchor)
})

onUnmounted(() => window.removeEventListener('keydown', onKeydown))


function scrollToRow(index: number) {
  nextTick(() => {
    const box = scrollBox.value
    const el = box?.querySelector<HTMLElement>(`[data-row="${index}"]`)
    if (!box || !el) return
    box.scrollTop = el.offsetTop - box.clientHeight / 2 + el.clientHeight / 2
  })
}

function scrollToBottom() {
  nextTick(() => {
    const box = scrollBox.value
    if (box) box.scrollTop = box.scrollHeight
  })
}

/** Pin a row by its index and bring it into view. */
function focusRow(rowIndex: number | null) {
  if (rowIndex === null) return
  followTail.value = false
  selectedSeq.value = logRows.value[rowIndex]?.seq ?? null
  scrollToRow(rowIndex)
}

function gotoMatch(delta: number) {
  focusRow(matchCursor.step(delta))
}

function gotoException(delta: number) {
  focusRow(exceptionCursor.step(delta))
}

/** Pin a row and make the address bar a link back to it. */
function selectRow(seq: number) {
  if (seq <= 0) return
  selectedSeq.value = seq
  followTail.value = false
  router.replace({ query: { ...route.query, logSeq: String(seq) } })
}

/** Promote the client-side search into the server-side filter over the whole log. */
function promoteSearchToFilter() {
  filter.value = search.query.value
  resetAndFetch()
}

function clearFilter() {
  filter.value = ''
  resetAndFetch()
}

function onScroll() {
  const box = scrollBox.value
  if (!box) return
  followTail.value = box.scrollHeight - box.scrollTop - box.clientHeight < FOLLOW_TAIL_SLACK_PX
}

// A start date means "read forward from here", so the interesting rows are at
// the top; only a tailing view chases the end of the stream.
watch(() => logRows.value.length, () => {
  if (followTail.value && !absoluteSince.value) scrollToBottom()
})

function isoTimestamp(entry: LogEntry): string {
  const ms = timestampMs(entry.timestamp)
  return ms ? new Date(ms).toISOString() : ''
}

// Serialize the currently loaded lines (respecting the active server-side
// filter and any expanded context) as a JSON array, one object per line.
function logsAsJson(): string {
  // In EXACT scope the backend omits per-row keys (they all equal the queried
  // source), so fall back to the query source for those rows.
  const exactSource = matchScope.value === 'EXACT' ? sourceInput.value : ''
  const rows = logRows.value.map((row) => ({
    seq: row.seq,
    timestamp: isoTimestamp(row.entry),
    level: logLevelName(row.entry.level),
    source: row.entry.key || exactSource,
    message: row.entry.data ?? '',
  }))
  return JSON.stringify(rows, null, 2)
}

const { copied, error: copyError, copy } = useCopyToClipboard()

function copyLogs() {
  copy(logsAsJson())
}

function onKeydown(e: KeyboardEvent) {
  // Only the visible viewer responds; a hidden tab keeps its handler inert.
  if (!scrollBox.value?.offsetParent) return
  if (e.metaKey || e.ctrlKey || e.altKey) return
  const target = e.target as HTMLElement | null
  const typing = !!target
    && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA'
      || target.tagName === 'SELECT' || target.isContentEditable)
  if (e.key === '/' && !typing) {
    e.preventDefault()
    searchInput.value?.focus()
    return
  }
  if (typing) return
  const bindings: Record<string, () => void> = {
    n: () => gotoMatch(1),
    N: () => gotoMatch(-1),
    e: () => gotoException(1),
    E: () => gotoException(-1),
  }
  const action = bindings[e.key]
  if (!action) return
  e.preventDefault()
  action()
}

function rowClasses(row: LogRow, index: number): string[] {
  const classes = ['border-l-2']
  if (row.isException) classes.push('border-status-danger', 'bg-status-danger-bg')
  else {
    const name = logLevelName(row.entry.level)
    if (name === 'error' || name === 'critical') classes.push('border-status-danger')
    else if (name === 'warning') classes.push('border-status-warning')
    else classes.push('border-transparent')
    // Zebra striping, dimmed for lines pulled in only as context.
    if (row.isContext) classes.push('bg-surface-sunken', 'text-text-secondary')
    else if (index % 2 === 1) classes.push('bg-surface-raised')
  }
  if (logLevelName(row.entry.level) === 'debug' && !row.isException) classes.push('text-text-muted')
  if (row.seq > 0 && row.seq === selectedSeq.value) classes.push('ring-1', 'ring-inset', 'ring-accent')
  else if (index === matchCursor.target.value) classes.push('ring-1', 'ring-inset', 'ring-accent-border')
  return classes
}

defineExpose({ selectedAttemptId })
</script>

<template>
  <div class="space-y-2">
    <!-- Query row: which log stream to read (source + scope + attempt) -->
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

    <!-- Filter row: server-side narrowing — non-matching lines never arrive -->
    <div class="flex flex-wrap items-center gap-2 sm:gap-3 text-sm">
      <input
        v-model="filter"
        type="text"
        title="Server-side filter: drops every line that does not contain this text"
        placeholder="Filter: keep only lines containing… (Enter)"
        class="w-full sm:w-56 px-3 py-1.5 bg-surface border border-surface-border rounded
               text-sm font-mono placeholder:text-text-muted
               focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
        @keyup.enter="resetAndFetch"
      />
      <select v-model="level" title="Minimum severity" class="px-2 py-1.5 border border-surface-border rounded text-sm">
        <option value="debug">Debug</option>
        <option value="info">Info</option>
        <option value="warning">Warning</option>
        <option value="error">Error</option>
      </select>
      <select
        :value="absoluteSince ? CUSTOM_PRESET : presetMs"
        title="Only show logs newer than this"
        class="px-2 py-1.5 border border-surface-border rounded text-sm"
        @change="selectPreset(Number(($event.target as HTMLSelectElement).value))"
      >
        <option v-if="absoluteSince" :value="CUSTOM_PRESET">Custom</option>
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
      <select v-model.number="tailLines" class="px-2 py-1.5 border border-surface-border rounded text-sm">
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
      <span class="ml-auto text-xs text-text-muted font-mono">{{ logRows.length }} lines</span>
    </div>

    <div
      v-if="errorMsg"
      class="px-3 py-2 text-sm text-status-danger bg-status-danger-bg rounded border border-status-danger-border"
    >
      {{ errorMsg }}
    </div>

    <!-- The find bar, the filter notice and the log body read as one panel, so
         they share a border and sit outside the toolbar's vertical rhythm. -->
    <div class="rounded-lg border border-surface-border overflow-hidden">
    <!-- Find bar: client-side search over the loaded lines, plus failure nav -->
    <div class="flex flex-wrap items-center gap-2 text-sm border-b border-surface-border
                bg-surface-raised px-2 py-1.5">
      <div class="relative">
        <input
          ref="searchInput"
          v-model="search.query.value"
          type="text"
          spellcheck="false"
          title="Highlights matches in the lines already loaded, keeping their context. Press / to focus, Enter or n for the next match."
          placeholder="Search loaded lines… (/)"
          class="w-56 pl-3 pr-16 py-1 bg-surface border border-surface-border rounded
                 text-sm font-mono placeholder:text-text-muted
                 focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
          :class="search.error.value ? 'border-status-danger' : ''"
          @keydown.enter.prevent="gotoMatch($event.shiftKey ? -1 : 1)"
          @keydown.esc.prevent="search.query.value = ''"
        />
        <div class="absolute inset-y-0 right-1 flex items-center gap-0.5">
          <button
            title="Match case"
            class="px-1 rounded text-xs font-mono hover:bg-surface-sunken"
            :class="search.caseSensitive.value ? 'text-accent' : 'text-text-muted'"
            @click="search.caseSensitive.value = !search.caseSensitive.value"
          >Aa</button>
          <button
            title="Regular expression"
            class="px-1 rounded text-xs font-mono hover:bg-surface-sunken"
            :class="search.useRegex.value ? 'text-accent' : 'text-text-muted'"
            @click="search.useRegex.value = !search.useRegex.value"
          >.*</button>
        </div>
      </div>

      <template v-if="search.applied.value && !search.error.value">
        <span class="text-xs font-mono text-text-muted tabular-nums">
          {{ matchIndices.length
            ? `${matchCursor.position.value >= 0 ? matchCursor.position.value + 1 : '–'} / ${matchIndices.length}`
            : 'no matches' }}
        </span>
        <button
          class="px-1.5 py-0.5 border border-surface-border rounded text-xs hover:bg-surface-sunken disabled:opacity-40"
          title="Previous match (Shift+Enter or N)"
          :disabled="!matchIndices.length"
          @click="gotoMatch(-1)"
        >↑</button>
        <button
          class="px-1.5 py-0.5 border border-surface-border rounded text-xs hover:bg-surface-sunken disabled:opacity-40"
          title="Next match (Enter or n)"
          :disabled="!matchIndices.length"
          @click="gotoMatch(1)"
        >↓</button>
        <button
          v-if="!matchIndices.length"
          class="px-1.5 py-0.5 border border-surface-border rounded text-xs text-accent hover:bg-surface-sunken"
          title="No match among the loaded lines. Re-query the whole log with this text as a server-side filter."
          @click="promoteSearchToFilter"
        >Filter the full log</button>
      </template>
      <span v-else-if="search.error.value" class="text-xs text-status-danger">{{ search.error.value }}</span>

      <div class="flex items-center gap-1 ml-auto">
        <button
          class="px-1.5 py-0.5 border border-surface-border rounded text-xs hover:bg-surface-sunken
                 disabled:opacity-40 disabled:hover:bg-transparent"
          :class="exceptionIndices.length ? 'text-status-danger' : 'text-text-muted'"
          :title="exceptionIndices.length
            ? 'Jump to the next traceback or fatal error (e)'
            : 'No traceback or fatal error among the loaded lines'"
          :disabled="!exceptionIndices.length"
          @click="gotoException(1)"
        >
          ⚠ {{ exceptionLabel }}
        </button>
        <button
          class="px-1.5 py-0.5 border border-surface-border rounded text-xs hover:bg-surface-sunken disabled:opacity-40"
          title="Previous exception (E)"
          :disabled="!exceptionIndices.length"
          @click="gotoException(-1)"
        >↑</button>
        <button
          class="px-2 py-0.5 border border-surface-border rounded text-xs hover:bg-surface-sunken"
          :class="copied ? 'text-status-success' : copyError ? 'text-status-danger' : 'text-text-muted'"
          :disabled="!logRows.length"
          title="Copy the loaded lines to the clipboard as JSON"
          @click="copyLogs"
        >{{ copied ? 'Copied' : copyError ? 'Failed' : 'Copy JSON' }}</button>
        <button
          class="px-2 py-0.5 border border-surface-border rounded text-xs hover:bg-surface-sunken"
          :class="wrap ? 'text-accent' : 'text-text-muted'"
          title="Wrap long lines"
          @click="wrap = !wrap"
        >Wrap</button>
        <button
          class="px-2 py-0.5 border border-surface-border rounded text-xs hover:bg-surface-sunken"
          :class="followTail ? 'text-accent' : 'text-text-muted'"
          title="Keep the newest line in view as logs arrive"
          @click="followTail = true; scrollToBottom()"
        >Follow</button>
      </div>
    </div>

      <div
        v-if="filterActive"
        class="flex items-center gap-2 px-3 py-1 text-xs border-b border-surface-border
               bg-surface-raised text-text-secondary"
      >
        <span>
          Filtered to lines containing <code class="font-mono text-text">{{ filter }}</code> — surrounding
          lines are hidden. Use <span class="font-mono">⋯</span> on a row to bring its context back.
        </span>
        <button class="ml-auto text-accent hover:underline" @click="clearFilter">Clear filter</button>
      </div>

      <div
        ref="scrollBox"
        class="relative overflow-auto bg-surface"
        :style="{ maxHeight: maxHeight }"
        @scroll.passive="onScroll"
      >
        <div v-if="loading && logRows.length === 0" class="py-12 text-center text-text-muted text-sm">
          Loading logs...
        </div>
        <div v-else-if="logRows.length === 0" class="py-12 text-center text-text-muted text-sm">
          No log entries
        </div>
        <!-- Sizing the list to its widest line (rather than each row to its own
             content) keeps the zebra stripes full-width when scrolled right. -->
        <div v-else :class="wrap ? '' : 'min-w-max'">
          <div
            v-for="(row, i) in logRows"
            :key="row.seq > 0 ? row.seq : `i${i}`"
            :data-row="i"
            class="group flex items-start gap-2 px-2 py-0.5 font-mono text-xs leading-relaxed
                   hover:bg-surface-sunken"
            :class="rowClasses(row, i)"
          >
            <button
              v-if="row.seq > 0"
              class="shrink-0 w-4 text-text-muted opacity-0 group-hover:opacity-100
                     focus-visible:opacity-100 hover:text-accent disabled:opacity-40"
              :title="`Show the ${CONTEXT_LINES} lines either side of this one, unfiltered`"
              :disabled="contextPending"
              @click="loadContext(row.seq)"
            >⋯</button>
            <span v-else class="shrink-0 w-4" />
            <RouterLink
              v-if="row.taskRef && props.taskId"
              :to="`/job/${encodeURIComponent(props.taskId)}/task/${encodeURIComponent(row.taskRef.taskId)}`"
              class="shrink-0 text-accent hover:underline"
              :title="row.taskRef.taskId"
            >
              T{{ row.taskRef.taskIndex }}
            </RouterLink>
            <button
              class="shrink-0 text-text-muted tabular-nums hover:text-accent hover:underline"
              :title="`${isoTimestamp(row.entry)} — click to pin and link to this line`"
              @click="selectRow(row.seq)"
            >{{ formatLogTime(timestampMs(row.entry.timestamp), timeZone === 'utc') }}</button>
            <span
              :class="wrap ? 'whitespace-pre-wrap break-words flex-1' : 'whitespace-pre'"
            ><template
              v-for="(seg, j) in row.segments"
              :key="j"
            ><RouterLink
              v-if="seg.to"
              :to="seg.to"
              class="text-accent hover:underline"
              :class="seg.match ? 'bg-status-warning-bg' : ''"
            >{{ seg.text }}</RouterLink><a
              v-else-if="seg.href"
              :href="seg.href"
              target="_blank"
              rel="noopener noreferrer"
              class="text-accent hover:underline"
              :class="seg.match ? 'bg-status-warning-bg' : ''"
            >{{ seg.text }}</a><mark
              v-else-if="seg.match"
              class="bg-status-warning-bg text-text rounded-sm"
            >{{ seg.text }}</mark><template v-else>{{ seg.text }}</template></template></span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
