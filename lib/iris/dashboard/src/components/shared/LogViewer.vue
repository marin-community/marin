<script setup lang="ts">
import { ref, computed, nextTick, onMounted, onUnmounted, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { logServiceRpcCall } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { useIndexCursor } from '@/composables/useIndexCursor'
import { useLogSearch } from '@/composables/useLogSearch'
import { useCopyToClipboard } from '@/composables/useCopyToClipboard'
import { useTimeWindow } from '@/composables/useTimeWindow'
import TimeRangeMenu from '@/components/shared/TimeRangeMenu.vue'
import { isFederated, type FetchLogsResponse, type LogEntry, type TaskAttempt } from '@/types/rpc'
import { timestampMs, logLevelName, formatLogTime, type TimeZoneName } from '@/utils/formatting'
import { parseLogLinks } from '@/utils/logLinks'
import { groupNearbyIndices, highlightSegments, isExceptionEntry, type HighlightedSegment } from '@/utils/logSearch'
import { CONTROL, CONTROL_OFF, CONTROL_ON, CONTROL_SM, FIELD, SELECT } from '@/styles/controls'

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
// Lines fetched either side of a row, or of a run of filter hits, when expanding
// context. The default is `grep -C 25`; the other two are the "just the
// neighbours" and "show me the whole episode" ends of the same question.
const CONTEXT_LINES_DEFAULT = 25
const CONTEXT_LINE_CHOICES = [10, CONTEXT_LINES_DEFAULT, 100]
// Bound the spliced-in context set; the oldest expansions are evicted first.
const MAX_CONTEXT_ROWS = 5_000
// Expanding a filtered set costs two reads per run of hits, so bound the runs: a
// filter that matches all over the stream must not turn one click into hundreds
// of fetches. The panel says so when it stops short.
const MAX_CONTEXT_RUNS = 10
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

// Server-side narrowing: `filter` becomes FetchLogs.regex, so non-matching
// lines never arrive. Distinct from `search` below, which only marks lines.
const filter = ref('')
const level = ref('info')
const pageLines = ref(500)
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

// Timezone for rendering log timestamps and reading typed instants. Stored
// timestamps are UTC epoch ms regardless; 'local' shows the browser's zone
// (default), 'utc' lines the prefix up with the UTC timestamps processes embed
// in their raw log lines.
const timeZone = ref<TimeZoneName>('local')

const { presetMs, sinceInstant, absolute: absoluteSince, sinceMs, setSinceMs, selectPreset } = useTimeWindow()

const entries = ref<LogEntry[]>([])
// Lines pulled in by expanding context, keyed by seq so they dedupe against
// `entries` and splice into the right place. Iteration follows first insertion,
// so an expansion evicts the earliest-expanded window first.
const contextEntries = ref<Map<number, LogEntry>>(new Map())
const contextLines = ref(CONTEXT_LINES_DEFAULT)
const contextPending = ref(false)
// Set when an expansion covered only the first MAX_CONTEXT_RUNS runs of hits.
const contextTruncated = ref(false)
const loading = ref(false)
const errorMsg = ref<string | null>(null)
// proto JSON encodes int64 as string; 0/"0" both mean "no cursor".
const cursor = ref<string | number | null>(null)

const wrap = ref(true)
const followTail = ref(true)
const selectedSeq = ref<number | null>(null)
const scrollBox = ref<HTMLElement | null>(null)

// Where the window sits in the stream, under the current query. Each flag is the
// end the matching step has run out of, so it also says whether that step is
// still offered; `reachedNewest` additionally gates polling, which has nowhere
// to put new lines while the window is paged back.
const paging = ref(false)
const reachedOldest = ref(false)
const reachedNewest = ref(false)

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

const contextActive = computed(() => contextEntries.value.size > 0)

/** The clock span of the loaded window, labelling what the pager moves. */
const loadedSpan = computed(() => {
  const rows = logRows.value
  if (rows.length === 0) return ''
  const first = timestampMs(rows[0].entry.timestamp)
  const last = timestampMs(rows[rows.length - 1].entry.timestamp)
  if (!first || !last) return ''
  // Seconds are enough at this scale; the per-row column carries milliseconds.
  const clock = (ms: number) => formatLogTime(ms, timeZone.value).slice(0, 8)
  return `${clock(first)}–${clock(last)}`
})

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

function requestSinceMs(): number | undefined {
  const ms = sinceMs()
  if (ms === undefined || !absoluteSince.value) return ms
  // FetchLogs applies sinceMs as an exclusive bound. The date picker describes
  // an inclusive start time, so move the wire bound back by one millisecond.
  return Math.max(0, ms - 1)
}

function baseRequest() {
  return {
    ...sourceRequest(),
    regex: filter.value || undefined,
    minLevel: level.value ? level.value.toUpperCase() : undefined,
    sinceMs: requestSinceMs(),
  }
}

/**
 * Read one page, plus a probe row past its far end.
 *
 * A page that comes back exactly the size it asked for says nothing about
 * whether the store had more; asking for one row more than fits does. The probe
 * is trimmed off before the page is shown, so it only ever answers whether the
 * step in that direction is still available.
 */
async function fetchPage(request: Record<string, unknown>, limit: number, tail: boolean) {
  const resp = await logServiceRpcCall<FetchLogsResponse>('FetchLogs', { ...request, maxLines: limit + 1, tail })
  const rows = resp.entries ?? []
  const more = rows.length > limit
  // A tailing read returns the newest limit + 1 rows, so the probe is the first
  // of them; a forward read returns the oldest, so the probe is the last.
  if (!more) return { rows, more }
  return { rows: tail ? rows.slice(1) : rows.slice(0, limit), more }
}

/** Where a poll resumes after this page: the last row shown, never the probe. */
function pageCursor(rows: LogEntry[]): string | null {
  return rows.length > 0 ? String(seqOf(rows[rows.length - 1])) : null
}

/**
 * Load the page at one end of the query and make it the whole window.
 *
 * `tail` picks the end. It defaults to the one a fresh query opens on: an
 * explicit start date means "read forward from here", so the first page is the
 * oldest one at/after sinceMs, while relative presets and the default view stay
 * anchored to now and open on the newest page. Following the stream asks for the
 * newest page either way — `sinceMs` only bounds the query from below, so it
 * constrains a tailing read without contradicting it.
 */
async function fetchTail(tail = !absoluteSince.value) {
  if (!sourceInput.value) {
    entries.value = []
    return
  }
  const gen = ++generation
  loading.value = true
  errorMsg.value = null
  try {
    const { rows, more } = await fetchPage(baseRequest(), pageLines.value, tail)
    if (gen !== generation) return
    collapseContext()
    entries.value = rows
    cursor.value = pageCursor(rows)
    if (tail) {
      reachedNewest.value = true
      reachedOldest.value = !more
    } else {
      // Nothing older belongs to a query already bounded below by a start time.
      reachedOldest.value = true
      reachedNewest.value = !more
    }
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
  // Only a window that already holds the newest line can absorb newer ones.
  // Paged back into the stream, polling has nowhere to put them; it picks up
  // again by itself once Follow returns the window to the tail.
  if (!reachedNewest.value) return
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
      const trimmed = combined.length > MAX_RETAINED_LINES
      entries.value = trimmed ? combined.slice(combined.length - MAX_RETAINED_LINES) : combined
      // Dropping the oldest rows to stay within the DOM budget gives up the
      // beginning of the stream, so the backward step is live again.
      if (trimmed) reachedOldest.value = false
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

const oldestSeq = computed(() => (entries.value.length ? seqOf(entries.value[0]) : 0))
const newestSeq = computed(() => (entries.value.length ? seqOf(entries.value[entries.value.length - 1]) : 0))

/**
 * Show the page of lines immediately before or after the one on screen.
 *
 * The window moves rather than grows, so the line count and span beside these
 * controls always describe exactly what is displayed, and the pair stays
 * symmetric: whichever end you are not at is a step you can take. A page keeps
 * the active filter, so paging a filtered view steps between hits rather than
 * between raw lines.
 */
async function loadPage(direction: 'older' | 'newer') {
  const older = direction === 'older'
  const anchor = older ? oldestSeq.value : newestSeq.value
  if (paging.value || anchor <= 0) return
  paging.value = true
  followTail.value = false
  const gen = ++generation
  try {
    const bound = older ? { untilCursor: anchor } : { cursor: anchor }
    const { rows, more } = await fetchPage({ ...baseRequest(), ...bound }, pageLines.value, older)
    if (gen !== generation) return
    errorMsg.value = null
    if (rows.length === 0) {
      if (older) reachedOldest.value = true
      else reachedNewest.value = true
      return
    }
    // Context was fetched around rows this page no longer holds.
    collapseContext()
    entries.value = rows
    cursor.value = pageCursor(rows)
    // The end just stepped away from is back in play either way.
    if (older) {
      reachedOldest.value = !more
      reachedNewest.value = false
    } else {
      reachedNewest.value = !more
      reachedOldest.value = false
    }
    // Land where reading continues: at the end of an earlier page, at the start
    // of a later one.
    if (older) scrollToBottom()
    else nextTick(() => { if (scrollBox.value) scrollBox.value.scrollTop = 0 })
  } catch (e) {
    if (gen !== generation) return
    errorMsg.value = e instanceof Error ? e.message : String(e)
  } finally {
    paging.value = false
  }
}

/** Put the window back on the live tail and track it. */
async function followNewest() {
  followTail.value = true
  if (!reachedNewest.value) await fetchTail(true)
  scrollToBottom()
}

/** Drop the pinned row and everything expanded around it. */
function clearAnchor() {
  selectedSeq.value = null
  collapseContext()
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
  reachedOldest.value = false
  reachedNewest.value = false
  clearAnchor()
  matchCursor.reset()
  exceptionCursor.reset()
  await fetchTail()
}

/** Drop everything expanded so far, leaving the query's own rows. */
function collapseContext() {
  contextEntries.value = new Map()
  contextTruncated.value = false
}

/**
 * The unfiltered neighbourhood of the seq range `[start, end]`.
 *
 * Reads are bounded by seq but limited by count, and the stream selector is
 * applied server-side, so `maxLines` counts rows of *this* source rather than
 * store rows. That is what makes "the 25 lines before this one" mean what it
 * says on a store shared by every task in the cluster.
 */
function contextRequests(start: number, end: number, lines: number) {
  return [
    // Exclusive, so tailing it yields the rows ending just before the range.
    { ...sourceRequest(), untilCursor: start, tail: true, maxLines: lines },
    // Also exclusive: start - 1 is the tightest bound that still returns the
    // first row. Reading forward from there covers the range's interior — the
    // rows the filter dropped — and `lines` more past its end.
    {
      ...sourceRequest(),
      cursor: start - 1,
      tail: false,
      maxLines: Math.min(end - start + lines + 1, MAX_CONTEXT_ROWS),
    },
  ]
}

/** Merge fetched rows into the spliced set, evicting the earliest expansions. */
function spliceContext(fetched: LogEntry[][]) {
  const next = new Map(contextEntries.value)
  for (const batch of fetched) for (const entry of batch) next.set(seqOf(entry), entry)
  while (next.size > MAX_CONTEXT_ROWS) {
    const oldest = next.keys().next().value
    if (oldest === undefined) break
    next.delete(oldest)
  }
  contextEntries.value = next
}

async function fetchContext(ranges: { start: number; end: number }[]) {
  if (contextPending.value || ranges.length === 0) return
  contextPending.value = true
  try {
    const requests = ranges.flatMap((range) => contextRequests(range.start, range.end, contextLines.value))
    const responses = await Promise.all(
      requests.map((request) => logServiceRpcCall<FetchLogsResponse>('FetchLogs', request)),
    )
    spliceContext(responses.map((resp) => resp.entries ?? []))
    errorMsg.value = null
  } catch (e) {
    errorMsg.value = e instanceof Error ? e.message : String(e)
  } finally {
    contextPending.value = false
  }
}

/** Splice the neighbourhood of one row into the view. */
async function loadContext(seq: number) {
  if (seq <= 0) return
  await fetchContext([{ start: seq, end: seq }])
}

/** Ascending seqs collapsed into runs, merging any whose windows would overlap. */
function contextRuns(seqs: number[], lines: number): { start: number; end: number }[] {
  const runs: { start: number; end: number }[] = []
  for (const seq of seqs) {
    const last = runs[runs.length - 1]
    if (last !== undefined && seq - last.end <= lines) last.end = seq
    else runs.push({ start: seq, end: seq })
  }
  return runs
}

/**
 * Bring back the lines the filter hides, around every loaded hit at once.
 *
 * The per-row control answers "what happened around this line". With a filter on
 * and dozens of hits, asking that one row at a time is the tedious part, so this
 * is the same expansion over the whole result set — `grep -C` on what is loaded.
 */
async function expandFilteredContext() {
  if (contextPending.value) return
  const seqs = entries.value.map(seqOf).filter((seq) => seq > 0)
  const runs = contextRuns(seqs, contextLines.value)
  contextTruncated.value = runs.length > MAX_CONTEXT_RUNS
  await fetchContext(runs.slice(0, MAX_CONTEXT_RUNS))
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

// Free-text fields (source key, regex filter) apply on Enter, not on every
// keystroke. The discrete selectors below refetch immediately on change. The
// match-scope select refetches via @change rather than a watch, so the
// reassignment in applyDefaults() doesn't fire a redundant second fetch.
watch(selectedAttemptId, applyDefaults)
watch(pageLines, resetAndFetch)
watch(level, resetAndFetch)
watch([presetMs, sinceInstant], resetAndFetch)
// The zone is not in the query: an absolute bound is held as an instant, so
// switching clocks re-labels it and the timestamp column without moving either.

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

/** Set the log time bound to this row. */
function setStartTime(entry: LogEntry) {
  const ms = timestampMs(entry.timestamp)
  if (ms > 0) setSinceMs(ms)
}

/** Promote the client-side search into the server-side filter over the whole log. */
function promoteSearchToFilter() {
  const pattern = search.useRegex.value
    ? search.query.value
    : search.query.value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  filter.value = search.caseSensitive.value ? pattern : `(?i)${pattern}`
  resetAndFetch()
}

function clearFilter() {
  filter.value = ''
  resetAndFetch()
}

function onScroll() {
  const box = scrollBox.value
  if (!box) return
  const atEnd = box.scrollHeight - box.scrollTop - box.clientHeight < FOLLOW_TAIL_SLACK_PX
  // Reaching the end of a page that is not the live tail is not following it;
  // paging back lands at the bottom of an earlier window on purpose.
  followTail.value = atEnd && reachedNewest.value
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
    <!-- Stream row: which keys this panel reads -->
    <div class="flex flex-wrap items-center gap-2">
      <input
        v-model="sourceInput"
        type="text"
        spellcheck="false"
        placeholder="Source key, e.g. /alice/job/ or /system/worker/… (Enter to apply)"
        :class="[FIELD, 'h-7 w-full text-sm sm:w-96']"
        @keyup.enter="resetAndFetch"
      />
      <select
        v-model="matchScope"
        title="How the source is matched against log keys"
        :class="SELECT"
        @change="resetAndFetch"
      >
        <option value="EXACT">Exact</option>
        <option value="PREFIX">Prefix</option>
        <option value="REGEX">Regex</option>
      </select>
      <select v-if="attempts && attempts.length > 0" v-model.number="selectedAttemptId" :class="SELECT">
        <option :value="-1">All attempts</option>
        <option v-for="a in attempts" :key="a.attemptId" :value="a.attemptId">
          Attempt {{ a.attemptId }}
        </option>
      </select>
    </div>

    <!-- Line row: which of that stream's lines to fetch -->
    <div class="flex flex-wrap items-center gap-2">
      <input
        v-model="filter"
        type="text"
        data-log-filter
        title="Server-side regex filter: drops every line that does not match"
        placeholder="Filter regex… (Enter)"
        :class="[FIELD, 'h-7 w-full text-sm sm:w-56']"
        @keyup.enter="resetAndFetch"
      />
      <select v-model="level" title="Minimum severity" :class="SELECT">
        <option value="debug">Debug</option>
        <option value="info">Info</option>
        <option value="warning">Warning</option>
        <option value="error">Error</option>
      </select>
      <TimeRangeMenu
        :preset-ms="presetMs"
        :since-instant="sinceInstant"
        :time-zone="timeZone"
        @select-preset="selectPreset"
        @set-since="setSinceMs"
        @set-time-zone="timeZone = $event"
      />
      <select
        v-model.number="pageLines"
        data-log-page-size
        title="How many lines one page holds"
        :class="SELECT"
      >
        <option :value="100">100 per page</option>
        <option :value="500">500 per page</option>
        <option :value="1000">1,000 per page</option>
        <option :value="5000">5,000 per page</option>
        <option :value="10000">10,000 per page</option>
      </select>
      <button
        type="button"
        :class="[CONTROL, autoRefreshActive ? CONTROL_ON : CONTROL_OFF]"
        :title="autoRefreshActive
          ? `Polling for new lines every ${POLL_INTERVAL_MS / 1000}s`
          : 'Paused; the view holds the lines already fetched'"
        @click="toggleAutoRefresh"
      >{{ autoRefreshActive ? 'Auto ⟳' : 'Paused' }}</button>
    </div>

    <div
      v-if="errorMsg"
      class="px-3 py-2 text-sm text-status-danger bg-status-danger-bg rounded border border-status-danger-border"
    >
      {{ errorMsg }}
    </div>

    <!-- The view bar, the context notice and the log body read as one panel, so
         they share a border and sit outside the toolbar's vertical rhythm. -->
    <div class="rounded-lg border border-surface-border overflow-hidden">
      <!-- View bar: everything that acts on the lines already loaded. Controls
           are grouped by what they answer, and each group is one flex item, so
           a narrow viewport wraps them in whole groups rather than scattering
           them individually. -->
      <div class="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-surface-border
                  bg-surface-raised px-2 py-1.5">
        <div class="relative">
          <input
            ref="searchInput"
            v-model="search.query.value"
            type="text"
            spellcheck="false"
            data-log-search
            title="Highlights matches in the lines already loaded, keeping their context. Press / to focus, Enter or n for the next match."
            placeholder="Search loaded lines… (/)"
            :class="[FIELD, 'h-6 w-64 pr-16 text-xs', search.error.value ? 'border-status-danger' : '']"
            @keydown.enter.prevent="gotoMatch($event.shiftKey ? -1 : 1)"
            @keydown.esc.prevent="search.query.value = ''"
          />
          <div class="absolute inset-y-0 right-1 flex items-center gap-0.5">
            <button
              type="button"
              title="Match case"
              class="rounded px-1 font-mono text-xs hover:bg-surface-sunken"
              :class="search.caseSensitive.value ? 'text-accent' : 'text-text-muted'"
              @click="search.caseSensitive.value = !search.caseSensitive.value"
            >Aa</button>
            <button
              type="button"
              title="Regular expression"
              class="rounded px-1 font-mono text-xs hover:bg-surface-sunken"
              :class="search.useRegex.value ? 'text-accent' : 'text-text-muted'"
              @click="search.useRegex.value = !search.useRegex.value"
            >.*</button>
          </div>
        </div>

        <div v-if="search.applied.value && !search.error.value" class="inline-flex items-center gap-1">
          <span class="font-mono text-xs tabular-nums text-text-muted">
            {{ matchIndices.length
              ? `${matchCursor.position.value >= 0 ? matchCursor.position.value + 1 : '–'} / ${matchIndices.length}`
              : 'no matches' }}
          </span>
          <button
            type="button"
            :class="[CONTROL_SM, CONTROL_OFF]"
            title="Previous match (Shift+Enter or N)"
            :disabled="!matchIndices.length"
            @click="gotoMatch(-1)"
          >↑</button>
          <button
            type="button"
            :class="[CONTROL_SM, CONTROL_OFF]"
            title="Next match (Enter or n)"
            :disabled="!matchIndices.length"
            @click="gotoMatch(1)"
          >↓</button>
          <button
            v-if="!matchIndices.length"
            type="button"
            :class="[CONTROL_SM, 'border-accent-border bg-surface text-accent hover:bg-surface-sunken']"
            title="No match among the loaded lines. Re-query the whole log with this text as a server-side filter."
            @click="promoteSearchToFilter"
          >Filter the full log</button>
        </div>
        <span v-else-if="search.error.value" class="text-xs text-status-danger">{{ search.error.value }}</span>

        <div class="inline-flex items-center gap-1">
          <button
            type="button"
            :class="[CONTROL_SM, exceptionIndices.length
              ? 'border-status-danger-border bg-surface text-status-danger hover:bg-surface-sunken'
              : 'border-surface-border bg-surface text-text-muted']"
            :title="exceptionIndices.length
              ? 'Jump to the next traceback or fatal error (e)'
              : 'No traceback or fatal error among the loaded lines'"
            :disabled="!exceptionIndices.length"
            @click="gotoException(1)"
          >⚠ {{ exceptionLabel }}</button>
          <button
            type="button"
            :class="[CONTROL_SM, CONTROL_OFF]"
            title="Previous exception (E)"
            :disabled="!exceptionIndices.length"
            @click="gotoException(-1)"
          >↑</button>
        </div>

        <!-- Pager: the two steps flank the window they move, so which way each
             one goes is legible without reading the labels. -->
        <div class="ml-auto inline-flex items-center gap-1">
          <button
            type="button"
            :class="[CONTROL_SM, CONTROL_OFF]"
            :disabled="paging || reachedOldest || !entries.length"
            :title="reachedOldest
              ? 'Beginning of the stream'
              : `Load the ${pageLines} lines before this window`"
            @click="loadPage('older')"
          >← Older</button>
          <span class="whitespace-nowrap px-1 font-mono text-xs tabular-nums text-text-muted">
            {{ logRows.length }} lines<template v-if="loadedSpan"> · {{ loadedSpan }}</template>
          </span>
          <button
            type="button"
            :class="[CONTROL_SM, CONTROL_OFF]"
            :disabled="paging || reachedNewest || !entries.length"
            :title="reachedNewest
              ? 'Caught up with the newest line'
              : `Load the ${pageLines} lines after this window`"
            @click="loadPage('newer')"
          >Newer →</button>
        </div>

        <div class="inline-flex items-center gap-1">
          <button
            type="button"
            :class="[CONTROL_SM, wrap ? CONTROL_ON : CONTROL_OFF]"
            title="Wrap long lines"
            @click="wrap = !wrap"
          >Wrap</button>
          <button
            type="button"
            :class="[CONTROL_SM, followTail ? CONTROL_ON : CONTROL_OFF]"
            title="Return to the newest line and keep it in view as logs arrive"
            @click="followNewest"
          >Follow</button>
          <button
            type="button"
            :class="[CONTROL_SM, copied ? 'border-status-success-border bg-surface text-status-success'
              : copyError ? 'border-status-danger-border bg-surface text-status-danger' : CONTROL_OFF]"
            :disabled="!logRows.length"
            title="Copy the loaded lines to the clipboard as JSON"
            @click="copyLogs"
          >{{ copied ? 'Copied' : copyError ? 'Failed' : 'Copy JSON' }}</button>
        </div>
      </div>

      <!-- Context notice: what the filter is hiding, and how to get it back. -->
      <div
        v-if="filterActive || contextActive"
        class="flex flex-wrap items-center gap-x-2 gap-y-1 border-b border-surface-border
               bg-surface-raised px-3 py-1 text-xs text-text-secondary"
      >
        <span v-if="filterActive">
          Filtered to lines matching <code class="font-mono text-text">{{ filter }}</code> — surrounding
          lines are hidden.
        </span>
        <span v-else>Showing the lines expanded around pinned rows.</span>
        <span v-if="contextTruncated" class="text-status-warning">
          Expanded the first {{ MAX_CONTEXT_RUNS }} runs of hits only.
        </span>

        <div class="ml-auto inline-flex items-center gap-1">
          <select
            v-model.number="contextLines"
            title="Lines kept either side of a hit, here and on a row's ⋯"
            class="h-6 rounded border border-surface-border bg-surface px-1 text-xs text-text"
          >
            <option v-for="n in CONTEXT_LINE_CHOICES" :key="n" :value="n">± {{ n }} lines</option>
          </select>
          <button
            v-if="filterActive"
            type="button"
            :class="[CONTROL_SM, CONTROL_OFF]"
            :disabled="contextPending || !entries.length"
            title="Bring back the hidden lines around every loaded hit at once"
            @click="expandFilteredContext"
          >{{ contextPending ? 'Expanding…' : 'Expand all' }}</button>
          <button
            v-if="contextActive"
            type="button"
            :class="[CONTROL_SM, CONTROL_OFF]"
            title="Drop the expanded lines and show only the query's own hits"
            @click="collapseContext"
          >Collapse</button>
          <button v-if="filterActive" type="button" class="px-1 text-accent hover:underline" @click="clearFilter">
            Clear filter
          </button>
        </div>
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
              class="shrink-0 w-4 select-none text-text-muted opacity-0 group-hover:opacity-100
                     focus-visible:opacity-100 hover:text-accent disabled:opacity-40"
              :title="`Show the ${contextLines} lines either side of this one, unfiltered`"
              :disabled="contextPending"
              @click="loadContext(row.seq)"
            >⋯</button>
            <span v-else class="shrink-0 w-4 select-none" />
            <template v-if="showTaskLinks">
              <RouterLink
                v-if="row.taskRef && props.taskId"
                :to="`/job/${encodeURIComponent(props.taskId)}/task/${encodeURIComponent(row.taskRef.taskId)}`"
                class="shrink-0 w-12 overflow-hidden select-none text-right tabular-nums
                       text-accent hover:underline"
                :title="row.taskRef.taskId"
              >T{{ row.taskRef.taskIndex }}</RouterLink>
              <span v-else class="shrink-0 w-12 select-none" />
            </template>
            <span class="shrink-0 select-none inline-flex items-center gap-1">
              <button
                data-log-permalink
                class="text-text-muted tabular-nums hover:text-accent hover:underline"
                :title="`${isoTimestamp(row.entry)} — click to pin and link to this line`"
                @click="selectRow(row.seq)"
              >{{ formatLogTime(timestampMs(row.entry.timestamp), timeZone) }}</button>
              <button
                data-log-start
                class="text-text-muted opacity-50 hover:opacity-100 hover:text-accent"
                :title="`${isoTimestamp(row.entry)} — show logs from this time`"
                :aria-label="`Show logs from ${isoTimestamp(row.entry)}`"
                @click="setStartTime(row.entry)"
              >▶</button>
            </span>
            <!-- The gutter is select-none: each item is a flex item, so a browser
                 would otherwise copy every one of them onto its own line. -->
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
