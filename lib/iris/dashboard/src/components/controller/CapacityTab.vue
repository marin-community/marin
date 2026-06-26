<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { controllerRpcCall, useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import { SLICE_STATUS_STYLES, SLICE_STATUS_SUMMARY_ORDER, DIVERGING_COLORS, type SliceStatusStyle } from '@/types/status'
import type {
  GetAutoscalerStatusResponse,
  GetSchedulerStateResponse,
  ListUsersResponse,
  ListJobsResponse,
  AutoscalerStatus,
  ScaleGroupStatus,
  SliceInfo,
  GroupRoutingStatus,
  ProtoTimestamp,
  SchedulerUserBudget,
  UserSummary,
  PendingTaskBucket,
  RunningTaskBucket,
  JobStatus,
  JobQuery,
} from '@/types/rpc'
import { timestampMs, formatRelativeTime, bandDisplayName, bandColor } from '@/utils/formatting'
import { buildSliceView, type SliceJob, type SliceStatus, type SliceView } from '@/utils/slices'
import SliceList from '@/components/controller/SliceList.vue'
import FleetOverview from '@/components/controller/FleetOverview.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import LogViewer from '@/components/shared/LogViewer.vue'
import LoadingSpinner from '@/components/shared/LoadingSpinner.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import MetricCard from '@/components/shared/MetricCard.vue'

// ===========================================================================
// RPC + auto-refresh
//
// One auto-refresh drives every panel so the autoscaler, scheduler, user and
// pending-job views always describe the same moment. The pending-jobs list is a
// paginated/searchable ListJobs query, so it's fetched imperatively.
// ===========================================================================

const { data: autoscalerData, loading: autoscalerLoading, error: autoscalerError, refresh: refreshAutoscaler } =
  useControllerRpc<GetAutoscalerStatusResponse>('GetAutoscalerStatus')
const { data: schedulerData, error: schedulerError, refresh: refreshScheduler } =
  useControllerRpc<GetSchedulerStateResponse>('GetSchedulerState')
const { data: usersData, error: usersError, refresh: refreshUsers } =
  useControllerRpc<ListUsersResponse>('ListUsers')

// Updated on every refresh so relative slice ages advance with the data.
const nowMs = ref(Date.now())

async function refreshAll() {
  await Promise.all([refreshAutoscaler(), refreshScheduler(), refreshUsers(), fetchPendingJobs()])
  nowMs.value = Date.now()
}
useAutoRefresh(refreshAll, DEFAULT_REFRESH_MS)
onMounted(refreshAll)

// ===========================================================================
// Pending jobs query (searchable / paginated)
// ===========================================================================

const PENDING_PAGE_SIZE = 25
const pendingPage = ref(0)
const pendingSearch = ref('')
const pendingSearchInput = ref('')
const pendingJobs = ref<JobStatus[]>([])
const pendingTotal = ref(0)
const pendingLoading = ref(false)
const pendingError = ref<string | null>(null)

async function fetchPendingJobs() {
  pendingLoading.value = true
  pendingError.value = null
  try {
    const query: JobQuery = {
      scope: 'JOB_QUERY_SCOPE_ROOTS',
      stateFilter: 'pending',
      offset: pendingPage.value * PENDING_PAGE_SIZE,
      limit: PENDING_PAGE_SIZE,
      sortField: 'JOB_SORT_FIELD_DATE',
      sortDirection: 'SORT_DIRECTION_DESC',
    }
    if (pendingSearch.value.trim()) {
      query.nameFilter = pendingSearch.value.trim()
    }
    const resp = await controllerRpcCall<ListJobsResponse>('ListJobs', { query })
    pendingJobs.value = resp.jobs ?? []
    pendingTotal.value = resp.totalCount ?? 0
    // Clamp the page if the pending queue shrank underneath us during a refresh.
    const maxPage = Math.max(0, Math.ceil(pendingTotal.value / PENDING_PAGE_SIZE) - 1)
    if (pendingPage.value > maxPage) {
      pendingPage.value = maxPage
    }
  } catch (e) {
    pendingError.value = e instanceof Error ? e.message : String(e)
  } finally {
    pendingLoading.value = false
  }
}

const pendingTotalPages = computed(() =>
  Math.max(1, Math.ceil(pendingTotal.value / PENDING_PAGE_SIZE))
)

function applyPendingSearch() {
  pendingSearch.value = pendingSearchInput.value
  pendingPage.value = 0
  fetchPendingJobs()
}

watch(pendingPage, () => fetchPendingJobs())

// ===========================================================================
// Top-level state
// ===========================================================================

const loading = computed(() => autoscalerLoading.value && !autoscalerData.value)
const error = computed(() => autoscalerError.value || schedulerError.value || usersError.value)

const autoscaler = computed<AutoscalerStatus | null>(() => autoscalerData.value?.status ?? null)
const groups = computed(() => autoscaler.value?.groups ?? [])
const routing = computed(() => autoscaler.value?.lastRoutingDecision ?? null)
const unmetEntries = computed(() => routing.value?.unmetEntries ?? [])
const actions = computed(() => (autoscaler.value?.recentActions ?? []).slice().reverse())

// worker_id → jobs running on that host, from the scheduler running buckets the
// users table already fetches. SliceList renders these as per-job links in the
// expanded slice rows; status classification does not depend on it.
const sliceWorkerJobs = computed<Map<string, SliceJob[]>>(() => {
  const map = new Map<string, SliceJob[]>()
  for (const bucket of (schedulerData.value?.runningBuckets ?? []) as RunningTaskBucket[]) {
    if (!bucket.workerId || !bucket.jobId) continue
    if (!map.has(bucket.workerId)) map.set(bucket.workerId, [])
    map.get(bucket.workerId)!.push({
      jobId: bucket.jobId,
      userId: bucket.userId,
      taskCount: bucket.count,
      hostCount: 1,
    })
  }
  return map
})

// One view-model per slice, shared by the summary strip, the per-group badges, and
// the expanded list — so the row summary can never disagree with the detail.
const allSliceViews = computed<SliceView[]>(() =>
  groups.value.flatMap(g => (g.slices ?? []).map(s => buildSliceView(s, sliceWorkerJobs.value, nowMs.value)))
)
function countSliceStatus(views: SliceView[], status: SliceStatus): number {
  return views.reduce((n, v) => n + (v.status === status ? 1 : 0), 0)
}

const groupIndex = computed(() => {
  const index: Record<string, ScaleGroupStatus> = {}
  for (const g of groups.value) {
    if (g.name) index[g.name] = g
  }
  return index
})

// ===========================================================================
// Capacity summary metrics
//
// Slice-granular: lifecycle totals from sliceStateCounts, the capacity split
// from each slice's server-stamped capacity_status.
// ===========================================================================

const sliceTotals = computed<Record<string, number>>(() => {
  const totals: Record<string, number> = {}
  for (const g of groups.value) {
    for (const [state, count] of Object.entries(g.sliceStateCounts ?? {})) {
      totals[state] = (totals[state] ?? 0) + (count ?? 0)
    }
  }
  return totals
})
const totalSlices = computed(() => Object.values(sliceTotals.value).reduce((a, b) => a + b, 0))
// Idle spare = healthy slices eligible for scale-down. Degraded = ready slices
// with missing/unhealthy hosts. Both are slice-granular, straight off the shared
// capacity classification.
const totalIdle = computed(() => countSliceStatus(allSliceViews.value, 'idle'))
const totalDegradedSlices = computed(() => countSliceStatus(allSliceViews.value, 'degraded'))
const onlineGroups = computed(() =>
  groups.value.filter(g =>
    Object.values(g.sliceStateCounts ?? {}).reduce((a, b) => a + b, 0) > 0
  ).length
)
const totalDemand = computed(() =>
  groups.value.reduce((n, g) => n + (g.currentDemand ?? 0), 0)
)
const launchPlanned = computed(() => {
  const statuses = routing.value?.groupStatuses ?? []
  if (statuses.length > 0) {
    return statuses.reduce((n, gs) => n + (gs.launch ?? 0), 0)
  }
  return Object.values(routing.value?.groupToLaunch ?? {}).reduce((n, v) => n + (v ?? 0), 0)
})
const lastEvalMs = computed(() => timestampMs(autoscaler.value?.lastEvaluation))

function formatSliceSummary(totals: Record<string, number>): string {
  const total = Object.values(totals).reduce((a, b) => a + b, 0)
  if (total === 0) return '0'
  const order = ['ready', 'requesting', 'booting', 'initializing', 'failed']
  const parts = order
    .filter(state => (totals[state] ?? 0) > 0)
    .map(state => `${totals[state]} ${state}`)
  return parts.length > 0 ? parts.join(', ') : `${total}`
}

// ===========================================================================
// Pools table (the cohesive core)
//
// Rows are the routing decision's per-group statuses, grouped into quota pools
// and ordered by allocation tier. Pool-tier monotonicity (a low tier blocking
// higher tiers) has no server equivalent, so it stays client-side.
// ===========================================================================

const expandedSlices = ref<Set<string>>(new Set())
// Explicit per-pool open/closed intent (set on user click). Pools with no slices and
// no demand default to collapsed — just their one-line state summary — while the rest
// default to expanded. An override recorded here wins over that default.
const poolOverrides = ref<Map<string, boolean>>(new Map())
// Pools whose idle (zero-slice, idle-decision) tiers have been revealed. By default an
// expanded pool shows only its active tiers plus a "+N idle sizes" toggle.
const expandedIdleSizes = ref<Set<string>>(new Set())

function toggleSet(set: Set<string>, key: string): Set<string> {
  const next = new Set(set)
  next.has(key) ? next.delete(key) : next.add(key)
  return next
}
function toggleSlices(name: string) { expandedSlices.value = toggleSet(expandedSlices.value, name) }
function toggleIdleSizes(pool: string) { expandedIdleSizes.value = toggleSet(expandedIdleSizes.value, pool) }

const sortedGroupStatuses = computed<GroupRoutingStatus[]>(() => {
  const statuses = routing.value?.groupStatuses ?? []
  return statuses.slice().sort((a, b) => {
    const pa = a.priority ?? 100
    const pb = b.priority ?? 100
    if (pa !== pb) return pa - pb
    return (a.group ?? '').localeCompare(b.group ?? '')
  })
})

interface PoolSection {
  pool: string
  groups: GroupRoutingStatus[]
  blockedAtTier: number | null  // lowest tier in quota_exceeded/backoff, or null
}

const poolSections = computed<PoolSection[]>(() => {
  const poolMap = new Map<string, GroupRoutingStatus[]>()
  const unpooled: GroupRoutingStatus[] = []

  for (const gs of sortedGroupStatuses.value) {
    const pool = groupIndex.value[gs.group]?.quotaPool
    if (pool) {
      if (!poolMap.has(pool)) poolMap.set(pool, [])
      poolMap.get(pool)!.push(gs)
    } else {
      unpooled.push(gs)
    }
  }

  const sections: PoolSection[] = []
  for (const [pool, poolGroups] of poolMap) {
    poolGroups.sort((a, b) => {
      const ta = groupIndex.value[a.group]?.allocationTier ?? 0
      const tb = groupIndex.value[b.group]?.allocationTier ?? 0
      return ta - tb
    })

    let blockedAtTier: number | null = null
    for (const gs of poolGroups) {
      const group = groupIndex.value[gs.group]
      if (!group) continue
      const tier = group.allocationTier ?? 0
      const status = group.availabilityStatus
      if (tier > 0 && (status === 'quota_exceeded' || status === 'backoff')) {
        if (blockedAtTier === null || tier < blockedAtTier) blockedAtTier = tier
      }
    }
    sections.push({ pool, groups: poolGroups, blockedAtTier })
  }

  if (unpooled.length > 0) {
    sections.push({ pool: '__unpooled', groups: unpooled, blockedAtTier: null })
  }
  return sections
})

function isTierBlocked(gs: GroupRoutingStatus, section: PoolSection): boolean {
  if (!section.blockedAtTier) return false
  const tier = groupIndex.value[gs.group]?.allocationTier ?? 0
  return tier > section.blockedAtTier
}

function tierLabel(gs: GroupRoutingStatus): string {
  const tier = groupIndex.value[gs.group]?.allocationTier ?? 0
  return tier > 0 ? `T${tier}` : ''
}

function isInactiveRow(gs: GroupRoutingStatus): boolean {
  const counts = groupIndex.value[gs.group]?.sliceStateCounts ?? {}
  const total = Object.values(counts).reduce((a, b) => a + b, 0)
  return total === 0 && (gs.decision ?? 'idle') === 'idle'
}

// -- Per-group accessors (read server fields directly) --

function group(name: string): ScaleGroupStatus | undefined { return groupIndex.value[name] }
function groupFailures(name: string): number { return group(name)?.consecutiveFailures ?? 0 }
function groupSlices(name: string): SliceInfo[] { return group(name)?.slices ?? [] }
function groupHasSlices(name: string): boolean { return groupSlices(name).length > 0 }
function groupSliceCounts(name: string): Record<string, number> { return group(name)?.sliceStateCounts ?? {} }
function groupDemand(name: string): number { return group(name)?.currentDemand ?? 0 }

// Per-group slice view-models, built from the same buildSliceView the expanded
// list uses, so the row summary and the detail rows can never disagree.
function groupSliceViews(name: string): SliceView[] {
  return groupSlices(name).map(s => buildSliceView(s, sliceWorkerJobs.value, nowMs.value))
}

interface SliceStatusCount { status: SliceStatus; count: number; style: SliceStatusStyle }

// Non-ready lifecycle states have authoritative counts in sliceStateCounts and may
// have no SliceInfo row — `requesting` in particular counts pending scale-ups that
// have not been granted a slice handle yet.
const NON_READY_LIFECYCLE: SliceStatus[] = ['requesting', 'booting', 'initializing', 'failed']

// One slice-granular chip per present status, in the canonical display order.
// Ready slices are split into capacity statuses from their per-slice views; the
// non-ready states are read from sliceStateCounts so in-flight scale-ups still
// show even before they materialize as slices.
function groupStatusSummary(name: string): SliceStatusCount[] {
  const counts = new Map<SliceStatus, number>()
  for (const v of groupSliceViews(name)) {
    if (v.lifecycle === 'ready') counts.set(v.status, (counts.get(v.status) ?? 0) + 1)
  }
  const lifecycle = groupSliceCounts(name)
  for (const state of NON_READY_LIFECYCLE) {
    const n = lifecycle[state] ?? 0
    if (n > 0) counts.set(state, (counts.get(state) ?? 0) + n)
  }
  return SLICE_STATUS_SUMMARY_ORDER
    .filter(s => (counts.get(s) ?? 0) > 0)
    .map(s => ({ status: s, count: counts.get(s)!, style: SLICE_STATUS_STYLES[s] }))
}

// Total slices the group holds, from the same status rollup the chips use (real ready
// slices + in-flight lifecycle counts), so the tier ladder agrees with the chips.
function groupSliceCountTotal(name: string): number {
  return groupStatusSummary(name).reduce((n, c) => n + c.count, 0)
}

// Free = healthy slices that can take work now (available + idle). Both counts
// are slice-granular — a fully-booked healthy slice is `in_use`, not free.
function groupFreeSlices(name: string): number {
  return groupSliceViews(name).reduce(
    (n, v) => n + (v.status === 'available' || v.status === 'idle' ? 1 : 0),
    0,
  )
}
function groupDegradedSliceCount(name: string): number {
  return countSliceStatus(groupSliceViews(name), 'degraded')
}

// -- Pool-level rollups (the one-line state summary on each pool header) --

// Aggregate the slice-status chips across every group in a pool, in canonical order,
// so a collapsed pool still says what it holds (e.g. "52 in use · 3 booting").
function poolStatusSummary(section: PoolSection): SliceStatusCount[] {
  const counts = new Map<SliceStatus, number>()
  for (const gs of section.groups) {
    for (const c of groupStatusSummary(gs.group)) {
      counts.set(c.status, (counts.get(c.status) ?? 0) + c.count)
    }
  }
  return SLICE_STATUS_SUMMARY_ORDER
    .filter(s => (counts.get(s) ?? 0) > 0)
    .map(s => ({ status: s, count: counts.get(s)!, style: SLICE_STATUS_STYLES[s] }))
}
function poolDemand(section: PoolSection): number {
  return section.groups.reduce((n, gs) => n + groupDemand(gs.group), 0)
}
function poolLaunch(section: PoolSection): number {
  return section.groups.reduce((n, gs) => n + (gs.launch ?? 0), 0)
}
function poolSliceTotal(section: PoolSection): number {
  return poolStatusSummary(section).reduce((n, c) => n + c.count, 0)
}

// A group needs attention when its availability is constrained even if it holds no
// slices yet — these must stay visible rather than collapse into "no slices".
function groupNeedsAttention(name: string): boolean {
  const status = group(name)?.availabilityStatus
  return status === 'quota_exceeded' || status === 'backoff' || status === 'at_capacity' || status === 'requesting'
}
function groupIsActive(gs: GroupRoutingStatus): boolean {
  const decision = gs.decision ?? 'idle'
  return (decision !== 'idle' && decision !== '') || groupNeedsAttention(gs.group)
}

// A pool is "active" — and so expanded by default — when it has materialized slices,
// outstanding demand, a planned launch, a tier-blocking condition, or any group with a
// non-idle decision or constrained availability. Empty inert pools collapse to their
// one-line header so the table stays scannable.
function poolHasActivity(section: PoolSection): boolean {
  if (poolSliceTotal(section) > 0 || poolDemand(section) > 0 || poolLaunch(section) > 0 || section.blockedAtTier != null) {
    return true
  }
  return section.groups.some(groupIsActive)
}
function isPoolCollapsed(section: PoolSection): boolean {
  const override = poolOverrides.value.get(section.pool)
  return override !== undefined ? override : !poolHasActivity(section)
}
function togglePool(section: PoolSection) {
  const next = new Map(poolOverrides.value)
  next.set(section.pool, !isPoolCollapsed(section))
  poolOverrides.value = next
}

// Tiers worth showing by default: any with slices, demand, a non-idle decision, or
// constrained availability. The fully-idle remainder hides behind a "show N idle sizes"
// toggle (but never hide all of a pool's tiers — keep them if none are active).
function activeGroups(section: PoolSection): GroupRoutingStatus[] {
  return section.groups.filter(gs => !isInactiveRow(gs) || groupIsActive(gs))
}
function visibleGroups(section: PoolSection): GroupRoutingStatus[] {
  if (expandedIdleSizes.value.has(section.pool)) return section.groups
  const active = activeGroups(section)
  return active.length > 0 ? active : section.groups
}
function idleSizeCount(section: PoolSection): number {
  const active = activeGroups(section)
  return active.length > 0 ? section.groups.length - active.length : 0
}

// Shared layout/typography for the slice-status chips; each chip adds its own
// tone colors and a status dot on top of this base.
const BADGE_BASE = 'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-semibold border'

// -- Availability badge --

interface AvailabilityBadge { label: string; classes: string }

function groupAvailabilityBadge(g: ScaleGroupStatus, section?: PoolSection): AvailabilityBadge | null {
  const status = g.availabilityStatus
  const blockedMs = timestampMs(g.blockedUntil)
  const cooldownMs = timestampMs(g.scaleUpCooldownUntil)
  const now = Date.now()

  if (section?.blockedAtTier) {
    const tier = g.allocationTier ?? 0
    if (tier > section.blockedAtTier) {
      return { label: 'tier-blocked', classes: 'bg-status-danger-bg text-status-danger border-status-danger-border opacity-60' }
    }
  }
  if (status === 'requesting') {
    return { label: 'in-flight', classes: 'bg-status-purple-bg text-status-purple border-status-purple-border' }
  }
  if (status === 'backoff') {
    const label = blockedMs && blockedMs > now ? `backoff ${Math.ceil((blockedMs - now) / 1000)}s` : 'backoff'
    return { label, classes: 'bg-status-orange-bg text-status-orange border-status-orange-border' }
  }
  if (status === 'quota_exceeded') {
    const label = blockedMs && blockedMs > now ? `quota exceeded ${Math.ceil((blockedMs - now) / 1000)}s` : 'quota exceeded'
    return { label, classes: 'bg-status-danger-bg text-status-danger border-status-danger-border' }
  }
  if (status === 'at_capacity') {
    return { label: 'at capacity', classes: 'bg-status-warning-bg text-status-warning border-status-warning-border' }
  }
  if (cooldownMs && cooldownMs > now) {
    return { label: `cooldown ${Math.ceil((cooldownMs - now) / 1000)}s`, classes: 'bg-accent-subtle text-accent border-accent-border' }
  }
  return null
}

// -- Decision / reason --

function formatDecision(decision?: string): string {
  return (decision ?? 'idle').replace('_', ' ')
}

function decisionClasses(decision?: string): string {
  switch (decision ?? 'idle') {
    case 'selected': return 'text-status-success font-semibold'
    case 'scale_up': return 'text-accent font-semibold'
    case 'idle': return 'text-text-muted'
    case 'at_capacity': return 'text-status-warning'
    case 'backoff': return 'text-status-orange'
    case 'quota_exceeded': return 'text-status-danger'
    default: return 'text-text-secondary'
  }
}

function groupReasonText(gs: GroupRoutingStatus): string {
  let reason = gs.reason ?? ''
  const g = group(gs.group)
  if (g && (g.availabilityStatus === 'backoff' || g.availabilityStatus === 'quota_exceeded')) {
    const blockedMs = timestampMs(g.blockedUntil)
    if (blockedMs && blockedMs > Date.now()) {
      const secsLeft = Math.ceil((blockedMs - Date.now()) / 1000)
      reason = reason ? `${reason} (unblocks in ${secsLeft}s)` : `unblocks in ${secsLeft}s`
    }
  }
  return reason || '-'
}

/**
 * Row-level reconciliation note that correlates the two stories that the
 * autoscaler and scheduler used to tell separately:
 *   - free slices AND unmet demand → a real placement bug.
 *   - degraded slices AND unmet demand → workers recovering, not a bug.
 */
interface ReconcileNote { text: string; classes: string }

function groupReconcileNote(name: string): ReconcileNote | null {
  const demand = groupDemand(name)
  if (demand <= 0) return null
  if (groupFreeSlices(name) > 0) {
    return {
      text: 'free slices with unmet demand — scheduler not placing on healthy slices',
      classes: 'text-status-warning',
    }
  }
  if (groupDegradedSliceCount(name) > 0) {
    return {
      text: 'demand blocked on degraded slices — recovery in progress',
      classes: 'text-status-orange',
    }
  }
  return null
}

// ===========================================================================
// Unmet demand
// ===========================================================================

function taskIdToJob(taskId: string): string {
  if (!taskId) return 'unknown'
  const idx = taskId.lastIndexOf('/')
  return idx <= 0 ? taskId : taskId.slice(0, idx)
}

interface UnmetDemandRow {
  job: string
  entryCount: number
  exampleTask: string | null
  reasonCounts: Record<string, number>
  accelerators: Set<string>
}

const aggregatedUnmet = computed<UnmetDemandRow[]>(() => {
  const byJob = new Map<string, UnmetDemandRow>()
  for (const u of unmetEntries.value) {
    const entry = u.entry ?? {}
    const reason = u.reason ?? 'unknown'
    const taskIds = entry.taskIds ?? []
    const job = entry.coscheduleGroupId ?? taskIdToJob(taskIds[0] ?? '')
    if (!byJob.has(job)) {
      byJob.set(job, { job, entryCount: 0, exampleTask: null, reasonCounts: {}, accelerators: new Set() })
    }
    const row = byJob.get(job)!
    row.entryCount += 1
    if (!row.exampleTask && taskIds.length > 0) row.exampleTask = taskIds[0]
    row.reasonCounts[reason] = (row.reasonCounts[reason] ?? 0) + 1
    const deviceStr = [entry.deviceType, entry.deviceVariant].filter(Boolean).join(':') || 'unknown'
    row.accelerators.add(deviceStr)
  }
  return Array.from(byJob.values()).sort((a, b) => a.job.localeCompare(b.job))
})

function formatReasonCounts(counts: Record<string, number>): string {
  const entries = Object.entries(counts)
  if (entries.length === 0) return '-'
  return entries.map(([reason, count]) => `${reason.replace(/^[a-z_]+:\s*/, '')} (${count})`).join(', ')
}

// ===========================================================================
// Pending jobs → effective band annotation
// ===========================================================================

const pendingJobBand = computed<Map<string, string>>(() => {
  const out = new Map<string, string>()
  for (const bucket of schedulerData.value?.pendingBuckets ?? []) {
    if (!out.has(bucket.jobId)) out.set(bucket.jobId, bucket.band)
  }
  return out
})

// ===========================================================================
// Users & quotas
// ===========================================================================

const TERMINAL_JOB_STATES = new Set(['succeeded', 'failed', 'killed', 'worker_failed', 'preempted'])
const BANDS = ['PRIORITY_BAND_PRODUCTION', 'PRIORITY_BAND_INTERACTIVE', 'PRIORITY_BAND_BATCH'] as const
type Band = typeof BANDS[number]

const userBudgets = computed<SchedulerUserBudget[]>(() => schedulerData.value?.userBudgets ?? [])
const users = computed<UserSummary[]>(() => usersData.value?.users ?? [])

interface BandBreakdown {
  running: Record<Band, number>
  pending: Record<Band, number>
}

function emptyBandBreakdown(): BandBreakdown {
  const running = Object.fromEntries(BANDS.map(b => [b, 0])) as Record<Band, number>
  const pending = Object.fromEntries(BANDS.map(b => [b, 0])) as Record<Band, number>
  return { running, pending }
}

function bandBreakdownTotal(b: BandBreakdown): number {
  return BANDS.reduce((acc, band) => acc + b.running[band] + b.pending[band], 0)
}

// Per-user task counts per effective band, split running vs pending. Band is a
// per-task attribute after downgrades, so it's derived from the task buckets.
const userBandCounts = computed<Map<string, BandBreakdown>>(() => {
  const out = new Map<string, BandBreakdown>()
  for (const bucket of (schedulerData.value?.pendingBuckets ?? []) as PendingTaskBucket[]) {
    const band = bucket.band as Band
    if (!BANDS.includes(band)) continue
    const entry = out.get(bucket.userId) ?? emptyBandBreakdown()
    entry.pending[band] += bucket.count
    out.set(bucket.userId, entry)
  }
  for (const bucket of (schedulerData.value?.runningBuckets ?? []) as RunningTaskBucket[]) {
    const band = bucket.band as Band
    if (!BANDS.includes(band)) continue
    const entry = out.get(bucket.userId) ?? emptyBandBreakdown()
    entry.running[band] += bucket.count
    out.set(bucket.userId, entry)
  }
  return out
})

interface MergedUser {
  userId: string
  activeJobs: number
  runningJobs: number
  pendingJobs: number
  runningTasks: number
  totalTasks: number
  budgetSpent: string
  budgetLimit: string
  utilizationPercent: number
  maxBand: string
  effectiveBand: string
  hasBudget: boolean
  bands: BandBreakdown
}

const mergedUsers = computed<MergedUser[]>(() => {
  const budgetMap = new Map<string, SchedulerUserBudget>()
  for (const b of userBudgets.value) budgetMap.set(b.userId, b)

  const userMap = new Map<string, UserSummary>()
  for (const u of users.value) userMap.set(u.user, u)

  const allUserIds = new Set([...budgetMap.keys(), ...userMap.keys()])
  const result: MergedUser[] = []

  for (const userId of allUserIds) {
    const user = userMap.get(userId)
    const budget = budgetMap.get(userId)
    const jobCounts = user?.jobStateCounts ?? {}
    const taskCounts = user?.taskStateCounts ?? {}

    const activeJobs = Object.entries(jobCounts)
      .filter(([state]) => !TERMINAL_JOB_STATES.has(state))
      .reduce((acc, [, count]) => acc + count, 0)

    result.push({
      userId,
      activeJobs,
      runningJobs: jobCounts['running'] ?? 0,
      pendingJobs: (jobCounts['pending'] ?? 0) + (jobCounts['unschedulable'] ?? 0),
      runningTasks: taskCounts['running'] ?? 0,
      totalTasks: Object.values(taskCounts).reduce((a, b) => a + b, 0),
      budgetSpent: budget?.budgetSpent ?? '-',
      budgetLimit: budget?.budgetLimit ?? '-',
      utilizationPercent: budget?.utilizationPercent ?? 0,
      maxBand: budget?.maxBand ?? '',
      effectiveBand: budget?.effectiveBand ?? '',
      hasBudget: !!budget,
      bands: userBandCounts.value.get(userId) ?? emptyBandBreakdown(),
    })
  }

  result.sort((a, b) => b.activeJobs - a.activeJobs || b.runningTasks - a.runningTasks || a.userId.localeCompare(b.userId))
  return result
})

function utilizationStyle(pct: number): Record<string, string> {
  const clamped = Math.min(pct, 120)
  const idx = Math.round((clamped / 120) * (DIVERGING_COLORS.length - 1))
  const colorIdx = DIVERGING_COLORS.length - 1 - Math.max(0, Math.min(idx, DIVERGING_COLORS.length - 1))
  return { color: DIVERGING_COLORS[colorIdx] }
}

// ===========================================================================
// Diagnostics (recent actions)
// ===========================================================================

function formatActionTime(ts?: ProtoTimestamp): string {
  const ms = timestampMs(ts)
  return ms ? new Date(ms).toLocaleTimeString() : '-'
}

function actionTypeClasses(actionType?: string): string {
  switch (actionType) {
    case 'scale_up': return 'text-status-success font-semibold'
    case 'scale_down': return 'text-status-warning font-semibold'
    case 'delete': return 'text-status-danger font-semibold'
    default: return 'text-text font-semibold'
  }
}

function actionStatusClasses(status?: string): string {
  switch (status) {
    case 'pending': return 'text-status-warning'
    case 'failed': return 'text-status-danger'
    default: return 'text-status-success'
  }
}

function sliceIdShort(sliceId?: string): string {
  if (!sliceId) return ''
  return sliceId.length > 24 ? `${sliceId.slice(0, 20)}...` : sliceId
}
</script>

<template>
  <!-- Loading -->
  <LoadingSpinner v-if="loading" label="Loading capacity & scheduling…" />

  <!-- Error -->
  <div
    v-else-if="error"
    class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
  >
    {{ error }}
  </div>

  <!-- Autoscaler disabled -->
  <div v-else-if="!autoscaler" class="space-y-4">
    <EmptyState message="Autoscaler: Disabled" icon="⏸" />
  </div>

  <div v-else class="space-y-8">

    <!-- ===== Capacity summary strip ===== -->
    <section>
      <div class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
        <MetricCard size="sm" :value="`${onlineGroups} / ${groups.length}`" label="Pools online" />
        <MetricCard size="sm" :value="totalSlices" label="Slices" :detail="formatSliceSummary(sliceTotals)" />
        <MetricCard
          size="sm"
          :value="totalIdle"
          label="Idle spare"
          :variant="totalIdle > 0 ? 'warning' : 'default'"
        />
        <MetricCard
          size="sm"
          :value="totalDegradedSlices"
          label="Degraded slices"
          :variant="totalDegradedSlices > 0 ? 'orange' : 'default'"
          :detail="totalDegradedSlices > 0 ? 'unschedulable' : undefined"
        />
        <MetricCard size="sm" :value="totalDemand" label="Demand" :variant="totalDemand > 0 ? 'accent' : 'default'" />
        <MetricCard size="sm" :value="launchPlanned" label="Launch planned" :variant="launchPlanned > 0 ? 'accent' : 'default'" />
        <MetricCard
          size="sm"
          :value="aggregatedUnmet.length"
          label="Unmet jobs"
          :variant="aggregatedUnmet.length > 0 ? 'danger' : 'default'"
        />
        <MetricCard size="sm" :value="formatRelativeTime(lastEvalMs)" label="Last evaluation" />
      </div>
    </section>

    <!-- ===== Fleet overview — what we have, where ===== -->
    <FleetOverview :groups="groups" :running-buckets="schedulerData?.runningBuckets ?? []" />

    <!-- ===== Pools — capacity & routing ===== -->
    <section>
      <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">
        Pools — Capacity &amp; Routing
      </h3>

      <div v-if="!routing" class="text-sm text-text-muted py-4">
        No routing decision yet
      </div>

      <div v-else class="overflow-x-auto rounded-lg border border-surface-border">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border bg-surface">
              <th scope="col" class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left w-16">Priority</th>
              <th scope="col" class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Group</th>
              <th scope="col" class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Slices</th>
              <th scope="col" class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-right w-20">Demand</th>
              <th scope="col" class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-right w-20">Assigned</th>
              <th scope="col" class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-right w-20">Launch</th>
              <th scope="col" class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Decision</th>
              <th scope="col" class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Reason</th>
            </tr>
          </thead>
          <tbody>
            <template v-for="section in poolSections" :key="section.pool || '__unpooled'">
              <!-- Pool header row: toggle + name + tier chain on the left, an
                   always-visible one-line state summary (slice chips + demand) on
                   the right so a collapsed pool still tells its story. -->
              <tr class="bg-surface border-b border-surface-border hover:bg-surface-raised">
                <td colspan="8" class="px-3 py-1.5">
                  <div class="flex items-center justify-between gap-3">
                    <div class="flex items-center gap-2 min-w-0">
                      <button
                        type="button"
                        class="inline-flex items-center gap-2 text-left cursor-pointer hover:opacity-80"
                        :aria-expanded="!isPoolCollapsed(section)"
                        :aria-label="(isPoolCollapsed(section) ? 'Expand ' : 'Collapse ') + (section.pool === '__unpooled' ? 'unpooled groups' : 'pool ' + section.pool)"
                        @click="togglePool(section)"
                      >
                        <span class="text-[10px] text-text-muted">
                          {{ isPoolCollapsed(section) ? '▶' : '▼' }}
                        </span>
                        <span class="text-xs font-semibold uppercase tracking-wider text-text-secondary whitespace-nowrap">
                          {{ section.pool === '__unpooled' ? 'Unpooled' : `Pool: ${section.pool}` }}
                        </span>
                      </button>
                      <span
                        v-if="section.blockedAtTier"
                        class="inline-flex items-center px-1.5 py-0.5 rounded text-xs border
                               bg-status-danger-bg text-status-danger border-status-danger-border"
                      >
                        blocked at tier {{ section.blockedAtTier }}+
                      </span>
                      <!-- Tier ladder: slice count at each fallback tier (left = first
                           tried). Position encodes the tier; the number is how many
                           slices sit there. Hover for the tier label and group. -->
                      <span v-if="section.pool !== '__unpooled'" class="flex items-center gap-0.5 text-xs text-text-muted ml-2">
                        <template v-for="(gs, idx) in section.groups" :key="gs.group">
                          <span v-if="idx > 0" class="text-text-muted mx-0.5">&rarr;</span>
                          <span
                            :title="`${tierLabel(gs) || 'tier'} · ${gs.group} · ${groupSliceCountTotal(gs.group)} slice${groupSliceCountTotal(gs.group) === 1 ? '' : 's'}`"
                            :class="[
                              'px-1 py-0.5 rounded border text-[11px] font-mono tabular-nums',
                              isTierBlocked(gs, section)
                                ? 'bg-status-danger-bg text-status-danger border-status-danger-border line-through'
                                : group(gs.group)?.availabilityStatus === 'quota_exceeded'
                                  ? 'bg-status-danger-bg text-status-danger border-status-danger-border'
                                  : group(gs.group)?.availabilityStatus === 'backoff'
                                    ? 'bg-status-orange-bg text-status-orange border-status-orange-border'
                                    : groupSliceCountTotal(gs.group) > 0
                                      ? 'bg-surface border-surface-border text-text-secondary'
                                      : 'bg-surface border-surface-border text-text-muted',
                            ]"
                          >
                            {{ groupSliceCountTotal(gs.group) }}
                          </span>
                        </template>
                      </span>
                    </div>
                    <div class="flex items-center gap-1.5 flex-shrink-0">
                      <template v-if="poolStatusSummary(section).length">
                        <span
                          v-for="b in poolStatusSummary(section)"
                          :key="b.status"
                          :class="[BADGE_BASE, b.style.bg, b.style.text, b.style.border]"
                          :title="`${b.count} ${b.style.label} slice${b.count > 1 ? 's' : ''} — ${b.style.description}`"
                        >
                          <span class="w-1.5 h-1.5 rounded-full" :class="b.style.dot" />
                          {{ b.count }} {{ b.style.label }}
                        </span>
                      </template>
                      <span v-else class="text-xs text-text-muted">no slices</span>
                      <span
                        v-if="poolDemand(section) > 0"
                        class="inline-flex items-center px-1.5 py-0.5 rounded text-xs border bg-accent-subtle text-accent border-accent-border"
                      >
                        demand {{ poolDemand(section) }}
                      </span>
                      <span
                        v-if="poolLaunch(section) > 0"
                        class="inline-flex items-center px-1.5 py-0.5 rounded text-xs border bg-accent-subtle text-accent border-accent-border"
                      >
                        launch {{ poolLaunch(section) }}
                      </span>
                    </div>
                  </div>
                </td>
              </tr>

              <template v-for="gs in visibleGroups(section)" :key="gs.group">
                <!-- Main row -->
                <tr
                  v-if="!isPoolCollapsed(section)"
                  :class="[
                    'border-b border-surface-border-subtle hover:bg-surface-raised transition-colors',
                    isInactiveRow(gs) ? 'opacity-50' : '',
                    isTierBlocked(gs, section) ? 'opacity-40' : '',
                  ]"
                >
                  <!-- Priority -->
                  <td class="px-3 py-2 text-[13px] font-mono text-text-muted align-top">
                    {{ gs.priority ?? 100 }}
                  </td>

                  <!-- Group name + badges -->
                  <td class="px-3 py-2 text-[13px] align-top">
                    <div>
                      <span class="font-semibold">{{ gs.group }}</span>
                      <span
                        v-if="groupFailures(gs.group) > 0"
                        class="ml-2 inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-xs
                               bg-status-danger-bg text-status-danger border border-status-danger-border"
                      >
                        &#x26a0; {{ groupFailures(gs.group) }} fail{{ groupFailures(gs.group) > 1 ? 's' : '' }}
                      </span>
                    </div>
                    <div v-if="group(gs.group) && groupAvailabilityBadge(group(gs.group)!, section)" class="mt-0.5">
                      <span
                        :class="[
                          'inline-flex items-center px-1.5 py-0.5 rounded text-xs border',
                          groupAvailabilityBadge(group(gs.group)!, section)!.classes,
                        ]"
                      >
                        {{ groupAvailabilityBadge(group(gs.group)!, section)!.label }}
                      </span>
                    </div>
                  </td>

                  <!-- Slices (expandable) + schedulable/degraded/idle badges -->
                  <td class="px-3 py-2 text-[13px] align-top">
                    <!-- Slice-granular status chips: one per present status, each a
                         count of SLICES (never per-host), so counts line up with the
                         expanded list and the total slice count. Expandable into the
                         per-slice list only when the group has materialized slices;
                         a group with only in-flight (requesting) scale-ups still shows
                         its chips but has nothing to drill into yet. -->
                    <component
                      :is="groupHasSlices(gs.group) ? 'button' : 'span'"
                      v-if="groupStatusSummary(gs.group).length"
                      class="inline-flex items-center gap-1 flex-wrap text-left"
                      :class="groupHasSlices(gs.group) ? 'cursor-pointer hover:opacity-80' : ''"
                      :type="groupHasSlices(gs.group) ? 'button' : undefined"
                      :aria-expanded="groupHasSlices(gs.group) ? expandedSlices.has(gs.group) : undefined"
                      :aria-label="groupHasSlices(gs.group)
                        ? (expandedSlices.has(gs.group) ? 'Hide' : 'Show') + ' slices for ' + gs.group
                        : undefined"
                      @click="groupHasSlices(gs.group) && toggleSlices(gs.group)"
                    >
                      <span v-if="groupHasSlices(gs.group)" class="text-[10px] text-text-muted">
                        {{ expandedSlices.has(gs.group) ? '▼' : '▶' }}
                      </span>
                      <span class="inline-flex items-center gap-1 flex-wrap">
                        <span
                          v-for="b in groupStatusSummary(gs.group)"
                          :key="b.status"
                          :class="[BADGE_BASE, b.style.bg, b.style.text, b.style.border]"
                          :title="`${b.count} ${b.style.label} slice${b.count > 1 ? 's' : ''} — ${b.style.description}`"
                        >
                          <span class="w-1.5 h-1.5 rounded-full" :class="b.style.dot" />
                          {{ b.count }} {{ b.style.label }}
                        </span>
                      </span>
                    </component>
                    <span v-else class="text-text-muted">-</span>

                    <!-- Reconciliation note: free capacity vs unmet demand -->
                    <div v-if="groupReconcileNote(gs.group)" class="mt-1 text-[11px]" :class="groupReconcileNote(gs.group)!.classes">
                      &#x26a0; {{ groupReconcileNote(gs.group)!.text }}
                    </div>
                  </td>

                  <!-- Demand -->
                  <td class="px-3 py-2 text-[13px] text-right font-mono align-top">
                    {{ groupDemand(gs.group) || '' }}
                  </td>

                  <!-- Assigned -->
                  <td class="px-3 py-2 text-[13px] text-right font-mono align-top">{{ gs.assigned ?? 0 }}</td>

                  <!-- Launch -->
                  <td class="px-3 py-2 text-[13px] text-right font-mono align-top">{{ gs.launch ?? 0 }}</td>

                  <!-- Decision -->
                  <td class="px-3 py-2 text-[13px] align-top">
                    <span :class="decisionClasses(gs.decision)">{{ formatDecision(gs.decision) }}</span>
                  </td>

                  <!-- Reason -->
                  <td class="px-3 py-2 text-[13px] text-text-secondary max-w-xs truncate align-top" :title="groupReasonText(gs)">
                    {{ groupReasonText(gs) }}
                  </td>
                </tr>

                <!-- Slice detail (expanded) -->
                <tr v-if="expandedSlices.has(gs.group) && groupHasSlices(gs.group) && !isPoolCollapsed(section)" class="bg-surface-sunken">
                  <td colspan="8" class="px-6 py-3">
                    <SliceList :slices="groupSlices(gs.group)" :worker-jobs="sliceWorkerJobs" :now="nowMs" />
                  </td>
                </tr>
              </template>

              <!-- Idle-tier toggle: an active pool hides its fully-idle sizes here. -->
              <tr v-if="!isPoolCollapsed(section) && idleSizeCount(section) > 0" class="border-b border-surface-border-subtle">
                <td colspan="8" class="px-3 py-1">
                  <button
                    type="button"
                    class="ml-6 text-xs text-text-muted hover:text-text-secondary cursor-pointer"
                    @click="toggleIdleSizes(section.pool)"
                  >
                    {{ expandedIdleSizes.has(section.pool)
                      ? `hide ${idleSizeCount(section)} idle size${idleSizeCount(section) > 1 ? 's' : ''}`
                      : `show ${idleSizeCount(section)} idle size${idleSizeCount(section) > 1 ? 's' : ''}` }}
                  </button>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </section>

    <!-- ===== Unmet demand ===== -->
    <section v-if="aggregatedUnmet.length > 0">
      <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">
        Unmet Demand
      </h3>
      <div class="overflow-x-auto rounded-lg border border-surface-border">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border bg-surface">
              <th scope="col" class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Job</th>
              <th scope="col" class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Reasons</th>
              <th scope="col" class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-right w-20">Entries</th>
              <th scope="col" class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Example Task</th>
              <th scope="col" class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Accelerator</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="row in aggregatedUnmet"
              :key="row.job"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td class="px-3 py-2 text-[13px] font-semibold">{{ row.job }}</td>
              <td class="px-3 py-2 text-[13px] text-text-secondary">{{ formatReasonCounts(row.reasonCounts) }}</td>
              <td class="px-3 py-2 text-[13px] text-right font-mono">{{ row.entryCount }}</td>
              <td class="px-3 py-2 text-[13px] font-mono text-text-muted truncate max-w-xs" :title="row.exampleTask ?? undefined">
                {{ row.exampleTask ?? '-' }}
              </td>
              <td class="px-3 py-2 text-[13px] font-mono">
                {{ row.accelerators.size === 1 ? [...row.accelerators][0] : 'mixed' }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- ===== Pending jobs (not scheduling) ===== -->
    <section>
      <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">
        Pending Jobs ({{ pendingTotal }})
      </h3>
      <div class="flex items-center gap-3 mb-3">
        <form class="flex items-center gap-2" @submit.prevent="applyPendingSearch">
          <input
            v-model="pendingSearchInput"
            type="text"
            placeholder="Search by job name..."
            aria-label="Search pending jobs by name"
            class="w-64 px-3 py-1.5 bg-surface border border-surface-border rounded
                   text-sm font-mono placeholder:text-text-muted
                   focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
          />
          <button
            type="submit"
            class="px-3 py-1.5 text-sm border border-surface-border rounded hover:bg-surface-raised text-text-secondary"
          >
            Search
          </button>
          <button
            v-if="pendingSearch"
            type="button"
            class="px-3 py-1.5 text-sm border border-surface-border rounded hover:bg-surface-raised text-text-muted"
            @click="pendingSearchInput = ''; applyPendingSearch()"
          >
            Clear
          </button>
        </form>
      </div>

      <div
        v-if="pendingError"
        class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
      >
        {{ pendingError }}
      </div>

      <LoadingSpinner v-if="pendingLoading && pendingJobs.length === 0" size="sm" />

      <EmptyState v-else-if="pendingJobs.length === 0" message="No pending jobs" />

      <div v-else class="overflow-x-auto">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border">
              <th scope="col" class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Job</th>
              <th scope="col" class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">User</th>
              <th scope="col" class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">State</th>
              <th scope="col" class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Band</th>
              <th scope="col" class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Pending Reason</th>
              <th scope="col" class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Submitted</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="job in pendingJobs"
              :key="job.jobId"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td class="px-3 py-2 text-[13px] font-mono">
                <RouterLink :to="`/job/${encodeURIComponent(job.jobId)}`" class="text-accent hover:underline">
                  {{ job.name || job.jobId }}
                </RouterLink>
              </td>
              <td class="px-3 py-2 text-[13px]">{{ job.jobId.split('/')[0] }}</td>
              <td class="px-3 py-2 text-[13px]">
                <StatusBadge :status="job.state" size="sm" />
              </td>
              <td class="px-3 py-2 text-[13px]">
                <span v-if="pendingJobBand.get(job.jobId)" :class="bandColor(pendingJobBand.get(job.jobId))">
                  {{ bandDisplayName(pendingJobBand.get(job.jobId)) }}
                </span>
                <span v-else class="text-text-muted">-</span>
              </td>
              <td class="px-3 py-2 text-[13px] text-status-warning max-w-md truncate" :title="job.pendingReason ?? ''">
                {{ job.pendingReason || '-' }}
              </td>
              <td class="px-3 py-2 text-[13px] font-mono text-text-secondary">
                {{ job.submittedAt ? formatRelativeTime(timestampMs(job.submittedAt)) : '-' }}
              </td>
            </tr>
          </tbody>
        </table>
        <!-- Pagination -->
        <div v-if="pendingTotalPages > 1" class="flex items-center justify-between px-3 py-2 text-xs text-text-secondary border-t border-surface-border">
          <span>
            {{ pendingPage * PENDING_PAGE_SIZE + 1 }}&ndash;{{ Math.min((pendingPage + 1) * PENDING_PAGE_SIZE, pendingTotal) }}
            of {{ pendingTotal }} jobs
          </span>
          <div class="flex items-center gap-1">
            <button
              :disabled="pendingPage === 0"
              class="px-2 py-1 rounded hover:bg-surface-raised disabled:opacity-30 disabled:cursor-not-allowed"
              @click="pendingPage--"
            >
              &larr; Prev
            </button>
            <span class="px-2 font-mono">{{ pendingPage + 1 }} / {{ pendingTotalPages }}</span>
            <button
              :disabled="pendingPage >= pendingTotalPages - 1"
              class="px-2 py-1 rounded hover:bg-surface-raised disabled:opacity-30 disabled:cursor-not-allowed"
              @click="pendingPage++"
            >
              Next &rarr;
            </button>
          </div>
        </div>
      </div>
    </section>

    <!-- ===== Users & quotas ===== -->
    <section>
      <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">Users &amp; Quotas</h3>
      <EmptyState v-if="mergedUsers.length === 0" message="No users" />
      <div v-else class="overflow-x-auto">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border">
              <th scope="col" class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">User</th>
              <th scope="col" class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Active Jobs</th>
              <th scope="col" class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Running</th>
              <th scope="col" class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Pending</th>
              <th scope="col" class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Running Tasks</th>
              <th scope="col" class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary" title="Running / pending tasks per effective priority band">By Band</th>
              <th scope="col" class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Total Tasks</th>
              <th scope="col" class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Spent</th>
              <th scope="col" class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Limit</th>
              <th scope="col" class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Utilization</th>
              <th scope="col" class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Band</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="user in mergedUsers"
              :key="user.userId"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td class="px-3 py-2 text-[13px] font-mono">
                <RouterLink
                  v-if="user.userId"
                  :to="{ path: '/', query: { user: user.userId } }"
                  class="text-accent hover:underline"
                >{{ user.userId }}</RouterLink>
                <span v-else class="text-text-muted">(unknown)</span>
              </td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">{{ user.activeJobs }}</td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">
                <span :class="user.runningJobs > 0 ? 'text-accent font-semibold' : ''">{{ user.runningJobs }}</span>
              </td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">
                <span :class="user.pendingJobs > 0 ? 'text-status-warning' : ''">{{ user.pendingJobs }}</span>
              </td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">
                <span :class="user.runningTasks > 0 ? 'text-accent font-semibold' : ''">{{ user.runningTasks }}</span>
              </td>
              <td class="px-3 py-2 text-[13px] whitespace-nowrap">
                <template v-for="band in BANDS" :key="band">
                  <span
                    v-if="user.bands.running[band] || user.bands.pending[band]"
                    class="mr-2 tabular-nums"
                    :title="bandDisplayName(band) + ': ' + user.bands.running[band] + ' running / ' + user.bands.pending[band] + ' pending'"
                  >
                    <span :class="bandColor(band)">{{ bandDisplayName(band).charAt(0) }}</span>
                    <span class="text-accent">{{ user.bands.running[band] }}</span>
                    <span class="text-text-muted">/</span>
                    <span :class="user.bands.pending[band] > 0 ? 'text-status-warning' : 'text-text-muted'">{{ user.bands.pending[band] }}</span>
                  </span>
                </template>
                <span v-if="bandBreakdownTotal(user.bands) === 0" class="text-text-muted">-</span>
              </td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">{{ user.totalTasks }}</td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">
                {{ user.hasBudget ? user.budgetSpent : '-' }}
              </td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums">
                {{ !user.hasBudget ? '-' : user.budgetLimit === '0' ? 'Unlimited' : user.budgetLimit }}
              </td>
              <td class="px-3 py-2 text-[13px] text-right tabular-nums font-semibold" :style="user.hasBudget ? utilizationStyle(user.utilizationPercent) : {}">
                {{ !user.hasBudget ? '-' : user.budgetLimit === '0' ? '-' : user.utilizationPercent.toFixed(1) + '%' }}
              </td>
              <td class="px-3 py-2 text-[13px]">
                <template v-if="user.hasBudget">
                  <span :class="bandColor(user.effectiveBand)">{{ bandDisplayName(user.effectiveBand) }}</span>
                  <span
                    v-if="user.maxBand !== user.effectiveBand"
                    class="ml-1 text-xs text-status-warning"
                  >
                    (max: {{ bandDisplayName(user.maxBand) }})
                  </span>
                </template>
                <span v-else class="text-text-muted">-</span>
              </td>
            </tr>
          </tbody>
        </table>
        <div class="px-3 py-2 text-xs text-text-secondary border-t border-surface-border">
          {{ mergedUsers.length }} user{{ mergedUsers.length !== 1 ? 's' : '' }}
        </div>
      </div>
    </section>

    <!-- ===== Diagnostics (secondary) ===== -->
    <details class="rounded-lg border border-surface-border bg-surface">
      <summary class="px-4 py-2 cursor-pointer text-sm font-semibold text-text-secondary uppercase tracking-wider select-none">
        Diagnostics — recent actions &amp; logs
      </summary>
      <div class="px-4 py-3 space-y-6 border-t border-surface-border">
        <!-- Recent autoscaler actions -->
        <div>
          <h4 class="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-2">Recent Actions</h4>
          <div v-if="actions.length === 0" class="text-sm text-text-muted py-2">No recent actions</div>
          <div v-else class="rounded-lg border border-surface-border bg-surface-sunken divide-y divide-surface-border-subtle">
            <div
              v-for="(action, i) in actions"
              :key="i"
              class="flex items-center gap-3 px-4 py-2 text-[13px] hover:bg-surface-raised transition-colors"
            >
              <span class="font-mono text-text-muted w-20 flex-shrink-0">{{ formatActionTime(action.timestamp) }}</span>
              <span :class="actionTypeClasses(action.actionType)">
                {{ (action.actionType ?? 'unknown').replace('_', ' ') }}
              </span>
              <span
                v-if="action.status && action.status !== 'completed'"
                :class="['text-xs', actionStatusClasses(action.status)]"
              >
                [{{ action.status }}]
              </span>
              <span class="font-semibold">{{ action.scaleGroup }}</span>
              <span v-if="action.sliceId" class="font-mono text-text-muted text-xs" :title="action.sliceId">
                [{{ sliceIdShort(action.sliceId) }}]
              </span>
              <span v-if="action.reason" class="text-text-secondary">- {{ action.reason }}</span>
            </div>
          </div>
        </div>

        <!-- Controller logs -->
        <div>
          <h4 class="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-2">Controller Logs</h4>
          <LogViewer source="controller" max-height="40vh" />
        </div>
      </div>
    </details>
  </div>
</template>
