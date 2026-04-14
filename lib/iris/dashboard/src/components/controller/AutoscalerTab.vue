<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import { SLICE_STATE_STYLES, SLICE_BADGE_ORDER, CATEGORICAL_COLORS, vmStateToName } from '@/types/status'
import type {
  GetAutoscalerStatusResponse,
  GetSchedulerStateResponse,
  SchedulerRunningTask,
  AutoscalerStatus,
  ScaleGroupStatus,
  SliceInfo,
  VmInfo,
  GroupRoutingStatus,
  UnmetDemand,
  AutoscalerAction,
  ProtoTimestamp,
} from '@/types/rpc'
import { timestampMs, formatRelativeTime, formatDuration } from '@/utils/formatting'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import MetricCard from '@/components/shared/MetricCard.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import LogViewer from '@/components/shared/LogViewer.vue'

// -- RPC + auto-refresh --

const { data, loading, error, refresh: refreshAutoscaler } = useControllerRpc<GetAutoscalerStatusResponse>('GetAutoscalerStatus')
const { data: schedulerData, refresh: refreshScheduler } = useControllerRpc<GetSchedulerStateResponse>('GetSchedulerState')

async function refresh() {
  await Promise.all([refreshAutoscaler(), refreshScheduler()])
}
useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
onMounted(refresh)

// -- Expand/collapse state --

const expandedDemand = ref<Set<string>>(new Set())
const expandedSlices = ref<Set<string>>(new Set())
const collapsedPools = ref<Set<string>>(new Set())

function togglePool(pool: string) {
  const next = new Set(collapsedPools.value)
  next.has(pool) ? next.delete(pool) : next.add(pool)
  collapsedPools.value = next
}

function toggleDemand(name: string) {
  const next = new Set(expandedDemand.value)
  next.has(name) ? next.delete(name) : next.add(name)
  expandedDemand.value = next
}

function toggleSlices(name: string) {
  const next = new Set(expandedSlices.value)
  next.has(name) ? next.delete(name) : next.add(name)
  expandedSlices.value = next
}

// -- Helpers --

function formatActionTime(ts?: ProtoTimestamp): string {
  const ms = timestampMs(ts)
  if (!ms) return '-'
  return new Date(ms).toLocaleTimeString()
}

const RESERVATION_RE = /^(.+):reservation:\d+$/

function taskIdToJob(taskId: string): string {
  if (!taskId) return 'unknown'
  const rsvMatch = taskId.match(RESERVATION_RE)
  if (rsvMatch) return rsvMatch[1]
  const idx = taskId.lastIndexOf('/')
  if (idx <= 0) return taskId
  return taskId.slice(0, idx)
}

function isReservationEntry(entry: { taskIds?: string[] }): boolean {
  const taskIds = entry.taskIds ?? []
  return taskIds.length > 0 && RESERVATION_RE.test(taskIds[0])
}

function formatIdleSince(lastActive?: ProtoTimestamp): string {
  const ms = timestampMs(lastActive)
  if (!ms) return 'never active'
  const elapsed = Date.now() - ms
  if (elapsed < 60000) return `${Math.floor(elapsed / 1000)}s`
  if (elapsed < 3600000) return `${Math.floor(elapsed / 60000)}m`
  return `${Math.floor(elapsed / 3600000)}h ${Math.floor((elapsed % 3600000) / 60000)}m`
}

function formatIdleThreshold(ms: number): string {
  if (!ms) return '?'
  return ms >= 60000 ? `${Math.floor(ms / 60000)}m` : `${Math.floor(ms / 1000)}s`
}

// Slice state badge styling and order are imported from @/types/status so the
// dashboard legend can stay in sync with a single canonical definition.

// -- Group availability status --

interface AvailabilityBadge {
  label: string
  classes: string
}

function groupAvailabilityBadge(group: ScaleGroupStatus, section?: PoolSection): AvailabilityBadge | null {
  const status = group.availabilityStatus
  const blockedMs = timestampMs(group.blockedUntil)
  const cooldownMs = timestampMs(group.scaleUpCooldownUntil)
  const now = Date.now()

  // Check for tier-blocked state (pool monotonicity)
  if (section && section.blockedAtTier) {
    const tier = group.config?.allocationTier ?? 0
    if (tier > section.blockedAtTier) {
      return { label: 'tier-blocked', classes: 'bg-status-danger-bg text-status-danger border-status-danger-border opacity-60' }
    }
  }

  if (status === 'requesting') {
    return { label: 'in-flight', classes: 'bg-status-purple-bg text-status-purple border-status-purple-border' }
  }
  if (status === 'backoff') {
    const label = blockedMs && blockedMs > now
      ? `backoff ${Math.ceil((blockedMs - now) / 1000)}s`
      : 'backoff'
    return { label, classes: 'bg-status-orange-bg text-status-orange border-status-orange-border' }
  }
  if (status === 'quota_exceeded') {
    const label = blockedMs && blockedMs > now
      ? `quota exceeded ${Math.ceil((blockedMs - now) / 1000)}s`
      : 'quota exceeded'
    return { label, classes: 'bg-status-danger-bg text-status-danger border-status-danger-border' }
  }
  if (status === 'at_capacity') {
    return { label: 'at capacity', classes: 'bg-status-warning-bg text-status-warning border-status-warning-border' }
  }
  if (cooldownMs && cooldownMs > now) {
    const secs = Math.ceil((cooldownMs - now) / 1000)
    return { label: `cooldown ${secs}s`, classes: 'bg-accent-subtle text-accent border-accent-border' }
  }
  return null
}

// -- Computed data --

const autoscaler = computed<AutoscalerStatus | null>(() => data.value?.status ?? null)

const groups = computed(() => autoscaler.value?.groups ?? [])
const routing = computed(() => autoscaler.value?.lastRoutingDecision ?? null)
const unmetEntries = computed(() => routing.value?.unmetEntries ?? [])
const actions = computed(() => (autoscaler.value?.recentActions ?? []).slice().reverse())

const groupIndex = computed(() => {
  const index: Record<string, ScaleGroupStatus> = {}
  for (const g of groups.value) {
    if (g.name) index[g.name] = g
  }
  return index
})

// -- Status bar metrics --

function sumSliceCounts(gs: ScaleGroupStatus[]): Record<string, number> {
  const totals: Record<string, number> = {}
  for (const g of gs) {
    for (const [state, count] of Object.entries(g.sliceStateCounts ?? {})) {
      totals[state] = (totals[state] ?? 0) + (count ?? 0)
    }
  }
  return totals
}

function countIdleSlices(gs: ScaleGroupStatus[]): number {
  let idle = 0
  for (const g of gs) {
    for (const slice of g.slices ?? []) {
      if (slice.idle) idle++
    }
  }
  return idle
}

const sliceTotals = computed(() => sumSliceCounts(groups.value))
const totalSlices = computed(() => Object.values(sliceTotals.value).reduce((a, b) => a + b, 0))
const totalIdle = computed(() => countIdleSlices(groups.value))
const onlineGroups = computed(() =>
  groups.value.filter(g => {
    const counts = g.sliceStateCounts ?? {}
    return Object.values(counts).reduce((a, b) => a + b, 0) > 0
  }).length
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
  const parts: string[] = []
  for (const state of order) {
    const n = totals[state] ?? 0
    if (n > 0) parts.push(`${n} ${state}`)
  }
  return `${total} (${parts.join(', ')})`
}

// -- Waterfall routing rows --

const sortedGroupStatuses = computed<GroupRoutingStatus[]>(() => {
  const statuses = routing.value?.groupStatuses ?? []
  return statuses.slice().sort((a, b) => {
    const pa = a.priority ?? 100
    const pb = b.priority ?? 100
    if (pa !== pb) return pa - pb
    return (a.group ?? '').localeCompare(b.group ?? '')
  })
})

// Pool grouping for tier chain display
interface PoolSection {
  pool: string
  groups: GroupRoutingStatus[]
  blockedAtTier: number | null  // lowest tier in quota_exceeded/backoff, or null
}

const poolSections = computed<PoolSection[]>(() => {
  const poolMap = new Map<string, GroupRoutingStatus[]>()
  const unpooled: GroupRoutingStatus[] = []

  for (const gs of sortedGroupStatuses.value) {
    const group = groupIndex.value[gs.group]
    const pool = group?.config?.quotaPool
    if (pool) {
      if (!poolMap.has(pool)) poolMap.set(pool, [])
      poolMap.get(pool)!.push(gs)
    } else {
      unpooled.push(gs)
    }
  }

  const sections: PoolSection[] = []
  for (const [pool, poolGroups] of poolMap) {
    // Sort by allocation_tier within the pool
    poolGroups.sort((a, b) => {
      const ta = groupIndex.value[a.group]?.config?.allocationTier ?? 0
      const tb = groupIndex.value[b.group]?.config?.allocationTier ?? 0
      return ta - tb
    })

    // Find the lowest blocked tier
    let blockedAtTier: number | null = null
    for (const gs of poolGroups) {
      const group = groupIndex.value[gs.group]
      if (!group) continue
      const tier = group.config?.allocationTier ?? 0
      const status = group.availabilityStatus
      if (tier > 0 && (status === 'quota_exceeded' || status === 'backoff')) {
        if (blockedAtTier === null || tier < blockedAtTier) {
          blockedAtTier = tier
        }
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
  const group = groupIndex.value[gs.group]
  const tier = group?.config?.allocationTier ?? 0
  return tier > section.blockedAtTier
}

function tierLabel(gs: GroupRoutingStatus): string {
  const group = groupIndex.value[gs.group]
  const tier = group?.config?.allocationTier ?? 0
  return tier > 0 ? `T${tier}` : ''
}

function isInactiveRow(gs: GroupRoutingStatus): boolean {
  const group = groupIndex.value[gs.group]
  const counts = group?.sliceStateCounts ?? {}
  const totalGroupSlices = Object.values(counts).reduce((a, b) => a + b, 0)
  return totalGroupSlices === 0 && (gs.decision ?? 'idle') === 'idle'
}

function groupFailures(groupName: string): number {
  return groupIndex.value[groupName]?.consecutiveFailures ?? 0
}

function groupSliceCounts(groupName: string): Record<string, number> {
  return groupIndex.value[groupName]?.sliceStateCounts ?? {}
}

function groupIdleCount(groupName: string): number {
  const slices = groupIndex.value[groupName]?.slices ?? []
  return slices.filter(s => s.idle).length
}

/** True if any VM in the slice is currently running at least one task. */
function sliceInUse(slice: SliceInfo): boolean {
  return (slice.vms ?? []).some(vm => (vm.runningTaskCount ?? 0) > 0)
}

/**
 * Count slices in a group that are currently in use (ready + have at least one
 * task assigned). The backend's sliceStateCounts only tracks lifecycle states —
 * "in-use" is orthogonal and computed from the per-VM runningTaskCount.
 */
function groupInUseCount(groupName: string): number {
  const slices = groupIndex.value[groupName]?.slices ?? []
  return slices.filter(sliceInUse).length
}

/**
 * Synthetic slice counts for badge rendering. Splits the `ready` bucket from
 * the backend's lifecycle counts into available ready (R) and in-use (U) so
 * operators can tell at a glance which slices are occupied vs. free.
 *
 * The backend tracks lifecycle (REQUESTING/BOOTING/INITIALIZING/READY/FAILED)
 * but "in use" is orthogonal — it's computed here by summing per-VM
 * runningTaskCount across the group's slices. R + U == the backend's ready
 * count (when all ready slices have lifecycle state information).
 */
function groupBadgeCounts(groupName: string): Record<string, number> {
  const counts = { ...groupSliceCounts(groupName) }
  const inUse = groupInUseCount(groupName)
  if (inUse > 0) {
    counts.in_use = inUse
    const ready = counts.ready ?? 0
    counts.ready = Math.max(0, ready - inUse)
  }
  return counts
}

function groupSlices(groupName: string): SliceInfo[] {
  return groupIndex.value[groupName]?.slices ?? []
}

function groupHasSlices(groupName: string): boolean {
  return groupSlices(groupName).length > 0
}

function groupDemand(groupName: string): number {
  return groupIndex.value[groupName]?.currentDemand ?? 0
}

function formatDecision(decision?: string): string {
  return (decision ?? 'idle').replace('_', ' ')
}

function decisionClasses(decision?: string): string {
  const d = decision ?? 'idle'
  switch (d) {
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
  const group = groupIndex.value[gs.group]
  if (group && (group.availabilityStatus === 'backoff' || group.availabilityStatus === 'quota_exceeded')) {
    const blockedMs = timestampMs(group.blockedUntil)
    if (blockedMs && blockedMs > Date.now()) {
      const secsLeft = Math.ceil((blockedMs - Date.now()) / 1000)
      reason = reason ? `${reason} (unblocks in ${secsLeft}s)` : `unblocks in ${secsLeft}s`
    }
  }
  return reason || '-'
}

// -- Demand breakdown --

interface JobDemandRow {
  job: string
  taskEntries: number
  reservationEntries: number
}

function aggregateDemandByJob(groupName: string): JobDemandRow[] {
  // The routing decision doesn't have routedEntries in the new proto format,
  // but demand comes from the group's slices/tasks. We use currentDemand as a fallback.
  // This aggregation is a no-op if there are no routed entries available.
  return [{ job: groupName, taskEntries: groupDemand(groupName), reservationEntries: 0 }]
}

// -- Unmet demand aggregation --

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
  return entries.map(([reason, count]) => {
    const display = reason.replace(/^[a-z_]+:\s*/, '')
    return `${display} (${count})`
  }).join(', ')
}

// -- Action styling --

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

function idleThresholdMs(groupName: string): number {
  return parseInt(groupIndex.value[groupName]?.idleThresholdMs ?? '0', 10)
}

// -- Fleet overview --

interface RegionCount {
  region: string
  count: number
}

interface BandCount {
  band: string
  count: number
}

interface RegionCapacity {
  region: string
  status: 'available' | 'limited' | 'blocked'
  detail: string
}

interface FleetChipSummary {
  chip: string
  total: number
  inUse: number
  avgUptimeMs: number | null
  regions: RegionCount[]
  bands: BandCount[]
  capacity: RegionCapacity[]
}

/** Map workerId → list of bands consuming that worker. */
const workerBands = computed<Map<string, Map<string, number>>>(() => {
  const map = new Map<string, Map<string, number>>()
  for (const task of (schedulerData.value?.runningTasks ?? []) as SchedulerRunningTask[]) {
    if (!task.workerId || !task.effectiveBand) continue
    const band = task.effectiveBand.replace(/^PRIORITY_BAND_/, '').toLowerCase()
    if (!map.has(task.workerId)) map.set(task.workerId, new Map())
    const bands = map.get(task.workerId)!
    bands.set(band, (bands.get(band) ?? 0) + 1)
  }
  return map
})

/** Map workerId → running tasks currently assigned to that worker. */
const workerTasks = computed<Map<string, SchedulerRunningTask[]>>(() => {
  const map = new Map<string, SchedulerRunningTask[]>()
  for (const task of (schedulerData.value?.runningTasks ?? []) as SchedulerRunningTask[]) {
    if (!task.workerId) continue
    if (!map.has(task.workerId)) map.set(task.workerId, [])
    map.get(task.workerId)!.push(task)
  }
  return map
})

function tasksForVm(vm: VmInfo): SchedulerRunningTask[] {
  if (!vm.workerId) return []
  return workerTasks.value.get(vm.workerId) ?? []
}

/** Extract chip type + size from scale group name.
 *  e.g. "TPU_V5E_PREEMPTIBLE_16_US_EAST" → "v5e-16"
 *       "TPU_V5P_SERVING_64_US_CENTRAL"  → "v5p-64"
 *       "CPU_VM_E2_HIGHMEM_2_ON..."      → "cpu-e2"
 *       "GPU_A100_8_US_CENTRAL"          → "A100-8"
 */
function chipFromGroupName(name: string): string | null {
  // Normalize separators so hyphens and underscores are interchangeable
  const norm = name.toUpperCase().replace(/-/g, '_')
  // TPU: extract chip and size (first number after class)
  const tpuMatch = norm.match(/TPU_(V\d+[A-Z]?)_(?:PREEMPTIBLE|SERVING|ON_DEMAND|RESERVED)_(\d+)/)
  if (tpuMatch) return `${tpuMatch[1].toLowerCase()}-${tpuMatch[2]}`
  // TPU without recognized class — try chip + first number
  const tpuFallback = norm.match(/TPU_(V\d+[A-Z]?)_\w+_(\d+)/)
  if (tpuFallback) return `${tpuFallback[1].toLowerCase()}-${tpuFallback[2]}`
  // GPU: variant + count
  const gpuMatch = norm.match(/(A100|H100|H200|L4|L40S?|B200)_(\d+)/)
  if (gpuMatch) return `${gpuMatch[1]}-${gpuMatch[2]}`
  // CPU — skip
  if (norm.startsWith('CPU')) return null
  return name
}

/** Extract region from scale group name.
 *  e.g. "TPU_V5E_PREEMPTIBLE_16_US_WEST4_A" → "us-west4"
 *       "TPU_V5P-PREEMPTIBLE_64-US-EASTS-A"  → "us-easts"
 *       "CPU_VM_E2_HIGHMEM_2_ON_DEMAND"       → "on-demand"
 */
function regionFromGroupName(name: string): string {
  const norm = name.toUpperCase().replace(/-/g, '_')
  // TPU: everything after the size number, drop trailing zone letter (e.g. _A, _B)
  const tpuMatch = norm.match(/TPU_V\d+[A-Z]?_(?:PREEMPTIBLE|SERVING|ON_DEMAND|RESERVED)_\d+_(.+)/)
  if (tpuMatch) {
    let region = tpuMatch[1]
    // Strip trailing single-letter zone suffix (e.g. _A, _B, _C)
    region = region.replace(/_[A-Z]$/, '')
    return region.toLowerCase().replace(/_/g, '-')
  }
  // Fallback: try to grab region-like suffix after the last number
  const fallback = norm.match(/_\d+_(.+)/)
  if (fallback) return fallback[1].toLowerCase().replace(/_/g, '-')
  return 'unknown'
}

// CATEGORICAL_COLORS imported from @/types/status

const fleetSummary = computed<FleetChipSummary[]>(() => {
  const now = Date.now()
  const chips = new Map<string, { total: number; inUse: number; uptimes: number[]; regions: Map<string, number>; bands: Map<string, number>; capacityByRegion: Map<string, { statuses: string[]; failures: number }> }>()

  for (const g of groups.value) {
    const chip = chipFromGroupName(g.name)
    if (chip == null) continue
    const region = regionFromGroupName(g.name)
    const entry = chips.get(chip) ?? { total: 0, inUse: 0, uptimes: [], regions: new Map(), bands: new Map<string, number>(), capacityByRegion: new Map<string, { statuses: string[]; failures: number }>() }

    const readyCount = g.sliceStateCounts?.['ready'] ?? 0
    const readySlices = (g.slices ?? []).filter(s => {
      const state = s.state ?? (s.vms?.length ? 'ready' : '')
      return state === 'ready' || state === 'SLICE_STATE_READY'
    })

    // Count slices (logical machines), not individual VMs.
    const sliceCount = readySlices.length > 0 ? readySlices.length : readyCount
    entry.total += sliceCount
    entry.inUse += readySlices.filter(s => sliceInUse(s)).length
    entry.regions.set(region, (entry.regions.get(region) ?? 0) + sliceCount)

    // Track capacity/availability per region
    const capEntry = entry.capacityByRegion.get(region) ?? { statuses: [], failures: 0 }
    if (g.availabilityStatus) capEntry.statuses.push(g.availabilityStatus)
    capEntry.failures += g.consecutiveFailures ?? 0
    entry.capacityByRegion.set(region, capEntry)

    // Collect band usage from scheduler running tasks via workerId join.
    // Assign each slice to a single dominant band (the band with the most
    // task-count across its VMs) so band shares partition in-use slices
    // rather than double-counting slices that host multiple bands.
    for (const slice of readySlices) {
      const sliceBandCounts = new Map<string, number>()
      for (const vm of slice.vms ?? []) {
        if (!vm.workerId) continue
        const bands = workerBands.value.get(vm.workerId)
        if (!bands) continue
        for (const [band, count] of bands) {
          sliceBandCounts.set(band, (sliceBandCounts.get(band) ?? 0) + count)
        }
      }
      let topBand: string | null = null
      let topCount = 0
      for (const [band, count] of sliceBandCounts) {
        if (count > topCount) {
          topBand = band
          topCount = count
        }
      }
      if (topBand) {
        entry.bands.set(topBand, (entry.bands.get(topBand) ?? 0) + 1)
      }
    }

    // Average uptime: use the earliest VM createdAt per slice as the slice uptime
    for (const slice of readySlices) {
      const vmTimes = (slice.vms ?? [])
        .map(vm => timestampMs(vm.createdAt))
        .filter((ms): ms is number => ms != null && ms > 0)
      if (vmTimes.length > 0) {
        const earliest = Math.min(...vmTimes)
        entry.uptimes.push(now - earliest)
      }
    }
    chips.set(chip, entry)
  }

  return Array.from(chips.entries())
    .filter(([, c]) => c.total > 0)
    .map(([chip, c]) => ({
      chip,
      total: c.total,
      inUse: c.inUse,
      avgUptimeMs: c.uptimes.length > 0
        ? c.uptimes.reduce((a, b) => a + b, 0) / c.uptimes.length
        : null,
      regions: Array.from(c.regions.entries())
        .map(([region, count]) => ({ region, count }))
        .sort((a, b) => b.count - a.count),
      bands: Array.from(c.bands.entries())
        .map(([band, count]) => ({ band, count }))
        .sort((a, b) => b.count - a.count),
      capacity: Array.from(c.capacityByRegion.entries()).map(([region, cap]) => {
        const hasQuotaExceeded = cap.statuses.includes('quota_exceeded')
        const hasBackoff = cap.statuses.includes('backoff')
        const hasAtCapacity = cap.statuses.includes('at_capacity')
        if (hasQuotaExceeded) {
          return { region, status: 'blocked' as const, detail: 'At Region Quota' }
        }
        if (hasAtCapacity || hasBackoff) {
          return { region, status: 'limited' as const, detail: 'At TRC Capacity' }
        }
        return { region, status: 'available' as const, detail: 'Compute Potentially Available' }
      }).sort((a, b) => {
        const order = { blocked: 0, limited: 1, available: 2 }
        return order[a.status] - order[b.status]
      }),
    }))
    .sort((a, b) => b.total - a.total)
})

const BAND_COLORS: Record<string, string> = {
  production: 'bg-status-danger',
  interactive: 'bg-accent',
  batch: 'bg-text-muted',
}

function bandColor(band: string): string {
  return BAND_COLORS[band] ?? 'bg-status-purple'
}

function capacityTooltip(capacity: RegionCapacity[]): string {
  if (capacity.length === 0) return ''
  return 'Capacity:\n' + capacity
    .map(c => {
      const icon = c.status === 'available' ? '✓' : c.status === 'limited' ? '~' : '✗'
      return `  ${icon} ${c.region}: ${c.detail}`
    })
    .join('\n')
}

/** Get a stable color index for a region across all chip types. */
const allRegions = computed<Map<string, number>>(() => {
  const seen = new Map<string, number>()
  // Collect all regions sorted by total count descending for stable ordering
  const regionTotals = new Map<string, number>()
  for (const c of fleetSummary.value) {
    for (const r of c.regions) {
      regionTotals.set(r.region, (regionTotals.get(r.region) ?? 0) + r.count)
    }
  }
  const sorted = Array.from(regionTotals.entries()).sort((a, b) => b[1] - a[1])
  for (const [region] of sorted) {
    seen.set(region, seen.size)
  }
  return seen
})

function regionColor(region: string): string {
  const idx = allRegions.value.get(region) ?? 0
  return CATEGORICAL_COLORS[idx % CATEGORICAL_COLORS.length]
}

function formatUptimeShort(ms: number | null): string {
  if (ms == null) return '-'
  const secs = Math.floor(ms / 1000)
  if (secs < 3600) return `${Math.floor(secs / 60)}m`
  const hours = Math.floor(secs / 3600)
  if (hours < 24) return `${hours}h ${Math.floor((secs % 3600) / 60)}m`
  return `${Math.floor(hours / 24)}d ${hours % 24}h`
}
</script>

<template>
  <!-- Loading state -->
  <div v-if="loading && !data" class="flex items-center justify-center py-12 text-text-muted text-sm">
    Loading autoscaler status...
  </div>

  <!-- Error state -->
  <div
    v-else-if="error"
    class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
  >
    {{ error }}
  </div>

  <!-- Disabled state -->
  <div v-else-if="!autoscaler" class="space-y-4">
    <EmptyState message="Autoscaler: Disabled" icon="⏸" />
  </div>

  <!-- Main content -->
  <div v-else class="space-y-6">

    <!-- ===== Status Bar ===== -->
    <div class="flex flex-wrap items-center gap-3 text-sm">
      <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span
          :class="[
            'w-2 h-2 rounded-full',
            totalSlices > 0 ? 'bg-status-success' : 'bg-text-muted',
          ]"
        />
        <span class="text-text-secondary">Groups:</span>
        <span class="font-semibold font-mono">{{ onlineGroups }} / {{ groups.length }}</span>
      </div>

      <div class="px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span class="text-text-secondary">Slices:</span>
        <span class="font-semibold font-mono ml-1">{{ formatSliceSummary(sliceTotals) }}</span>
      </div>

      <div
        v-if="totalIdle > 0"
        class="px-3 py-1.5 rounded-lg bg-status-warning-bg border border-status-warning-border"
      >
        <span class="text-text-secondary">Idle:</span>
        <span class="font-semibold font-mono text-status-warning ml-1">{{ totalIdle }}</span>
      </div>

      <div class="px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span class="text-text-secondary">Demand:</span>
        <span class="font-semibold font-mono ml-1">{{ totalDemand }}</span>
      </div>

      <div class="px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span class="text-text-secondary">Launch Planned:</span>
        <span class="font-semibold font-mono ml-1">{{ launchPlanned }}</span>
      </div>

      <div
        v-if="unmetEntries.length > 0"
        class="px-3 py-1.5 rounded-lg bg-status-danger-bg border border-status-danger-border"
      >
        <span class="text-text-secondary">Unmet:</span>
        <span class="font-semibold font-mono text-status-danger ml-1">{{ unmetEntries.length }}</span>
      </div>

      <div class="px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span class="text-text-secondary">Last Decision:</span>
        <span class="font-semibold font-mono ml-1">{{ formatRelativeTime(lastEvalMs) }}</span>
      </div>
    </div>

    <!-- ===== Fleet Overview ===== -->
    <section v-if="fleetSummary.length > 0">
      <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">
        Fleet Overview
      </h3>
      <div class="grid grid-cols-4 gap-2">
        <div
          v-for="c in fleetSummary"
          :key="c.chip"
          class="rounded-lg border border-surface-border bg-surface px-4 py-2"
          :title="capacityTooltip(c.capacity)"
        >
          <div class="flex items-baseline gap-[0.4vw] mb-1" style="font-size: clamp(10px, 0.75vw, 14px)">
            <span class="font-semibold font-mono tabular-nums text-text" style="font-size: clamp(14px, 1.1vw, 22px)">{{ c.total }}</span>
            <span class="font-medium text-text-secondary uppercase whitespace-nowrap">{{ c.chip }}</span>
            <span class="tabular-nums" :class="c.total > 0 && c.inUse === c.total ? 'text-status-warning' : 'text-text-muted'">
              {{ c.total > 0 ? Math.round(c.inUse / c.total * 100) : 0 }}% in use
            </span>
            <span class="text-text-muted tabular-nums whitespace-nowrap">
              avg uptime {{ formatUptimeShort(c.avgUptimeMs) }}
            </span>
          </div>
          <!-- Region bar -->
          <div class="flex rounded-full overflow-hidden bg-surface-sunken" style="height: clamp(4px, 0.4vw, 8px)">
            <div
              v-for="r in c.regions"
              :key="r.region"
              class="transition-all"
              :style="{ width: (r.count / c.total * 100) + '%', backgroundColor: regionColor(r.region) }"
            />
          </div>
          <div class="flex flex-wrap gap-x-[0.5vw] gap-y-0.5 mt-0.5" style="font-size: clamp(8px, 0.6vw, 11px)">
            <span
              v-for="r in c.regions"
              :key="r.region"
              class="text-text-muted flex items-center gap-0.5"
            >
              <span class="rounded-full inline-block" :style="{ backgroundColor: regionColor(r.region), width: 'clamp(4px, 0.4vw, 8px)', height: 'clamp(4px, 0.4vw, 8px)' }" />
              {{ r.region }} ({{ r.count }})
            </span>
          </div>
          <!-- Priority band breakdown -->
          <div v-if="c.bands.length > 0" class="mt-0.5 text-text-muted" style="font-size: clamp(8px, 0.6vw, 11px)">
            <span v-for="(b, i) in c.bands" :key="b.band">
              <span v-if="i > 0">, </span>
              {{ c.total > 0 ? Math.round(b.count / c.total * 100) : 0 }}% {{ b.band }}
            </span>
          </div>
        </div>
      </div>
    </section>

    <!-- ===== Waterfall Routing ===== -->
    <section>
      <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">
        Waterfall Routing
      </h3>

      <div v-if="!routing" class="text-sm text-text-muted py-4">
        No routing decision yet
      </div>

      <div v-else class="overflow-x-auto rounded-lg border border-surface-border">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border bg-surface">
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left w-16">Priority</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Group</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Slices</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-right w-20">Demand</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-right w-20">Assigned</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-right w-20">Launch</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Decision</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Reason</th>
            </tr>
          </thead>
          <tbody>
            <template v-for="section in poolSections" :key="section.pool || '__unpooled'">
              <!-- Pool header row -->
              <tr class="bg-surface border-b border-surface-border cursor-pointer hover:bg-surface-raised" @click="togglePool(section.pool)">
                <td colspan="8" class="px-3 py-1.5">
                  <div class="flex items-center gap-2">
                    <span class="text-[10px] text-text-muted">
                      {{ collapsedPools.has(section.pool) ? '▶' : '▼' }}
                    </span>
                    <span class="text-xs font-semibold uppercase tracking-wider text-text-secondary">
                      {{ section.pool === '__unpooled' ? 'Unpooled' : `Pool: ${section.pool}` }}
                    </span>
                    <span
                      v-if="section.blockedAtTier"
                      class="inline-flex items-center px-1.5 py-0.5 rounded text-xs border
                             bg-status-danger-bg text-status-danger border-status-danger-border"
                    >
                      blocked at tier {{ section.blockedAtTier }}+
                    </span>
                    <!-- Tier chain visualization (not shown for unpooled groups) -->
                    <span v-if="section.pool !== '__unpooled'" class="flex items-center gap-0.5 text-xs text-text-muted ml-2">
                      <template v-for="(gs, idx) in section.groups" :key="gs.group">
                        <span v-if="idx > 0" class="text-text-muted mx-0.5">&rarr;</span>
                        <span
                          :class="[
                            'px-1 py-0.5 rounded border text-[11px] font-mono',
                            isTierBlocked(gs, section)
                              ? 'bg-status-danger-bg text-status-danger border-status-danger-border line-through'
                              : groupIndex[gs.group]?.availabilityStatus === 'quota_exceeded'
                                ? 'bg-status-danger-bg text-status-danger border-status-danger-border'
                                : groupIndex[gs.group]?.availabilityStatus === 'backoff'
                                  ? 'bg-status-orange-bg text-status-orange border-status-orange-border'
                                  : 'bg-surface border-surface-border text-text-secondary',
                          ]"
                        >
                          {{ tierLabel(gs) }}
                        </span>
                      </template>
                    </span>
                  </div>
                </td>
              </tr>

            <template v-for="gs in section.groups" :key="gs.group">
              <!-- Main row -->
              <tr
                v-if="!collapsedPools.has(section.pool)"
                :class="[
                  'border-b border-surface-border-subtle hover:bg-surface-raised transition-colors',
                  isInactiveRow(gs) ? 'opacity-50' : '',
                  isTierBlocked(gs, section) ? 'opacity-40' : '',
                ]"
              >
                <!-- Priority -->
                <td class="px-3 py-2 text-[13px] font-mono text-text-muted">
                  {{ gs.priority ?? 100 }}
                </td>

                <!-- Group name + badges -->
                <td class="px-3 py-2 text-[13px]">
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
                  <div v-if="groupIndex[gs.group] && groupAvailabilityBadge(groupIndex[gs.group], section)" class="mt-0.5">
                    <span
                      :class="[
                        'inline-flex items-center px-1.5 py-0.5 rounded text-xs border',
                        groupAvailabilityBadge(groupIndex[gs.group], section)!.classes,
                      ]"
                    >
                      {{ groupAvailabilityBadge(groupIndex[gs.group], section)!.label }}
                    </span>
                  </div>
                </td>

                <!-- Slices (expandable) -->
                <td class="px-3 py-2 text-[13px]">
                  <button
                    v-if="groupHasSlices(gs.group)"
                    class="inline-flex items-center gap-1 cursor-pointer hover:opacity-80"
                    @click="toggleSlices(gs.group)"
                  >
                    <span class="text-[10px] text-text-muted">
                      {{ expandedSlices.has(gs.group) ? '▼' : '▶' }}
                    </span>
                    <span class="inline-flex items-center gap-1">
                      <template v-for="state in SLICE_BADGE_ORDER" :key="state">
                        <span
                          v-if="(groupBadgeCounts(gs.group)[state] ?? 0) > 0"
                          :class="[
                            'inline-flex items-center px-1.5 py-0.5 rounded text-xs font-semibold border',
                            SLICE_STATE_STYLES[state].bg,
                            SLICE_STATE_STYLES[state].text,
                            SLICE_STATE_STYLES[state].border,
                          ]"
                          :title="SLICE_STATE_STYLES[state].label"
                        >
                          {{ groupBadgeCounts(gs.group)[state] }}{{ SLICE_STATE_STYLES[state].letter }}
                        </span>
                      </template>
                      <span
                        v-if="groupIdleCount(gs.group) > 0"
                        class="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-semibold border
                               bg-status-warning-bg text-status-warning border-status-warning-border"
                        :title="`${groupIdleCount(gs.group)} idle slice${groupIdleCount(gs.group) > 1 ? 's' : ''}`"
                      >
                        {{ groupIdleCount(gs.group) }} idle
                      </span>
                    </span>
                  </button>
                  <span v-else class="text-text-muted">-</span>
                </td>

                <!-- Demand (expandable if > 0) -->
                <td class="px-3 py-2 text-[13px] text-right font-mono">
                  <button
                    v-if="groupDemand(gs.group) > 0"
                    class="cursor-pointer hover:opacity-80"
                    @click="toggleDemand(gs.group)"
                  >
                    <span class="text-[10px] text-text-muted mr-1">
                      {{ expandedDemand.has(gs.group) ? '▼' : '▶' }}
                    </span>
                    {{ groupDemand(gs.group) }}
                  </button>
                  <span v-else>{{ groupDemand(gs.group) || '' }}</span>
                </td>

                <!-- Assigned -->
                <td class="px-3 py-2 text-[13px] text-right font-mono">
                  {{ gs.assigned ?? 0 }}
                </td>

                <!-- Launch -->
                <td class="px-3 py-2 text-[13px] text-right font-mono">
                  {{ gs.launch ?? 0 }}
                </td>

                <!-- Decision -->
                <td class="px-3 py-2 text-[13px]">
                  <span :class="decisionClasses(gs.decision)">
                    {{ formatDecision(gs.decision) }}
                  </span>
                </td>

                <!-- Reason -->
                <td class="px-3 py-2 text-[13px] text-text-secondary max-w-xs truncate" :title="groupReasonText(gs)">
                  {{ groupReasonText(gs) }}
                </td>
              </tr>

              <!-- Slice detail (expanded) -->
              <tr v-if="expandedSlices.has(gs.group) && groupHasSlices(gs.group) && (!collapsedPools.has(section.pool))" class="bg-surface-sunken">
                <td colspan="8" class="px-6 py-3">
                  <div class="space-y-1.5">
                    <div
                      v-for="slice in groupSlices(gs.group)"
                      :key="slice.sliceId"
                      class="flex items-center gap-3 text-xs"
                    >
                      <span class="font-mono text-text-muted w-48 truncate" :title="slice.sliceId">
                        {{ slice.sliceId }}
                      </span>
                      <span class="text-text-secondary w-16">
                        {{ (slice.vms ?? []).length }} vm{{ (slice.vms ?? []).length !== 1 ? 's' : '' }}
                      </span>
                      <span v-if="slice.idle" class="inline-flex items-center px-1.5 py-0.5 rounded border text-xs font-semibold bg-status-warning-bg text-status-warning border-status-warning-border" :title="`Idle for ${formatIdleSince(slice.lastActive)}, threshold ${formatIdleThreshold(idleThresholdMs(gs.group))}`">
                        idle {{ formatIdleSince(slice.lastActive) }}
                      </span>
                      <span
                        v-else-if="sliceInUse(slice)"
                        :class="[
                          'inline-flex items-center px-1.5 py-0.5 rounded border text-xs font-semibold',
                          SLICE_STATE_STYLES.in_use.bg,
                          SLICE_STATE_STYLES.in_use.text,
                          SLICE_STATE_STYLES.in_use.border,
                        ]"
                        :title="SLICE_STATE_STYLES.in_use.label"
                      >
                        in use
                      </span>
                      <StatusBadge
                        v-else-if="(slice.vms ?? []).length > 0"
                        :status="vmStateToName(slice.vms![0].state)"
                        size="sm"
                        :show-dot="false"
                      />
                      <span v-else class="text-text-muted">unknown</span>
                      <span class="text-text-muted text-[11px]">
                        {{ timestampMs(slice.createdAt) ? formatRelativeTime(timestampMs(slice.createdAt)) : '-' }}
                      </span>
                      <!-- Per-VM task counts + job links -->
                      <span v-if="(slice.vms ?? []).length > 0" class="text-text-muted text-[11px] flex flex-wrap items-center gap-x-1.5 gap-y-0.5">
                        <template v-for="(vm, vi) in (slice.vms ?? [])" :key="vm.vmId">
                          <span v-if="vi > 0" class="text-text-muted">·</span>
                          <span :title="`${vm.vmId}: ${vm.runningTaskCount ?? 0} tasks`">
                            vm{{ vi }}: {{ vm.runningTaskCount ?? 0 }}t
                          </span>
                          <template v-if="tasksForVm(vm).length > 0">
                            <RouterLink
                              v-for="task in tasksForVm(vm)"
                              :key="task.taskId"
                              :to="`/job/${encodeURIComponent(task.jobId)}`"
                              class="text-accent hover:underline font-mono"
                              :title="`${task.jobId} (user: ${task.userId})`"
                            >
                              {{ task.jobId }}<span v-if="task.userId" class="text-text-muted"> · {{ task.userId }}</span>
                            </RouterLink>
                          </template>
                        </template>
                      </span>
                    </div>
                  </div>
                </td>
              </tr>

              <!-- Demand detail (expanded) -->
              <tr v-if="expandedDemand.has(gs.group) && groupDemand(gs.group) > 0 && (!collapsedPools.has(section.pool))" class="bg-surface-sunken">
                <td colspan="8" class="px-6 py-3">
                  <div class="space-y-1">
                    <div
                      v-for="row in aggregateDemandByJob(gs.group)"
                      :key="row.job"
                      class="flex items-center gap-3 text-xs"
                    >
                      <span class="font-mono text-text truncate max-w-xs">{{ row.job }}</span>
                      <span class="text-text-muted">
                        <template v-if="row.taskEntries > 0">{{ row.taskEntries }} task{{ row.taskEntries > 1 ? 's' : '' }}</template>
                        <template v-if="row.taskEntries > 0 && row.reservationEntries > 0">, </template>
                        <template v-if="row.reservationEntries > 0">{{ row.reservationEntries }} rsv</template>
                      </span>
                    </div>
                  </div>
                </td>
              </tr>
            </template>
            </template>
          </tbody>
        </table>
      </div>
    </section>

    <!-- ===== Unmet Demand ===== -->
    <section v-if="unmetEntries.length > 0">
      <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">
        Unmet Demand
      </h3>

      <div class="overflow-x-auto rounded-lg border border-surface-border">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border bg-surface">
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Job</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Reasons</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-right w-20">Entries</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Example Task</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Accelerator</th>
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

    <!-- ===== Recent Actions ===== -->
    <section>
      <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">
        Recent Actions
      </h3>

      <div v-if="actions.length === 0" class="text-sm text-text-muted py-4">
        No recent actions
      </div>

      <div v-else class="rounded-lg border border-surface-border bg-surface divide-y divide-surface-border-subtle">
        <div
          v-for="(action, i) in actions"
          :key="i"
          class="flex items-center gap-3 px-4 py-2 text-[13px] hover:bg-surface-raised transition-colors"
        >
          <span class="font-mono text-text-muted w-20 flex-shrink-0">
            {{ formatActionTime(action.timestamp) }}
          </span>
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
          <span v-if="action.reason" class="text-text-secondary">
            - {{ action.reason }}
          </span>
        </div>
      </div>
    </section>

    <!-- ===== Autoscaler Logs ===== -->
    <section>
      <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">
        Autoscaler Logs
      </h3>
      <LogViewer source="controller" max-height="40vh" />
    </section>
  </div>
</template>
