<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import { resourceRpcCall } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import { LOCAL_CLUSTER } from '@/types/rpc'
import type {
  ResourceCapacityBackend,
  ResourceCapacityPeer,
  ResourceCapacityPlacement,
  ResourceCapacityScalingGroup,
  ResourceCapacitySlice,
  ResourceCapacityUnmetDemand,
  ResourceGetCapacityStatusResponse,
  ResourceJobSummary,
  ResourceListJobsResponse,
  ResourceListUsersResponse,
  ResourceSourceStatus,
  ResourceUserSummary,
} from '@/types/rpc'
import { formatRelativeTime, timestampMs } from '@/utils/formatting'
import EmptyState from '@/components/shared/EmptyState.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import SourceWarnings from '@/components/shared/SourceWarnings.vue'
import MetricCard from '@/components/shared/MetricCard.vue'

const INVENTORY_PAGE_SIZE = 100
const route = useRoute()
const scalingGroupFilter = ref(typeof route.query.group === 'string' ? route.query.group : '')
const loading = ref(false)
const errors = ref<string[]>([])
const users = ref<ResourceUserSummary[]>([])
const pendingJobs = ref<ResourceJobSummary[]>([])
const pendingJobsTruncated = ref(false)
const backends = ref<ResourceCapacityBackend[]>([])
const peers = ref<ResourceCapacityPeer[]>([])
const placements = ref<ResourceCapacityPlacement[]>([])
const unroutableJobCount = ref(0)
const unroutableJobs = ref<Array<{ jobId: string; reason: string }>>([])
const sourceStatuses = ref<ResourceSourceStatus[]>([])
const expandedGroups = ref<Set<string>>(new Set())

const backendId = computed(() => typeof route.query.backend === 'string' ? route.query.backend : '')
const peerId = computed(() => typeof route.query.cluster === 'string' ? route.query.cluster : '')

function stateCount(user: ResourceUserSummary, ...states: string[]): number {
  return states.reduce((count, state) => count + (user.taskStateCounts?.[state] ?? 0), 0)
}

const visibleBackends = computed(() => peerId.value && peerId.value !== LOCAL_CLUSTER
  ? []
  : backends.value.filter(backend => !backendId.value || backend.backendId === backendId.value))
const visiblePeers = computed(() => backendId.value
  ? []
  : peers.value.filter(peer => !peerId.value || (peerId.value !== LOCAL_CLUSTER && peer.peerId === peerId.value)))

interface GroupRow {
  key: string
  backend: ResourceCapacityBackend
  group: ResourceCapacityScalingGroup
}

const groups = computed<GroupRow[]>(() => visibleBackends.value.flatMap(backend =>
  (backend.scalingGroups ?? [])
    .filter(group => !scalingGroupFilter.value || group.name.toLowerCase().includes(scalingGroupFilter.value.toLowerCase()))
    .map(group => ({ key: `${backend.backendId}\u0000${group.name}`, backend, group })),
))

const allSlices = computed(() => groups.value.flatMap(row => row.group.slices ?? []))
const readySlices = computed(() => allSlices.value.filter(slice => lifecycle(slice) === 'ready').length)
const idleSlices = computed(() => allSlices.value.filter(slice => capacityState(slice) === 'idle').length)
const degradedSlices = computed(() => allSlices.value.filter(slice => capacityState(slice) === 'degraded').length)
const demand = computed(() => groups.value.reduce((count, row) => count + (row.group.currentDemand ?? 0), 0))
const plannedLaunches = computed(() => groups.value.reduce(
  (count, row) => count + (routingFor(row)?.launch ?? 0),
  0,
))
const lastEvaluation = computed(() => Math.max(
  0,
  ...visibleBackends.value.map(backend => timestampMs(backend.lastEvaluation) ?? 0),
))

interface FleetVariant {
  key: string
  variant: string
  region: string
  total: number
  inUse: number
  available: number
  degraded: number
}

const fleetVariants = computed<FleetVariant[]>(() => {
  const result = new Map<string, FleetVariant>()
  for (const { group } of groups.value) {
    const variant = group.deviceVariant || group.deviceType || 'CPU'
    const region = group.region || 'unknown'
    const key = `${variant}\u0000${region}`
    const row = result.get(key) ?? { key, variant, region, total: 0, inUse: 0, available: 0, degraded: 0 }
    for (const slice of group.slices ?? []) {
      if (lifecycle(slice) !== 'ready') continue
      row.total += 1
      const state = capacityState(slice)
      if (state === 'in_use') row.inUse += 1
      if (state === 'available') row.available += 1
      if (state === 'degraded') row.degraded += 1
    }
    result.set(key, row)
  }
  return [...result.values()].sort((left, right) => left.key.localeCompare(right.key))
})

const unmet = computed<ResourceCapacityUnmetDemand[]>(() => visibleBackends.value.flatMap(
  backend => backend.routing?.unmet ?? [],
))
const recentActions = computed(() => visibleBackends.value.flatMap(backend =>
  (backend.recentActions ?? []).map(action => ({ backendId: backend.backendId, action })),
).sort((left, right) => (timestampMs(right.action.timestamp) ?? 0) - (timestampMs(left.action.timestamp) ?? 0)).slice(0, 20))

function resultError(result: PromiseRejectedResult): string {
  return result.reason instanceof Error ? result.reason.message : String(result.reason)
}

async function refresh() {
  loading.value = true
  errors.value = []
  const [capacityResult, userResult, jobResult] = await Promise.allSettled([
    resourceRpcCall<ResourceGetCapacityStatusResponse>('GetCapacityStatus'),
    resourceRpcCall<ResourceListUsersResponse>('ListUsers'),
    resourceRpcCall<ResourceListJobsResponse>('ListJobs', {
      query: {
        states: ['JOB_STATE_PENDING', 'JOB_STATE_BUILDING', 'JOB_STATE_UNSCHEDULABLE'],
        backendId: backendId.value || undefined,
        executionClusterId: peerId.value || undefined,
        topLevelOnly: true,
        page: { pageSize: INVENTORY_PAGE_SIZE },
      },
    }),
  ])
  if (capacityResult.status === 'fulfilled') {
    backends.value = capacityResult.value.backends ?? []
    peers.value = capacityResult.value.peers ?? []
    placements.value = capacityResult.value.runningPlacements ?? []
    unroutableJobCount.value = capacityResult.value.unroutableJobCount ?? 0
    unroutableJobs.value = capacityResult.value.unroutableJobs ?? []
    sourceStatuses.value = capacityResult.value.sourceStatuses ?? []
  } else errors.value.push(resultError(capacityResult))
  if (userResult.status === 'fulfilled') users.value = userResult.value.users ?? []
  else errors.value.push(resultError(userResult))
  if (jobResult.status === 'fulfilled') {
    pendingJobs.value = jobResult.value.jobs ?? []
    pendingJobsTruncated.value = Boolean(jobResult.value.page?.nextPageToken)
  } else errors.value.push(resultError(jobResult))
  loading.value = false
}

function lifecycle(slice: ResourceCapacitySlice): string {
  return (slice.summary.lifecycle || 'unknown').replace('SLICE_LIFECYCLE_', '').toLowerCase()
}

function capacityState(slice: ResourceCapacitySlice): string {
  return (slice.summary.capacityState || 'unknown').replace('SLICE_CAPACITY_STATE_', '').toLowerCase()
}

function routingFor(row: GroupRow) {
  return row.backend.routing?.groups?.find(item => item.scalingGroupId === row.group.name)
}

function toggleGroup(key: string) {
  const next = new Set(expandedGroups.value)
  if (next.has(key)) next.delete(key)
  else next.add(key)
  expandedGroups.value = next
}

interface SliceJob {
  jobId: string
  userId: string
  taskCount: number
  hostCount: number
}

function sliceJobs(slice: ResourceCapacitySlice): SliceJob[] {
  const workerIds = new Set((slice.members ?? []).map(member => member.workerId).filter(Boolean))
  const jobs = new Map<string, SliceJob>()
  for (const placement of placements.value) {
    if (!workerIds.has(placement.workerId)) continue
    const row = jobs.get(placement.jobId) ?? {
      jobId: placement.jobId,
      userId: placement.userId,
      taskCount: 0,
      hostCount: 0,
    }
    row.taskCount += placement.taskCount
    row.hostCount += 1
    jobs.set(placement.jobId, row)
  }
  return [...jobs.values()].sort((left, right) => right.taskCount - left.taskCount)
}

function backendHealth(backend: ResourceCapacityBackend): string {
  if (Object.keys(backend.capacityHealth ?? {}).some(key => ['quota_exceeded', 'backoff'].includes(key))) {
    return 'blocked'
  }
  if ((backend.capacityHealth?.degraded ?? 0) > 0) return 'degraded'
  if ((backend.workerCount ?? 0) > 0 || backend.kubernetes) return 'healthy'
  return 'no capacity observed'
}

function peerContact(peer: ResourceCapacityPeer): string {
  const value = Number(peer.lastContactMs ?? 0)
  return value ? formatRelativeTime(value) : 'never'
}

function applyFilter() {
  expandedGroups.value = new Set()
}

watch(() => [route.query.backend, route.query.cluster], refresh)
onMounted(refresh)
useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
</script>

<template>
  <section class="space-y-6">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h2 class="text-xl font-semibold">Capacity</h2>
        <p class="text-sm text-text-muted">Backend-owned pools, slice inventory, routing decisions, and demand.</p>
      </div>
      <form class="flex gap-2" @submit.prevent="applyFilter">
        <input v-model="scalingGroupFilter" class="px-3 py-1.5 text-sm border rounded bg-surface" placeholder="Scaling group" />
        <button class="px-3 py-1.5 text-sm border rounded">Filter</button>
      </form>
    </div>

    <div v-for="message in errors" :key="message" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">
      {{ message }}
    </div>
    <SourceWarnings :statuses="sourceStatuses" />

    <div v-if="unroutableJobCount" class="rounded border border-status-danger-border bg-status-danger-bg px-4 py-3 text-sm text-status-danger">
      <span class="font-semibold">{{ unroutableJobCount }} unroutable job{{ unroutableJobCount === 1 ? '' : 's' }}</span>
      <span v-if="unroutableJobs.length"> — {{ unroutableJobs.slice(0, 3).map(item => `${item.jobId}: ${item.reason}`).join(' · ') }}</span>
    </div>

    <div class="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
      <MetricCard size="sm" :value="visibleBackends.length" label="Backends" />
      <MetricCard size="sm" :value="allSlices.length" label="Slices" />
      <MetricCard size="sm" :value="readySlices" label="Ready" :variant="readySlices ? 'success' : 'default'" />
      <MetricCard size="sm" :value="idleSlices" label="Idle spare" />
      <MetricCard size="sm" :value="degradedSlices" label="Degraded" :variant="degradedSlices ? 'warning' : 'default'" />
      <MetricCard size="sm" :value="demand" label="Demand" :variant="demand ? 'accent' : 'default'" />
      <MetricCard size="sm" :value="plannedLaunches" label="Launch planned" />
      <MetricCard size="sm" :value="lastEvaluation ? formatRelativeTime(lastEvaluation) : 'never'" label="Last evaluation" />
    </div>

    <section v-if="fleetVariants.length">
      <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Fleet Overview</h3>
      <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div v-for="variant in fleetVariants" :key="variant.key" class="rounded border border-surface-border bg-surface p-3">
          <div class="flex items-baseline justify-between gap-2">
            <span class="font-mono text-lg font-semibold">{{ variant.total }} {{ variant.variant }}</span>
            <span class="text-xs text-text-muted">{{ variant.region }}</span>
          </div>
          <div class="mt-2 h-2 overflow-hidden rounded bg-surface-sunken">
            <div class="h-full bg-status-success" :style="{ width: `${variant.total ? variant.inUse / variant.total * 100 : 0}%` }" />
          </div>
          <div class="mt-1 flex justify-between text-xs text-text-muted">
            <span>{{ variant.inUse }} in use</span><span>{{ variant.available }} available<span v-if="variant.degraded"> · {{ variant.degraded }} degraded</span></span>
          </div>
        </div>
      </div>
    </section>

    <section>
      <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Execution Targets</h3>
      <EmptyState v-if="!loading && visibleBackends.length === 0 && visiblePeers.length === 0" message="No execution targets reported" />
      <div v-else class="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        <div v-for="backend in visibleBackends" :key="backend.backendId" class="rounded border border-surface-border bg-surface p-4">
          <div class="flex items-center justify-between gap-2">
            <div><span class="font-semibold">{{ backend.name || backend.backendId }}</span><span class="ml-2 font-mono text-xs text-text-muted">{{ backend.kind }}</span></div>
            <StatusBadge :status="backendHealth(backend)" size="sm" />
          </div>
          <div class="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
            <div><div class="font-mono text-base">{{ backend.workerCount ?? 0 }}</div><div class="text-text-muted">nodes</div></div>
            <div><div class="font-mono text-base">{{ backend.runningTaskCount ?? 0 }}</div><div class="text-text-muted">running</div></div>
            <div><div class="font-mono text-base">{{ backend.pendingTaskCount ?? 0 }}</div><div class="text-text-muted">pending</div></div>
          </div>
          <div v-if="backend.availability" class="mt-3 text-xs text-text-muted">
            <span v-for="(total, token) in backend.availability.totalAmounts" :key="token" class="mr-3">
              {{ token }}: {{ backend.availability.amounts?.[token] ?? '0' }} / {{ total }} free
            </span>
          </div>
          <div v-if="backend.kubernetes" class="mt-3 border-t border-surface-border-subtle pt-3 text-xs text-text-muted">
            {{ backend.kubernetes.schedulableNodes ?? 0 }} / {{ backend.kubernetes.totalNodes ?? 0 }} schedulable nodes
            <span v-if="backend.kubernetes.namespace"> · namespace {{ backend.kubernetes.namespace }}</span>
            <div v-if="backend.kubernetes.pools?.length" class="mt-2">
              {{ backend.kubernetes.pools.map(pool => `${pool.name}: ${pool.currentNodes ?? 0}/${pool.targetNodes ?? 0}`).join(' · ') }}
            </div>
          </div>
        </div>
        <div v-for="peer in visiblePeers" :key="peer.peerId" class="rounded border border-surface-border bg-surface p-4">
          <div class="flex items-center justify-between gap-2"><span class="font-semibold">{{ peer.peerId }}</span><StatusBadge :status="peer.reachable ? 'reachable' : 'unreachable'" size="sm" /></div>
          <div class="mt-2 text-xs text-text-muted">Federation peer · last contact {{ peerContact(peer) }}</div>
          <div class="mt-3 text-sm">{{ peer.backends?.length ?? 0 }} backend{{ (peer.backends?.length ?? 0) === 1 ? '' : 's' }} · {{ peer.activeFederatedJobs ?? 0 }} active jobs</div>
          <RouterLink :to="{ path: '/', query: { cluster: peer.peerId } }" class="mt-3 inline-block text-sm text-accent hover:underline">View jobs</RouterLink>
        </div>
      </div>
    </section>

    <section>
      <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Pools — Capacity &amp; Routing</h3>
      <EmptyState v-if="!loading && groups.length === 0" message="No autoscaled pools reported" />
      <div v-else class="overflow-x-auto rounded border border-surface-border">
        <table class="w-full border-collapse text-sm">
          <thead><tr class="border-b border-surface-border text-left text-xs uppercase text-text-secondary">
            <th class="px-3 py-2">Pool / tier</th><th class="px-3 py-2">Device / region</th><th class="px-3 py-2">Slices</th>
            <th class="px-3 py-2 text-right">Demand</th><th class="px-3 py-2 text-right">Assigned</th><th class="px-3 py-2 text-right">Launch</th><th class="px-3 py-2">Decision</th>
          </tr></thead>
          <tbody>
            <template v-for="row in groups" :key="row.key">
              <tr class="border-b border-surface-border-subtle align-top">
                <td class="px-3 py-2"><button class="text-left font-mono text-accent hover:underline" @click="toggleGroup(row.key)">{{ row.group.quotaPool || row.group.name }}</button><div class="text-xs text-text-muted">{{ row.group.name }}<span v-if="row.group.allocationTier"> · tier {{ row.group.allocationTier }}</span></div></td>
                <td class="px-3 py-2">{{ row.group.deviceVariant || row.group.deviceType || 'CPU' }}<div class="text-xs text-text-muted">{{ row.group.region || 'unknown' }}</div></td>
                <td class="px-3 py-2"><span class="font-mono">{{ row.group.slices?.length ?? 0 }}</span><span v-if="row.group.availabilityStatus" class="ml-2"><StatusBadge :status="row.group.availabilityStatus" size="sm" /></span></td>
                <td class="px-3 py-2 text-right font-mono">{{ row.group.currentDemand ?? 0 }}</td>
                <td class="px-3 py-2 text-right font-mono">{{ routingFor(row)?.assigned ?? 0 }}</td>
                <td class="px-3 py-2 text-right font-mono">{{ routingFor(row)?.launch ?? 0 }}</td>
                <td class="max-w-sm px-3 py-2"><span>{{ routingFor(row)?.decision || '—' }}</span><div class="text-xs text-text-muted">{{ routingFor(row)?.reason || row.group.availabilityReason }}</div></td>
              </tr>
              <tr v-if="expandedGroups.has(row.key)" class="border-b border-surface-border bg-surface-sunken/40">
                <td colspan="7" class="px-4 py-3">
                  <EmptyState v-if="!(row.group.slices?.length)" message="No slices reported" />
                  <div v-else class="space-y-2">
                    <div v-for="slice in row.group.slices" :key="slice.summary.identity.sliceUid" class="grid items-center gap-3 rounded bg-surface px-3 py-2 text-xs md:grid-cols-[110px_1fr_120px_2fr]">
                      <StatusBadge :status="capacityState(slice) === 'unknown' ? lifecycle(slice) : capacityState(slice)" size="sm" />
                      <span class="truncate font-mono" :title="slice.summary.identity.key.resourceId">{{ slice.summary.identity.key.resourceId }}</span>
                      <span class="text-text-muted">{{ slice.summary.healthyMemberCount ?? 0 }}/{{ slice.summary.observedMemberCount }} healthy · {{ slice.summary.runningTaskCount ?? 0 }} tasks</span>
                      <div class="min-w-0">
                        <template v-if="sliceJobs(slice).length">
                          <RouterLink v-for="job in sliceJobs(slice)" :key="job.jobId" :to="{ name: 'job-detail', params: { clusterId: slice.summary.identity.key.clusterId, jobId: job.jobId } }" class="mr-3 font-mono text-accent hover:underline">{{ job.jobId }} ×{{ job.taskCount }}</RouterLink>
                        </template>
                        <span v-else-if="capacityState(slice) === 'idle'" class="text-status-warning">idle {{ formatRelativeTime(timestampMs(slice.summary.lastActiveAt)) }}</span>
                        <span v-else class="text-text-muted">{{ slice.summary.errorMessage || `${slice.members?.length ?? 0} observed members` }}</span>
                      </div>
                    </div>
                  </div>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </section>

    <section v-if="unmet.length || recentActions.length" class="grid gap-6 xl:grid-cols-2">
      <div>
        <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Unmet Demand</h3>
        <EmptyState v-if="!unmet.length" message="No unmet demand" />
        <div v-else class="space-y-2">
          <div v-for="(item, index) in unmet" :key="index" class="rounded border border-surface-border px-3 py-2 text-sm">
            <span class="font-mono">{{ item.entry?.deviceVariant || item.entry?.deviceType || 'unspecified device' }}</span>
            <span class="ml-2 text-text-muted">{{ item.entry?.taskIds?.length ?? 0 }} tasks</span>
            <div class="mt-1 text-xs text-status-warning">{{ item.reason }}</div>
          </div>
        </div>
      </div>
      <div>
        <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Recent Capacity Actions</h3>
        <div class="space-y-2">
          <div v-for="item in recentActions" :key="`${item.backendId}-${item.action.timestamp?.epochMs}-${item.action.sliceId}`" class="flex gap-3 rounded border border-surface-border px-3 py-2 text-sm">
            <span class="font-mono">{{ item.action.actionType }}</span><span>{{ item.action.scalingGroupId }}</span><span class="min-w-0 flex-1 truncate text-text-muted">{{ item.action.reason }}</span><span class="text-xs text-text-muted">{{ formatRelativeTime(timestampMs(item.action.timestamp)) }}</span>
          </div>
        </div>
      </div>
    </section>

    <section>
      <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Pending Jobs{{ pendingJobsTruncated ? '+' : '' }}</h3>
      <EmptyState v-if="!loading && pendingJobs.length === 0" message="No pending jobs" />
      <div v-else class="overflow-x-auto rounded border border-surface-border">
        <table class="w-full border-collapse text-sm">
          <thead><tr class="border-b border-surface-border text-left text-xs uppercase text-text-secondary"><th class="px-3 py-2">Job</th><th class="px-3 py-2">User</th><th class="px-3 py-2">State</th><th class="px-3 py-2">Backend</th><th class="px-3 py-2">Diagnostic</th></tr></thead>
          <tbody><tr v-for="job in pendingJobs" :key="job.identity.jobUid" class="border-b border-surface-border-subtle">
            <td class="px-3 py-2"><RouterLink :to="{ name: 'job-detail', params: { clusterId: job.identity.key.clusterId, jobId: job.identity.key.resourceId } }" class="font-mono text-accent hover:underline">{{ job.identity.key.resourceId }}</RouterLink></td>
            <td class="px-3 py-2"><RouterLink :to="{ path: '/', query: { user: job.ownerId } }" class="font-mono text-accent hover:underline">{{ job.ownerId }}</RouterLink></td>
            <td class="px-3 py-2"><StatusBadge :status="job.state" size="sm" /></td><td class="px-3 py-2 font-mono">{{ job.backendId || job.executionClusterId }}</td><td class="max-w-xl px-3 py-2 text-xs text-status-warning">{{ job.pendingReason || 'Awaiting placement' }}</td>
          </tr></tbody>
        </table>
      </div>
    </section>

    <section>
      <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Users</h3>
      <div class="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        <RouterLink v-for="user in users" :key="user.userId" :to="{ path: '/', query: { user: user.userId } }" class="flex items-center justify-between rounded border border-surface-border bg-surface px-3 py-2 text-sm hover:border-accent">
          <span class="font-mono text-accent">{{ user.userId }}</span><span class="text-xs text-text-muted">{{ stateCount(user, 'running') }} running · {{ stateCount(user, 'pending', 'assigned', 'building', 'unschedulable') }} waiting</span>
        </RouterLink>
      </div>
    </section>
  </section>
</template>
