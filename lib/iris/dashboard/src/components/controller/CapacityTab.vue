<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import { resourceRpcCall } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type {
  ResourceDescribeSliceResponse,
  ResourceJobSummary,
  ResourceListJobsResponse,
  ResourceListNodesResponse,
  ResourceListSlicesResponse,
  ResourceListUsersResponse,
  ResourceNodeSummary,
  ResourceSliceDetail,
  ResourceSliceSummary,
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
const scalingGroup = ref(typeof route.query.group === 'string' ? route.query.group : '')
const loading = ref(false)
const errors = ref<string[]>([])
const users = ref<ResourceUserSummary[]>([])
const pendingJobs = ref<ResourceJobSummary[]>([])
const nodes = ref<ResourceNodeSummary[]>([])
const slices = ref<ResourceSliceSummary[]>([])
const sourceStatuses = ref<ResourceSourceStatus[]>([])
const pendingJobsTruncated = ref(false)
const nodesTruncated = ref(false)
const slicesTruncated = ref(false)
const selectedSlice = ref<ResourceSliceDetail | null>(null)
const sliceError = ref<string | null>(null)

const backendId = computed(() => typeof route.query.backend === 'string' ? route.query.backend : '')

function stateCount(user: ResourceUserSummary, ...states: string[]): number {
  return states.reduce((count, state) => count + (user.taskStateCounts?.[state] ?? 0), 0)
}

const runningTasks = computed(() => users.value.reduce((count, user) => count + stateCount(user, 'running'), 0))
const waitingTasks = computed(() => users.value.reduce(
  (count, user) => count + stateCount(user, 'pending', 'assigned', 'building', 'unschedulable'),
  0,
))
const activeJobs = computed(() => users.value.reduce(
  (count, user) => count + Object.values(user.jobStateCounts ?? {}).reduce((subtotal, value) => subtotal + value, 0),
  0,
))
const visibleNodes = computed(() => scalingGroup.value
  ? nodes.value.filter(node => node.scalingGroupId === scalingGroup.value)
  : nodes.value)

interface GroupRow {
  key: string
  backendId: string
  scalingGroupId: string
  nodeCount: number
  readyNodeCount: number
  runningTaskCount: number
  sliceCount: number
  readySliceCount: number
  observedMemberCount: number
}

const groups = computed<GroupRow[]>(() => {
  const result = new Map<string, GroupRow>()
  function row(backend: string, group: string): GroupRow {
    const key = `${backend}\u0000${group}`
    let value = result.get(key)
    if (!value) {
      value = {
        key,
        backendId: backend,
        scalingGroupId: group || '(ungrouped)',
        nodeCount: 0,
        readyNodeCount: 0,
        runningTaskCount: 0,
        sliceCount: 0,
        readySliceCount: 0,
        observedMemberCount: 0,
      }
      result.set(key, value)
    }
    return value
  }
  for (const node of visibleNodes.value) {
    const value = row(node.identity.backendId, node.scalingGroupId ?? '')
    value.nodeCount += 1
    if (node.health.includes('READY')) value.readyNodeCount += 1
    value.runningTaskCount += node.runningTaskCount
  }
  for (const slice of slices.value) {
    const value = row(slice.identity.backendId, slice.scalingGroupId)
    value.sliceCount += 1
    if (slice.lifecycle.includes('READY')) value.readySliceCount += 1
    value.observedMemberCount += slice.observedMemberCount
  }
  return [...result.values()].sort((left, right) => left.key.localeCompare(right.key))
})

function resultError(result: PromiseRejectedResult): string {
  return result.reason instanceof Error ? result.reason.message : String(result.reason)
}

async function refresh() {
  loading.value = true
  errors.value = []
  const backend = backendId.value || undefined
  const [userResult, jobResult, nodeResult, sliceResult] = await Promise.allSettled([
    resourceRpcCall<ResourceListUsersResponse>('ListUsers'),
    resourceRpcCall<ResourceListJobsResponse>('ListJobs', {
      query: {
        states: ['JOB_STATE_PENDING', 'JOB_STATE_BUILDING', 'JOB_STATE_UNSCHEDULABLE'],
        backendId: backend,
        topLevelOnly: true,
        page: { pageSize: INVENTORY_PAGE_SIZE },
      },
    }),
    resourceRpcCall<ResourceListNodesResponse>('ListNodes', {
      query: { backendId: backend, page: { pageSize: INVENTORY_PAGE_SIZE } },
    }),
    resourceRpcCall<ResourceListSlicesResponse>('ListSlices', {
      query: {
        backendId: backend,
        scalingGroupId: scalingGroup.value || undefined,
        page: { pageSize: INVENTORY_PAGE_SIZE },
      },
    }),
  ])
  if (userResult.status === 'fulfilled') users.value = userResult.value.users ?? []
  else errors.value.push(resultError(userResult))
  if (jobResult.status === 'fulfilled') {
    pendingJobs.value = jobResult.value.jobs ?? []
    pendingJobsTruncated.value = Boolean(jobResult.value.page?.nextPageToken)
  } else errors.value.push(resultError(jobResult))
  if (nodeResult.status === 'fulfilled') {
    nodes.value = nodeResult.value.nodes ?? []
    nodesTruncated.value = Boolean(nodeResult.value.page?.nextPageToken)
  } else errors.value.push(resultError(nodeResult))
  if (sliceResult.status === 'fulfilled') {
    slices.value = sliceResult.value.slices ?? []
    slicesTruncated.value = Boolean(sliceResult.value.page?.nextPageToken)
  } else errors.value.push(resultError(sliceResult))
  sourceStatuses.value = [nodeResult, sliceResult].flatMap(result =>
    result.status === 'fulfilled' ? result.value.page?.sourceStatuses ?? [] : [],
  )
  loading.value = false
}

async function describeSlice(slice: ResourceSliceSummary) {
  sliceError.value = null
  try {
    selectedSlice.value = (
      await resourceRpcCall<ResourceDescribeSliceResponse>('DescribeSlice', { slice: slice.identity })
    ).slice ?? null
  } catch (cause) {
    sliceError.value = cause instanceof Error ? cause.message : String(cause)
  }
}

function applyFilter() {
  selectedSlice.value = null
  void refresh()
}

onMounted(refresh)
useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
</script>

<template>
  <section class="space-y-6">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h2 class="text-xl font-semibold">Capacity &amp; Scheduling</h2>
        <p class="text-sm text-text-muted">Active demand, observed nodes, scaling groups, and slices.</p>
      </div>
      <form class="flex gap-2" @submit.prevent="applyFilter">
        <input
          v-model="scalingGroup"
          class="px-3 py-1.5 text-sm border rounded bg-surface"
          placeholder="Scaling group"
        />
        <button class="px-3 py-1.5 text-sm border rounded">Filter</button>
      </form>
    </div>

    <div v-for="message in errors" :key="message" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">
      {{ message }}
    </div>
    <SourceWarnings :statuses="sourceStatuses" />

    <div class="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
      <MetricCard size="sm" :value="activeJobs" label="Active jobs" :variant="activeJobs ? 'accent' : 'default'" />
      <MetricCard size="sm" :value="runningTasks" label="Running tasks" :variant="runningTasks ? 'success' : 'default'" />
      <MetricCard size="sm" :value="waitingTasks" label="Waiting tasks" :variant="waitingTasks ? 'warning' : 'default'" />
      <MetricCard size="sm" :value="visibleNodes.length + (nodesTruncated ? '+' : '')" label="Observed nodes" />
      <MetricCard size="sm" :value="slices.length + (slicesTruncated ? '+' : '')" label="Observed slices" />
      <MetricCard size="sm" :value="users.length" label="Users" />
    </div>

    <section>
      <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Scaling Groups</h3>
      <EmptyState v-if="!loading && groups.length === 0" message="No nodes or slices reported" />
      <div v-else class="overflow-x-auto rounded border border-surface-border">
        <table class="w-full border-collapse text-sm">
          <thead>
            <tr class="border-b border-surface-border text-left text-xs uppercase text-text-secondary">
              <th class="px-3 py-2">Backend</th>
              <th class="px-3 py-2">Scaling group</th>
              <th class="px-3 py-2 text-right">Ready nodes</th>
              <th class="px-3 py-2 text-right">Running tasks</th>
              <th class="px-3 py-2 text-right">Ready slices</th>
              <th class="px-3 py-2 text-right">Members</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="group in groups" :key="group.key" class="border-b border-surface-border-subtle">
              <td class="px-3 py-2 font-mono">{{ group.backendId }}</td>
              <td class="px-3 py-2 font-mono">{{ group.scalingGroupId }}</td>
              <td class="px-3 py-2 text-right">{{ group.readyNodeCount }} / {{ group.nodeCount }}</td>
              <td class="px-3 py-2 text-right">{{ group.runningTaskCount }}</td>
              <td class="px-3 py-2 text-right">{{ group.readySliceCount }} / {{ group.sliceCount }}</td>
              <td class="px-3 py-2 text-right">{{ group.observedMemberCount }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-if="nodesTruncated || slicesTruncated" class="mt-2 text-xs text-text-muted">
        Showing the first {{ INVENTORY_PAGE_SIZE }} matching nodes or slices; narrow by backend or scaling group for a complete view.
      </p>
    </section>

    <section>
      <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">
        Pending Jobs{{ pendingJobsTruncated ? '+' : '' }}
      </h3>
      <EmptyState v-if="!loading && pendingJobs.length === 0" message="No pending jobs" />
      <div v-else class="overflow-x-auto rounded border border-surface-border">
        <table class="w-full border-collapse text-sm">
          <thead>
            <tr class="border-b border-surface-border text-left text-xs uppercase text-text-secondary">
              <th class="px-3 py-2">Job</th><th class="px-3 py-2">User</th><th class="px-3 py-2">State</th>
              <th class="px-3 py-2">Backend</th><th class="px-3 py-2">Diagnostic</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="job in pendingJobs" :key="job.identity.jobUid" class="border-b border-surface-border-subtle">
              <td class="px-3 py-2">
                <RouterLink
                  :to="{ name: 'job-detail', params: { clusterId: job.identity.key.clusterId, jobId: job.identity.key.resourceId } }"
                  class="font-mono text-accent hover:underline"
                >{{ job.identity.key.resourceId }}</RouterLink>
              </td>
              <td class="px-3 py-2"><RouterLink :to="{ path: '/', query: { user: job.ownerId } }" class="font-mono text-accent hover:underline">{{ job.ownerId }}</RouterLink></td>
              <td class="px-3 py-2"><StatusBadge :status="job.state" size="sm" /></td>
              <td class="px-3 py-2 font-mono">{{ job.backendId || job.executionClusterId }}</td>
              <td class="max-w-xl px-3 py-2 text-xs text-status-warning">{{ job.pendingReason || 'Awaiting placement' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="grid gap-6 xl:grid-cols-2">
      <div>
        <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Users</h3>
        <EmptyState v-if="!loading && users.length === 0" message="No users" />
        <div v-else class="overflow-x-auto rounded border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead><tr class="border-b border-surface-border text-left text-xs uppercase text-text-secondary">
              <th class="px-3 py-2">User</th><th class="px-3 py-2 text-right">Active jobs</th>
              <th class="px-3 py-2 text-right">Running</th><th class="px-3 py-2 text-right">Waiting</th>
            </tr></thead>
            <tbody>
              <tr v-for="user in users" :key="user.userId" class="border-b border-surface-border-subtle">
                <td class="px-3 py-2"><RouterLink :to="{ path: '/', query: { user: user.userId } }" class="font-mono text-accent hover:underline">{{ user.userId }}</RouterLink></td>
                <td class="px-3 py-2 text-right">{{ Object.values(user.jobStateCounts ?? {}).reduce((count, value) => count + value, 0) }}</td>
                <td class="px-3 py-2 text-right">{{ stateCount(user, 'running') }}</td>
                <td class="px-3 py-2 text-right">{{ stateCount(user, 'pending', 'assigned', 'building', 'unschedulable') }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Slices</h3>
        <EmptyState v-if="!loading && slices.length === 0" message="No slices reported" />
        <div v-else class="overflow-x-auto rounded border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead><tr class="border-b border-surface-border text-left text-xs uppercase text-text-secondary">
              <th class="px-3 py-2">Slice</th><th class="px-3 py-2">Lifecycle</th>
              <th class="px-3 py-2 text-right">Members</th><th class="px-3 py-2">Observed</th>
            </tr></thead>
            <tbody>
              <tr v-for="slice in slices" :key="slice.identity.sliceUid" class="border-b border-surface-border-subtle">
                <td class="px-3 py-2"><button class="font-mono text-accent hover:underline" @click="describeSlice(slice)">{{ slice.identity.key.resourceId }}</button></td>
                <td class="px-3 py-2"><StatusBadge :status="slice.lifecycle" size="sm" /></td>
                <td class="px-3 py-2 text-right">{{ slice.membershipState.includes('UNKNOWN') ? 'unknown' : slice.observedMemberCount }}</td>
                <td class="px-3 py-2">{{ formatRelativeTime(timestampMs(slice.observedAt)) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <div v-if="sliceError" class="text-sm text-status-danger">{{ sliceError }}</div>
    <section v-if="selectedSlice" class="rounded border border-surface-border p-4">
      <div class="mb-3 flex items-center gap-3">
        <h3 class="font-mono font-semibold">{{ selectedSlice.summary.identity.key.resourceId }}</h3>
        <StatusBadge :status="selectedSlice.summary.lifecycle" size="sm" />
        <button class="ml-auto text-sm text-text-muted" @click="selectedSlice = null">Close</button>
      </div>
      <SourceWarnings :statuses="selectedSlice.sourceStatuses" />
      <EmptyState v-if="(selectedSlice.members ?? []).length === 0" message="No observed members" />
      <div v-else class="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
        <div v-for="member in selectedSlice.members" :key="member.providerNodeId" class="rounded bg-surface-sunken p-3 text-sm">
          <RouterLink
            v-if="member.node"
            :to="{ name: 'node-detail', params: { clusterId: member.node.key.clusterId, backendId: member.node.backendId, nodeUid: member.node.nodeUid, nodeId: member.node.key.resourceId } }"
            class="font-mono text-accent hover:underline"
          >{{ member.node.key.resourceId }}</RouterLink>
          <span v-else class="font-mono">{{ member.providerNodeId }}</span>
          <div class="mt-1 text-xs text-text-muted">Observed {{ formatRelativeTime(timestampMs(member.observedAt)) }}</div>
        </div>
      </div>
    </section>
  </section>
</template>
