<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { useLogServerStatsRpc } from '@/composables/useRpc'
import {
  RESOURCE_MESSAGES,
  RESOURCE_TYPES,
  updateResource,
  useGetResource,
  useListResources,
} from '@/composables/useResources'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type {
  ResourceActionResponse,
  ResourceActionReceipt,
  ResourceAttemptDetail,
  ResourceEndpointSummary,
  ResourceJobDetail,
  ResourceTaskDetail,
} from '@/types/rpc'
import { formatBytes, formatDuration, formatTimestamp, timestampMs } from '@/utils/formatting'
import { decodeArrowIpc } from '@/utils/arrow'
import { detailSql } from '@/utils/taskStatus'
import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import LogViewer from '@/components/shared/LogViewer.vue'
import EndpointLink from '@/components/shared/EndpointLink.vue'
import ProfileLink from '@/components/shared/ProfileLink.vue'
import SourceWarnings from '@/components/shared/SourceWarnings.vue'
import ActivityTimeline from '@/components/shared/ActivityTimeline.vue'
import ProfileButtons from '@/components/shared/ProfileButtons.vue'
import { useAttemptProfileAction } from '@/composables/useProfileAction'
import ResourceGauge from '@/components/shared/ResourceGauge.vue'
import Sparkline from '@/components/shared/Sparkline.vue'
import MarkdownRenderer from '@/components/shared/MarkdownRenderer.vue'

const TASK_REFRESH_MS = 10_000

const props = defineProps<{ clusterId: string; taskId: string }>()
const key = computed(() => ({
  clusterId: props.clusterId,
  kind: 'RESOURCE_KIND_TASK',
  resourceId: props.taskId,
}))
const taskRef = computed(() => ({ authorityClusterId: props.clusterId, type: RESOURCE_TYPES.task, id: props.taskId }))
const { data, loading, error, refresh } = useGetResource<ResourceTaskDetail>(() => taskRef.value, 'FULL')
const task = computed(() => data.value)
const { data: jobData, refresh: refreshJob } = useGetResource<ResourceJobDetail>(() => ({
  authorityClusterId: task.value?.summary.job.key.clusterId ?? props.clusterId,
  type: RESOURCE_TYPES.job,
  id: task.value?.summary.job.key.resourceId ?? props.taskId.slice(0, props.taskId.lastIndexOf('/')),
}), 'FULL')
const selectedAttempt = ref<number | undefined>()
const attemptNumber = computed(() => selectedAttempt.value ?? task.value?.summary.currentAttempt?.attemptNumber)
const { data: attemptData, error: attemptError, refresh: refreshAttempt } =
  useGetResource<ResourceAttemptDetail>(() => ({
    authorityClusterId: props.clusterId,
    type: RESOURCE_TYPES.attempt,
    id: `${props.taskId}:${attemptNumber.value ?? 'current'}`,
  }), 'FULL')
const { data: endpointData, error: endpointError, refresh: refreshEndpoints } =
  useListResources<ResourceEndpointSummary>(
    RESOURCE_TYPES.endpoint,
    RESOURCE_MESSAGES.endpointQuery,
    () => ({ task: key.value, page: { pageSize: 100 } }),
    'BASIC',
  )

interface QueryResponse { arrowIpc?: string }
interface UsageRow {
  cpu_millicores?: number
  memory_mb?: number
  memory_peak_mb?: number
  disk_mb?: number
}
interface StatusRow { status_text_detail_md?: string }

function sqlString(value: string): string {
  return `'${value.replace(/'/g, "''")}'`
}

const { data: usageData, error: usageError, refresh: refreshUsage } = useLogServerStatsRpc<QueryResponse>(
  'Query',
  () => ({
    sql: attemptNumber.value === undefined
      ? 'SELECT cpu_millicores, memory_mb, memory_peak_mb, disk_mb FROM "iris.task" WHERE false'
      : `SELECT cpu_millicores, memory_mb, memory_peak_mb, disk_mb
FROM "iris.task"
WHERE task_id = ${sqlString(props.taskId)} AND attempt_id = ${attemptNumber.value}
ORDER BY ts DESC
LIMIT 60`,
  }),
)
const { data: statusData, refresh: refreshStatus } = useLogServerStatsRpc<QueryResponse>(
  'Query',
  () => ({ sql: detailSql(props.taskId) }),
)
const action = ref<ResourceActionResponse | null>(null)
const actionError = ref<string | null>(null)
const acting = ref(false)

const summary = computed(() => task.value?.summary)
const selected = computed(() => attemptData.value)
const { profiling, profile } = useAttemptProfileAction(
  () => selected.value?.summary.identity,
  () => `${props.taskId}:${attemptNumber.value ?? 0}`,
)
const endpoints = computed<ResourceEndpointSummary[]>(() => endpointData.value?.items ?? [])
const usageRows = computed(() => (
  decodeArrowIpc(usageData.value?.arrowIpc).rows as UsageRow[]
))
const latestUsage = computed(() => usageRows.value[0])
const cpuHistory = computed(() => usageRows.value.map(row => Number(row.cpu_millicores ?? 0) / 1_000).reverse())
const memoryHistory = computed(() => usageRows.value.map(row => Number(row.memory_mb ?? 0)).reverse())
const cpuUsed = computed(() => Number(latestUsage.value?.cpu_millicores ?? 0) / 1_000)
const memoryUsed = computed(() => Number(latestUsage.value?.memory_mb ?? 0) * 1024 * 1024)
const diskUsed = computed(() => Number(latestUsage.value?.disk_mb ?? 0) * 1024 * 1024)
const memoryPeak = computed(() => Number(latestUsage.value?.memory_peak_mb ?? 0) * 1024 * 1024)
const cpuLimit = computed(() => Number(jobData.value?.spec.resources?.cpuMillicores ?? 0) / 1_000)
const memoryLimit = computed(() => Number(jobData.value?.spec.resources?.memoryBytes ?? 0))
const diskLimit = computed(() => Number(jobData.value?.spec.resources?.diskBytes ?? 0))
const cpuGaugeLimit = computed(() => Math.max(cpuLimit.value, cpuUsed.value, 1))
const memoryGaugeLimit = computed(() => Math.max(memoryLimit.value, memoryPeak.value, memoryUsed.value, 1))
const diskGaugeLimit = computed(() => Math.max(diskLimit.value, diskUsed.value, 1))
const detailedStatus = computed(() => {
  const rows = decodeArrowIpc(statusData.value?.arrowIpc).rows as StatusRow[]
  return rows[0]?.status_text_detail_md ?? ''
})
const backTo = computed(() => task.value
  ? {
      name: 'job-detail',
      params: {
        clusterId: task.value.summary.job.key.clusterId,
        jobId: task.value.summary.job.key.resourceId,
      },
    }
  : '/')
const taskDuration = computed(() => formatDuration(
  timestampMs(summary.value?.startedAt),
  timestampMs(summary.value?.finishedAt) || undefined,
))
const attemptDuration = computed(() => formatDuration(
  timestampMs(selected.value?.summary.startedAt),
  timestampMs(selected.value?.summary.finishedAt) || undefined,
))
const logCluster = computed(() => {
  const executionCluster = summary.value?.executionClusterId
  return executionCluster && executionCluster !== props.clusterId ? executionCluster : undefined
})

function nodeRoute() {
  const node = summary.value?.currentNode
  if (!node) return '/nodes'
  return {
    name: 'node-detail',
    params: {
      clusterId: node.key.clusterId,
      backendId: node.backendId,
      nodeUid: node.nodeUid,
      nodeId: node.key.resourceId,
    },
  }
}

async function refreshPage() {
  await refresh()
  await Promise.all([refreshJob(), refreshEndpoints(), refreshUsage(), refreshStatus()])
  if (attemptNumber.value !== undefined) await refreshAttempt()
}

async function retryTask() {
  const current = summary.value?.currentAttempt
  if (!current || acting.value) return
  acting.value = true
  actionError.value = null
  try {
    const operation = await updateResource<ResourceActionReceipt>(
      { ...taskRef.value, uid: summary.value!.identity.taskUid },
      'PENDING',
      {},
      {
        type: RESOURCE_MESSAGES.retryTaskRequest,
        value: { expectedAttemptUid: current.attemptUid },
      },
    )
    action.value = { receipt: operation.result }
    selectedAttempt.value = undefined
    await refreshPage()
  } catch (cause) {
    actionError.value = cause instanceof Error ? cause.message : String(cause)
  } finally {
    acting.value = false
  }
}

async function terminateAttempt() {
  const attempt = selected.value?.summary.identity
  if (!attempt || acting.value) return
  acting.value = true
  actionError.value = null
  try {
    const operation = await updateResource<ResourceActionReceipt>({
      authorityClusterId: attempt.task.clusterId,
      type: RESOURCE_TYPES.attempt,
      id: `${attempt.task.resourceId}:${attempt.attemptNumber}`,
      uid: attempt.attemptUid,
    }, 'CANCELLED')
    action.value = { receipt: operation.result }
    await refreshPage()
  } catch (cause) {
    actionError.value = cause instanceof Error ? cause.message : String(cause)
  } finally {
    acting.value = false
  }
}

async function selectAttempt(number: number) {
  selectedAttempt.value = number
  await Promise.all([refreshAttempt(), refreshUsage()])
}

onMounted(refreshPage)
useAutoRefresh(refreshPage, TASK_REFRESH_MS)
</script>

<template>
  <PageShell :title="taskId" :back-to="backTo" back-label="Job">
    <div v-if="error" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">{{ error }}</div>
    <div v-else-if="loading && !task" class="text-sm text-text-muted">Loading task…</div>
    <div v-else-if="task" class="space-y-6">
      <SourceWarnings :statuses="task.sourceStatuses" />

      <div
        v-if="task.rootCauseHighlights?.length"
        class="px-4 py-3 rounded border border-status-danger-border bg-status-danger-bg"
      >
        <h3 class="mb-2 text-sm font-semibold text-status-danger">Likely Root Cause</h3>
        <pre class="whitespace-pre-wrap break-all font-mono text-xs text-status-danger">{{ task.rootCauseHighlights.join('\n') }}</pre>
      </div>
      <div
        v-if="summary!.errorMessage"
        class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border border-status-danger-border"
      >
        <span class="font-semibold">Error:</span> {{ summary!.errorMessage }}
      </div>
      <div
        v-if="summary!.statusMessage"
        class="px-4 py-3 whitespace-pre-wrap text-sm text-status-warning bg-status-warning-bg rounded border border-status-warning-border"
      >
        {{ summary!.statusMessage }}
      </div>

      <div class="flex flex-wrap items-center gap-3">
        <StatusBadge :status="summary!.state" />
        <RouterLink :to="backTo" class="font-mono text-sm text-accent hover:underline">
          {{ summary!.job.key.resourceId }}
        </RouterLink>
        <button
          class="ml-auto px-3 py-1.5 text-sm border rounded disabled:opacity-40"
          :disabled="acting || !summary!.currentAttempt"
          @click="retryTask"
        >
          Retry current attempt
        </button>
        <button
          class="px-3 py-1.5 text-sm border border-status-danger-border text-status-danger rounded disabled:opacity-40"
          :disabled="acting || !selected"
          @click="terminateAttempt"
        >
          Terminate selected attempt
        </button>
      </div>
      <div v-if="action?.receipt" class="p-3 border rounded text-sm">
        Action <span class="font-mono">{{ action.receipt.actionId }}</span>: {{ action.receipt.state }}
      </div>
      <div v-if="actionError" class="text-sm text-status-danger">{{ actionError }}</div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <InfoCard title="Status">
          <InfoRow label="State"><StatusBadge :status="summary!.state" size="sm" /></InfoRow>
          <InfoRow label="Submitted"><span class="font-mono">{{ formatTimestamp(summary!.submittedAt) }}</span></InfoRow>
          <InfoRow label="Started"><span class="font-mono">{{ formatTimestamp(summary!.startedAt) }}</span></InfoRow>
          <InfoRow label="Finished"><span class="font-mono">{{ formatTimestamp(summary!.finishedAt) }}</span></InfoRow>
          <InfoRow label="Duration"><span class="font-mono">{{ taskDuration }}</span></InfoRow>
        </InfoCard>

        <InfoCard title="Placement">
          <InfoRow label="Backend"><span class="font-mono">{{ summary!.backendId || '—' }}</span></InfoRow>
          <InfoRow label="Cluster"><span class="font-mono">{{ summary!.executionClusterId }}</span></InfoRow>
          <InfoRow label="Node">
            <RouterLink v-if="summary!.currentNode" :to="nodeRoute()" class="font-mono text-accent hover:underline">
              {{ summary!.currentNode.key.resourceId }}
            </RouterLink>
            <span v-else>—</span>
          </InfoRow>
          <InfoRow label="Current attempt">{{ summary!.currentAttempt?.attemptNumber ?? '—' }}</InfoRow>
        </InfoCard>

        <InfoCard title="History">
          <InfoRow label="Attempts">{{ task.attempts?.length ?? 0 }}</InfoRow>
          <InfoRow label="Failures">{{ summary!.failureCount }}</InfoRow>
          <InfoRow label="Preemptions">{{ summary!.preemptionCount }}</InfoRow>
          <InfoRow label="Task UID"><span class="font-mono text-xs">{{ summary!.identity.taskUid }}</span></InfoRow>
        </InfoCard>
      </div>

      <div class="grid gap-4 lg:grid-cols-2">
        <InfoCard title="Live Resources">
          <div v-if="latestUsage" class="space-y-4">
            <div class="flex items-center gap-3">
              <div class="flex-1"><ResourceGauge label="CPU" :used="cpuUsed" :total="cpuGaugeLimit" unit="cores" /></div>
              <div class="w-24"><Sparkline :data="cpuHistory" /></div>
            </div>
            <div class="flex items-center gap-3">
              <div class="flex-1"><ResourceGauge label="Memory" :used="memoryUsed" :total="memoryGaugeLimit" unit="bytes" /></div>
              <div class="w-24"><Sparkline :data="memoryHistory" /></div>
            </div>
            <ResourceGauge label="Disk" :used="diskUsed" :total="diskGaugeLimit" unit="bytes" />
            <div class="text-xs text-text-muted">Peak memory {{ formatBytes(memoryPeak) }}</div>
          </div>
          <div v-else-if="usageError" class="text-sm text-status-danger">{{ usageError }}</div>
          <div v-else class="text-sm text-text-muted">No resource measurements recorded for this attempt.</div>
        </InfoCard>

        <InfoCard title="Task Status Detail">
          <MarkdownRenderer v-if="detailedStatus" :content="detailedStatus" />
          <div v-else class="text-sm text-text-muted">No detailed status reported.</div>
        </InfoCard>
      </div>

      <InfoCard v-if="endpoints.length || endpointError" title="Endpoints">
        <div v-if="endpointError" class="text-sm text-status-danger">{{ endpointError }}</div>
        <div v-else class="flex flex-wrap gap-3">
          <EndpointLink v-for="endpoint in endpoints" :key="endpoint.endpointId" :name="endpoint.name" />
        </div>
      </InfoCard>

      <section>
        <h3 class="mb-2 text-sm font-semibold uppercase tracking-wider text-text-secondary">Attempts</h3>
        <EmptyState v-if="(task.attempts ?? []).length === 0" message="No attempts" />
        <div v-else class="overflow-x-auto rounded border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead>
              <tr class="border-b border-surface-border text-left text-xs uppercase text-text-secondary">
                <th class="px-3 py-2">Attempt</th>
                <th class="px-3 py-2">State</th>
                <th class="px-3 py-2">Node</th>
                <th class="px-3 py-2">Started</th>
                <th class="px-3 py-2">Finished</th>
                <th class="px-3 py-2">Exit</th>
                <th class="px-3 py-2">Reason</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="attempt in task.attempts"
                :key="attempt.identity.attemptUid"
                class="cursor-pointer border-b border-surface-border-subtle hover:bg-surface-raised"
                :class="attempt.identity.attemptNumber === attemptNumber ? 'bg-accent-subtle' : ''"
                @click="selectAttempt(attempt.identity.attemptNumber)"
              >
                <td class="px-3 py-2 font-mono">{{ attempt.identity.attemptNumber }}</td>
                <td class="px-3 py-2"><StatusBadge :status="attempt.state" size="sm" /></td>
                <td class="px-3 py-2 font-mono">{{ attempt.node?.key.resourceId || '—' }}</td>
                <td class="px-3 py-2 font-mono">{{ formatTimestamp(attempt.startedAt) }}</td>
                <td class="px-3 py-2 font-mono">{{ formatTimestamp(attempt.finishedAt) }}</td>
                <td class="px-3 py-2 font-mono">{{ attempt.exitCode ?? '—' }}</td>
                <td class="max-w-md px-3 py-2 text-xs text-status-danger">{{ attempt.terminalReason || attempt.errorMessage || '—' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section v-if="selected" class="rounded border border-surface-border p-4 text-sm">
        <div class="mb-3 flex items-center gap-3">
          <h3 class="font-semibold">Attempt {{ selected.summary.identity.attemptNumber }}</h3>
          <StatusBadge :status="selected.summary.state" size="sm" />
          <span class="ml-auto font-mono text-xs text-text-muted">{{ selected.summary.identity.attemptUid }}</span>
        </div>
        <dl class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div><dt class="text-xs text-text-muted">Duration</dt><dd class="font-mono">{{ attemptDuration }}</dd></div>
          <div><dt class="text-xs text-text-muted">Backend</dt><dd class="font-mono">{{ selected.summary.backendId || '—' }}</dd></div>
          <div><dt class="text-xs text-text-muted">Runtime</dt><dd class="font-mono">{{ selected.runtime?.providerKind || '—' }}</dd></div>
          <div><dt class="text-xs text-text-muted">Runtime object</dt><dd class="font-mono break-all">{{ selected.runtime?.name || selected.runtime?.containerId || '—' }}</dd></div>
        </dl>
        <div v-if="attemptError" class="mt-3 text-status-danger">{{ attemptError }}</div>
        <SourceWarnings :statuses="selected.sourceStatuses" />
        <div class="mt-4">
          <ProfileButtons :profiling="profiling" @profile="profile" />
        </div>
      </section>

      <ActivityTimeline :target="key" :attempt-uid="selected?.summary.identity.attemptUid" />

      <section>
        <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Logs</h3>
        <LogViewer :task-id="taskId" :cluster="logCluster" :authority-cluster="clusterId" />
      </section>
      <ProfileLink :source="taskId" />
    </div>
  </PageShell>
</template>
