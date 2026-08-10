<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { resourceRpcCall, useResourceRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type {
  ResourceActionResponse,
  ResourceDescribeAttemptResponse,
  ResourceDescribeTaskResponse,
  ResourceEndpointSummary,
  ResourceListEndpointsResponse,
} from '@/types/rpc'
import { formatDuration, formatTimestamp, timestampMs } from '@/utils/formatting'
import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import LogViewer from '@/components/shared/LogViewer.vue'
import EndpointLink from '@/components/shared/EndpointLink.vue'
import ProfileLink from '@/components/shared/ProfileLink.vue'
import SourceWarnings from '@/components/shared/SourceWarnings.vue'

const TASK_REFRESH_MS = 10_000

const props = defineProps<{ clusterId: string; taskId: string }>()
const key = computed(() => ({
  clusterId: props.clusterId,
  kind: 'RESOURCE_KIND_TASK',
  resourceId: props.taskId,
}))
const { data, loading, error, refresh } = useResourceRpc<ResourceDescribeTaskResponse>(
  'DescribeTask',
  () => ({ task: key.value }),
)
const task = computed(() => data.value?.task)
const selectedAttempt = ref<number | undefined>()
const attemptNumber = computed(() => selectedAttempt.value ?? task.value?.summary.currentAttempt?.attemptNumber)
const { data: attemptData, error: attemptError, refresh: refreshAttempt } =
  useResourceRpc<ResourceDescribeAttemptResponse>('DescribeAttempt', () => ({
    attempt: { task: key.value, attemptNumber: attemptNumber.value },
  }))
const { data: endpointData, error: endpointError, refresh: refreshEndpoints } =
  useResourceRpc<ResourceListEndpointsResponse>('ListEndpoints', () => ({
    query: { task: key.value, page: { pageSize: 100 } },
  }))
const action = ref<ResourceActionResponse | null>(null)
const actionError = ref<string | null>(null)
const acting = ref(false)

const summary = computed(() => task.value?.summary)
const selected = computed(() => attemptData.value?.attempt)
const endpoints = computed<ResourceEndpointSummary[]>(() => endpointData.value?.endpoints ?? [])
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
  await refreshEndpoints()
  if (attemptNumber.value !== undefined) await refreshAttempt()
}

async function retryTask() {
  const current = summary.value?.currentAttempt
  if (!current || acting.value) return
  acting.value = true
  actionError.value = null
  try {
    action.value = await resourceRpcCall<ResourceActionResponse>('RetryTask', {
      task: summary.value!.identity,
      expectedAttemptUid: current.attemptUid,
      idempotencyKey: crypto.randomUUID(),
    })
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
    action.value = await resourceRpcCall<ResourceActionResponse>('TerminateAttempt', {
      attempt,
      idempotencyKey: crypto.randomUUID(),
    })
    await refreshPage()
  } catch (cause) {
    actionError.value = cause instanceof Error ? cause.message : String(cause)
  } finally {
    acting.value = false
  }
}

async function selectAttempt(number: number) {
  selectedAttempt.value = number
  await refreshAttempt()
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
      </section>

      <section>
        <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Logs</h3>
        <LogViewer :task-id="taskId" :cluster="logCluster" />
      </section>
      <ProfileLink :source="taskId" />
    </div>
  </PageShell>
</template>
