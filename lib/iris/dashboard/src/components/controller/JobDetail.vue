<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { resourceRpcCall, useResourceRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { loadJobTasks } from '@/components/controller/jobTaskPages'
import type {
  Constraint,
  ResourceActionResponse,
  ResourceDescribeJobResponse,
  ResourceJobSummary,
  ResourceListJobsResponse,
  ResourceListTasksResponse,
  ResourceTaskSummary,
} from '@/types/rpc'
import {
  formatBytes,
  formatCpuMillicores,
  formatDeviceConfig,
  formatDuration,
  formatTimestamp,
  timestampMs,
} from '@/utils/formatting'
import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import ConstraintChip from '@/components/shared/ConstraintChip.vue'
import LogViewer from '@/components/shared/LogViewer.vue'

const TASK_PAGE_SIZE = 50
const JOB_REFRESH_MS = 10_000
const TERMINAL_STATES = new Set([
  'succeeded',
  'failed',
  'killed',
  'worker_failed',
  'unschedulable',
  'preempted',
  'cosched_failed',
])

const props = defineProps<{ clusterId: string; jobId: string }>()
const key = computed(() => ({
  clusterId: props.clusterId,
  kind: 'RESOURCE_KIND_JOB',
  resourceId: props.jobId,
}))
const { data, loading, error, refresh } = useResourceRpc<ResourceDescribeJobResponse>(
  'DescribeJob',
  () => ({ job: key.value }),
)
const job = computed(() => data.value?.job)
const tasks = ref<ResourceTaskSummary[]>([])
const childJobs = ref<ResourceJobSummary[]>([])
const taskLoading = ref(false)
const taskError = ref<string | null>(null)
const childError = ref<string | null>(null)
const action = ref<ResourceActionResponse | null>(null)
const actionError = ref<string | null>(null)
const acting = ref(false)
const taskSearch = ref('')
const stateFilter = ref('')
const taskPage = ref(0)

const summary = computed(() => job.value?.summary)
const spec = computed(() => job.value?.spec)
const backTo = computed(() => summary.value?.ownerId
  ? { path: '/', query: { user: summary.value.ownerId } }
  : '/')
const logCluster = computed(() => {
  const executionCluster = summary.value?.executionClusterId
  return executionCluster && executionCluster !== props.clusterId ? executionCluster : undefined
})

function stateName(state: string): string {
  return state.toLowerCase().replace(/^job_state_/, '').replace(/^task_state_/, '')
}

const isTerminal = computed(() => TERMINAL_STATES.has(stateName(summary.value?.state ?? '')))
const duration = computed(() => {
  const started = timestampMs(summary.value?.startedAt)
  const finished = timestampMs(summary.value?.finishedAt)
  return formatDuration(started, finished || undefined)
})
const taskCounts = computed(() => {
  const counts: Record<string, number> = {}
  for (const task of tasks.value) {
    const state = stateName(task.state)
    counts[state] = (counts[state] ?? 0) + 1
  }
  return counts
})
const totalFailures = computed(() => tasks.value.reduce((count, task) => count + task.failureCount, 0))
const totalPreemptions = computed(() => tasks.value.reduce((count, task) => count + task.preemptionCount, 0))
const taskStates = computed(() => [...new Set(tasks.value.map(task => stateName(task.state)))].sort())
const filteredTasks = computed(() => {
  const needle = taskSearch.value.trim().toLowerCase()
  return tasks.value.filter(task => {
    const matchesState = !stateFilter.value || stateName(task.state) === stateFilter.value
    const text = `${task.identity.key.resourceId} ${task.statusMessage ?? ''} ${task.errorMessage ?? ''}`.toLowerCase()
    return matchesState && (!needle || text.includes(needle))
  })
})
const totalTaskPages = computed(() => Math.max(1, Math.ceil(filteredTasks.value.length / TASK_PAGE_SIZE)))
const visibleTasks = computed(() => {
  const start = taskPage.value * TASK_PAGE_SIZE
  return filteredTasks.value.slice(start, start + TASK_PAGE_SIZE)
})

watch([taskSearch, stateFilter], () => { taskPage.value = 0 })
watch(totalTaskPages, pages => {
  if (taskPage.value >= pages) taskPage.value = pages - 1
})

function attributeValue(value?: { stringValue?: string; intValue?: string; floatValue?: string }): string {
  return value?.stringValue ?? value?.intValue ?? value?.floatValue ?? ''
}

function constraintText(constraint: Constraint): string {
  const values = constraint.values?.map(attributeValue) ?? []
  const operand = values.length ? values.join(', ') : attributeValue(constraint.value)
  return [constraint.key, constraint.op.replace('CONSTRAINT_OP_', '').toLowerCase(), operand]
    .filter(Boolean)
    .join(' ')
}

function taskRoute(task: ResourceTaskSummary) {
  return {
    name: 'task-detail',
    params: {
      clusterId: task.identity.key.clusterId,
      taskId: task.identity.key.resourceId,
    },
  }
}

function jobRoute(item: ResourceJobSummary) {
  return {
    name: 'job-detail',
    params: {
      clusterId: item.identity.key.clusterId,
      jobId: item.identity.key.resourceId,
    },
  }
}

function nodeRoute(task: ResourceTaskSummary) {
  const node = task.currentNode
  if (!node) return undefined
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

function taskDuration(task: ResourceTaskSummary): string {
  return formatDuration(timestampMs(task.startedAt), timestampMs(task.finishedAt) || undefined)
}

async function loadChildJobs(): Promise<ResourceJobSummary[]> {
  const items: ResourceJobSummary[] = []
  let pageToken: string | undefined
  do {
    const response = await resourceRpcCall<ResourceListJobsResponse>('ListJobs', {
      query: { jobIdPrefix: `${props.jobId}/`, page: { pageSize: 100, pageToken } },
    })
    items.push(...(response.jobs ?? []))
    pageToken = response.page?.nextPageToken || undefined
  } while (pageToken)
  return items
}

async function refreshPage() {
  await refresh()
  if (!job.value) return
  taskLoading.value = true
  taskError.value = null
  childError.value = null
  const taskRequest = loadJobTasks(job.value.summary.numTasks, pageToken =>
    resourceRpcCall<ResourceListTasksResponse>('ListTasks', {
      query: { job: key.value, page: { pageSize: 100, pageToken } },
    }),
  )
  const [taskResult, childResult] = await Promise.allSettled([taskRequest, loadChildJobs()])
  if (taskResult.status === 'fulfilled') tasks.value = taskResult.value
  else taskError.value = taskResult.reason instanceof Error ? taskResult.reason.message : String(taskResult.reason)
  if (childResult.status === 'fulfilled') childJobs.value = childResult.value
  else childError.value = childResult.reason instanceof Error ? childResult.reason.message : String(childResult.reason)
  taskLoading.value = false
}

async function cancelJob() {
  if (!job.value || acting.value) return
  acting.value = true
  actionError.value = null
  try {
    action.value = await resourceRpcCall<ResourceActionResponse>('CancelJob', {
      job: job.value.summary.identity,
      idempotencyKey: crypto.randomUUID(),
    })
    await refreshPage()
  } catch (cause) {
    actionError.value = cause instanceof Error ? cause.message : String(cause)
  } finally {
    acting.value = false
  }
}

onMounted(refreshPage)
useAutoRefresh(refreshPage, JOB_REFRESH_MS)
</script>

<template>
  <PageShell :title="spec?.name || jobId" :back-to="backTo" back-label="Jobs">
    <p v-if="spec?.name && spec.name !== jobId" class="-mt-4 mb-6 font-mono text-sm text-text-secondary">
      {{ jobId }}
    </p>

    <div v-if="error" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">
      {{ error }}
    </div>
    <div v-else-if="loading && !job" class="text-sm text-text-muted">Loading job…</div>
    <div v-else-if="job" class="space-y-6">
      <div
        v-if="summary?.errorMessage"
        class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border border-status-danger-border"
      >
        <span class="font-semibold">Error:</span> {{ summary.errorMessage }}
      </div>
      <div
        v-if="summary?.pendingReason"
        class="px-4 py-3 text-sm text-status-warning bg-status-warning-bg rounded border border-status-warning-border"
      >
        <div class="font-semibold">Scheduling diagnostic</div>
        <pre class="mt-2 whitespace-pre-wrap font-mono text-xs">{{ summary.pendingReason }}</pre>
      </div>

      <div class="flex flex-wrap items-center gap-3">
        <StatusBadge :status="summary!.state" />
        <span class="font-mono text-sm text-text-muted">
          {{ summary!.backendId || summary!.executionClusterId }}
        </span>
        <button
          class="ml-auto px-3 py-1.5 text-sm border border-status-danger-border text-status-danger rounded disabled:opacity-40"
          :disabled="acting || isTerminal"
          @click="cancelJob"
        >
          {{ acting ? 'Cancelling…' : 'Cancel job' }}
        </button>
      </div>
      <div v-if="action?.receipt" class="p-3 border rounded text-sm">
        Action <span class="font-mono">{{ action.receipt.actionId }}</span>: {{ action.receipt.state }}
      </div>
      <div v-if="actionError" class="text-sm text-status-danger">{{ actionError }}</div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <InfoCard title="Job Status">
          <InfoRow label="State"><StatusBadge :status="summary!.state" size="sm" /></InfoRow>
          <InfoRow label="Submitted"><span class="font-mono">{{ formatTimestamp(summary!.submittedAt) }}</span></InfoRow>
          <InfoRow label="Started"><span class="font-mono">{{ formatTimestamp(summary!.startedAt) }}</span></InfoRow>
          <InfoRow label="Finished"><span class="font-mono">{{ formatTimestamp(summary!.finishedAt) }}</span></InfoRow>
          <InfoRow label="Duration"><span class="font-mono">{{ duration }}</span></InfoRow>
          <InfoRow label="Failures">{{ totalFailures }} / max {{ spec?.maxTaskFailures ?? 0 }}</InfoRow>
          <InfoRow label="Preemptions">{{ totalPreemptions }}</InfoRow>
          <InfoRow label="Priority">{{ (spec?.priorityBand ?? '—').replace('PRIORITY_BAND_', '').toLowerCase() }}</InfoRow>
        </InfoCard>

        <InfoCard title="Task Summary">
          <InfoRow label="Total">{{ tasks.length || summary!.numTasks }}</InfoRow>
          <InfoRow label="Succeeded">{{ taskCounts.succeeded ?? 0 }}</InfoRow>
          <InfoRow label="Running">{{ taskCounts.running ?? 0 }}</InfoRow>
          <InfoRow label="Building">{{ taskCounts.building ?? 0 }}</InfoRow>
          <InfoRow label="Assigned">{{ taskCounts.assigned ?? 0 }}</InfoRow>
          <InfoRow label="Pending">{{ taskCounts.pending ?? 0 }}</InfoRow>
          <InfoRow label="Failed">{{ taskCounts.failed ?? 0 }}</InfoRow>
        </InfoCard>

        <InfoCard title="Resources (per task)">
          <InfoRow label="CPU">{{ formatCpuMillicores(spec?.resources?.cpuMillicores) }}</InfoRow>
          <InfoRow label="Memory">{{ formatBytes(Number(spec?.resources?.memoryBytes ?? 0)) }}</InfoRow>
          <InfoRow label="Disk">{{ formatBytes(Number(spec?.resources?.diskBytes ?? 0)) }}</InfoRow>
          <InfoRow label="Accelerator">{{ formatDeviceConfig(spec?.resources?.device) ?? 'CPU' }}</InfoRow>
          <InfoRow label="Replicas">{{ spec?.replicas ?? summary!.numTasks }}</InfoRow>
          <InfoRow label="Backend"><span class="font-mono">{{ summary!.backendId || '—' }}</span></InfoRow>
          <InfoRow label="Cluster"><span class="font-mono">{{ summary!.executionClusterId }}</span></InfoRow>
        </InfoCard>
      </div>

      <details
        v-if="(spec?.constraints ?? []).length"
        class="rounded-lg border border-surface-border bg-surface"
        :open="(taskCounts.pending ?? 0) > 0"
      >
        <summary class="cursor-pointer px-4 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary">
          Scheduling — {{ spec!.constraints!.length }} constraint{{ spec!.constraints!.length === 1 ? '' : 's' }}
        </summary>
        <div class="flex flex-wrap gap-2 border-t border-surface-border px-4 py-3">
          <ConstraintChip
            v-for="constraint in spec!.constraints"
            :key="constraintText(constraint)"
            :constraint="constraintText(constraint)"
          />
        </div>
      </details>

      <details class="rounded-lg border border-surface-border bg-surface">
        <summary class="cursor-pointer px-4 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary">
          Job Request — command, setup &amp; environment
        </summary>
        <div class="space-y-3 border-t border-surface-border px-4 py-3 text-sm">
          <div v-if="spec?.entrypoint?.runCommand?.argv?.length">
            <div class="text-xs text-text-muted">Command</div>
            <pre class="mt-1 whitespace-pre-wrap break-all rounded bg-surface-sunken px-2 py-1 font-mono text-xs">{{ spec.entrypoint.runCommand.argv.join(' ') }}</pre>
          </div>
          <div v-if="spec?.entrypoint?.setupCommands?.length">
            <div class="text-xs text-text-muted">Runtime setup</div>
            <pre class="mt-1 whitespace-pre-wrap break-all rounded bg-surface-sunken px-2 py-1 font-mono text-xs">{{ spec.entrypoint.setupCommands.join('\n') }}</pre>
          </div>
          <div v-if="spec?.environment?.setupScripts?.length">
            <div class="text-xs text-text-muted">Environment setup</div>
            <pre class="mt-1 whitespace-pre-wrap break-all rounded bg-surface-sunken px-2 py-1 font-mono text-xs">{{ spec.environment.setupScripts.join('\n') }}</pre>
          </div>
          <div v-if="Object.keys(spec?.environment?.envVars ?? {}).length">
            <div class="text-xs text-text-muted">Environment variables</div>
            <div class="mt-1 flex flex-wrap gap-1.5">
              <span
                v-for="(value, name) in spec!.environment!.envVars"
                :key="name"
                class="rounded bg-surface-sunken px-2 py-0.5 font-mono text-xs"
              >{{ name }}={{ value }}</span>
            </div>
          </div>
          <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div><div class="text-xs text-text-muted">Task image</div><span class="font-mono break-all">{{ spec?.taskImage || '—' }}</span></div>
            <div><div class="text-xs text-text-muted">Bundle</div><span class="font-mono">{{ spec?.bundleId || '—' }}</span></div>
            <div><div class="text-xs text-text-muted">Failure retries</div>{{ spec?.maxRetriesFailure ?? 0 }} / task</div>
            <div><div class="text-xs text-text-muted">Preemption retries</div>{{ spec?.maxRetriesPreemption ?? 0 }} / task</div>
          </div>
        </div>
      </details>

      <section v-if="childError || childJobs.length">
        <h3 class="mb-2 text-sm font-semibold uppercase tracking-wider text-text-secondary">Descendant Jobs</h3>
        <div v-if="childError" class="text-sm text-status-danger">{{ childError }}</div>
        <div v-else class="overflow-x-auto rounded border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead><tr class="border-b border-surface-border text-left text-xs uppercase text-text-secondary">
              <th class="px-3 py-2">Job</th><th class="px-3 py-2">State</th><th class="px-3 py-2 text-right">Tasks</th>
            </tr></thead>
            <tbody>
              <tr v-for="child in childJobs" :key="child.identity.jobUid" class="border-b border-surface-border-subtle">
                <td class="px-3 py-2">
                  <RouterLink
                    :to="jobRoute(child)"
                    class="font-mono text-accent hover:underline"
                    :style="{ paddingLeft: `${Math.max(0, child.identity.key.resourceId.split('/').length - props.jobId.split('/').length - 1) * 1.25}rem` }"
                  >{{ child.identity.key.resourceId }}</RouterLink>
                </td>
                <td class="px-3 py-2"><StatusBadge :status="child.state" size="sm" /></td>
                <td class="px-3 py-2 text-right">{{ child.numTasks }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <div class="mb-3 flex flex-wrap items-center gap-2">
          <h3 class="mr-auto text-sm font-semibold uppercase tracking-wider text-text-secondary">Tasks</h3>
          <input
            v-model="taskSearch"
            class="w-56 rounded border border-surface-border bg-surface px-3 py-1.5 text-sm"
            placeholder="Filter tasks…"
          />
          <select v-model="stateFilter" class="rounded border border-surface-border bg-surface px-3 py-1.5 text-sm">
            <option value="">All states</option>
            <option v-for="state in taskStates" :key="state" :value="state">{{ state }}</option>
          </select>
        </div>
        <div v-if="taskError" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">{{ taskError }}</div>
        <div v-else-if="taskLoading && tasks.length === 0" class="text-sm text-text-muted">Loading tasks…</div>
        <EmptyState v-else-if="filteredTasks.length === 0" message="No matching tasks" />
        <div v-else class="overflow-x-auto rounded border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead>
              <tr class="border-b border-surface-border text-left text-xs uppercase text-text-secondary">
                <th class="px-3 py-2">Task</th>
                <th class="px-3 py-2">State</th>
                <th class="px-3 py-2">Attempt</th>
                <th class="px-3 py-2">Node</th>
                <th class="px-3 py-2">Started</th>
                <th class="px-3 py-2">Duration</th>
                <th class="px-3 py-2 text-right">Failures</th>
                <th class="px-3 py-2 text-right">Preemptions</th>
              </tr>
            </thead>
            <tbody>
              <template v-for="task in visibleTasks" :key="task.identity.taskUid">
                <tr class="border-b border-surface-border-subtle hover:bg-surface-raised">
                  <td class="px-3 py-2"><RouterLink :to="taskRoute(task)" class="font-mono text-accent hover:underline">{{ task.identity.key.resourceId }}</RouterLink></td>
                  <td class="px-3 py-2"><StatusBadge :status="task.state" size="sm" /></td>
                  <td class="px-3 py-2 font-mono">{{ task.currentAttempt?.attemptNumber ?? '—' }}</td>
                  <td class="px-3 py-2">
                    <RouterLink v-if="task.currentNode" :to="nodeRoute(task)!" class="font-mono text-accent hover:underline">{{ task.currentNode.key.resourceId }}</RouterLink>
                    <span v-else>—</span>
                  </td>
                  <td class="px-3 py-2 font-mono">{{ formatTimestamp(task.startedAt) }}</td>
                  <td class="px-3 py-2 font-mono">{{ taskDuration(task) }}</td>
                  <td class="px-3 py-2 text-right">{{ task.failureCount }}</td>
                  <td class="px-3 py-2 text-right">{{ task.preemptionCount }}</td>
                </tr>
                <tr v-if="task.statusMessage || task.errorMessage" class="border-b border-surface-border-subtle bg-surface-sunken">
                  <td colspan="8" class="px-3 py-2 text-xs" :class="task.errorMessage ? 'text-status-danger' : 'text-text-secondary'">
                    {{ task.errorMessage || task.statusMessage }}
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
          <div v-if="totalTaskPages > 1" class="flex items-center justify-between border-t border-surface-border px-3 py-2 text-xs text-text-secondary">
            <span>{{ taskPage * TASK_PAGE_SIZE + 1 }}–{{ Math.min((taskPage + 1) * TASK_PAGE_SIZE, filteredTasks.length) }} of {{ filteredTasks.length }}</span>
            <div class="flex items-center gap-1">
              <button :disabled="taskPage === 0" class="rounded px-2 py-1 hover:bg-surface-raised disabled:opacity-30" @click="taskPage--">← Prev</button>
              <span class="px-2 font-mono">{{ taskPage + 1 }} / {{ totalTaskPages }}</span>
              <button :disabled="taskPage >= totalTaskPages - 1" class="rounded px-2 py-1 hover:bg-surface-raised disabled:opacity-30" @click="taskPage++">Next →</button>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Job Logs</h3>
        <LogViewer :task-id="jobId" :cluster="logCluster" :authority-cluster="clusterId" />
      </section>
    </div>
  </PageShell>
</template>
