<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { resourceRpcCall, useResourceRpc } from '@/composables/useRpc'
import { loadJobTasks } from '@/components/controller/jobTaskPages'
import type {
  ResourceActionResponse,
  Constraint,
  ResourceDescribeJobResponse,
  ResourceListTasksResponse,
  ResourceTaskSummary,
} from '@/types/rpc'
import { formatBytes, formatRelativeTime, timestampMs } from '@/utils/formatting'
import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import ActivityTimeline from '@/components/shared/ActivityTimeline.vue'
import ConstraintChip from '@/components/shared/ConstraintChip.vue'

const props = defineProps<{ clusterId: string; jobId: string }>()
const key = computed(() => ({ clusterId: props.clusterId, kind: 'RESOURCE_KIND_JOB', resourceId: props.jobId }))
const { data, loading, error, refresh } = useResourceRpc<ResourceDescribeJobResponse>('DescribeJob', () => ({ job: key.value }))
const job = computed(() => data.value?.job)
const tasks = ref<ResourceTaskSummary[]>([])
const taskLoading = ref(false)
const taskError = ref<string | null>(null)
const action = ref<ResourceActionResponse | null>(null)
const actionError = ref<string | null>(null)
const acting = ref(false)

function attributeValue(value?: { stringValue?: string; intValue?: string; floatValue?: string }): string {
  return value?.stringValue ?? value?.intValue ?? value?.floatValue ?? ''
}

function constraintText(constraint: Constraint): string {
  const values = constraint.values?.map(attributeValue) ?? []
  const operand = values.length ? values.join(', ') : attributeValue(constraint.value)
  return [constraint.key, constraint.op.replace('CONSTRAINT_OP_', '').toLowerCase(), operand].filter(Boolean).join(' ')
}

function taskRoute(task: ResourceTaskSummary) {
  return { name: 'task-detail', params: { clusterId: task.identity.key.clusterId, taskId: task.identity.key.resourceId } }
}

async function refreshPage() {
  taskLoading.value = true
  taskError.value = null
  await refresh()
  if (!job.value) {
    taskLoading.value = false
    return
  }
  try {
    tasks.value = await loadJobTasks(job.value.summary.numTasks, pageToken =>
      resourceRpcCall<ResourceListTasksResponse>('ListTasks', {
        query: { job: key.value, page: { pageSize: 100, pageToken } },
      }),
    )
  } catch (e) {
    taskError.value = e instanceof Error ? e.message : String(e)
  } finally {
    taskLoading.value = false
  }
}

async function cancelJob() {
  if (!job.value || acting.value) return
  acting.value = true; actionError.value = null
  try {
    action.value = await resourceRpcCall<ResourceActionResponse>('CancelJob', {
      job: job.value.summary.identity,
      idempotencyKey: crypto.randomUUID(),
    })
  } catch (e) { actionError.value = e instanceof Error ? e.message : String(e) }
  finally { acting.value = false }
}
onMounted(refreshPage)
</script>

<template>
  <PageShell :title="jobId" back-to="/" back-label="Jobs">
    <div v-if="error" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">{{ error }}</div>
    <div v-else-if="loading && !job" class="text-sm text-text-muted">Loading job…</div>
    <div v-else-if="job" class="space-y-6">
      <div class="flex flex-wrap items-center gap-3">
        <StatusBadge :status="job.summary.state" />
        <span class="text-sm text-text-muted">{{ job.summary.ownerId }} · {{ job.summary.backendId || job.summary.executionClusterId }}</span>
        <button class="ml-auto px-3 py-1.5 text-sm border border-status-danger-border text-status-danger rounded disabled:opacity-40" :disabled="acting" @click="cancelJob">
          Cancel job
        </button>
      </div>
      <div v-if="action?.receipt" class="p-3 border rounded text-sm">Action <span class="font-mono">{{ action.receipt.actionId }}</span>: {{ action.receipt.state }}</div>
      <div v-if="actionError" class="text-sm text-status-danger">{{ actionError }}</div>
      <div v-if="job.summary.pendingReason" class="px-4 py-3 text-sm text-status-warning bg-status-warning-bg rounded border">
        {{ job.summary.pendingReason }}
      </div>
      <div class="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Tasks</div>{{ job.summary.numTasks }}</div>
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Submitted</div>{{ formatRelativeTime(timestampMs(job.summary.submittedAt)) }}</div>
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Backend</div><span class="font-mono">{{ job.summary.backendId || '—' }}</span></div>
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Bundle</div><span class="font-mono">{{ job.spec.bundleId || '—' }}</span></div>
      </div>
      <section class="p-4 border rounded space-y-3">
        <h3 class="font-semibold">Specification</h3>
        <div class="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
          <div><div class="text-xs text-text-muted">Name</div>{{ job.spec.name || '—' }}</div>
          <div><div class="text-xs text-text-muted">Replicas</div>{{ job.spec.replicas ?? job.summary.numTasks }}</div>
          <div><div class="text-xs text-text-muted">CPU</div>{{ job.spec.resources?.cpuMillicores ?? 0 }}m</div>
          <div><div class="text-xs text-text-muted">Memory</div>{{ formatBytes(Number(job.spec.resources?.memoryBytes ?? 0)) }}</div>
        </div>
        <div v-if="(job.spec.constraints ?? []).length" class="flex flex-wrap gap-2">
          <ConstraintChip v-for="constraint in job.spec.constraints" :key="constraintText(constraint)" :constraint="constraintText(constraint)" />
        </div>
      </section>
      <section>
        <h3 class="font-semibold mb-2">Tasks</h3>
        <div v-if="taskError" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">{{ taskError }}</div>
        <div v-else-if="taskLoading" class="text-sm text-text-muted">Loading tasks…</div>
        <EmptyState v-else-if="tasks.length === 0" message="No tasks" />
        <div v-else class="border rounded divide-y">
          <RouterLink v-for="task in tasks" :key="task.identity.taskUid" :to="taskRoute(task)" class="flex items-center justify-between p-3 hover:bg-surface-raised">
            <span class="font-mono">{{ task.identity.key.resourceId }}</span><StatusBadge :status="task.state" size="sm" />
          </RouterLink>
        </div>
      </section>
      <ActivityTimeline :target="key" />
    </div>
  </PageShell>
</template>
