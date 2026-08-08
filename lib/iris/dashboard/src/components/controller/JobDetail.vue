<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { resourceRpcCall, useResourceRpc } from '@/composables/useRpc'
import type {
  ResourceActionResponse,
  ResourceDescribeJobResponse,
  ResourceListTasksResponse,
  ResourceTaskSummary,
} from '@/types/rpc'
import { formatRelativeTime, timestampMs } from '@/utils/formatting'
import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import ActivityTimeline from '@/components/shared/ActivityTimeline.vue'

const props = defineProps<{ clusterId: string; jobId: string }>()
const key = computed(() => ({ clusterId: props.clusterId, kind: 'RESOURCE_KIND_JOB', resourceId: props.jobId }))
const { data, loading, error, refresh } = useResourceRpc<ResourceDescribeJobResponse>('DescribeJob', () => ({ job: key.value }))
const { data: taskData, refresh: refreshTasks } = useResourceRpc<ResourceListTasksResponse>('ListTasks', () => ({
  query: { job: key.value, page: { pageSize: 100 } },
}))
const job = computed(() => data.value?.job)
const tasks = computed(() => taskData.value?.tasks ?? [])
const action = ref<ResourceActionResponse | null>(null)
const actionError = ref<string | null>(null)
const acting = ref(false)

function taskRoute(task: ResourceTaskSummary) {
  return { name: 'task-detail', params: { clusterId: task.identity.key.clusterId, taskId: task.identity.key.resourceId } }
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
onMounted(() => Promise.all([refresh(), refreshTasks()]))
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
      <div class="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Tasks</div>{{ job.summary.numTasks }}</div>
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Submitted</div>{{ formatRelativeTime(timestampMs(job.summary.submittedAt)) }}</div>
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Backend</div><span class="font-mono">{{ job.summary.backendId || '—' }}</span></div>
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Bundle</div><span class="font-mono">{{ job.spec.bundleId || '—' }}</span></div>
      </div>
      <section>
        <h3 class="font-semibold mb-2">Tasks</h3>
        <EmptyState v-if="tasks.length === 0" message="No tasks" />
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
