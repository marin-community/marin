<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { resourceRpcCall, useResourceRpc } from '@/composables/useRpc'
import type { ResourceActionResponse, ResourceDescribeAttemptResponse, ResourceDescribeTaskResponse } from '@/types/rpc'
import { formatRelativeTime, timestampMs } from '@/utils/formatting'
import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import ActivityTimeline from '@/components/shared/ActivityTimeline.vue'
import SourceWarnings from '@/components/shared/SourceWarnings.vue'

const props = defineProps<{ clusterId: string; taskId: string }>()
const key = computed(() => ({ clusterId: props.clusterId, kind: 'RESOURCE_KIND_TASK', resourceId: props.taskId }))
const { data, loading, error, refresh } = useResourceRpc<ResourceDescribeTaskResponse>('DescribeTask', () => ({ task: key.value }))
const task = computed(() => data.value?.task)
const selectedAttempt = ref<number | undefined>()
const attemptNumber = computed(() => selectedAttempt.value ?? task.value?.summary.currentAttempt?.attemptNumber)
const { data: attemptData, refresh: refreshAttempt } = useResourceRpc<ResourceDescribeAttemptResponse>('DescribeAttempt', () => ({
  attempt: { task: key.value, attemptNumber: attemptNumber.value },
}))
const action = ref<ResourceActionResponse | null>(null)
const actionError = ref<string | null>(null)

async function retryTask() {
  const summary = task.value?.summary
  if (!summary?.currentAttempt) return
  try {
    action.value = await resourceRpcCall<ResourceActionResponse>('RetryTask', {
      task: summary.identity,
      expectedAttemptUid: summary.currentAttempt.attemptUid,
      idempotencyKey: crypto.randomUUID(),
    })
  } catch (e) { actionError.value = e instanceof Error ? e.message : String(e) }
}
async function terminateAttempt() {
  const attempt = attemptData.value?.attempt?.summary.identity
  if (!attempt) return
  try {
    action.value = await resourceRpcCall<ResourceActionResponse>('TerminateAttempt', {
      attempt,
      idempotencyKey: crypto.randomUUID(),
    })
  } catch (e) { actionError.value = e instanceof Error ? e.message : String(e) }
}
async function selectAttempt(number: number) { selectedAttempt.value = number; await refreshAttempt() }
onMounted(async () => { await refresh(); if (attemptNumber.value !== undefined) await refreshAttempt() })
</script>

<template>
  <PageShell :title="taskId" :back-to="task ? { name: 'job-detail', params: { clusterId: task.summary.job.key.clusterId, jobId: task.summary.job.key.resourceId } } : '/tasks'" back-label="Job">
    <div v-if="error" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">{{ error }}</div>
    <div v-else-if="loading && !task" class="text-sm text-text-muted">Loading task…</div>
    <div v-else-if="task" class="space-y-6">
      <SourceWarnings :statuses="task.sourceStatuses" />
      <div class="flex flex-wrap items-center gap-3">
        <StatusBadge :status="task.summary.state" />
        <span class="text-sm text-text-muted">{{ task.summary.backendId || task.summary.executionClusterId }}</span>
        <button class="ml-auto px-3 py-1.5 text-sm border rounded" :disabled="!task.summary.currentAttempt" @click="retryTask">Retry current attempt</button>
        <button class="px-3 py-1.5 text-sm border border-status-danger-border text-status-danger rounded" :disabled="!attemptData?.attempt" @click="terminateAttempt">Terminate attempt</button>
      </div>
      <div v-if="action?.receipt" class="p-3 border rounded text-sm">Action <span class="font-mono">{{ action.receipt.actionId }}</span>: {{ action.receipt.state }}</div>
      <div v-if="actionError" class="text-sm text-status-danger">{{ actionError }}</div>
      <div class="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Submitted</div>{{ formatRelativeTime(timestampMs(task.summary.submittedAt)) }}</div>
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Failures</div>{{ task.summary.failureCount }}</div>
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Preemptions</div>{{ task.summary.preemptionCount }}</div>
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Node</div>
          <RouterLink v-if="task.summary.currentNode" class="font-mono text-accent hover:underline" :to="{ name: 'node-detail', params: { clusterId: task.summary.currentNode.key.clusterId, backendId: task.summary.currentNode.backendId, nodeUid: task.summary.currentNode.nodeUid, nodeId: task.summary.currentNode.key.resourceId } }">{{ task.summary.currentNode.key.resourceId }}</RouterLink><span v-else>—</span>
        </div>
      </div>
      <div v-if="task.summary.statusMessage || task.summary.errorMessage" class="p-3 border rounded text-sm">{{ task.summary.statusMessage || task.summary.errorMessage }}</div>
      <section>
        <h3 class="font-semibold mb-2">Attempts</h3>
        <EmptyState v-if="(task.attempts ?? []).length === 0" message="No attempts" />
        <div v-else class="border rounded divide-y">
          <button v-for="attempt in task.attempts" :key="attempt.identity.attemptUid" class="w-full flex items-center justify-between p-3 hover:bg-surface-raised" @click="selectAttempt(attempt.identity.attemptNumber)">
            <span class="font-mono">Attempt {{ attempt.identity.attemptNumber }}</span><StatusBadge :status="attempt.state" size="sm" />
          </button>
        </div>
      </section>
      <section v-if="attemptData?.attempt" class="p-4 border rounded space-y-2 text-sm">
        <h3 class="font-semibold">Attempt {{ attemptData.attempt.summary.identity.attemptNumber }}</h3>
        <div>UID <span class="font-mono">{{ attemptData.attempt.summary.identity.attemptUid }}</span></div>
        <div v-if="attemptData.attempt.runtime">Runtime <span class="font-mono">{{ attemptData.attempt.runtime.providerKind }} / {{ attemptData.attempt.runtime.name || attemptData.attempt.runtime.containerId }}</span></div>
        <div v-if="attemptData.attempt.summary.terminalReason" class="text-status-danger">{{ attemptData.attempt.summary.terminalReason }}</div>
      </section>
      <ActivityTimeline :target="key" :attempt-uid="attemptData?.attempt?.summary.identity.attemptUid" />
    </div>
  </PageShell>
</template>
