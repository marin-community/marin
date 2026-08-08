<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { useResourceRpc } from '@/composables/useRpc'
import type { ResourceDescribeNodeResponse } from '@/types/rpc'
import { formatBytes, formatRelativeTime, timestampMs } from '@/utils/formatting'
import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import SourceWarnings from '@/components/shared/SourceWarnings.vue'

const props = defineProps<{ clusterId: string; backendId: string; nodeUid: string; nodeId: string }>()
const { data, loading, error, refresh } = useResourceRpc<ResourceDescribeNodeResponse>('DescribeNode', () => ({
  node: {
    key: { clusterId: props.clusterId, kind: 'RESOURCE_KIND_NODE', resourceId: props.nodeId },
    backendId: props.backendId,
    nodeUid: props.nodeUid,
  },
}))
const node = computed(() => data.value?.node)
function attributeValue(value: { stringValue?: string; integerValue?: string; floatValue?: number }): string {
  return value.stringValue ?? value.integerValue ?? String(value.floatValue ?? '')
}
onMounted(refresh)
</script>

<template>
  <PageShell :title="nodeId" back-to="/nodes" back-label="Nodes">
    <div v-if="error" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">{{ error }}</div>
    <div v-else-if="loading && !node" class="text-sm text-text-muted">Loading node…</div>
    <div v-else-if="node" class="space-y-6">
      <SourceWarnings :statuses="node.sourceStatuses" />
      <div class="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Health</div><StatusBadge :status="node.summary.health" size="sm" /></div>
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Backend</div><div class="font-mono">{{ node.summary.identity.backendId }}</div></div>
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Address</div><div class="font-mono">{{ node.address || '—' }}</div></div>
        <div class="p-3 border rounded"><div class="text-xs text-text-muted">Observed</div><div>{{ formatRelativeTime(timestampMs(node.summary.observedAt)) }}</div></div>
      </div>
      <div class="p-4 border rounded space-y-2">
        <h3 class="font-semibold">Capacity</h3>
        <div class="text-sm">CPU {{ node.summary.capacity?.cpuMillicores ?? '0' }}m · Memory {{ formatBytes(Number(node.summary.capacity?.memoryBytes ?? 0)) }} · {{ node.summary.capacity?.acceleratorCount ?? 0 }}× {{ node.summary.capacity?.acceleratorVariant || node.summary.capacity?.acceleratorKind || 'accelerator' }}</div>
      </div>
      <div class="p-4 border rounded">
        <h3 class="font-semibold mb-2">Attributes</h3>
        <dl class="grid sm:grid-cols-2 gap-2 text-sm">
          <template v-for="attribute in node.attributes ?? []" :key="attribute.key">
            <dt class="font-mono text-text-muted">{{ attribute.key }}</dt><dd>{{ attributeValue(attribute) }}</dd>
          </template>
        </dl>
      </div>
      <div>
        <h3 class="font-semibold mb-2">Recent attempts</h3>
        <EmptyState v-if="(node.recentAttempts ?? []).length === 0" message="No recent attempts" />
        <div v-else class="divide-y border rounded">
          <RouterLink v-for="attempt in node.recentAttempts" :key="attempt.identity.attemptUid"
            :to="{ name: 'task-detail', params: { clusterId: attempt.identity.task.clusterId, taskId: attempt.identity.task.resourceId } }"
            class="flex justify-between p-3 hover:bg-surface-raised">
            <span class="font-mono">{{ attempt.identity.task.resourceId }} #{{ attempt.identity.attemptNumber }}</span>
            <StatusBadge :status="attempt.state" size="sm" />
          </RouterLink>
        </div>
      </div>
    </div>
  </PageShell>
</template>
