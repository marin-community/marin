<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { useLogServerStatsRpc, useResourceRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import type { ResourceDescribeNodeResponse } from '@/types/rpc'
import { formatBytes, formatTimestamp } from '@/utils/formatting'
import { decodeArrowIpc } from '@/utils/arrow'
import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import SourceWarnings from '@/components/shared/SourceWarnings.vue'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import LogViewer from '@/components/shared/LogViewer.vue'
import ResourceGauge from '@/components/shared/ResourceGauge.vue'
import Sparkline from '@/components/shared/Sparkline.vue'

const NODE_REFRESH_MS = 10_000

const props = defineProps<{ clusterId: string; backendId: string; nodeUid: string; nodeId: string }>()
const { data, loading, error, refresh } = useResourceRpc<ResourceDescribeNodeResponse>('DescribeNode', () => ({
  node: {
    key: { clusterId: props.clusterId, kind: 'RESOURCE_KIND_NODE', resourceId: props.nodeId },
    backendId: props.backendId,
    nodeUid: props.nodeUid,
  },
}))
const node = computed(() => data.value?.node)

interface QueryResponse { arrowIpc?: string }
interface WorkerUsageRow {
  cpu_pct?: number
  mem_bytes?: number
  mem_total_bytes?: number
  disk_used_bytes?: number
  disk_total_bytes?: number
}

function sqlString(value: string): string {
  return `'${value.replace(/'/g, "''")}'`
}

const { data: usageData, error: usageError, refresh: refreshUsage } = useLogServerStatsRpc<QueryResponse>(
  'Query',
  () => ({
    sql: `SELECT cpu_pct, mem_bytes, mem_total_bytes, disk_used_bytes, disk_total_bytes
FROM "iris.worker"
WHERE worker_id = ${sqlString(props.nodeId)}
ORDER BY ts DESC
LIMIT 60`,
  }),
)
const usageRows = computed(() => decodeArrowIpc(usageData.value?.arrowIpc).rows as WorkerUsageRow[])
const latestUsage = computed(() => usageRows.value[0])
const cpuHistory = computed(() => usageRows.value.map(row => Number(row.cpu_pct ?? 0)).reverse())
const memoryHistory = computed(() => usageRows.value.map(row => Number(row.mem_bytes ?? 0)).reverse())

async function refreshPage() {
  await Promise.all([refresh(), refreshUsage()])
}

function attributeValue(value: { stringValue?: string; integerValue?: string; floatValue?: number }): string {
  return value.stringValue ?? value.integerValue ?? String(value.floatValue ?? '')
}

onMounted(refreshPage)
useAutoRefresh(refreshPage, NODE_REFRESH_MS)
</script>

<template>
  <PageShell :title="nodeId" back-to="/nodes" back-label="Nodes">
    <div v-if="error" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded border">{{ error }}</div>
    <div v-else-if="loading && !node" class="text-sm text-text-muted">Loading node…</div>
    <div v-else-if="node" class="space-y-6">
      <SourceWarnings :statuses="node.sourceStatuses" />

      <div class="flex flex-wrap items-center gap-3">
        <StatusBadge :status="node.summary.health" />
        <span class="font-mono text-sm text-text-muted">{{ node.summary.identity.backendId }}</span>
        <span v-if="!node.summary.schedulable" class="rounded bg-status-warning-bg px-2 py-1 text-xs text-status-warning">
          Unschedulable
        </span>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <InfoCard title="Node Status">
          <InfoRow label="Health"><StatusBadge :status="node.summary.health" size="sm" /></InfoRow>
          <InfoRow label="Schedulable">{{ node.summary.schedulable ? 'yes' : 'no' }}</InfoRow>
          <InfoRow label="Address"><span class="font-mono">{{ node.address || '—' }}</span></InfoRow>
          <InfoRow label="Region">{{ node.summary.region || '—' }}</InfoRow>
          <InfoRow label="Observed"><span class="font-mono">{{ formatTimestamp(node.summary.observedAt) }}</span></InfoRow>
        </InfoCard>

        <InfoCard title="Capacity">
          <InfoRow label="CPU">{{ node.summary.capacity?.cpuMillicores ?? '0' }}m</InfoRow>
          <InfoRow label="Memory">{{ formatBytes(Number(node.summary.capacity?.memoryBytes ?? 0)) }}</InfoRow>
          <InfoRow label="Disk">{{ formatBytes(Number(node.summary.capacity?.diskBytes ?? 0)) }}</InfoRow>
          <InfoRow label="Accelerator">
            {{ node.summary.capacity?.acceleratorCount ?? 0 }}×
            {{ node.summary.capacity?.acceleratorVariant || node.summary.capacity?.acceleratorKind || 'none' }}
          </InfoRow>
        </InfoCard>

        <InfoCard title="Placement">
          <InfoRow label="Backend"><span class="font-mono">{{ node.summary.identity.backendId }}</span></InfoRow>
          <InfoRow label="Scaling group"><span class="font-mono">{{ node.summary.scalingGroupId || '—' }}</span></InfoRow>
          <InfoRow label="Slice"><span class="font-mono">{{ node.summary.slice?.key.resourceId || '—' }}</span></InfoRow>
          <InfoRow label="Running tasks">{{ node.summary.runningTaskCount }}</InfoRow>
          <InfoRow label="Node UID"><span class="font-mono text-xs">{{ node.summary.identity.nodeUid }}</span></InfoRow>
        </InfoCard>
      </div>

      <section class="rounded border border-surface-border p-4">
        <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Attributes</h3>
        <EmptyState v-if="(node.attributes ?? []).length === 0" message="No attributes reported" />
        <dl v-else class="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2 lg:grid-cols-3">
          <template v-for="attribute in node.attributes" :key="attribute.key">
            <dt class="font-mono text-text-muted">{{ attribute.key }}</dt>
            <dd class="break-all">{{ attributeValue(attribute) }}</dd>
          </template>
        </dl>
      </section>

      <section class="grid gap-4 lg:grid-cols-2">
        <InfoCard title="Live Resources">
          <div v-if="latestUsage" class="space-y-4">
            <div class="flex items-center gap-3">
              <div class="flex-1">
                <ResourceGauge label="CPU" :used="Number(latestUsage.cpu_pct ?? 0)" :total="100" unit="%" />
              </div>
              <div class="w-24"><Sparkline :data="cpuHistory" /></div>
            </div>
            <div class="flex items-center gap-3">
              <div class="flex-1">
                <ResourceGauge
                  label="Memory"
                  :used="Number(latestUsage.mem_bytes ?? 0)"
                  :total="Number(latestUsage.mem_total_bytes ?? 0)"
                  unit="bytes"
                />
              </div>
              <div class="w-24"><Sparkline :data="memoryHistory" /></div>
            </div>
            <ResourceGauge
              label="Disk"
              :used="Number(latestUsage.disk_used_bytes ?? 0)"
              :total="Number(latestUsage.disk_total_bytes ?? 0)"
              unit="bytes"
            />
          </div>
          <div v-else-if="usageError" class="text-sm text-status-danger">{{ usageError }}</div>
          <div v-else class="text-sm text-text-muted">No worker measurements recorded for this node.</div>
        </InfoCard>

        <InfoCard title="Bootstrap Logs">
          <pre v-if="node.bootstrapLogs" class="max-h-80 overflow-auto whitespace-pre-wrap break-all font-mono text-xs">{{ node.bootstrapLogs }}</pre>
          <div v-else class="text-sm text-text-muted">No bootstrap logs recorded.</div>
        </InfoCard>
      </section>

      <section>
        <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Recent Attempts</h3>
        <EmptyState v-if="(node.recentAttempts ?? []).length === 0" message="No recent attempts" />
        <div v-else class="overflow-x-auto rounded border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead>
              <tr class="border-b border-surface-border text-left text-xs uppercase text-text-secondary">
                <th class="px-3 py-2">Task</th>
                <th class="px-3 py-2">Attempt</th>
                <th class="px-3 py-2">State</th>
                <th class="px-3 py-2">Started</th>
                <th class="px-3 py-2">Finished</th>
                <th class="px-3 py-2">Exit</th>
                <th class="px-3 py-2">Reason</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="attempt in node.recentAttempts" :key="attempt.identity.attemptUid" class="border-b border-surface-border-subtle">
                <td class="px-3 py-2">
                  <RouterLink
                    :to="{ name: 'task-detail', params: { clusterId: attempt.identity.task.clusterId, taskId: attempt.identity.task.resourceId } }"
                    class="font-mono text-accent hover:underline"
                  >{{ attempt.identity.task.resourceId }}</RouterLink>
                </td>
                <td class="px-3 py-2 font-mono">{{ attempt.identity.attemptNumber }}</td>
                <td class="px-3 py-2"><StatusBadge :status="attempt.state" size="sm" /></td>
                <td class="px-3 py-2 font-mono">{{ formatTimestamp(attempt.startedAt) }}</td>
                <td class="px-3 py-2 font-mono">{{ formatTimestamp(attempt.finishedAt) }}</td>
                <td class="px-3 py-2 font-mono">{{ attempt.exitCode ?? '—' }}</td>
                <td class="max-w-md px-3 py-2 text-xs text-status-danger">{{ attempt.terminalReason || attempt.errorMessage || '—' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h3 class="mb-3 text-sm font-semibold uppercase tracking-wider text-text-secondary">Worker Daemon Logs</h3>
        <LogViewer :worker-id="nodeId" :authority-cluster="clusterId" />
      </section>
    </div>
  </PageShell>
</template>
