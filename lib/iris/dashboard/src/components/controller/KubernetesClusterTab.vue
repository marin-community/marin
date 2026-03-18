<script setup lang="ts">
import { onMounted } from 'vue'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { timestampMs, formatRelativeTime } from '@/utils/formatting'
import MetricCard from '@/components/shared/MetricCard.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import type { GetKubernetesClusterStatusResponse } from '@/types/rpc'

const { data, loading, error, refresh } = useControllerRpc<GetKubernetesClusterStatusResponse>('GetKubernetesClusterStatus')
useAutoRefresh(refresh, 15_000)
onMounted(refresh)

function phaseClass(phase: string): string {
  switch (phase) {
    case 'Running': return 'text-green-600'
    case 'Succeeded': return 'text-blue-600'
    case 'Failed': return 'text-red-600'
    case 'Pending': return 'text-yellow-600'
    default: return 'text-text-muted'
  }
}

function formatTransition(ts?: { epochMs?: string }): string {
  if (!ts?.epochMs) return '-'
  const ms = timestampMs(ts as { epochMs: string })
  if (!ms) return '-'
  return formatRelativeTime(ms)
}
</script>

<template>
  <div class="space-y-6">
    <div v-if="loading && !data" class="text-text-muted">Loading cluster status...</div>
    <div v-else-if="error" class="text-red-600">{{ error }}</div>
    <template v-else-if="data">
      <!-- Summary cards -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Namespace" :value="data.namespace || '-'" />
        <MetricCard label="Total Nodes" :value="String(data.totalNodes ?? 0)" />
        <MetricCard label="Schedulable Nodes" :value="String(data.schedulableNodes ?? 0)" />
        <MetricCard label="Allocatable CPU" :value="data.allocatableCpu || '-'" />
        <MetricCard label="Allocatable Memory" :value="data.allocatableMemory || '-'" />
        <MetricCard label="Provider" :value="data.providerVersion || '-'" />
      </div>

      <!-- Pod statuses -->
      <div>
        <h2 class="text-lg font-semibold mb-3">Pod Statuses</h2>
        <EmptyState v-if="!data.podStatuses?.length" message="No iris-managed pods found." />
        <div v-else class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-text-muted border-b border-border">
                <th class="pb-2 pr-4">Pod Name</th>
                <th class="pb-2 pr-4">Task ID</th>
                <th class="pb-2 pr-4">Phase</th>
                <th class="pb-2 pr-4">Reason</th>
                <th class="pb-2 pr-4">Message</th>
                <th class="pb-2">Last Transition</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="pod in data.podStatuses"
                :key="pod.podName"
                class="border-b border-border last:border-0"
              >
                <td class="py-2 pr-4 font-mono text-xs">{{ pod.podName }}</td>
                <td class="py-2 pr-4 font-mono text-xs max-w-xs truncate">{{ pod.taskId || '-' }}</td>
                <td class="py-2 pr-4">
                  <span :class="phaseClass(pod.phase)">{{ pod.phase }}</span>
                </td>
                <td class="py-2 pr-4">{{ pod.reason || '-' }}</td>
                <td class="py-2 pr-4 max-w-xs truncate" :title="pod.message">{{ pod.message || '-' }}</td>
                <td class="py-2">{{ formatTransition(pod.lastTransition) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>
  </div>
</template>
