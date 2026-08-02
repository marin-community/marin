<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import ErrorBanner from '@/components/ErrorBanner.vue'
import MetricChart from '@/components/MetricChart.vue'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { selectedExecutionId } from '@/composables/usePipelineSelection'
import { useDashboardRpc } from '@/composables/useRpc'
import type { PipelineMetrics, PipelineStatus } from '@/types/dashboard'
import { formatBytes, formatNumber } from '@/utils/formatting'

const metricsRpc = useDashboardRpc<PipelineMetrics>('GetMetrics', () => ({
  executionId: selectedExecutionId.value,
  maxPoints: 300,
}))
const statusRpc = useDashboardRpc<PipelineStatus>('GetStatus', () => ({ executionId: selectedExecutionId.value }))
const points = computed(() => metricsRpc.data.value?.points ?? [])

async function refresh() {
  await Promise.all([metricsRpc.refresh(), statusRpc.refresh()])
}

useAutoRefresh(refresh, 10_000)
onMounted(refresh)
watch(selectedExecutionId, () => void refresh())
</script>

<template>
  <div class="space-y-4">
    <ErrorBanner :message="metricsRpc.error.value || statusRpc.error.value" />
    <div v-if="metricsRpc.data.value?.warning" class="rounded-lg border border-status-warning bg-status-warning-bg px-4 py-3 text-sm text-status-warning">
      {{ metricsRpc.data.value.warning }}
    </div>
    <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <div class="card p-4">
        <p class="text-xs text-text-muted">Live CPU</p>
        <p class="mt-1 text-2xl font-semibold">{{ formatNumber(statusRpc.data.value?.resources?.cpuCores) }} cores</p>
        <p class="mt-1 text-xs text-text-secondary">{{ formatNumber((statusRpc.data.value?.resources?.cpuUtilization ?? 0) * 100) }}% of worker capacity</p>
      </div>
      <div class="card p-4">
        <p class="text-xs text-text-muted">Live memory</p>
        <p class="mt-1 text-2xl font-semibold">{{ formatBytes(statusRpc.data.value?.resources?.memoryBytes) }}</p>
        <p class="mt-1 text-xs text-text-secondary">{{ formatNumber((statusRpc.data.value?.resources?.memoryUtilization ?? 0) * 100) }}% of worker capacity</p>
      </div>
      <div class="card p-4">
        <p class="text-xs text-text-muted">In flight</p>
        <p class="mt-1 text-2xl font-semibold">{{ statusRpc.data.value?.inFlightShards ?? 0 }}</p>
        <p class="mt-1 text-xs text-text-secondary">{{ statusRpc.data.value?.queuedShards ?? 0 }} queued shards</p>
      </div>
      <div class="card p-4">
        <p class="text-xs text-text-muted">Retries</p>
        <p class="mt-1 text-2xl font-semibold">{{ statusRpc.data.value?.retries ?? 0 }}</p>
        <p class="mt-1 text-xs text-text-secondary">Current stage attempts</p>
      </div>
    </div>
    <div class="grid gap-4 xl:grid-cols-2">
      <MetricChart title="Item throughput" :points="points" field="itemRate" unit="items / sec" />
      <MetricChart title="Byte throughput" :points="points" field="byteRate" unit="bytes / sec" />
      <MetricChart title="CPU utilization" :points="points" field="cpuCores" unit="CPU cores" />
      <MetricChart title="Memory utilization" :points="points" field="memoryBytes" unit="bytes" />
    </div>
  </div>
</template>
