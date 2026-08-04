<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import ErrorBanner from '@/components/ErrorBanner.vue'
import MetricChart from '@/components/MetricChart.vue'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { selectedExecutionId } from '@/composables/usePipelineSelection'
import { useDashboardApi } from '@/composables/useApi'
import type { PipelineMetrics, PipelineStatus } from '@/types/dashboard'
import { formatBytes, formatNumber } from '@/utils/formatting'

const metricsApi = useDashboardApi<PipelineMetrics>('metrics', () => ({
  execution_id: selectedExecutionId.value,
  max_points: 300,
}))
const statusApi = useDashboardApi<PipelineStatus>('status', () => ({ execution_id: selectedExecutionId.value }))
const points = computed(() => metricsApi.data.value?.points ?? [])

async function refresh() {
  await Promise.all([metricsApi.refresh(), statusApi.refresh()])
}

useAutoRefresh(refresh, 10_000)
onMounted(refresh)
watch(selectedExecutionId, () => void refresh())
</script>

<template>
  <div class="space-y-4">
    <ErrorBanner :message="metricsApi.error.value || statusApi.error.value" />
    <div v-if="metricsApi.data.value?.warning" class="rounded-lg border border-status-warning bg-status-warning-bg px-4 py-3 text-sm text-status-warning">
      {{ metricsApi.data.value.warning }}
    </div>
    <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <div class="card p-4">
        <p class="text-xs text-text-muted">Live CPU</p>
        <p class="mt-1 text-2xl font-semibold">{{ formatNumber(statusApi.data.value?.resources?.cpu_cores) }} cores</p>
        <p class="mt-1 text-xs text-text-secondary">{{ formatNumber((statusApi.data.value?.resources?.cpu_utilization ?? 0) * 100) }}% of worker capacity</p>
      </div>
      <div class="card p-4">
        <p class="text-xs text-text-muted">Live memory</p>
        <p class="mt-1 text-2xl font-semibold">{{ formatBytes(statusApi.data.value?.resources?.memory_bytes) }}</p>
        <p class="mt-1 text-xs text-text-secondary">{{ formatNumber((statusApi.data.value?.resources?.memory_utilization ?? 0) * 100) }}% of worker capacity</p>
      </div>
      <div class="card p-4">
        <p class="text-xs text-text-muted">In flight</p>
        <p class="mt-1 text-2xl font-semibold">{{ statusApi.data.value?.in_flight_shards ?? 0 }}</p>
        <p class="mt-1 text-xs text-text-secondary">{{ statusApi.data.value?.queued_shards ?? 0 }} queued shards</p>
      </div>
      <div class="card p-4">
        <p class="text-xs text-text-muted">Retries</p>
        <p class="mt-1 text-2xl font-semibold">{{ statusApi.data.value?.retries ?? 0 }}</p>
        <p class="mt-1 text-xs text-text-secondary">Current stage attempts</p>
      </div>
    </div>
    <div class="grid gap-4 xl:grid-cols-2">
      <MetricChart title="Item throughput" :points="points" field="item_rate" unit="items / sec" />
      <MetricChart title="Byte throughput" :points="points" field="byte_rate" unit="bytes / sec" />
      <MetricChart title="CPU utilization" :points="points" field="cpu_cores" unit="CPU cores" />
      <MetricChart title="Memory utilization" :points="points" field="memory_bytes" unit="bytes" />
    </div>
  </div>
</template>
