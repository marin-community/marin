<script setup lang="ts">
import { computed } from 'vue'
import type { PipelinePlan, PipelineStatus } from '@/types/dashboard'
import { formatBytes, formatCount, formatDuration, formatNumber, numeric } from '@/utils/formatting'

const props = defineProps<{ plan: PipelinePlan | null; status: PipelineStatus | null }>()

const elapsed = computed(() => {
  const start = numeric(props.status?.startedAtMs)
  if (!start) return '—'
  return formatDuration((numeric(props.status?.finishedAtMs) || Date.now()) - start)
})
const shardProgress = computed(() => `${props.status?.completedShards ?? 0}/${props.status?.totalShards ?? 0}`)
const workerCount = computed(() =>
  (props.status?.workerStates ?? []).reduce((sum, item) => sum + item.count, 0),
)
</script>

<template>
  <div class="grid grid-cols-2 gap-3 lg:grid-cols-6">
    <div class="card p-4">
      <p class="text-xs font-medium text-text-muted">Elapsed</p>
      <p class="mt-1 text-xl font-semibold">{{ elapsed }}</p>
    </div>
    <div class="card p-4">
      <p class="text-xs font-medium text-text-muted">Current shards</p>
      <p class="mt-1 text-xl font-semibold">{{ shardProgress }}</p>
    </div>
    <div class="card p-4">
      <p class="text-xs font-medium text-text-muted">Workers</p>
      <p class="mt-1 text-xl font-semibold">{{ workerCount }}/{{ status?.expectedWorkers ?? 0 }}</p>
    </div>
    <div class="card p-4">
      <p class="text-xs font-medium text-text-muted">CPU</p>
      <p class="mt-1 text-xl font-semibold">{{ formatNumber(status?.resources?.cpuCores) }} cores</p>
    </div>
    <div class="card p-4">
      <p class="text-xs font-medium text-text-muted">Memory</p>
      <p class="mt-1 text-xl font-semibold">{{ formatBytes(status?.resources?.memoryBytes) }}</p>
    </div>
    <div class="card p-4">
      <p class="text-xs font-medium text-text-muted">Source</p>
      <p class="mt-1 text-xl font-semibold">{{ formatCount(plan?.sourceItemCount) }}</p>
    </div>
  </div>
</template>
