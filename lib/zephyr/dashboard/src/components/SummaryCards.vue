<script setup lang="ts">
import { computed } from 'vue'
import type { PipelinePlan, PipelineStatus } from '@/types/dashboard'
import { formatBytes, formatCount, formatDuration, formatNumber } from '@/utils/formatting'

const props = defineProps<{ plan: PipelinePlan | null; status: PipelineStatus | null }>()

const elapsed = computed(() => {
  const start = props.status?.started_at_ms ?? 0
  if (!start) return '—'
  return formatDuration((props.status?.finished_at_ms || Date.now()) - start)
})
const shardProgress = computed(() => `${props.status?.completed_shards ?? 0}/${props.status?.total_shards ?? 0}`)
const workerCount = computed(() =>
  (props.status?.worker_states ?? []).find((item) => item.state === 'active')?.count ?? 0,
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
      <p class="mt-1 text-xl font-semibold">{{ workerCount }}/{{ status?.expected_workers ?? 0 }}</p>
    </div>
    <div class="card p-4">
      <p class="text-xs font-medium text-text-muted">CPU</p>
      <p class="mt-1 text-xl font-semibold">{{ formatNumber(status?.resources?.cpu_cores) }} cores</p>
    </div>
    <div class="card p-4">
      <p class="text-xs font-medium text-text-muted">Memory</p>
      <p class="mt-1 text-xl font-semibold">{{ formatBytes(status?.resources?.memory_bytes) }}</p>
    </div>
    <div class="card p-4">
      <p class="text-xs font-medium text-text-muted">Source</p>
      <p class="mt-1 text-xl font-semibold">{{ formatCount(plan?.source_item_count) }}</p>
    </div>
  </div>
</template>
