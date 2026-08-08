<script setup lang="ts">
import { computed } from 'vue'
import type { ResourceSourceStatus } from '@/types/rpc'

const props = defineProps<{ statuses?: ResourceSourceStatus[] }>()
const warnings = computed(() =>
  (props.statuses ?? []).filter(status => !['SOURCE_STATE_AVAILABLE', 'available'].includes(status.state)),
)
</script>

<template>
  <div v-if="warnings.length" class="space-y-1">
    <div v-for="status in warnings" :key="status.sourceId" class="px-3 py-2 text-sm text-status-warning bg-status-warning-bg border border-status-warning-border rounded">
      <span class="font-mono">{{ status.sourceId }}</span>:
      {{ status.errorMessage || status.errorCode || status.state }}
      <span v-if="status.freshness" class="text-text-muted">({{ status.freshness.toLowerCase().replace('freshness_', '') }})</span>
    </div>
  </div>
</template>
