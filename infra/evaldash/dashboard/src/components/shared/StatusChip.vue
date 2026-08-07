<script setup lang="ts">
import { computed } from 'vue'
import { RUN_STATUS } from '@/types/api'

const props = defineProps<{
  status: string
}>()

// Artifact and infrastructure failures are warnings: neither is a model/eval regression.
const STYLES: Record<string, string> = {
  [RUN_STATUS.SUCCEEDED]: 'bg-status-success-bg text-status-success border-status-success-border',
  [RUN_STATUS.FAILED]: 'bg-status-danger-bg text-status-danger border-status-danger-border',
  [RUN_STATUS.ARTIFACT_FAILED]: 'bg-status-warning-bg text-status-warning border-status-warning-border',
  [RUN_STATUS.INFRA_FAILED]: 'bg-status-warning-bg text-status-warning border-status-warning-border',
}

const chipClass = computed(
  () => STYLES[props.status] ?? 'bg-surface-sunken text-text-secondary border-surface-border',
)

const label = computed(() => props.status.replace(/_/g, ' '))
</script>

<template>
  <span
    class="inline-block rounded px-1.5 py-0.5 text-xs font-medium border whitespace-nowrap"
    :class="chipClass"
  >{{ label }}</span>
</template>
