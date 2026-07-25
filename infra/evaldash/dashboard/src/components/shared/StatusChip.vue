<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  status: string
}>()

// infra_failed is deliberately a distinct colour from failed: an infrastructure
// fault (preemption, OOM, host loss) is not a model/eval regression.
const STYLES: Record<string, string> = {
  succeeded: 'bg-status-success-bg text-status-success border-status-success-border',
  failed: 'bg-status-danger-bg text-status-danger border-status-danger-border',
  infra_failed: 'bg-status-warning-bg text-status-warning border-status-warning-border',
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
