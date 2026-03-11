<script setup lang="ts">
const props = withDefaults(defineProps<{
  value: string | number
  label: string
  detail?: string
  variant?: string
}>(), {
  variant: 'default',
})

const VARIANT_CLASSES: Record<string, string> = {
  default: 'text-text',
  accent: 'text-accent',
  success: 'text-status-success',
  warning: 'text-status-warning',
  danger: 'text-status-danger',
  purple: 'text-status-purple',
  orange: 'text-status-orange',
}

function valueClass(): string {
  return VARIANT_CLASSES[props.variant] ?? VARIANT_CLASSES.default
}
</script>

<template>
  <div class="rounded-lg border border-surface-border bg-surface px-4 py-3">
    <div :class="['text-2xl font-semibold font-mono tabular-nums', valueClass()]">
      {{ value }}
    </div>
    <div class="text-xs font-medium text-text-secondary mt-1 uppercase tracking-wider">
      {{ label }}
    </div>
    <div v-if="detail" class="text-xs text-text-muted mt-0.5">
      {{ detail }}
    </div>
  </div>
</template>
