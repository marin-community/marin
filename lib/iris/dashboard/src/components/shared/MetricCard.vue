<script setup lang="ts">
const props = withDefaults(defineProps<{
  value: string | number
  label: string
  detail?: string
  variant?: string
  size?: 'sm' | 'md'
}>(), {
  variant: 'default',
  size: 'md',
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
  <div :class="['rounded-lg border border-surface-border bg-surface', size === 'sm' ? 'px-3 py-1.5' : 'px-4 py-3']">
    <div :class="[size === 'sm' ? 'text-lg' : 'text-2xl', 'font-semibold font-mono tabular-nums leading-tight', valueClass()]">
      {{ value }}
    </div>
    <div :class="['font-medium text-text-secondary uppercase tracking-wider', size === 'sm' ? 'text-[10px] mt-0.5' : 'text-xs mt-1']">
      {{ label }}
    </div>
    <div v-if="detail" :class="['text-text-muted', size === 'sm' ? 'text-[10px] mt-0' : 'text-xs mt-0.5']">
      {{ detail }}
    </div>
  </div>
</template>
