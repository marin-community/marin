<script setup lang="ts">
import { computed } from 'vue'
import { shortEnum } from '@/utils/formatting'

const props = defineProps<{ value?: string; prefix?: string }>()

const label = computed(() => shortEnum(props.value, props.prefix ?? ''))
const tone = computed(() => {
  const value = label.value
  if (value.includes('succeeded') || value === 'active') return 'bg-status-success-bg text-status-success'
  if (value.includes('failed') || value.includes('stopping')) return 'bg-status-danger-bg text-status-danger'
  if (value.includes('running') || value.includes('waiting')) return 'bg-status-warning-bg text-status-warning'
  return 'bg-surface-sunken text-text-secondary'
})
</script>

<template>
  <span :class="['inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold capitalize', tone]">
    <span class="h-1.5 w-1.5 rounded-full bg-current" />
    {{ label }}
  </span>
</template>
