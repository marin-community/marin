<script setup lang="ts">
import { computed } from 'vue'
import { statusColors, stateDisplayName, stateToName } from '@/types/status'

const props = withDefaults(
  defineProps<{
    status: string
    size?: 'sm' | 'md'
    showDot?: boolean
  }>(),
  {
    size: 'md',
    showDot: true,
  }
)

const normalizedState = computed(() => stateToName(props.status))
const colors = computed(() => statusColors(normalizedState.value))
const displayName = computed(() => stateDisplayName(normalizedState.value))

const badgePadding = computed(() =>
  props.size === 'sm' ? 'px-2 py-0.5' : 'px-2.5 py-0.5'
)
</script>

<template>
  <span
    :class="[
      'inline-flex items-center gap-1.5 rounded-full border',
      'text-xs font-semibold tracking-wide uppercase',
      badgePadding,
      colors.text,
      colors.bg,
      colors.border,
    ]"
  >
    <span
      v-if="showDot"
      :class="['w-1.5 h-1.5 rounded-full flex-shrink-0', colors.dot]"
    />
    {{ displayName }}
  </span>
</template>
