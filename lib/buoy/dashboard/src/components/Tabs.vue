<script setup lang="ts">
import type { TabId } from '../types'

defineProps<{ active: TabId; hasProfile: boolean }>()
defineEmits<{ change: [tab: TabId] }>()

const TABS: { id: TabId; label: string }[] = [
  { id: 'summary', label: 'summary' },
  { id: 'charts', label: 'charts' },
  { id: 'profile', label: 'profile' },
]
</script>

<template>
  <nav class="flex gap-1 border-b border-surface-border bg-surface-raised px-6">
    <button
      v-for="t in TABS"
      v-show="t.id !== 'profile' || hasProfile"
      :key="t.id"
      class="border-b-2 px-4 py-2 text-sm"
      :class="active === t.id ? 'border-accent font-semibold text-accent' : 'border-transparent text-text-secondary'"
      @click="$emit('change', t.id)"
    >
      {{ t.label }}
    </button>
  </nav>
</template>
