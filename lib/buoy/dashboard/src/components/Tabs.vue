<script setup lang="ts">
import type { TabId } from '../types'

defineProps<{ active: TabId; hasProfile: boolean; live?: string | null }>()
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
    <span v-if="live" class="ml-auto flex items-center gap-1 self-center pr-1 text-xs font-semibold text-status-success">
      <span class="inline-block h-2 w-2 animate-pulse rounded-full bg-status-success"></span>
      live · updated {{ live }}
    </span>
  </nav>
</template>
