<script setup lang="ts">
import { RouterLink } from 'vue-router'

export interface Tab {
  key: string
  label: string
  to: string
}

defineProps<{
  tabs: Tab[]
  activeTab: string
}>()

const emit = defineEmits<{
  refresh: []
}>()
</script>

<template>
  <nav class="border-b border-surface-border bg-surface">
    <div class="max-w-7xl mx-auto px-6 flex items-center">
      <div class="flex gap-0">
        <RouterLink
          v-for="tab in tabs"
          :key="tab.key"
          :to="tab.to"
          :class="[
            'px-4 py-3 text-sm font-medium border-b-2 transition-colors',
            activeTab === tab.key
              ? 'text-accent border-accent font-semibold'
              : 'text-text-secondary border-transparent hover:text-text hover:bg-surface-sunken',
          ]"
        >
          {{ tab.label }}
        </RouterLink>
      </div>
      <div class="ml-auto flex items-center gap-3">
        <slot />
        <button
          class="px-3 py-1.5 text-sm text-text-secondary border border-surface-border rounded
                 hover:bg-surface-sunken hover:text-text transition-colors"
          @click="emit('refresh')"
        >
          Refresh
        </button>
      </div>
    </div>
  </nav>
</template>
