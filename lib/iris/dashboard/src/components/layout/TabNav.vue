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
    <div class="max-w-7xl mx-auto px-3 sm:px-6 flex items-center gap-3">
      <!-- Tab strip scrolls horizontally inside this container so the page itself
           never needs to scroll. shrink-0 on each link prevents them from squishing. -->
      <div class="flex gap-0 flex-1 min-w-0 overflow-x-auto whitespace-nowrap">
        <RouterLink
          v-for="tab in tabs"
          :key="tab.key"
          :to="tab.to"
          :class="[
            'px-4 py-3 text-sm font-medium border-b-2 transition-colors shrink-0',
            activeTab === tab.key
              ? 'text-accent border-accent font-semibold'
              : 'text-text-secondary border-transparent hover:text-text hover:bg-surface-sunken',
          ]"
        >
          {{ tab.label }}
        </RouterLink>
      </div>
      <div class="flex items-center gap-3 shrink-0">
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
