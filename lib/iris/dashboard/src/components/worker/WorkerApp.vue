<script setup lang="ts">
import { ref } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import AppHeader from '@/components/layout/AppHeader.vue'
import DashboardLegend from '@/components/shared/DashboardLegend.vue'

const router = useRouter()
const refreshKey = ref(0)
const legendOpen = ref(false)

function refresh() {
  refreshKey.value++
}
</script>

<template>
  <div class="min-h-screen bg-surface-raised">
    <AppHeader title="Iris Worker Dashboard">
      <RouterLink
        to="/"
        class="text-xs text-text-secondary hover:text-text px-2 py-1 rounded hover:bg-surface-raised"
      >
        Status
      </RouterLink>
      <button
        class="flex items-center justify-center w-7 h-7 rounded-full border border-surface-border
               text-text-secondary hover:text-text hover:bg-surface-raised transition-colors text-sm font-semibold"
        aria-label="Show dashboard legend"
        title="Dashboard legend"
        @click="legendOpen = true"
      >
        ?
      </button>
      <button
        class="flex items-center gap-1.5 px-3 py-1.5 text-xs border border-surface-border rounded
               hover:bg-surface-raised text-text-secondary"
        @click="refresh"
      >
        <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
          <path d="M21 3v5h-5" />
          <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
          <path d="M3 21v-5h5" />
        </svg>
        Refresh
      </button>
    </AppHeader>
    <DashboardLegend v-if="legendOpen" @close="legendOpen = false" />
    <main class="max-w-7xl mx-auto px-6 py-6">
      <router-view :key="refreshKey" />
    </main>
  </div>
</template>
