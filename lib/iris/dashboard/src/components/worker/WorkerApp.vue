<script setup lang="ts">
import { ref } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import AppHeader from '@/components/layout/AppHeader.vue'
import DashboardLegend from '@/components/shared/DashboardLegend.vue'
import { useDarkMode } from '@/composables/useDarkMode'

const router = useRouter()
const refreshKey = ref(0)
const legendOpen = ref(false)
const { isDark, toggle: toggleDark } = useDarkMode()

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
               text-text-secondary hover:text-text hover:bg-surface-raised transition-colors text-sm"
        :aria-label="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
        :title="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
        @click="toggleDark"
      >
        <svg v-if="isDark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-4 h-4">
          <path d="M10 2a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 2zm0 13a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 15zm-8-5a.75.75 0 01.75-.75h1.5a.75.75 0 010 1.5h-1.5A.75.75 0 012 10zm13 0a.75.75 0 01.75-.75h1.5a.75.75 0 010 1.5h-1.5A.75.75 0 0115 10zM4.343 4.343a.75.75 0 011.06 0l1.061 1.06a.75.75 0 01-1.06 1.061l-1.061-1.06a.75.75 0 010-1.06zm9.193 9.193a.75.75 0 011.06 0l1.061 1.06a.75.75 0 01-1.06 1.061l-1.061-1.06a.75.75 0 010-1.06zM4.343 15.657a.75.75 0 010-1.06l1.06-1.061a.75.75 0 111.061 1.06l-1.06 1.061a.75.75 0 01-1.06 0zm9.193-9.193a.75.75 0 010-1.06l1.06-1.061a.75.75 0 111.061 1.06l-1.06 1.061a.75.75 0 01-1.06 0zM10 7a3 3 0 100 6 3 3 0 000-6zm-4 3a4 4 0 118 0 4 4 0 01-8 0z" />
        </svg>
        <svg v-else xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-4 h-4">
          <path fill-rule="evenodd" d="M7.455 2.004a.75.75 0 01.26.77 7 7 0 009.958 7.967.75.75 0 011.067.853A8.5 8.5 0 1110.239 1.87a.75.75 0 01-.784.135h-.001z" clip-rule="evenodd" />
        </svg>
      </button>
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
