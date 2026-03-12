<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import AppHeader from '@/components/layout/AppHeader.vue'
import TabNav, { type Tab } from '@/components/layout/TabNav.vue'

const route = useRoute()

const TABS: Tab[] = [
  { key: 'jobs', label: 'Jobs', to: '/' },
  { key: 'users', label: 'Users', to: '/users' },
  { key: 'fleet', label: 'Workers', to: '/fleet' },
  { key: 'endpoints', label: 'Endpoints', to: '/endpoints' },
  { key: 'autoscaler', label: 'Autoscaler', to: '/autoscaler' },
  { key: 'status', label: 'Status', to: '/status' },
  { key: 'transactions', label: 'Transactions', to: '/transactions' },
]

const PATH_TO_TAB: Record<string, string> = {
  '/': 'jobs',
  '/users': 'users',
  '/fleet': 'fleet',
  '/endpoints': 'endpoints',
  '/autoscaler': 'autoscaler',
  '/status': 'status',
  '/transactions': 'transactions',
}

const activeTab = computed(() => {
  const path = route.path
  // Exact match first
  if (PATH_TO_TAB[path]) return PATH_TO_TAB[path]
  // Detail pages map to their parent tab
  if (path.startsWith('/job')) return 'jobs'
  if (path.startsWith('/worker')) return 'fleet'
  return 'jobs'
})

// Detail pages hide the tab nav to show breadcrumb navigation instead
const isDetailPage = computed(() => {
  return route.path.includes('/job/') || route.path.includes('/worker/')
})
</script>

<template>
  <div class="min-h-screen bg-surface-raised">
    <AppHeader title="Iris Controller Dashboard" />
    <TabNav
      v-if="!isDetailPage"
      :tabs="TABS"
      :active-tab="activeTab"
    />
    <main class="max-w-7xl mx-auto px-6 py-6">
      <router-view />
    </main>
  </div>
</template>
