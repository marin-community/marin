<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getAuthToken, clearAuthToken } from '@/composables/useRpc'
import AppHeader from '@/components/layout/AppHeader.vue'
import TabNav, { type Tab } from '@/components/layout/TabNav.vue'

const route = useRoute()
const router = useRouter()

const authEnabled = ref(false)

const TABS: Tab[] = [
  { key: 'jobs', label: 'Jobs', to: '/' },
  { key: 'users', label: 'Users', to: '/users' },
  { key: 'fleet', label: 'Workers', to: '/fleet' },
  { key: 'endpoints', label: 'Endpoints', to: '/endpoints' },
  { key: 'autoscaler', label: 'Autoscaler', to: '/autoscaler' },
  { key: 'status', label: 'Status', to: '/status' },
  { key: 'transactions', label: 'Transactions', to: '/transactions' },
  { key: 'account', label: 'Account', to: '/account' },
]

const PATH_TO_TAB: Record<string, string> = {
  '/': 'jobs',
  '/users': 'users',
  '/fleet': 'fleet',
  '/endpoints': 'endpoints',
  '/autoscaler': 'autoscaler',
  '/status': 'status',
  '/transactions': 'transactions',
  '/account': 'account',
}

const activeTab = computed(() => {
  const path = route.path
  if (PATH_TO_TAB[path]) return PATH_TO_TAB[path]
  if (path.startsWith('/job')) return 'jobs'
  if (path.startsWith('/worker')) return 'fleet'
  return 'jobs'
})

// Detail pages hide the tab nav to show breadcrumb navigation instead
const isDetailPage = computed(() => {
  return route.path.includes('/job/') || route.path.includes('/worker/')
})

const isLoginPage = computed(() => route.path === '/login')

function onAuthRequired() {
  router.push('/login')
}

function logout() {
  clearAuthToken()
  router.push('/login')
}

onMounted(async () => {
  window.addEventListener('iris-auth-required', onAuthRequired)

  try {
    const resp = await fetch('/auth/config')
    if (resp.ok) {
      const config = await resp.json()
      authEnabled.value = config.auth_enabled ?? false
    }
  } catch {
    // Auth config endpoint unavailable — assume no auth
  }

  if (authEnabled.value && !getAuthToken() && route.path !== '/login') {
    router.push('/login')
  }
})

onUnmounted(() => {
  window.removeEventListener('iris-auth-required', onAuthRequired)
})
</script>

<template>
  <div v-if="isLoginPage">
    <router-view />
  </div>
  <div v-else class="min-h-screen bg-surface-raised">
    <AppHeader title="Iris Controller Dashboard">
      <button
        v-if="authEnabled"
        class="text-sm text-text-muted hover:text-text transition-colors"
        @click="logout"
      >
        Logout
      </button>
    </AppHeader>
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
