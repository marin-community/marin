<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AppHeader from '@/components/layout/AppHeader.vue'
import DashboardLegend from '@/components/shared/DashboardLegend.vue'
import TabNav, { type Tab } from '@/components/layout/TabNav.vue'
import { useDarkMode } from '@/composables/useDarkMode'

const route = useRoute()
const router = useRouter()
const { isDark, toggle: toggleDark } = useDarkMode()

const authEnabled = ref(false)
const providerKind = ref<'worker' | 'kubernetes'>('worker')
const legendOpen = ref(false)

const WORKER_TABS: Tab[] = [
  { key: 'jobs', label: 'Jobs', to: '/' },
  { key: 'scheduler', label: 'Scheduler', to: '/scheduler' },
  { key: 'fleet', label: 'Workers', to: '/fleet' },
  { key: 'endpoints', label: 'Endpoints', to: '/endpoints' },
  { key: 'autoscaler', label: 'Autoscaler', to: '/autoscaler' },
  { key: 'account', label: 'Account', to: '/account' },
  { key: 'status', label: 'Status', to: '/status' },
]

const KUBERNETES_TABS: Tab[] = [
  { key: 'jobs', label: 'Jobs', to: '/' },
  { key: 'scheduler', label: 'Scheduler', to: '/scheduler' },
  { key: 'cluster', label: 'Cluster', to: '/cluster' },
  { key: 'endpoints', label: 'Endpoints', to: '/endpoints' },
  { key: 'account', label: 'Account', to: '/account' },
  { key: 'status', label: 'Status', to: '/status' },
]

const TABS = computed<Tab[]>(() =>
  providerKind.value === 'kubernetes' ? KUBERNETES_TABS : WORKER_TABS
)

const PATH_TO_TAB: Record<string, string> = {
  '/': 'jobs',
  '/scheduler': 'scheduler',
  '/fleet': 'fleet',
  '/cluster': 'cluster',
  '/endpoints': 'endpoints',
  '/autoscaler': 'autoscaler',
  '/account': 'account',
  '/status': 'status',
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
  return route.path.includes('/job/') || route.path.includes('/worker/') || route.path.startsWith('/system/')
})

const isLoginPage = computed(() => route.path === '/login')

function onAuthRequired() {
  router.push('/login')
}

async function logout() {
  await fetch('/auth/logout', { method: 'POST' })
  router.push('/login')
}

onMounted(async () => {
  window.addEventListener('iris-auth-required', onAuthRequired)

  let hasSession = false
  let authOptional = false
  try {
    const resp = await fetch('/auth/config')
    if (resp.ok) {
      const config = await resp.json()
      authEnabled.value = config.auth_enabled ?? false
      hasSession = config.has_session ?? false
      authOptional = config.optional ?? false
      providerKind.value = config.provider_kind === 'kubernetes' ? 'kubernetes' : 'worker'
    }
  } catch {
    // Auth config endpoint unavailable — assume no auth
  }

  if (authEnabled.value && !authOptional && !hasSession && route.path !== '/login') {
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
  <div v-else class="min-h-screen bg-surface-raised overflow-x-clip">
    <AppHeader title="Iris Controller Dashboard">
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
        v-if="authEnabled"
        class="text-sm text-text-muted hover:text-text transition-colors"
        @click="logout"
      >
        Logout
      </button>
    </AppHeader>
    <DashboardLegend v-if="legendOpen" @close="legendOpen = false" />
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
