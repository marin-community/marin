<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import AppHeader from '@/components/layout/AppHeader.vue'
import DashboardLegend from '@/components/shared/DashboardLegend.vue'
import TabNav, { type Tab } from '@/components/layout/TabNav.vue'
import BackendScope from '@/components/shared/BackendScope.vue'
import { useDarkMode } from '@/composables/useDarkMode'
import { useBackends } from '@/composables/useBackends'

const route = useRoute()
const { isDark, toggle: toggleDark } = useDarkMode()
const { backends, peers, fetchConfig, ensurePeers } = useBackends()

// Show the scope selector once there is more than one execution target to pick
// between — counting backends and federation peers, so a 1-backend + N-peer
// deployment still gets the selector.
const showScope = computed(() => backends.value.length + peers.value.length > 1)

// On an IAP cluster a 401 is an edge-session lapse recovered by a full-page
// reload through the edge. Direct loopback access has no browser login flow.
const authProvider = ref<string | null>(null)
const legendOpen = ref(false)

// sessionStorage key holding the epoch-ms of the last IAP re-auth reload.
const IAP_REAUTH_RELOAD_KEY = 'iris-iap-reauth-reload-ms'
// A second 401 within this window of a reload means the reload did not re-auth.
const IAP_REAUTH_RELOAD_WINDOW_MS = 15_000
// Set once we schedule a reload so concurrent polls cannot trigger several reloads.
let reloadingForAuth = false

// The Backends tab subsumes provider-specific cluster views.
const TABS: Tab[] = [
  { key: 'jobs', label: 'Jobs', to: '/' },
  { key: 'nodes', label: 'Nodes', to: '/nodes' },
  { key: 'capacity', label: 'Capacity & Scheduling', to: '/capacity' },
  { key: 'backends', label: 'Backends', to: '/backends' },
  { key: 'endpoints', label: 'Endpoints', to: '/endpoints' },
  { key: 'logs', label: 'Logs', to: '/logs' },
  { key: 'account', label: 'Account', to: '/account' },
  { key: 'status', label: 'Status', to: '/status' },
]

const PATH_TO_TAB = Object.fromEntries(TABS.map(tab => [tab.to, tab.key])) as Record<string, string>

const activeTab = computed(() => {
  const path = route.path
  if (PATH_TO_TAB[path]) return PATH_TO_TAB[path]
  if (path.startsWith('/job')) return 'jobs'
  if (path.startsWith('/task')) return 'jobs'
  if (path.startsWith('/node')) return 'nodes'
  return 'jobs'
})

// Detail pages hide the tab nav to show breadcrumb navigation instead
const isDetailPage = computed(() => {
  return route.path.startsWith('/job/') || route.path.startsWith('/task/') || route.path.startsWith('/node/') || route.path.startsWith('/system/')
})

function onAuthRequired() {
  // A 401 reached the SPA. On an IAP-fronted cluster this is almost always the
  // browser's IAP EDGE session lapsing: IAP answers a background XHR/POST (the
  // RPC polls, the 30s log-viewer FetchLogs) with 401 rather than the 302 it gives
  // a GET navigation, so iris never sees the request. The remedy is a full-page
  // reload — a GET that IAP redirects through its edge re-auth. Reload at most
  // once per window so revoked access or a persistent controller rejection does
  // not cause a loop.
  if (reloadingForAuth) return
  if (authProvider.value === 'iap') {
    const last = Number(sessionStorage.getItem(IAP_REAUTH_RELOAD_KEY) ?? '0')
    if (!Number.isFinite(last) || Date.now() - last > IAP_REAUTH_RELOAD_WINDOW_MS) {
      reloadingForAuth = true
      sessionStorage.setItem(IAP_REAUTH_RELOAD_KEY, String(Date.now()))
      window.location.reload()
      return
    }
  }
}

onMounted(async () => {
  window.addEventListener('iris-auth-required', onAuthRequired)

  try {
    // The provider selects IAP-specific 401 recovery; auth itself happens at
    // the edge or through direct loopback trust.
    const { provider } = await fetchConfig()
    authProvider.value = provider
  } catch {
    // Config endpoint unavailable — RPC views surface their own errors.
  }

  // Load the peer roster so the scope selector can count peers; inert (empty)
  // on a single-cluster deployment.
  void ensurePeers()
})

onUnmounted(() => {
  window.removeEventListener('iris-auth-required', onAuthRequired)
})
</script>

<template>
  <div class="min-h-screen bg-surface-raised overflow-x-clip">
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
    </AppHeader>
    <DashboardLegend v-if="legendOpen" @close="legendOpen = false" />
    <TabNav
      v-if="!isDetailPage"
      :tabs="TABS"
      :active-tab="activeTab"
    >
      <!-- Scope selector: visible with >1 execution target (backends + peers) -->
      <BackendScope v-if="showScope" />
    </TabNav>
    <main class="max-w-7xl mx-auto px-6 py-6">
      <router-view />
    </main>
  </div>
</template>
