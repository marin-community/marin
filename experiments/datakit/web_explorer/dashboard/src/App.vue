<script setup lang="ts">
import { onMounted, onUnmounted, ref, shallowRef } from 'vue'
import type { Component } from 'vue'
import DecontamTab from './components/DecontamTab.vue'
import DedupTab from './components/DedupTab.vue'
import NormalizedTab from './components/NormalizedTab.vue'
import OverviewTab from './components/OverviewTab.vue'
import QualityTab from './components/QualityTab.vue'
import SourcesTab from './components/SourcesTab.vue'
import StoreTab from './components/StoreTab.vue'
import { checkDuckyStatus, duckyAvailable } from './composables/duckyStatus'
import type { Overview } from './types'

const TABS: { key: string; label: string; component: Component }[] = [
  { key: 'overview', label: 'Overview', component: OverviewTab },
  { key: 'sources', label: 'Sources', component: SourcesTab },
  { key: 'normalized', label: 'Normalized', component: NormalizedTab },
  { key: 'decontam', label: 'Decontamination', component: DecontamTab },
  { key: 'dedup', label: 'Deduplication', component: DedupTab },
  { key: 'quality', label: 'Classifier', component: QualityTab },
  { key: 'store', label: 'Store', component: StoreTab },
]

const ov = shallowRef<Overview | null>(null)
const loadError = ref('')
const activeTab = ref(TABS[0])

const dark = ref(document.documentElement.classList.contains('dark'))
function toggleDark() {
  dark.value = !dark.value
  document.documentElement.classList.toggle('dark', dark.value)
  try {
    localStorage.setItem('datakit-explorer-dark-mode', String(dark.value))
  } catch (e) {
    /* ignore */
  }
}

const DUCKY_POLL_MS = 20_000
const SUMMARY_POLL_MS = 3_000
let duckyTimer: ReturnType<typeof setInterval> | undefined
let summaryTimer: ReturnType<typeof setInterval> | undefined

// The app counts per-source sizes in the background; poll until it's done,
// folding each partial result into `ov` so the leaderboard fills in live.
async function pollSourceSummary() {
  if (!ov.value) return
  try {
    const resp = await fetch('api/source-summary')
    const data = await resp.json()
    if (!resp.ok) return
    ov.value = { ...ov.value, source_summary: data.rows, source_summary_ready: data.ready }
    if (data.ready) {
      clearInterval(summaryTimer)
      summaryTimer = undefined
    }
  } catch (e) {
    /* transient; keep polling */
  }
}

onMounted(async () => {
  try {
    const resp = await fetch('api/overview')
    const data = await resp.json()
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`)
    ov.value = data as Overview
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : String(e)
  }
  if (ov.value && !ov.value.source_summary_ready) {
    summaryTimer = setInterval(pollSourceSummary, SUMMARY_POLL_MS)
  }
  void checkDuckyStatus()
  duckyTimer = setInterval(checkDuckyStatus, DUCKY_POLL_MS)
})

onUnmounted(() => {
  clearInterval(duckyTimer)
  clearInterval(summaryTimer)
})
</script>

<template>
  <div class="flex min-h-screen flex-col bg-surface-raised">
    <header class="flex items-center gap-2.5 border-b border-surface-border bg-surface px-5 py-2.5">
      <h1 class="text-base font-semibold">datakit explorer</h1>
      <span class="font-mono text-xs text-text-muted">{{ ov ? ov.store_path : 'loading…' }}</span>
      <span
        v-if="ov"
        class="rounded-full px-2 py-0.5 text-[11px]"
        :class="ov.verified ? 'bg-status-success/15 text-status-success' : 'bg-status-warning/15 text-status-warning'"
      >
        {{ ov.verified ? 'lineage verified' : 'lineage best-effort' }}
      </span>
      <button
        class="ml-auto rounded-md border border-surface-border px-2.5 py-1.5 text-sm text-text-secondary hover:bg-surface-raised"
        :title="dark ? 'Switch to light mode' : 'Switch to dark mode'"
        @click="toggleDark"
      >
        {{ dark ? '☀️' : '🌙' }}
      </button>
    </header>

    <div
      v-if="!duckyAvailable"
      class="border-b border-status-warning/40 bg-status-warning/10 px-5 py-1.5 text-[13px] text-status-warning"
    >
      ⚠ query backend (ducky) is currently unavailable — it's preemptible and may be rescheduling. Queries retry
      automatically for a while; results may be delayed or fail until it's back.
    </div>

    <nav class="flex gap-1 border-b border-surface-border bg-surface px-5">
      <button
        v-for="tab in TABS"
        :key="tab.key"
        class="rounded-t-md border border-transparent px-3.5 py-2 text-[13px]"
        :class="
          tab.key === activeTab.key
            ? 'border-surface-border border-b-transparent bg-surface-raised font-semibold text-text'
            : 'text-text-muted hover:text-text-secondary'
        "
        @click="activeTab = tab"
      >
        {{ tab.label }}
      </button>
    </nav>

    <main class="flex-1 px-5 py-4">
      <p v-if="loadError" class="text-sm text-status-danger">failed to load overview: {{ loadError }}</p>
      <KeepAlive v-else-if="ov">
        <component :is="activeTab.component" :ov="ov" />
      </KeepAlive>
      <p v-else class="text-sm text-text-muted">loading overview…</p>
    </main>
  </div>
</template>
