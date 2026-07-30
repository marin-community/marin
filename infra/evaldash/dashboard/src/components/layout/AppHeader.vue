<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { apiGet } from '@/composables/useApi'
import type { Meta } from '@/types/api'
import RefreshButton from '@/components/shared/RefreshButton.vue'

const dark = ref(false)
const currentUser = ref<string | null>(null)
const store = ref<string | null>(null)

onMounted(async () => {
  dark.value = document.documentElement.classList.contains('dark')
  try {
    const meta = await apiGet<Meta>('api/meta')
    currentUser.value = meta.current_user
    store.value = meta.store
  } catch {
    // Header identity is best-effort; the pages surface real errors.
  }
})

function toggleDark() {
  dark.value = !dark.value
  document.documentElement.classList.toggle('dark', dark.value)
  try {
    localStorage.setItem('evaldash-dark-mode', String(dark.value))
  } catch {}
}
</script>

<template>
  <header class="border-b border-surface-border bg-surface px-6 py-3 flex items-center justify-between">
    <div class="flex items-baseline gap-2.5 font-mono">
      <span
        class="w-2 h-2 rounded-full bg-status-success self-center shrink-0"
        style="box-shadow: 0 0 0 3px var(--c-status-success-bg)"
        title="Ingest is live"
      ></span>
      <h1 class="text-sm font-semibold tracking-wide">MARIN<span class="text-text-muted">·</span>EVALS</h1>
      <span class="text-[10px] uppercase tracking-[0.18em] text-text-muted">measurement&nbsp;log</span>
      <span
        v-if="store === 'memory'"
        class="text-xs px-1.5 py-0.5 rounded bg-status-warning-bg text-status-warning border border-status-warning-border"
        title="Local mode — serving from the record snapshot with no database"
      >local</span>
    </div>
    <div class="flex items-center gap-3">
      <span v-if="currentUser" class="text-xs text-text-muted font-mono">{{ currentUser }}</span>
      <RefreshButton />
      <button
        class="text-xs px-2 py-1 rounded border border-surface-border hover:bg-surface-raised"
        @click="toggleDark"
      >
        {{ dark ? '☀ light' : '☾ dark' }}
      </button>
    </div>
  </header>
</template>
