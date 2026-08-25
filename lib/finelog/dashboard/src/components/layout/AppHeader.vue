<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { timeZoneMode, type TimeZoneMode } from '@/composables/useDisplayPrefs'
import { timeZoneLabel } from '@/utils/formatting'

const MODES: TimeZoneMode[] = ['utc', 'local', 'raw']

const dark = ref(false)

onMounted(() => {
  dark.value = document.documentElement.classList.contains('dark')
})

function toggleDark() {
  dark.value = !dark.value
  document.documentElement.classList.toggle('dark', dark.value)
  try {
    localStorage.setItem('finelog-dark-mode', String(dark.value))
  } catch {}
}
</script>

<template>
  <header class="border-b border-surface-border bg-surface px-6 py-3 flex items-center justify-between">
    <div class="flex items-baseline gap-3">
      <h1 class="text-base font-semibold tracking-tight">Finelog</h1>
      <span class="text-xs text-text-muted font-mono">stats &amp; logs</span>
    </div>
    <div class="flex items-center gap-2">
      <div class="inline-flex rounded border border-surface-border overflow-hidden" role="group" aria-label="Timestamp display">
        <button
          v-for="mode in MODES"
          :key="mode"
          class="text-xs px-2 py-1"
          :class="timeZoneMode === mode ? 'bg-accent text-white' : 'hover:bg-surface-raised'"
          :title="`Show timestamps as ${timeZoneLabel(mode)}`"
          @click="timeZoneMode = mode"
        >{{ mode === 'raw' ? 'epoch' : mode }}</button>
      </div>
      <button
        class="text-xs px-2 py-1 rounded border border-surface-border hover:bg-surface-raised"
        @click="toggleDark"
      >
        {{ dark ? '☀ light' : '☾ dark' }}
      </button>
    </div>
  </header>
</template>
