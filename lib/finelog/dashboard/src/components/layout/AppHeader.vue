<script setup lang="ts">
import { ref, onMounted } from 'vue'

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
    <button
      class="text-xs px-2 py-1 rounded border border-surface-border hover:bg-surface-raised"
      @click="toggleDark"
    >
      {{ dark ? '☀ light' : '☾ dark' }}
    </button>
  </header>
</template>
