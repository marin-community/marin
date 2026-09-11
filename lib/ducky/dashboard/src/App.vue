<script setup lang="ts">
import { onMounted, ref } from 'vue'
import SqlEditor from './components/SqlEditor.vue'
import ResultTable from './components/ResultTable.vue'
import StatusBar from './components/StatusBar.vue'
import Catalog from './components/Catalog.vue'
import { useQuery } from './composables/useQuery'

const sqlText = ref('')
const useCache = ref(true)
const ttlDays = ref<number | null>(null)
const { phase, error, result, running, run } = useQuery()

function runQuery() {
  run(sqlText.value, useCache.value)
}

const dark = ref(document.documentElement.classList.contains('dark'))
function toggleDark() {
  dark.value = !dark.value
  document.documentElement.classList.toggle('dark', dark.value)
  try {
    localStorage.setItem('ducky-dark-mode', String(dark.value))
  } catch (e) {
    /* ignore */
  }
}

onMounted(async () => {
  try {
    const cfg = await (await fetch('api/config')).json()
    ttlDays.value = cfg.result_ttl_days ?? null
  } catch (e) {
    /* config is best-effort; the TTL note just won't show */
  }
})
</script>

<template>
  <div class="mx-auto flex min-h-screen max-w-6xl flex-col gap-4 px-6 py-6">
    <header class="flex items-center justify-between">
      <h1 class="flex items-center gap-2 text-xl font-semibold">
        <span>🦆</span><span>ducky</span>
        <span class="text-sm font-normal text-text-muted">ad-hoc DuckDB SQL</span>
      </h1>
      <button
        class="rounded-md border border-surface-border px-2.5 py-1.5 text-sm text-text-secondary hover:bg-surface-raised"
        :title="dark ? 'Switch to light mode' : 'Switch to dark mode'"
        @click="toggleDark"
      >
        {{ dark ? '☀️' : '🌙' }}
      </button>
    </header>

    <Catalog @select="sqlText = $event" />

    <SqlEditor v-model="sqlText" :dark="dark" @run="runQuery" />

    <div class="flex items-center gap-3">
      <button
        class="rounded-md bg-accent px-4 py-1.5 text-sm font-semibold text-white hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-60"
        :disabled="running"
        @click="runQuery"
      >
        {{ running ? 'Running…' : 'Run' }}
      </button>
      <span class="text-xs text-text-muted">⌘/Ctrl-Enter</span>
      <label class="flex items-center gap-1.5 text-sm text-text-secondary" title="Reuse a prior identical query's result; uncheck to force a fresh run">
        <input type="checkbox" v-model="useCache" />
        use cached results
      </label>
      <span v-if="phase === 'submitting'" class="text-sm text-text-muted">Submitting…</span>
      <span v-else-if="phase === 'running'" class="text-sm text-text-muted">Running…</span>
    </div>

    <p
      v-if="error"
      class="whitespace-pre-wrap rounded-lg border border-status-danger/40 bg-status-danger/10 px-3 py-2 font-mono text-[13px] text-status-danger"
    >
      {{ error }}
    </p>

    <template v-if="phase === 'done' && result">
      <StatusBar :result="result" :ttl-days="ttlDays" />
      <ResultTable :columns="result.columns" :rows="result.rows" />
    </template>
  </div>
</template>
