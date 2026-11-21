<script setup lang="ts">
/**
 * Entry point to the per-sample viewer that is not tied to one run: filter runs by model and
 * benchmark, then open any run's samples. The viewer itself lives at /runs/:id/samples.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatScore, formatTimestamp } from '@/utils/formatting'
import type { Meta, RunRow } from '@/types/api'
import StatusChip from '@/components/shared/StatusChip.vue'
import EmptyState from '@/components/shared/EmptyState.vue'

const router = useRouter()
const model = ref('')
const evalName = ref('')

const { data: meta, refresh: refreshMeta } = useApi<Meta>(() => 'api/meta')

function runsPath(): string {
  const params = new URLSearchParams()
  if (model.value) params.set('model', model.value)
  if (evalName.value) params.set('eval', evalName.value)
  params.set('limit', '200')
  return `api/runs?${params.toString()}`
}
const { data: runs, loading, refresh } = useApi<RunRow[]>(runsPath)

onMounted(() => {
  refreshMeta()
  refresh()
})
watch([model, evalName], refresh)
onViewRefresh(() => {
  refreshMeta()
  refresh()
})

// Samples are exported per run; succeeded runs are the ones worth inspecting first.
const rows = computed(() => (runs.value ?? []).filter((r) => r.status === 'succeeded'))

function openSamples(run: RunRow) {
  router.push(`/runs/${run.run_id}/samples`)
}
</script>

<template>
  <section>
    <div class="mb-4">
      <h2 class="text-lg font-semibold">Inspect</h2>
      <p class="text-xs text-text-muted mt-0.5">
        Open the per-question viewer for any run — its prompts, the model's answers, the grading, and (for agentic
        evals) the full step trajectory.
      </p>
    </div>

    <div class="flex flex-wrap items-end gap-4 mb-4">
      <label class="flex flex-col text-xs text-text-secondary gap-1">
        Model
        <select v-model="model" class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[12rem]">
          <option value="">All</option>
          <option v-for="m in meta?.models ?? []" :key="m" :value="m">{{ m }}</option>
        </select>
      </label>
      <label class="flex flex-col text-xs text-text-secondary gap-1">
        Benchmark
        <select v-model="evalName" class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[12rem]">
          <option value="">All</option>
          <option v-for="e in meta?.evals ?? []" :key="e" :value="e">{{ e }}</option>
        </select>
      </label>
      <span class="text-xs text-text-muted ml-auto">{{ rows.length }} succeeded runs</span>
    </div>

    <div v-if="loading && !runs" class="text-sm text-text-muted py-12 text-center">Loading…</div>
    <EmptyState v-else-if="!rows.length" icon="🔬" message="No runs match. Widen the filter." />

    <div v-else class="rounded-lg border border-surface-border overflow-hidden bg-surface">
      <button
        v-for="run in rows"
        :key="run.run_id"
        class="w-full flex items-center gap-4 px-4 py-2.5 border-b border-surface-border-subtle last:border-b-0 hover:bg-surface-raised text-left"
        @click="openSamples(run)"
      >
        <span class="font-mono text-[13px] font-semibold w-40 truncate">{{ run.model_name }}</span>
        <span v-if="run.version" class="font-mono text-[11px] text-text-muted">{{ run.version }}</span>
        <span class="font-mono text-[13px] w-36 truncate">{{ run.eval_name }}</span>
        <StatusChip :status="run.status" />
        <span class="font-mono text-[11px] text-text-muted ml-auto">{{ formatTimestamp(run.created_at) }}</span>
        <span class="text-accent text-xs whitespace-nowrap">open samples →</span>
      </button>
    </div>
  </section>
</template>
