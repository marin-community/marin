<script setup lang="ts">
/**
 * The model index: one card per model, each showing how much of the benchmark panel it has covered
 * and its measurement profile. No card carries a single headline score — a model's quality is not
 * one number, and the card has no room for the panel and missing-data policy one would need.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { fleetBest } from '@/utils/panel'
import type { Panel, PanelRow } from '@/types/api'
import EvalRail from '@/components/charts/EvalRail.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import ModelName from '@/components/shared/ModelName.vue'

const router = useRouter()
const route = useRoute()
const showArchived = ref(false)
const query = ref('')

const panelQuery = computed(() => {
  const params = new URLSearchParams()
  const raw = route.query.cohort
  const cohort = Array.isArray(raw) ? raw[0] : raw
  if (cohort) params.set('cohort', cohort)
  if (showArchived.value) params.set('include_archived', '1')
  return `api/panel?${params.toString()}`
})
const { data, loading, error, refresh } = useApi<Panel>(() => panelQuery.value)
watch(panelQuery, refresh)

onMounted(refresh)
onViewRefresh(refresh)

const tasks = computed(() => data.value?.panel ?? [])
const best = computed(() => fleetBest(data.value?.rows ?? [], tasks.value))

const models = computed<PanelRow[]>(() => {
  const q = query.value.trim().toLowerCase()
  return [...(data.value?.rows ?? [])]
    .filter((row) => !q || row.model.toLowerCase().includes(q))
    .sort((a, b) => {
      if (a.archived !== b.archived) return Number(a.archived) - Number(b.archived)
      if (a.covered !== b.covered) return b.covered - a.covered
      return a.model.localeCompare(b.model)
    })
})

function open(model: string) {
  router.push({
    path: `/models/${encodeURIComponent(model)}`,
    query: { cohort: route.query.cohort || data.value?.request.cohort_version },
  })
}
</script>

<template>
  <section>
    <div class="mb-4">
      <h2 class="text-lg font-semibold">Models</h2>
      <p class="text-xs text-text-muted mt-0.5">
        Every model indexed, with its benchmark coverage and profile. Open one for its cohorts, history, and runs.
      </p>
    </div>

    <div class="flex flex-wrap items-center gap-4 mb-4">
      <input
        v-model="query"
        type="search"
        aria-label="Filter models"
        placeholder="Filter models…"
        class="rounded border border-surface-border bg-surface px-3 py-1.5 text-sm font-mono w-full sm:w-80 min-w-0"
      />
      <label class="flex items-center gap-2 text-sm text-text-secondary">
        <input v-model="showArchived" type="checkbox" class="accent-accent" />
        Show archived
      </label>
      <span class="text-xs text-text-muted ml-auto">{{ models.length }} models</span>
    </div>

    <p v-if="loading" class="text-sm text-text-muted" role="status">Loading models…</p>
    <div v-else-if="error" role="alert" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2">
      Could not load models: {{ error }}
      <button type="button" class="ml-3 underline" @click="refresh">Retry</button>
    </div>
    <EmptyState v-else-if="data && models.length === 0" icon="🔎" message="No models match." />

    <div v-else class="grid gap-3" style="grid-template-columns: repeat(auto-fill, minmax(min(100%, 320px), 1fr))">
      <button
        v-for="row in models"
        :key="row.model"
        :aria-label="row.model"
        class="text-left rounded-lg border border-surface-border bg-surface p-4 hover:border-accent-border hover:bg-surface-raised transition-colors"
        :class="{ 'opacity-50': row.archived }"
        @click="open(row.model)"
      >
        <div class="space-y-2 mb-2">
          <ModelName :model="row.model" class="font-semibold text-sm text-accent" />
          <span
            class="block font-mono text-[11px] tabular-nums whitespace-nowrap"
            :class="row.covered <= 1 ? 'text-status-warning' : 'text-text-muted'"
            >{{ row.covered }}/{{ tasks.length }} benchmarks<span v-if="row.covered <= 1"> ⚠</span></span
          >
        </div>
        <div class="h-1.5 w-full rounded-full bg-surface-sunken overflow-hidden">
          <span
            class="block h-full rounded-full bg-accent"
            :style="{ width: `${(row.covered / (tasks.length || 1)) * 100}%` }"
          />
        </div>
        <div class="mt-3" @click.stop>
          <EvalRail :tasks="tasks" :cells="row.cells" :missing="row.missing" :best="best" :model="row.model" size="sm" />
        </div>
      </button>
    </div>
  </section>
</template>
