<script setup lang="ts">
/**
 * The model index: one card per model, each showing how much of the benchmark panel it has covered
 * and its measurement profile. The MRR shown on each card orders completed results; the panel's
 * opt-in aggregate and Compare page provide qualified score comparisons.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { fleetBest } from '@/utils/panel'
import { meanReciprocalRanks } from '@/utils/ranking'
import type { Panel, PanelRow } from '@/types/api'
import EvalRail from '@/components/charts/EvalRail.vue'
import EmptyState from '@/components/shared/EmptyState.vue'

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
const { data, refresh } = useApi<Panel>(() => panelQuery.value)
watch(panelQuery, refresh)

onMounted(refresh)
onViewRefresh(refresh)

const tasks = computed(() => data.value?.panel ?? [])
const best = computed(() => fleetBest(data.value?.rows ?? [], tasks.value))
const meanRanks = computed(() => meanReciprocalRanks(data.value?.rows ?? [], tasks.value))

const models = computed<PanelRow[]>(() => {
  const q = query.value.trim().toLowerCase()
  return [...(data.value?.rows ?? [])]
    .filter((row) => !q || row.model.toLowerCase().includes(q))
    .sort((a, b) => {
      if (a.archived !== b.archived) return Number(a.archived) - Number(b.archived)
      const rankA = meanRanks.value.get(a.model)
      const rankB = meanRanks.value.get(b.model)
      if (rankA === undefined || rankB === undefined) return Number(rankA === undefined) - Number(rankB === undefined)
      if (rankA !== rankB) return rankB - rankA
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
        Models ordered by mean reciprocal rank (MRR) across completed results in the selected cohort. Missing
        benchmarks are omitted from the mean; coverage is shown beside it. Open one for its cohorts, history, and runs.
      </p>
    </div>

    <div class="flex flex-wrap items-center gap-4 mb-4">
      <input
        v-model="query"
        type="search"
        placeholder="Filter models…"
        class="rounded border border-surface-border bg-surface px-3 py-1.5 text-sm font-mono min-w-[16rem]"
      />
      <label class="flex items-center gap-2 text-sm text-text-secondary">
        <input v-model="showArchived" type="checkbox" class="accent-accent" />
        Show archived
      </label>
      <span class="text-xs text-text-muted ml-auto">{{ models.length }} models</span>
    </div>

    <EmptyState v-if="data && models.length === 0" icon="🔎" message="No models match." />

    <div v-else class="grid gap-3" style="grid-template-columns: repeat(auto-fill, minmax(320px, 1fr))">
      <button
        v-for="row in models"
        :key="row.model"
        class="text-left rounded-lg border border-surface-border bg-surface p-4 hover:border-accent-border hover:bg-surface-raised transition-colors"
        :class="{ 'opacity-50': row.archived }"
        @click="open(row.model)"
      >
        <div class="flex items-baseline gap-2 mb-2">
          <span class="font-mono font-semibold text-sm text-accent truncate">{{ row.model }}</span>
          <span
            class="ml-auto font-mono text-[11px] tabular-nums whitespace-nowrap"
            :class="row.covered <= 1 ? 'text-status-warning' : 'text-text-muted'"
          >
            MRR {{ meanRanks.get(row.model)?.toFixed(3) ?? '—' }} · {{ row.covered }}/{{ tasks.length }} benchmarks<span
              v-if="row.covered <= 1"
            > ⚠</span>
          </span>
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
