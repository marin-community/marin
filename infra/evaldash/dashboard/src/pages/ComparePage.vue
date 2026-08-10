<script setup lang="ts">
/**
 * Head-to-head over the benchmarks the selected models share.
 *
 * Both the shared-panel score and each per-benchmark gap come from the server's statistics engine,
 * so a difference on screen is an interval rather than two bars a reader is invited to eyeball. This
 * page used to average whatever cells it had client-side and treat a missing shared cell as zero,
 * which turned a coverage gap into a score.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatCoverage, formatDelta, formatInterval, formatScore } from '@/utils/formatting'
import { scoreTint } from '@/utils/score'
import { isPartialCoverage } from '@/utils/panel'
import { FACETS, MAX_COMPARE } from '@/constants'
import type { Comparison, ComparisonRow, Meta, PanelCell } from '@/types/api'
import EmptyState from '@/components/shared/EmptyState.vue'
import ModelCompareChart from '@/components/charts/ModelCompareChart.vue'

const route = useRoute()
const router = useRouter()

const selected = ref<string[]>([])
const comparing = computed(() => selected.value.length >= 2)

// The panel's selection travels in the route, so a comparison launched from a narrowed panel keeps
// its benchmark set, cohort, and filters. Anything else in the query string is ignored.
const SELECTION_PARAMS = ['benchmarks', 'cohort', 'complete', 'min_coverage', ...FACETS] as const

const { data, error, refresh } = useApi<Comparison>(() => {
  const params = new URLSearchParams({ models: selected.value.join(',') })
  for (const name of SELECTION_PARAMS) {
    const raw = route.query[name]
    const value = Array.isArray(raw) ? raw[0] : raw
    if (value) params.set(name, value)
  }
  return `api/compare?${params.toString()}`
})
const { data: meta, refresh: refreshMeta } = useApi<Meta>(() => 'api/meta')

function fromQuery(): string[] {
  const raw = route.query.models
  const csv = Array.isArray(raw) ? raw[0] : raw
  return (csv ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .slice(0, MAX_COMPARE)
}

function load() {
  if (comparing.value) refresh()
}

onMounted(() => {
  selected.value = fromQuery()
  load()
  refreshMeta()
})
watch(
  () => route.query,
  () => {
    selected.value = fromQuery()
    load()
  },
  { deep: true },
)
onViewRefresh(() => {
  load()
  refreshMeta()
})

// Keep the inherited selection in the URL when the model set changes: it is what the comparison is
// computed over, and dropping it would silently switch to a different question mid-session.
function syncQuery() {
  const query = { ...route.query }
  if (selected.value.length) query.models = selected.value.join(',')
  else delete query.models
  router.replace({ path: '/compare', query })
}
function toggle(model: string) {
  const at = selected.value.indexOf(model)
  if (at >= 0) selected.value.splice(at, 1)
  else if (selected.value.length < MAX_COMPARE) selected.value.push(model)
  syncQuery()
  load()
}

const shared = computed(() => data.value?.shared ?? [])

// Ranked by the shared-panel aggregate's interval lower bound, like every other ordering in the app.
// A model missing one of the shared benchmarks has no aggregate and is not ranked.
const ranking = computed(() =>
  selected.value
    .map((model) => ({ model, aggregate: data.value?.aggregates[model] ?? null }))
    .filter((entry) => entry.aggregate !== null)
    .sort((a, b) => b.aggregate!.low - a.aggregate!.low),
)
const leader = computed(() => ranking.value[0] ?? null)

function cell(row: ComparisonRow, model: string): PanelCell | undefined {
  return row.cells[model]
}
function orderingLabel(row: ComparisonRow): string {
  const models = Object.keys(row.cells)
  if (models.length < 2) return 'one model only'
  return Object.values(row.differences).every((d) => d.separated) ? 'separated' : 'intervals overlap'
}
function gapLabel(row: ComparisonRow, model: string): string {
  const difference = row.differences[model]
  if (!difference) return ''
  return `${formatDelta(-difference.high)} to ${formatDelta(-difference.low)} vs ${row.leader}`
}

// The compared models over the union of their benchmarks, for the grouped bar chart.
const chartSeries = computed(() =>
  selected.value.map((model) => ({
    model,
    cells: Object.fromEntries(
      (data.value?.rows ?? []).filter((row) => row.cells[model]).map((row) => [row.benchmark, row.cells[model]]),
    ),
  })),
)
</script>

<template>
  <section>
    <div class="mb-4">
      <h2 class="text-lg font-semibold">Compare</h2>
      <p class="text-xs text-text-muted mt-0.5">
        Pick 2–{{ MAX_COMPARE }} models. The ranking scores them on their shared benchmarks only, and every gap comes
        with an interval, so neither a coverage difference nor sampling noise reads as a lead.
      </p>
    </div>

    <!-- model picker -->
    <div class="rounded-lg border border-surface-border bg-surface p-4 mb-5">
      <div class="font-mono text-[10px] uppercase tracking-widest text-text-muted mb-2">
        Models ({{ selected.length }}/{{ MAX_COMPARE }})
      </div>
      <div class="flex flex-wrap gap-2">
        <button
          v-for="m in meta?.models ?? []"
          :key="m"
          class="font-mono text-xs px-2.5 py-1 rounded-full border"
          :class="
            selected.includes(m)
              ? 'border-accent bg-accent-subtle text-text'
              : selected.length >= MAX_COMPARE
                ? 'border-surface-border-subtle text-text-muted opacity-50 cursor-not-allowed'
                : 'border-surface-border text-text-secondary hover:bg-surface-raised'
          "
          :disabled="!selected.includes(m) && selected.length >= MAX_COMPARE"
          @click="toggle(m)"
        >
          {{ m }}
        </button>
      </div>
    </div>

    <div
      v-if="error"
      class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2 mb-4"
    >
      {{ error }}
    </div>

    <EmptyState v-if="!comparing" icon="⚖" message="Pick at least two models to compare." />

    <div v-else-if="data" class="space-y-6">
      <!-- shared-benchmark ranking -->
      <div>
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">
          Shared-benchmark ranking
          <span class="font-normal normal-case text-text-muted">
            — {{ shared.length }} shared: <span class="font-mono">{{ shared.join(' · ') || 'none' }}</span>
          </span>
        </h3>
        <div
          v-if="!shared.length"
          class="text-sm text-text-muted rounded-lg border border-surface-border bg-surface p-4"
        >
          These models have no benchmark in common, so there is nothing to rank them on. The per-benchmark table below
          still shows where each one has run.
        </div>
        <template v-else>
          <div class="rounded-lg border border-surface-border overflow-hidden bg-surface">
            <div
              v-for="(entry, i) in ranking"
              :key="entry.model"
              class="flex items-center gap-3 px-4 py-2.5 border-b border-surface-border-subtle last:border-b-0"
            >
              <span class="font-mono text-text-muted tabular-nums w-5">{{ i + 1 }}</span>
              <button
                class="font-mono text-[13px] font-semibold text-accent hover:underline"
                @click="router.push(`/models/${encodeURIComponent(entry.model)}`)"
              >
                {{ entry.model }}
              </button>
              <span class="ml-auto text-right">
                <span class="font-mono text-base font-semibold tabular-nums">{{
                  formatScore(entry.aggregate!.value)
                }}</span>
                <span class="block font-mono text-[10px] text-text-muted tabular-nums leading-none">
                  {{ formatInterval(entry.aggregate!.low, entry.aggregate!.high) }}
                </span>
              </span>
              <span class="font-mono text-xs text-text-muted tabular-nums w-16 text-right">
                {{ i === 0 || !leader ? '' : formatDelta(entry.aggregate!.value - leader.aggregate!.value) }}
              </span>
            </div>
          </div>
          <p class="text-xs text-text-muted mt-2 leading-relaxed">
            Unweighted mean over the {{ shared.length }} shared benchmarks, each at its own primary metric, ranked by
            the interval's lower bound. The interval covers item sampling and any items a run attempted but never
            graded; it does not cover run-to-run variation, so a gap inside a few points is not a result.
          </p>
        </template>
      </div>

      <!-- per-benchmark table -->
      <div>
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Per-benchmark</h3>
        <div class="overflow-x-auto rounded-lg border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead>
              <tr
                class="border-b border-surface-border bg-surface-raised text-xs font-semibold uppercase tracking-wider text-text-secondary"
              >
                <th class="px-3 py-2 text-left">Benchmark</th>
                <th v-for="m in selected" :key="m" class="px-3 py-2 text-center font-mono normal-case">{{ m }}</th>
                <th class="px-3 py-2 text-left">Ordering</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in data.rows" :key="row.benchmark" class="border-b border-surface-border-subtle">
                <td class="px-3 py-2 font-mono text-[13px] whitespace-nowrap">
                  {{ row.benchmark }}
                  <span v-if="row.shared" class="ml-1 text-[10px] font-sans text-text-muted">shared</span>
                </td>
                <td v-for="m in selected" :key="m" class="p-1 text-center">
                  <div
                    v-if="cell(row, m)"
                    class="rounded px-2 py-1.5 leading-tight"
                    :class="row.leader === m ? 'ring-2 ring-inset' : ''"
                    :style="{
                      backgroundColor: scoreTint(cell(row, m)!.value),
                      ...(row.leader === m ? { '--tw-ring-color': 'var(--c-best)' } : {}),
                    }"
                    :title="`${cell(row, m)!.metric} · ${cell(row, m)!.n_scored} items graded`"
                  >
                    <span class="font-mono font-medium tabular-nums">{{ formatScore(cell(row, m)!.value) }}</span>
                    <span class="block font-mono text-[10px] text-text-muted tabular-nums leading-none">
                      {{ formatInterval(cell(row, m)!.low, cell(row, m)!.high) }}
                    </span>
                    <span
                      v-if="isPartialCoverage(cell(row, m)!)"
                      class="block font-mono text-[9px] leading-none text-status-warning"
                      >{{ formatCoverage(cell(row, m)!.coverage) }}</span
                    >
                    <span
                      v-else-if="row.differences[m]"
                      class="block font-mono text-[9px] leading-none text-text-muted"
                      :title="`95% interval for this model's gap to ${row.leader}`"
                      >{{ gapLabel(row, m) }}</span
                    >
                  </div>
                  <span v-else class="text-text-muted">—</span>
                </td>
                <td class="px-3 py-2 text-xs whitespace-nowrap">
                  <span :class="orderingLabel(row) === 'separated' ? 'text-text-secondary' : 'text-text-muted'">
                    {{ orderingLabel(row) }}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p class="text-xs text-text-muted mt-2 leading-relaxed">
          Under each score is the 95% interval for that model's gap to the benchmark's leader. Separated means every
          such interval clears zero, so the ordering on that benchmark holds at the 5% level. An interval spanning zero
          is not evidence of a tie: it means these runs cannot resolve the difference, and at the panel's coverage floor
          the ungraded items alone can be wide enough to prevent it.
        </p>
      </div>

      <!-- grouped bars -->
      <div>
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Per-benchmark bars</h3>
        <ModelCompareChart :benchmarks="data.benchmarks" :series="chartSeries" :models="selected" />
      </div>
    </div>
  </section>
</template>
