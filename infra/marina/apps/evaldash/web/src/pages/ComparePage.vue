<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import BenchmarkFocus from '@/components/comparison/BenchmarkFocus.vue'
import BenchmarkOverview from '@/components/comparison/BenchmarkOverview.vue'
import { benchmarkPage, orderBenchmarks, type BenchmarkScope } from '@/components/comparison/benchmarkIndex'
import { useComparisonData } from '@/components/comparison/useComparisonData'
import CohortNotice from '@/components/shared/CohortNotice.vue'
import ModelName from '@/components/shared/ModelName.vue'
import PolicyRejections from '@/components/shared/PolicyRejections.vue'
import SearchSelect from '@/components/shared/SearchSelect.vue'
import { useRouteQuery } from '@/composables/useRouteQuery'
import { FACETS, MAX_COMPARE } from '@/constants'
import { formatInterval, formatScore } from '@/utils/formatting'
import { ALL_COHORTS, UNVERSIONED_COHORT, cohortWarning } from '@/utils/panel'

const MODEL_OPTION_LIMIT = 12
const SELECTION_PARAMS = [
  'benchmarks', 'cohort', 'complete', 'min_coverage', 'min_benchmark_coverage', 'include_flagged', ...FACETS,
] as const

const route = useRoute()
const { location, pushQuery, replaceQuery } = useRouteQuery()

function param(key: string): string {
  const value = route.query[key]
  return (Array.isArray(value) ? value[0] : value) ?? ''
}

function names(key: string): string[] {
  return [...new Set(param(key).split(',').map((name) => name.trim()).filter(Boolean))]
}

const cohort = computed(() => param('cohort') || ALL_COHORTS)
const selectionQuery = computed(() => {
  const params = new URLSearchParams()
  for (const name of SELECTION_PARAMS) {
    const value = name === 'cohort' ? cohort.value : param(name)
    if (value) params.set(name, value)
  }
  return params.toString()
})
const desiredModels = computed(() => names('models'))
const { choices, results, selected, remaining, excluded, comparing } = useComparisonData(selectionQuery, desiredModels)
const { data: catalog, loading: catalogLoading, error: catalogError, refresh: refreshCatalog } = choices
const { data: comparison, loading: comparisonLoading, error: comparisonError, refresh: refreshComparison } = results

const cohortOptions = computed(() => [...new Set([
  ALL_COHORTS,
  ...[...(catalog.value?.versions ?? [])].reverse(),
  cohort.value,
])])
const warning = computed(() => catalog.value ? cohortWarning(cohort.value, catalog.value) : null)
const benchmarks = computed(() => names('benchmarks'))
const editModels = ref<boolean | null>(null)
const editorOpen = computed(() => editModels.value ?? (catalog.value ? selected.value.length < 2 : desiredModels.value.length < 2))

function cohortLabel(value: string) {
  if (value === ALL_COHORTS) return 'All cohorts'
  if (value === UNVERSIONED_COHORT) return 'Unversioned'
  return value
}

watch(() => catalog.value?.default_cohort, (value) => {
  if (value && !param('cohort')) void replaceQuery({ cohort: value })
})

function updateModels(models: string[]) {
  void replaceQuery({ models: models.join(','), bpage: '', focus: '' })
}

function addModel(model: string) {
  if (selected.value.length < MAX_COMPARE && remaining.value.includes(model)) {
    editModels.value = true
    updateModels([...selected.value, model])
  }
}

const ordered = computed(() => orderBenchmarks(comparison.value?.rows ?? []))
const benchmarkSearch = computed(() => param('benchmarkSearch'))
const focusName = computed(() => param('focus'))
const index = computed(() => benchmarkPage(ordered.value, {
  scope: param('show'), search: benchmarkSearch.value, page: param('bpage'), focus: focusName.value,
}))
const benchmarkIndex = ref<HTMLElement | null>(null)
const benchmarkDetail = ref<HTMLElement | null>(null)

function benchmarkLinkId(benchmark: string): string {
  return `compare-${encodeURIComponent(benchmark)}`
}

function benchmarkLocation(benchmark: string) {
  return { id: benchmarkLinkId(benchmark), to: location({ focus: benchmark }) }
}

watch(focusName, async (next, previous) => {
  await nextTick()
  if (next) {
    benchmarkDetail.value?.focus({ preventScroll: true })
    if (benchmarkIndex.value?.getClientRects().length === 0) benchmarkDetail.value?.scrollIntoView({ block: 'start' })
    return
  }
  if (benchmarkDetail.value?.getClientRects().length !== 0) return
  const source = document.getElementById(benchmarkLinkId(previous))
  source?.focus({ preventScroll: true })
  source?.scrollIntoView({ block: 'nearest' })
})

function setScope(show: BenchmarkScope) {
  if (show !== index.value.scope) void pushQuery({ show, bpage: '', focus: '' })
}

function setPage(next: number) {
  void pushQuery({ bpage: next > 1 ? String(next) : '' })
}

const ranking = computed(() => selected.value
  .flatMap((model) => {
    const aggregate = comparison.value?.aggregates[model]
    return aggregate ? [{ model, aggregate }] : []
  })
  .sort((a, b) => b.aggregate.low - a.aggregate.low),
)
</script>

<template>
  <section class="compare-workspace" aria-labelledby="compare-heading">
    <h1 id="compare-heading">Compare models</h1>

    <div class="comparison-selection">
      <div class="selection-toolbar">
        <button
          type="button"
          class="model-edit"
          aria-label="Edit models"
          aria-controls="comparison-model-editor"
          :aria-expanded="editorOpen"
          @click="editModels = !editorOpen"
        >Edit models <span v-if="selected.length">{{ selected.length }}</span></button>
        <label class="cohort-control">
          Cohort
          <select :value="cohort" @change="pushQuery({ cohort: ($event.target as HTMLSelectElement).value, bpage: '' })">
            <option v-for="version in cohortOptions" :key="version" :value="version">{{ cohortLabel(version) }}</option>
          </select>
        </label>
        <div class="benchmark-scope">
          <span>Benchmark scope</span>
          <p v-if="benchmarks.length <= 1">{{ benchmarks[0] || 'All settings' }}</p>
          <details v-else>
            <summary>{{ benchmarks.length }} settings</summary>
            <ul>
              <li v-for="benchmark in benchmarks" :key="benchmark">{{ benchmark }}</li>
            </ul>
          </details>
          <button
            v-if="benchmarks.length"
            type="button"
            class="text-link"
            @click="pushQuery({ benchmarks: '', bpage: '', focus: '' })"
          >Clear benchmark scope</button>
        </div>
      </div>
      <div v-if="catalogError" class="selection-controls comparison-message" role="alert">
        <p>Couldn’t load model choices. {{ catalogError }}</p>
        <button type="button" @click="refreshCatalog">Retry model choices</button>
      </div>
      <div v-show="editorOpen" id="comparison-model-editor" class="selection-controls">
        <div class="picker-column">
          <div class="selected-models" aria-label="Selected models" role="group">
            <button
              v-for="model in selected"
              :key="model"
              type="button"
              :aria-label="`Remove ${model}`"
              class="model-pill"
              @click="updateModels(selected.filter((value) => value !== model))"
            >
              <ModelName :model="model" />
              <span aria-hidden="true">×</span>
            </button>
          </div>
          <p v-if="catalogLoading && !catalog" class="comparison-message" role="status">Loading model choices…</p>
          <SearchSelect
            v-else-if="!catalogError && selected.length < MAX_COMPARE && remaining.length"
            label="Add model"
            placeholder="Search models…"
            :options="remaining"
            :max-options="MODEL_OPTION_LIMIT"
            model-value=""
            @update:model-value="addModel"
          >
            <template #option="{ option }"><ModelName :model="option" /></template>
          </SearchSelect>
          <p v-else-if="selected.length === MAX_COMPARE" class="comparison-message">
            {{ MAX_COMPARE }} models selected. Remove one to add another.
          </p>
          <p v-else-if="catalog && !selected.length" class="comparison-message">No eligible models in this selection.</p>
          <button v-if="desiredModels.length" type="button" class="text-link" @click="updateModels([])">Clear models</button>
          <details v-if="catalog && excluded.length" class="excluded-models">
            <summary>{{ excluded.length }} {{ excluded.length === 1 ? 'model unavailable' : 'models unavailable' }} in this selection</summary>
            <ul>
              <li v-for="model in excluded" :key="model"><ModelName :model="model" /></li>
            </ul>
          </details>
        </div>
      </div>
      <CohortNotice v-if="warning" :message="warning" />
    </div>

    <div class="comparison-results">
      <div v-if="comparing && comparisonError" class="result-state" role="alert">
        <p>Couldn’t load benchmark results. {{ comparisonError }}</p>
        <button type="button" @click="refreshComparison">Retry results</button>
      </div>
      <div v-else-if="catalog && !comparing" class="result-state">Pick at least two models to compare.</div>
      <div v-else-if="comparing && !comparison" class="result-state" role="status">Loading benchmark results…</div>
      <template v-else-if="comparison && comparing">
        <div class="benchmark-workspace" :class="{ 'benchmark-open': focusName }" :aria-busy="comparisonLoading">
          <section ref="benchmarkIndex" class="benchmark-index" aria-label="Benchmark index">
            <div class="index-controls">
              <h2>Benchmarks</h2>
              <label class="benchmark-search">
                <span class="sr-only">Search benchmarks</span>
                <input
                  type="search"
                  placeholder="Search benchmarks"
                  :value="benchmarkSearch"
                  @input="replaceQuery({ benchmarkSearch: ($event.target as HTMLInputElement).value, bpage: '', focus: '' })"
                />
              </label>
              <div class="scope-switch" role="group" aria-label="Benchmark coverage">
                <button type="button" :aria-pressed="index.scope === 'shared'" aria-label="Shared only" @click="setScope('shared')">
                  Shared only <span>{{ index.sharedCount }}</span>
                </button>
                <button type="button" :aria-pressed="index.scope === 'all'" aria-label="All recorded" @click="setScope('all')">
                  All recorded <span>{{ index.allCount }}</span>
                </button>
              </div>
            </div>
            <p v-if="index.allCount && !index.sharedCount" class="index-note">No setting has scores for every selected model.</p>
            <BenchmarkOverview
              v-if="index.rows.length"
              :rows="index.rows"
              :models="selected"
              :active="index.focused?.benchmark"
              :location="benchmarkLocation"
            />
            <p v-else class="index-note">{{ index.allCount ? 'No settings match.' : 'No results in this selection.' }}</p>
            <div class="results-range">
              <p role="status">
                <template v-if="index.rows.length">
                  {{ index.offset + 1 }}–{{ index.offset + index.rows.length }} of {{ index.total }} {{ index.total === 1 ? 'setting' : 'settings' }}
                </template>
                <template v-else>0 settings</template>
              </p>
              <nav v-if="index.pageCount > 1" class="compare-pagination" aria-label="Benchmark result pages">
                <button type="button" aria-label="Previous" :disabled="index.page === 1" @click="setPage(index.page - 1)">‹</button>
                <button type="button" aria-label="Next" :disabled="index.page === index.pageCount" @click="setPage(index.page + 1)">›</button>
              </nav>
            </div>
          </section>

          <div ref="benchmarkDetail" class="benchmark-detail" tabindex="-1">
            <RouterLink v-if="focusName" :to="location({ focus: '' })" class="back-link">
              <span aria-hidden="true">←</span> Back to benchmarks
            </RouterLink>
            <BenchmarkFocus v-if="index.focused" :key="index.focused.benchmark" :row="index.focused" :models="selected" />
            <div v-else class="result-state">{{ focusName ? 'No results for this setting in the current selection.' : 'Choose a benchmark setting to see its scores.' }}</div>
          </div>
        </div>

        <details v-if="ranking.length" class="comparison-notes">
          <summary>Shared-benchmark aggregate</summary>
          <p>Unweighted mean over {{ index.sharedCount }} shared benchmark settings, ranked by the lower bound of the 95% interval.</p>
          <ol>
            <li v-for="{ model, aggregate } in ranking" :key="model">
              <ModelName :model="model" />
              <span class="tabular-nums">{{ formatScore(aggregate) }} <span class="text-text-secondary">({{ formatInterval(aggregate) }})</span></span>
            </li>
          </ol>
        </details>
        <PolicyRejections :rejections="comparison.policy_rejections" scope="this comparison" class="mt-3" />
      </template>
    </div>
  </section>
</template>

<style scoped>
h1 {
  margin: 0 0 0.65rem;
  color: var(--c-text);
  font-size: clamp(1.55rem, 2.5vw, 2rem);
  font-weight: 650;
  letter-spacing: -0.045em;
  line-height: 1.15;
}

.comparison-selection,
.comparison-notes,
.result-state {
  background: var(--c-surface);
  border: 1px solid var(--c-surface-border);
  border-radius: 0.75rem;
}

.selection-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: start;
  gap: 0.4rem 1rem;
  padding: 0.4rem 0.75rem;
}

.model-edit {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  min-height: 2.15rem;
  color: var(--c-accent);
  font-size: 0.8125rem;
  font-weight: 600;
}

.model-edit[aria-expanded='true'] {
  background: var(--c-accent-subtle);
  border-color: var(--c-accent-border);
}

.model-edit span {
  font-family: var(--mono);
}

.selection-controls {
  padding: 0.75rem 1rem;
  border-top: 1px solid var(--c-surface-border-subtle);
}

.picker-column {
  max-width: 48rem;
  min-width: 0;
}

.selected-models {
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
  gap: 0.375rem;
}

.selected-models:not(:empty) {
  margin-bottom: 0.65rem;
}

.model-pill {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  min-width: 0;
  max-width: 100%;
  padding: 0.3rem 0.5rem;
  background: var(--c-accent-subtle);
  border: 1px solid var(--c-accent-border);
  border-radius: 0.4rem;
  color: var(--c-accent);
  text-align: start;
  font-size: 0.8rem;
}

.model-pill:hover {
  color: var(--c-accent-hover);
  border-color: var(--c-accent-hover);
}

.picker-column :deep(input[role='combobox']) {
  min-height: 2.5rem;
}

.cohort-control,
.benchmark-scope > span {
  color: var(--c-text-secondary);
  font-size: 0.75rem;
}

.cohort-control {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  min-width: 0;
}

.cohort-control > select {
  min-width: 0;
  max-width: min(24vw, 14rem);
  min-height: 2.15rem;
  color: var(--c-text);
  font-size: 0.8rem;
}

.benchmark-scope {
  min-width: 0;
  margin-block: auto;
  overflow-wrap: anywhere;
  font-size: 0.8125rem;
}

.benchmark-scope > p,
.benchmark-scope > details {
  display: inline-block;
  margin-inline: 0.5rem;
}

.benchmark-scope > details[open] {
  display: block;
  margin-inline: 0;
}

.benchmark-scope ul,
.excluded-models ul {
  margin-top: 0.35rem;
  padding-left: 1rem;
  list-style: disc;
}

.text-link {
  background: transparent;
  border: 0;
  padding: 0.2rem 0;
  color: var(--c-accent);
  font-size: 0.75rem;
}

.text-link:hover,
.back-link:hover {
  text-decoration: underline;
  text-underline-offset: 0.16em;
}

.excluded-models,
.comparison-message {
  color: var(--c-text-secondary);
  font-size: 0.8rem;
}

.comparison-results {
  margin-top: 0.7rem;
}

.benchmark-workspace {
  display: grid;
  grid-template-columns: minmax(0, 1.18fr) minmax(0, 1fr);
  gap: 0.75rem;
  align-items: start;
}

.benchmark-index {
  min-width: 0;
  background: var(--c-surface);
  border: 1px solid var(--c-surface-border);
  border-radius: 0.75rem;
}

.index-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem 0.6rem;
  padding: 0.55rem 0.75rem 0.3rem;
}

.index-controls h2 {
  font-size: 0.96rem;
  font-weight: 650;
}

.benchmark-search {
  flex: 1 1 8.5rem;
  min-width: 0;
}

.benchmark-search input {
  width: 100%;
  min-height: 2.05rem;
  padding-block: 0.3rem;
  font-size: 0.8rem;
}

.scope-switch,
.compare-pagination {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.3rem;
}

.scope-switch button,
.compare-pagination button {
  min-height: 2rem;
  padding: 0.25rem 0.55rem;
  font-size: 0.75rem;
  font-weight: 600;
  white-space: nowrap;
}

.scope-switch button[aria-pressed='true'] {
  background: var(--c-accent-subtle);
  border-color: var(--c-accent-border);
  color: var(--c-accent);
}

.scope-switch span {
  margin-inline-start: 0.2rem;
  font-family: var(--mono);
}

.index-note,
.results-range {
  padding: 0.6rem 1rem;
  border-top: 1px solid var(--c-surface-border);
  color: var(--c-text-secondary);
  font-size: 0.8rem;
}

.results-range {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.benchmark-detail {
  min-width: 0;
  scroll-margin-top: 0.5rem;
  outline-offset: 4px;
}

.back-link {
  display: none;
  margin: 0.125rem 0 0.6rem;
  color: var(--c-accent);
  font-size: 0.875rem;
}

.result-state {
  padding: 1.5rem 1rem;
  color: var(--c-text-secondary);
  font-size: 0.875rem;
}

.comparison-notes {
  margin-top: 1rem;
  padding: 0.65rem 1rem;
  color: var(--c-text-secondary);
  font-size: 0.8rem;
}

.comparison-notes p,
.comparison-notes ol {
  margin-top: 0.5rem;
}

.comparison-notes li {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.25rem 1rem;
  padding-block: 0.3rem;
}

@media (max-width: 800px) {
  .selection-toolbar {
    gap: 0.5rem 0.75rem;
  }

  .benchmark-workspace {
    display: block;
  }

  .benchmark-workspace.benchmark-open .benchmark-index,
  .benchmark-workspace:not(.benchmark-open) .benchmark-detail {
    display: none;
  }

  .back-link {
    display: inline-block;
  }

  .benchmark-search input,
  .cohort-control select,
  .picker-column :deep(input[role='combobox']) {
    font-size: 1rem;
  }

  .cohort-control select {
    max-width: min(35vw, 13rem);
  }
}

@media (max-width: 550px) {
  .index-controls {
    justify-content: space-between;
  }

  .benchmark-search {
    flex-basis: 100%;
    order: 1;
  }
}
</style>
