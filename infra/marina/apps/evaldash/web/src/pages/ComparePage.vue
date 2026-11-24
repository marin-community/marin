<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import BenchmarkFocus from '@/components/comparison/BenchmarkFocus.vue'
import ModelName from '@/components/shared/ModelName.vue'
import PolicyRejections from '@/components/shared/PolicyRejections.vue'
import SearchSelect from '@/components/shared/SearchSelect.vue'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { FACETS, MAX_COMPARE } from '@/constants'
import type { Comparison, ComparisonRow, Meta } from '@/types/api'
import { formatMeasurement } from '@/utils/formatting'
import { ALL_COHORTS, UNVERSIONED_COHORT, cohortWarning } from '@/utils/panel'

const PAGE_SIZE = 12
const MODEL_OPTION_LIMIT = 12
const SELECTION_PARAMS = [
  'benchmarks', 'cohort', 'complete', 'min_coverage', 'min_benchmark_coverage', 'include_flagged', ...FACETS,
] as const

type ComparisonModels = Pick<Meta, 'models' | 'default_cohort' | 'verified_cohorts' | 'versions'>
type RowScope = 'all' | 'shared'

const route = useRoute()
const router = useRouter()

function param(key: string): string {
  const value = route.query[key]
  return (Array.isArray(value) ? value[0] : value) ?? ''
}

function names(key: string): string[] {
  return [...new Set(param(key).split(',').map((name) => name.trim()).filter(Boolean))]
}

function comparisonLocation(values: Record<string, string>) {
  const query = { ...route.query }
  for (const [key, value] of Object.entries(values)) {
    if (value) query[key] = value
    else delete query[key]
  }
  return { name: 'compare', query }
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
const {
  data: catalog,
  loading: catalogLoading,
  error: catalogError,
  refresh: refreshCatalog,
} = useApi<ComparisonModels>(() => `api/comparison-models?${selectionQuery.value}`)
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

const desiredModels = computed(() => names('models'))
const modelSet = computed(() => new Set(catalog.value?.models ?? []))
const selected = computed(() => desiredModels.value.filter((model) => modelSet.value.has(model)).slice(0, MAX_COMPARE))
const remaining = computed(() => (catalog.value?.models ?? []).filter((model) => !selected.value.includes(model)))
const excludedModels = computed(() => desiredModels.value.filter((model) => !modelSet.value.has(model)))
const comparisonPath = computed(() => selected.value.length < 2
  ? ''
  : `api/compare?${selectionQuery.value}&models=${encodeURIComponent(selected.value.join(','))}`,
)
const {
  data: comparison,
  loading: comparisonLoading,
  error: comparisonError,
  refresh: refreshComparison,
} = useApi<Comparison>(() => comparisonPath.value)

watch(selectionQuery, () => {
  catalog.value = null
  void refreshCatalog()
}, { immediate: true })
watch(comparisonPath, (path) => {
  comparison.value = null
  if (path) void refreshComparison()
})
watch(() => catalog.value?.default_cohort, (value) => {
  if (value && !param('cohort')) void router.replace(comparisonLocation({ cohort: value }))
})
onViewRefresh(() => {
  void refreshCatalog()
  if (comparisonPath.value) void refreshComparison()
})

function updateModels(models: string[]) {
  void router.replace(comparisonLocation({ models: models.join(','), bpage: '', focus: '' }))
}

function addModel(model: string) {
  if (selected.value.length < MAX_COMPARE && remaining.value.includes(model)) {
    editModels.value = true
    updateModels([...selected.value, model])
  }
}

function modelCoverage(row: ComparisonRow): number {
  return Object.keys(row.cells).length
}

const rows = computed(() => [...(comparison.value?.rows ?? [])].sort((a, b) =>
  modelCoverage(b) - modelCoverage(a) || a.benchmark.localeCompare(b.benchmark),
))
const sharedRows = computed(() => rows.value.filter((row) => row.shared))
const scope = computed<RowScope>(() => {
  const requested = param('show')
  if (requested === 'shared' || requested === 'all') return requested
  return sharedRows.value.length ? 'shared' : 'all'
})
const query = computed(() => param('bq'))
const terms = computed(() => query.value.toLowerCase().trim().split(/\s+/).filter(Boolean))
const filteredRows = computed(() => (scope.value === 'shared' ? sharedRows.value : rows.value).filter((row) => {
  const label = row.benchmark.toLowerCase()
  return terms.value.every((term) => label.includes(term))
}))
const pageCount = computed(() => Math.max(1, Math.ceil(filteredRows.value.length / PAGE_SIZE)))
const page = computed(() => Math.min(Math.max(1, Number.parseInt(param('bpage'), 10) || 1), pageCount.value))
const offset = computed(() => (page.value - 1) * PAGE_SIZE)
const pageRows = computed(() => filteredRows.value.slice(offset.value, offset.value + PAGE_SIZE))
const focusName = computed(() => param('focus'))
const focusedRow = computed(() => focusName.value
  ? rows.value.find((row) => row.benchmark === focusName.value)
  : pageRows.value[0],
)
const benchmarkDetail = ref<HTMLElement | null>(null)

function benchmarkLinkId(benchmark: string): string {
  return `compare-${encodeURIComponent(benchmark)}`
}

watch(focusName, async (next, previous) => {
  await nextTick()
  const narrow = window.matchMedia('(max-width: 800px)').matches
  if (next) {
    benchmarkDetail.value?.focus({ preventScroll: true })
    if (narrow) benchmarkDetail.value?.scrollIntoView({ block: 'start' })
    return
  }
  if (!narrow) return
  const source = document.getElementById(benchmarkLinkId(previous))
  source?.focus({ preventScroll: true })
  source?.scrollIntoView({ block: 'nearest' })
})

function setScope(show: RowScope) {
  if (show !== scope.value) void router.push(comparisonLocation({ show, bpage: '' }))
}

function setPage(next: number) {
  void router.push(comparisonLocation({ bpage: next > 1 ? String(next) : '' }))
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
          <select :value="cohort" @change="router.push(comparisonLocation({ cohort: ($event.target as HTMLSelectElement).value, bpage: '' }))">
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
            @click="router.push(comparisonLocation({ benchmarks: '', bpage: '', focus: '' }))"
          >Clear benchmark scope</button>
        </div>
      </div>
      <div v-show="editorOpen || catalogError" id="comparison-model-editor" class="selection-controls">
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
          <div v-if="catalogError" class="comparison-message" role="alert">
            <p>Couldn’t load model choices. {{ catalogError }}</p>
            <button type="button" @click="refreshCatalog">Retry model choices</button>
          </div>
          <p v-else-if="catalogLoading && !catalog" class="comparison-message" role="status">Loading model choices…</p>
          <SearchSelect
            v-else-if="selected.length < MAX_COMPARE && remaining.length"
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
          <p v-else-if="!selected.length" class="comparison-message">No eligible models in this selection.</p>
          <button v-if="desiredModels.length" type="button" class="text-link" @click="updateModels([])">Clear models</button>
          <details v-if="catalog && excludedModels.length" class="excluded-models">
            <summary>{{ excludedModels.length }} {{ excludedModels.length === 1 ? 'model unavailable' : 'models unavailable' }} in this selection</summary>
            <ul>
              <li v-for="model in excludedModels" :key="model"><ModelName :model="model" /></li>
            </ul>
          </details>
        </div>
      </div>
      <p v-if="warning" class="selection-warning">{{ warning }}</p>
    </div>

    <div class="comparison-results">
      <div v-if="comparisonPath && comparisonError" class="result-state" role="alert">
        <p>Couldn’t load benchmark results. {{ comparisonError }}</p>
        <button type="button" @click="refreshComparison">Retry results</button>
      </div>
      <div v-else-if="catalog && selected.length < 2" class="result-state">Pick at least two models to compare.</div>
      <div v-else-if="comparisonPath && !comparison" class="result-state" role="status">Loading benchmark results…</div>
      <template v-else-if="comparison && comparisonPath">
        <div class="benchmark-workspace" :class="{ 'benchmark-open': focusName }" :aria-busy="comparisonLoading">
          <section class="benchmark-index" aria-label="Benchmark index">
            <div class="index-controls">
              <h2>Benchmarks</h2>
              <label class="benchmark-search">
                <span class="sr-only">Search benchmarks</span>
                <input
                  type="search"
                  placeholder="Search benchmarks"
                  :value="query"
                  @input="router.replace(comparisonLocation({ bq: ($event.target as HTMLInputElement).value, bpage: '', focus: '' }))"
                />
              </label>
              <div class="scope-switch" role="group" aria-label="Benchmark coverage">
                <button type="button" :aria-pressed="scope === 'shared'" aria-label="Shared only" @click="setScope('shared')">
                  Shared only <span>{{ sharedRows.length }}</span>
                </button>
                <button type="button" :aria-pressed="scope === 'all'" aria-label="All recorded" @click="setScope('all')">
                  All recorded <span>{{ rows.length }}</span>
                </button>
              </div>
            </div>
            <p v-if="rows.length && !sharedRows.length" class="index-note">No setting has scores for every selected model.</p>
            <ul v-if="pageRows.length" class="benchmark-list" aria-label="Benchmark settings">
              <li v-for="row in pageRows" :key="row.benchmark">
                <RouterLink
                  :id="benchmarkLinkId(row.benchmark)"
                  :to="comparisonLocation({ focus: row.benchmark })"
                  :aria-current="focusedRow?.benchmark === row.benchmark ? 'true' : undefined"
                  :aria-description="modelCoverage(row) + ' of ' + selected.length + ' models scored'"
                  class="benchmark-link"
                >
                  <span>{{ row.benchmark }}</span>
                  <span class="benchmark-coverage" aria-hidden="true">{{ modelCoverage(row) }}/{{ selected.length }} scored</span>
                  <span class="benchmark-chevron" aria-hidden="true">›</span>
                </RouterLink>
              </li>
            </ul>
            <p v-else class="index-note">{{ !rows.length ? 'No results in this selection.' : 'No settings match.' }}</p>
            <div class="results-range">
              <p role="status">
                <template v-if="pageRows.length">
                  {{ offset + 1 }}–{{ offset + pageRows.length }} of {{ filteredRows.length }} {{ filteredRows.length === 1 ? 'setting' : 'settings' }}
                </template>
                <template v-else>0 settings</template>
              </p>
              <nav v-if="pageCount > 1" class="compare-pagination" aria-label="Benchmark result pages">
                <button type="button" aria-label="Previous" :disabled="page === 1" @click="setPage(page - 1)">‹</button>
                <button type="button" aria-label="Next" :disabled="page === pageCount" @click="setPage(page + 1)">›</button>
              </nav>
            </div>
          </section>

          <div ref="benchmarkDetail" class="benchmark-detail" tabindex="-1">
            <RouterLink v-if="focusName" :to="comparisonLocation({ focus: '' })" class="back-link">
              <span aria-hidden="true">←</span> Back to benchmarks
            </RouterLink>
            <BenchmarkFocus v-if="focusedRow" :key="focusedRow.benchmark" :row="focusedRow" :models="selected" />
            <div v-else class="result-state">{{ focusName ? 'No results for this setting in the current selection.' : 'Choose a benchmark setting to see its scores.' }}</div>
          </div>
        </div>

        <details v-if="ranking.length" class="comparison-notes">
          <summary>Shared-benchmark aggregate</summary>
          <p>Unweighted mean over {{ sharedRows.length }} shared benchmark settings, ranked by the lower bound of the 95% interval.</p>
          <ol>
            <li v-for="{ model, aggregate } in ranking" :key="model">
              <ModelName :model="model" />
              <span class="tabular-nums">{{ formatMeasurement(aggregate.value, false) }} <span class="text-text-secondary">({{ formatMeasurement(aggregate.low, false) }}–{{ formatMeasurement(aggregate.high, false) }})</span></span>
            </li>
          </ol>
        </details>
        <PolicyRejections :rejections="comparison.policy_rejections" scope="this comparison" class="mt-3" />
      </template>
    </div>
  </section>
</template>

<style scoped>
.compare-workspace {
  padding-block: clamp(0.5rem, 2vw, 1.75rem);
}

h1 {
  margin: 0 0 1.25rem;
  color: var(--c-text);
  font-size: clamp(1.75rem, 3vw, 2.5rem);
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
  gap: 0.6rem 1.5rem;
  padding: 0.65rem 1rem;
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
.benchmark-link:hover,
.back-link:hover {
  text-decoration: underline;
  text-underline-offset: 0.16em;
}

.selection-warning {
  padding: 0.4rem 1rem;
  border-top: 1px solid var(--c-status-warning-border);
  border-radius: 0 0 0.75rem 0.75rem;
  color: var(--c-status-warning);
  background: var(--c-status-warning-bg);
  font-size: 0.78rem;
}

.excluded-models,
.comparison-message {
  color: var(--c-text-secondary);
  font-size: 0.8rem;
}

.comparison-results {
  margin-top: 1rem;
}

.benchmark-workspace {
  display: grid;
  grid-template-columns: minmax(16rem, 22rem) minmax(0, 1fr);
  gap: 1rem;
  align-items: start;
}

.benchmark-index {
  min-width: 0;
  background: var(--c-surface);
  border: 1px solid var(--c-surface-border);
  border-radius: 0.75rem;
}

.index-controls {
  display: grid;
  gap: 0.6rem;
  padding: 0.85rem 1rem;
}

.index-controls h2 {
  font-size: 1.05rem;
  font-weight: 650;
}

.benchmark-search input {
  width: 100%;
  min-height: 2.4rem;
  font-size: 0.875rem;
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

.benchmark-list {
  border-top: 1px solid var(--c-surface-border);
}

.benchmark-list li + li {
  border-top: 1px solid var(--c-surface-border-subtle);
}

.benchmark-link {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  align-items: center;
  min-height: 2.6rem;
  gap: 0.5rem;
  padding: 0.4rem 1rem;
  color: var(--c-text);
  font-size: 0.8125rem;
  overflow-wrap: anywhere;
}

.benchmark-link:hover {
  background: var(--c-surface-raised);
}

.benchmark-link[aria-current='true'] {
  box-shadow: inset 3px 0 var(--c-accent);
  background: var(--c-accent-subtle);
  color: var(--c-accent);
  font-weight: 650;
}

.benchmark-coverage {
  color: var(--c-text-secondary);
  font: 0.7rem var(--mono);
  white-space: nowrap;
}

.benchmark-chevron {
  color: var(--c-accent);
  font-size: 1rem;
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
</style>
