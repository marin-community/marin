<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import LaunchActivity from '@/components/runs/LaunchActivity.vue'
import RunCatalogTable from '@/components/runs/RunCatalogTable.vue'
import ModelName from '@/components/shared/ModelName.vue'
import SearchSelect from '@/components/shared/SearchSelect.vue'
import { useApi } from '@/composables/useApi'
import { useCatalogPage, useCatalogQuery } from '@/composables/useCatalogQuery'
import { onViewRefresh } from '@/composables/useRefresh'
import { MIXED_LAUNCH_STATUS, type CatalogFacets, type LaunchRow, type RunRow } from '@/types/api'
import { formatPageRange } from '@/utils/pagination'

const FACETS = [
  { key: 'model', label: 'Model', searchable: true },
  { key: 'eval', label: 'Eval', searchable: true },
  { key: 'version', label: 'Version', searchable: false },
  { key: 'status', label: 'Status', searchable: false },
  { key: 'user', label: 'User', searchable: false },
  { key: 'accelerator', label: 'Accelerator', searchable: false },
] as const

type View = 'launches' | 'runs'

const { queryValue, offset, query, search, submitSearch, pushQuery, changePage } = useCatalogQuery()

const group = computed(() => queryValue('group'))
const view = computed<View>(() => queryValue('view') === 'runs' || group.value ? 'runs' : 'launches')
const unit = computed(() => view.value === 'launches' ? 'launch' : 'run')
const pluralUnit = computed(() => view.value === 'launches' ? 'launches' : 'runs')

function setView(value: View) {
  if (value === view.value) return
  void pushQuery({ view: value, group: '', offset: '' })
}

const filtersOpen = ref(false)
const {
  data: facetData,
  loading: facetsLoading,
  error: facetsError,
  refresh: refreshFacets,
} = useApi<CatalogFacets>(() => 'api/catalog/facets')

const facets = computed(() => FACETS.map((facet) => {
  const value = queryValue(facet.key)
  let options = facetData.value?.run_facets[facet.key] ?? []
  if (facet.key === 'status' && view.value === 'runs') {
    options = options.filter((status) => status !== MIXED_LAUNCH_STATUS)
  }
  return { ...facet, value, options: value && !options.includes(value) ? [value, ...options] : options }
}))
const activeFacets = computed(() => facets.value.filter(({ value }) => value))

function setFacet(key: string, value: string) {
  void pushQuery({ [key]: value, offset: '' })
}

function clearFilters() {
  void pushQuery({ ...Object.fromEntries(FACETS.map(({ key }) => [key, ''])), offset: '' })
}

function resetSearch() {
  void pushQuery({
    ...Object.fromEntries(FACETS.map(({ key }) => [key, ''])),
    q: '',
    group: '',
    offset: '',
    ...(group.value ? { view: 'runs' } : {}),
  })
}

watch(filtersOpen, (open) => {
  if (open && !facetData.value && !facetsLoading.value) void refreshFacets()
})

const limitQuery = computed(() => queryValue('limit'))

function launchLocation(groupId: string) {
  return {
    name: 'runs',
    query: { view: 'runs', group: groupId, ...(limitQuery.value ? { limit: limitQuery.value } : {}) },
  }
}

const filtered = computed(() => query.value || group.value || activeFacets.value.length > 0)

function setLimit(event: Event) {
  const input = event.target as HTMLInputElement
  const value = input.valueAsNumber
  if (!input.validity.valid || !Number.isSafeInteger(value)) {
    input.value = String(limit.value)
    return
  }
  void pushQuery({ limit: String(value), offset: '' })
}

const catalogPath = computed(() => {
  const params = new URLSearchParams({
    view: view.value,
    offset: String(offset.value),
  })
  const requestedLimit = Number(limitQuery.value)
  if (Number.isSafeInteger(requestedLimit) && requestedLimit > 0) params.set('limit', String(requestedLimit))
  if (query.value) params.set('q', query.value)
  if (group.value) params.set('group', group.value)
  for (const { key, value } of activeFacets.value) params.set(key, value)
  return `api/catalog?${params}`
})
const { data: catalog, error, loading, refresh, limit, hasNext, showPagination } =
  useCatalogPage<LaunchRow | RunRow>(() => catalogPath.value, offset)
onViewRefresh(() => {
  void refresh()
  if (filtersOpen.value && !facetsLoading.value) void refreshFacets()
})

const launchRows = computed(() =>
  (catalog.value?.rows ?? []).filter((row): row is LaunchRow => !('run_id' in row)),
)
const runRows = computed(() =>
  (catalog.value?.rows ?? []).filter((row): row is RunRow => 'run_id' in row),
)
const resultLabel = computed(() => {
  if (!catalog.value) return error.value ? '' : `Loading ${pluralUnit.value}…`
  const { total, rows } = catalog.value
  const noun = total === 1 ? unit.value : pluralUnit.value
  return `${formatPageRange(offset.value, rows.length, total)} ${noun}`
})
</script>

<template>
  <section class="runs-workspace" aria-labelledby="evaluation-runs">
    <h1 id="evaluation-runs">Evaluation runs</h1>

    <div class="catalog">
      <div class="catalog-toolbar">
        <div class="view-switch" role="group" aria-label="Run catalog view">
          <button type="button" :aria-pressed="view === 'launches'" @click="setView('launches')">Launches</button>
          <button type="button" :aria-pressed="view === 'runs'" @click="setView('runs')">All runs</button>
        </div>
        <div class="catalog-controls">
          <form class="run-search" role="search" @submit.prevent="submitSearch">
            <label class="sr-only" for="search-runs">Search runs</label>
            <input
              id="search-runs"
              type="search"
              v-model="search"
              autocomplete="off"
              placeholder="Search runs"
            />
          </form>
          <button
            type="button"
            class="catalog-button filter-toggle"
            aria-label="Filters"
            aria-controls="run-filters"
            :aria-expanded="filtersOpen"
            @click="filtersOpen = !filtersOpen"
          >
            Filters
            <span v-if="activeFacets.length" class="filter-count" aria-hidden="true">{{ activeFacets.length }}</span>
            <span class="filter-chevron" aria-hidden="true"></span>
          </button>
        </div>
        <div class="catalog-summary">
          <p class="result-count" role="status">{{ resultLabel }}</p>
          <nav v-if="showPagination" class="catalog-pagination" aria-label="Result pages">
            <button type="button" class="catalog-button" :disabled="!catalog || loading || offset === 0" @click="changePage(offset - limit)">
              <span aria-hidden="true">‹</span> Previous
            </button>
            <button type="button" class="catalog-button" :disabled="loading || !hasNext" @click="changePage(offset + limit)">
              Next <span aria-hidden="true">›</span>
            </button>
          </nav>
        </div>
      </div>

      <section v-show="filtersOpen" id="run-filters" class="facet-panel" aria-label="Run filters">
        <div v-if="facetsError" class="catalog-error" role="alert">
          <p>Couldn’t load filter options. {{ facetsError }}</p>
          <button type="button" class="catalog-button" @click="refreshFacets">Retry filters</button>
        </div>
        <div v-else-if="!facetData" class="text-sm text-text-secondary" role="status">Loading filters…</div>
        <div v-else class="facet-grid">
          <template v-for="facet in facets" :key="facet.key">
            <SearchSelect
              v-if="facet.searchable"
              :label="facet.label"
              :options="facet.options"
              :model-value="facet.value"
              @update:model-value="setFacet(facet.key, $event)"
            >
              <template v-if="facet.key === 'model'" #option="{ option }"><ModelName :model="option" /></template>
            </SearchSelect>
            <label v-else class="facet-select">
              {{ facet.label }}
              <select :value="facet.value" @change="setFacet(facet.key, ($event.target as HTMLSelectElement).value)">
                <option value="">All</option>
                <option v-for="option in facet.options" :key="option" :value="option">
                  {{ facet.key === 'status' ? option.replace(/_/g, ' ') : option }}
                </option>
              </select>
            </label>
          </template>
          <label class="facet-select">
            Result limit
            <input :disabled="!catalog" :value="limit || undefined" type="number" min="1" @change="setLimit" />
          </label>
        </div>
      </section>

      <div v-if="group || activeFacets.length" class="active-filters" role="group" aria-label="Active filters">
        <div v-if="group" class="active-filter">
          <span><strong>Launch</strong> <span class="font-mono">{{ group }}</span></span>
          <button type="button" aria-label="Clear launch filter" @click="pushQuery({ group: '', view: 'runs', offset: '' })">×</button>
        </div>
        <div v-for="facet in activeFacets" :key="facet.key" class="active-filter">
          <span><strong>{{ facet.label }}</strong> {{ facet.value }}</span>
          <button type="button" :aria-label="`Remove ${facet.label.toLowerCase()} filter`" @click="setFacet(facet.key, '')">×</button>
        </div>
        <button v-if="activeFacets.length > 1" type="button" class="text-action" @click="clearFilters">Clear filters</button>
      </div>

      <div v-if="error" class="catalog-state" role="alert">
        <p>Couldn’t load {{ pluralUnit }}. {{ error }}</p>
        <button type="button" class="catalog-button" @click="refresh">Retry</button>
      </div>
      <LaunchActivity
        v-else-if="view === 'launches' && launchRows.length"
        :rows="launchRows"
        :launch-location="launchLocation"
        :aria-busy="loading"
      />
      <RunCatalogTable
        v-else-if="view === 'runs' && runRows.length"
        :rows="runRows"
        :group="group"
        :launch-location="launchLocation"
        :aria-busy="loading"
      />
      <div v-else-if="catalog && !loading" class="catalog-state">
        <template v-if="filtered">
          <p>No {{ pluralUnit }} match your search or filters.</p>
          <button type="button" class="catalog-button" @click="resetSearch">Reset search and filters</button>
        </template>
        <p v-else>No {{ pluralUnit }} recorded yet.</p>
      </div>
      <div v-else class="catalog-pending" aria-hidden="true"></div>
    </div>
  </section>
</template>

<style scoped>
h1 {
  margin: 0 0 0.72rem;
}

.catalog {
  color: var(--c-text);
  background: var(--c-surface);
  border: 1px solid var(--c-surface-border);
  border-radius: 0.8rem;
}

.catalog-controls,
.catalog-summary,
.catalog-pagination {
  display: flex;
  align-items: center;
}

.catalog-toolbar {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 0.65rem 0.85rem;
  padding: 0.55rem 0.85rem;
  border-bottom: 1px solid var(--c-surface-border);
}

.catalog-controls {
  justify-content: flex-end;
  gap: 0.5rem;
  min-width: 0;
}

.view-switch {
  display: inline-flex;
  flex: 0 0 auto;
  gap: 0.125rem;
  padding: 0.1875rem;
  background: var(--c-surface-sunken);
  border-radius: 0.55rem;
}

.view-switch button {
  min-height: 2rem;
  padding: 0.25rem 0.8rem;
  border: 1px solid transparent;
  border-radius: 0.375rem;
  background: transparent;
  color: var(--c-text-secondary);
  font-size: 0.8125rem;
  font-weight: 600;
  white-space: nowrap;
}

.view-switch button:hover {
  color: var(--c-text);
}

.view-switch button[aria-pressed='true'] {
  color: var(--c-accent);
  background: var(--c-accent-subtle);
  border-color: var(--c-accent-border);
}

.run-search {
  flex: 0 1 26rem;
  min-width: 0;
  position: relative;
}

.run-search::before {
  content: '';
  position: absolute;
  pointer-events: none;
  inset: 0.825rem auto auto 0.825rem;
  width: 0.625rem;
  height: 0.625rem;
  border: 1.5px solid var(--c-text-secondary);
  border-radius: 50%;
}

.run-search::after {
  content: '';
  position: absolute;
  pointer-events: none;
  inset: 1.4rem auto auto 1.34rem;
  width: 0.28rem;
  height: 1.5px;
  transform: rotate(45deg);
  background: var(--c-text-secondary);
}

.run-search input {
  width: 100%;
  min-height: 2.5rem;
  padding: 0.4rem 0.625rem 0.4rem 2.15rem;
  background: var(--c-surface);
  border: 1px solid var(--c-control-border);
  border-radius: 0.45rem;
  color: var(--c-text);
  font-size: 0.875rem;
}

.run-search input::placeholder {
  color: var(--c-text-secondary);
  opacity: 1;
}

.catalog-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.375rem;
  min-height: 2rem;
  padding: 0.25rem 0.625rem;
  border: 1px solid var(--c-control-border);
  border-radius: 0.4rem;
  background: var(--c-surface);
  color: var(--c-text-secondary);
  font-size: 0.8125rem;
  font-weight: 600;
  white-space: nowrap;
}

.catalog-button:hover:not(:disabled),
.catalog-button[aria-expanded='true'] {
  color: var(--c-accent);
  border-color: var(--c-accent-border);
  background: var(--c-accent-subtle);
}

.catalog-button:disabled {
  opacity: 0.5;
}

.filter-toggle {
  min-height: 2.5rem;
  padding-inline: 0.75rem;
}

.filter-count {
  padding-inline: 0.325rem;
  border-radius: 0.25rem;
  background: var(--c-accent-subtle);
  color: var(--c-accent);
  font-variant-numeric: tabular-nums;
}

.filter-chevron {
  width: 0.375rem;
  height: 0.375rem;
  margin: -0.125rem 0.125rem 0 0.125rem;
  border: solid currentColor;
  border-width: 0 1px 1px 0;
  transform: rotate(45deg);
}

.filter-toggle[aria-expanded='true'] .filter-chevron {
  margin-top: 0.2rem;
  transform: rotate(225deg);
}

.facet-panel {
  padding: 0.9rem 1rem 1rem;
  border-bottom: 1px solid var(--c-surface-border-subtle);
  background: var(--c-surface-raised);
}

.facet-grid {
  display: grid;
  grid-template-columns: repeat(10, minmax(0, 1fr));
  gap: 0.75rem;
  align-items: end;
}

.facet-grid > * {
  grid-column: span 2;
}

.facet-grid > :nth-child(-n+2) {
  grid-column: span 5;
}

.facet-select {
  display: grid;
  min-width: 0;
  gap: 0.25rem;
  color: var(--c-text-secondary);
  font-size: 0.75rem;
}

.facet-select > input,
.facet-select > select {
  min-height: 2.1rem;
  width: 100%;
  min-width: 0;
  padding: 0.25rem 0.5rem;
  background: var(--c-surface);
  border: 1px solid var(--c-control-border);
  border-radius: 0.25rem;
  color: var(--c-text);
  font-size: 0.875rem;
}

.facet-select > input {
  width: 7rem;
  max-width: 100%;
}

.facet-grid :deep(input[role='combobox']) {
  min-height: 2.1rem;
}

.active-filters {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.375rem 0.5rem;
  border-bottom: 1px solid var(--c-surface-border-subtle);
  padding: 0.55rem 1rem;
}

.active-filter {
  display: flex;
  align-items: center;
  min-width: 0;
  max-width: 100%;
  gap: 0.375rem;
  border: 1px solid var(--c-accent-border);
  border-radius: 0.375rem;
  padding-inline-start: 0.5rem;
  background: var(--c-accent-subtle);
  color: var(--c-accent);
  overflow-wrap: anywhere;
  font-size: 0.75rem;
}

.active-filter strong {
  margin-inline-end: 0.2rem;
  font-weight: 650;
}

.active-filter button {
  min-width: 1.6rem;
  min-height: 1.6rem;
  padding: 0;
  border: 0;
  background: transparent;
  font-size: 0.95rem;
}

.active-filter button:hover,
.text-action:hover {
  text-decoration: underline;
}

.text-action {
  border: 0;
  padding-inline: 0.2rem;
  background: transparent;
  color: var(--c-accent);
  font-size: 0.75rem;
}

.catalog-summary {
  justify-content: flex-end;
  gap: 0.6rem;
  white-space: nowrap;
}

.result-count {
  color: var(--c-text-secondary);
  font-size: 0.8125rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.catalog-pagination {
  gap: 0.375rem;
}

.catalog-state {
  min-height: 8rem;
  padding: 2.5rem 1rem;
  text-align: center;
  color: var(--c-text-secondary);
  font-size: 0.875rem;
}

.catalog-state .catalog-button {
  margin-top: 0.75rem;
}

.catalog-error {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.75rem;
  color: var(--c-status-danger);
  font-size: 0.875rem;
}

.catalog-pending {
  min-height: 8rem;
}

@media (max-width: 960px) {
  .catalog-toolbar {
    grid-template-columns: auto minmax(0, 1fr);
  }

  .catalog-summary {
    grid-column: 1 / -1;
    justify-content: space-between;
  }
}

@media (max-width: 650px) {
  .catalog-toolbar {
    display: flex;
    align-items: flex-start;
    flex-direction: column;
    gap: 0.625rem;
  }

  .catalog-controls {
    width: 100%;
  }

  .catalog-summary {
    width: 100%;
  }

  .run-search {
    flex: 1;
  }

  .run-search input {
    font-size: 1rem;
  }

  .facet-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .facet-grid > :nth-child(-n+2) {
    grid-column: span 4;
  }
}

@media (pointer: coarse) {
  .view-switch button,
  .catalog-button,
  .run-search input,
  .facet-select > input,
  .facet-select > select,
  .facet-grid :deep(input[role='combobox']) {
    min-height: 2.75rem;
  }

  .facet-select > input,
  .facet-select > select,
  .facet-grid :deep(input[role='combobox']) {
    font-size: 1rem;
  }
}
</style>
