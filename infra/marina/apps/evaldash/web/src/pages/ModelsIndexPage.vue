<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import ModelName from '@/components/shared/ModelName.vue'
import StatusChip from '@/components/shared/StatusChip.vue'
import { useCatalogPage, useCatalogQuery } from '@/composables/useCatalogQuery'
import { onViewRefresh } from '@/composables/useRefresh'
import type { ModelCatalogRow } from '@/types/api'
import { formatTimestamp } from '@/utils/formatting'
import { formatPageRange } from '@/utils/pagination'
import { ALL_COHORTS, UNVERSIONED_COHORT } from '@/utils/panel'

const ARCHIVE_FILTER = { key: 'include_archived', enabled: '1' }
const numberFormat = new Intl.NumberFormat()

const { queryValue, offset, query, search, submitSearch, updateCatalogQuery, changePage } = useCatalogQuery()

const cohort = computed(() => queryValue('cohort'))
const cohortLabel = computed(() => cohort.value === UNVERSIONED_COHORT ? 'Unversioned' : cohort.value)
const showArchived = computed({
  get: () => queryValue(ARCHIVE_FILTER.key) === ARCHIVE_FILTER.enabled,
  set: (checked: boolean) => {
    void updateCatalogQuery({ [ARCHIVE_FILTER.key]: checked ? ARCHIVE_FILTER.enabled : '' })
  },
})

const catalogPath = computed(() => {
  const params = new URLSearchParams()
  if (cohort.value) params.set('cohort', cohort.value)
  if (query.value) params.set('q', query.value)
  if (showArchived.value) params.set(ARCHIVE_FILTER.key, ARCHIVE_FILTER.enabled)
  params.set('offset', String(offset.value))
  return `api/catalog/models?${params}`
})
const { data, loading, error, refresh, limit, hasNext, showPagination } =
  useCatalogPage<ModelCatalogRow>(() => catalogPath.value, offset)
onViewRefresh(refresh)
const resultLabel = computed(() => {
  if (!data.value) return error.value ? '' : 'Loading models…'
  if (!data.value.rows.length) return 'No models match.'
  const { total, rows } = data.value
  return `${formatPageRange(offset.value, rows.length, total)} ${total === 1 ? 'model' : 'models'}`
})
</script>

<template>
  <section class="models-directory" aria-labelledby="model-directory-title">
    <h1 id="model-directory-title">Models</h1>

    <div class="directory">
      <div class="directory-controls">
        <form role="search" @submit.prevent="submitSearch">
          <label class="sr-only" for="filter-models">Filter models</label>
          <input
            id="filter-models"
            v-model="search"
            type="search"
            placeholder="Search models"
            autocomplete="off"
          />
        </form>
        <label class="archive-toggle">
          <input v-model="showArchived" type="checkbox" />
          Show archived
        </label>
        <button
          v-if="cohort && cohort !== ALL_COHORTS"
          type="button"
          class="directory-button"
          aria-label="Clear cohort filter"
          @click="updateCatalogQuery({ cohort: '' })"
        >{{ cohortLabel }} <span aria-hidden="true">×</span></button>
        <div class="directory-summary">
          <p role="status">{{ resultLabel }}</p>
          <nav v-if="showPagination" aria-label="Model pages">
            <button
              type="button"
              class="directory-button"
              :disabled="!data || loading || offset === 0"
              @click="changePage(offset - limit)"
            >‹ Previous</button>
            <button
              type="button"
              class="directory-button"
              :disabled="loading || !hasNext"
              @click="changePage(offset + limit)"
            >Next ›</button>
          </nav>
        </div>
      </div>

      <div v-if="error" role="alert" class="directory-error">
        <p>Couldn’t load models. {{ error }}</p>
        <button type="button" class="directory-button" @click="refresh">Retry</button>
      </div>
      <table v-else-if="data?.rows.length" class="directory-table max-[700px]:block" role="table" aria-label="Recorded models" :aria-busy="loading">
        <thead class="max-[700px]:sr-only" role="rowgroup">
          <tr role="row">
            <th scope="col" role="columnheader">Model</th>
            <th scope="col" role="columnheader" aria-sort="descending">Latest evaluation</th>
            <th scope="col" role="columnheader">Runs</th>
            <th scope="col" role="columnheader">Distinct evals</th>
          </tr>
        </thead>
        <tbody class="max-[700px]:block" role="rowgroup">
          <tr v-for="row in data.rows" :key="row.model" role="row">
            <th scope="row" role="rowheader" class="model-cell">
              <RouterLink :to="{ name: 'model', params: { model: row.model }, query: { cohort: cohort || undefined } }" :aria-label="row.model">
                <ModelName :model="row.model" />
              </RouterLink>
              <span v-if="row.archived" class="archived">Archived</span>
            </th>
            <td role="cell" class="latest-cell">
              <div class="latest">
                <RouterLink :to="{ name: 'run', params: { runId: row.latest_run.run_id } }">
                  {{ row.latest_run.eval_name || row.latest_run.run_id }}
                </RouterLink>
                <StatusChip :status="row.latest_run.status" />
              </div>
              <time :datetime="row.latest_run.created_at">{{ formatTimestamp(row.latest_run.created_at) }}</time>
            </td>
            <td role="cell">{{ numberFormat.format(row.run_count) }} <span class="mobile-unit">runs</span></td>
            <td role="cell">{{ numberFormat.format(row.benchmark_count) }} <span class="mobile-unit">distinct evals</span></td>
          </tr>
        </tbody>
      </table>
      <div v-else class="directory-pending" aria-hidden="true"></div>
    </div>
  </section>
</template>

<style scoped>
h1 {
  margin-bottom: .65rem;
}

.directory {
  border: 1px solid var(--c-surface-border);
  border-radius: 0.8rem;
  background: var(--c-surface);
}

.directory-controls,
.directory-summary,
.directory-summary nav,
.archive-toggle {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.55rem;
}

.directory-controls {
  padding: .55rem .85rem;
  border-bottom: 1px solid var(--c-surface-border);
  color: var(--c-text-secondary);
  font-size: 0.8125rem;
  gap: 0.75rem;
}

.directory-controls form {
  width: min(100%, 23rem);
}

.directory-controls input[type='search'] {
  width: 100%;
  min-height: 2.5rem;
  padding: 0.4rem 0.65rem;
  border: 1px solid var(--c-control-border);
  border-radius: 0.45rem;
  background: var(--c-surface);
  color: var(--c-text);
}

.archive-toggle {
  min-height: 2.25rem;
  white-space: nowrap;
}

.archive-toggle input {
  accent-color: var(--c-accent);
}

.directory-summary {
  justify-content: space-between;
  margin-inline-start: auto;
  font-variant-numeric: tabular-nums;
}

.directory-button {
  min-height: 2rem;
  max-width: 100%;
  padding: 0.25rem 0.625rem;
  border: 1px solid var(--c-control-border);
  border-radius: 0.4rem;
  color: var(--c-text-secondary);
  font-size: 0.8125rem;
  font-weight: 600;
  overflow-wrap: anywhere;
}

.directory-button:hover:not(:disabled) {
  border-color: var(--c-accent-border);
  background: var(--c-accent-subtle);
  color: var(--c-accent);
}

.directory-button:disabled {
  opacity: 0.5;
}

.directory-table {
  width: 100%;
  table-layout: fixed;
  color: var(--c-text-secondary);
  font-size: 0.8125rem;
}

.directory-table th,
.directory-table td {
  padding: .4rem .85rem;
  text-align: start;
  vertical-align: top;
  overflow-wrap: anywhere;
}

.directory-table thead {
  background: var(--c-surface-raised);
}

.directory-table thead th {
  font-size: 0.75rem;
  font-weight: 650;
}

.directory-table thead th:nth-child(1) { width: 42%; }
.directory-table thead th:nth-child(3),
.directory-table thead th:nth-child(4) { width: 13%; }

.directory-table tbody tr + tr {
  border-top: 1px solid var(--c-surface-border-subtle);
}

.directory-table tbody tr:hover,
.directory-table tbody tr:focus-within {
  background: color-mix(in srgb, var(--c-accent) 5%, var(--c-surface));
}

.directory-table a {
  color: var(--c-accent);
}

.directory-table a:hover {
  text-decoration: underline;
}

.model-cell {
  color: var(--c-text);
  font-weight: 650;
}

.latest {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.3rem 0.5rem;
}

.archived,
.latest-cell time {
  display: block;
  margin-top: 0.2rem;
  color: var(--c-text-secondary);
  font-size: 0.75rem;
  font-weight: 400;
}

.mobile-unit {
  display: none;
}

.directory-error,
.directory-pending {
  min-height: 5rem;
}

.directory-error {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.6rem;
  padding: 1rem;
  color: var(--c-status-danger);
  font-size: 0.875rem;
}

@media (width < 700px) {
  .directory-table tbody tr {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: 0.4rem 0.85rem;
    padding: 0.75rem 1rem;
  }

  .directory-table tbody th,
  .directory-table tbody td {
    padding: 0;
  }

  .model-cell,
  .latest-cell {
    grid-column: 1 / -1;
  }

  .mobile-unit {
    display: inline;
  }
}
</style>
