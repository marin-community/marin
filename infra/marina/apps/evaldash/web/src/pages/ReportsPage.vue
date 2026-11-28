<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { useRouteQuery } from '@/composables/useRouteQuery'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatTimestamp } from '@/utils/formatting'
import type { ReportListing, ReportSummary } from '@/types/api'
import EmptyState from '@/components/shared/EmptyState.vue'
import SearchSelect from '@/components/shared/SearchSelect.vue'
import ModelName from '@/components/shared/ModelName.vue'

const route = useRoute()
const { queryValue, pushQuery, replaceQuery } = useRouteQuery()

const model = computed(() => queryValue('model'))
const reporter = computed(() => queryValue('reporter'))
const filtersOpen = ref(false)
const filterLabel = computed(() => [model.value, reporter.value].filter(Boolean).join(' · '))

const { data, loading, error, refresh } = useApi<ReportListing>(() => 'api/playground-reports')
onMounted(refresh)
onViewRefresh(refresh)

const reports = computed(() => data.value?.reports ?? [])

function distinct(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))].sort()
}

const models = computed(() => distinct(reports.value.map((report) => report.model)))
const reporters = computed(() => distinct(reports.value.map((report) => report.reporter)))

const search = computed({
  get: () => queryValue('q'),
  set: (value: string) => replaceQuery({ q: value }),
})
const oldestFirst = computed({
  get: () => queryValue('order') === 'oldest',
  set: (value: boolean) => replaceQuery({ order: value ? 'oldest' : '' }),
})

function matchesSearch(r: ReportSummary, terms: string[]): boolean {
  const text = `${r.description}\n${r.model}\n${r.reporter}`.toLowerCase()
  return terms.every((term) => text.includes(term))
}

const shown = computed(() => {
  const terms = search.value.toLowerCase().split(/\s+/).filter(Boolean)
  const matching = reports.value.filter((r) =>
    (!model.value || r.model === model.value) &&
    (!reporter.value || r.reporter === reporter.value) &&
    matchesSearch(r, terms),
  )
  // The API returns newest first.
  return oldestFirst.value ? [...matching].reverse() : matching
})

const resultLabel = computed(() => {
  if (loading.value && !data.value) return 'Loading…'
  return `${shown.value.length} report${shown.value.length === 1 ? '' : 's'} shown`
})
</script>

<template>
  <section>
    <div class="flex items-baseline justify-between mb-4">
      <h1>Playground reports</h1>
    </div>

    <p class="mb-3 text-sm text-text-secondary">
      Feedback from the serve dashboard. New reports usually appear within 10 minutes.
    </p>
    <div class="report-toolbar">
      <label class="flex-1 min-w-32 max-w-md">
        <span class="sr-only">Search</span>
        <input v-model="search" type="search" placeholder="Search reports" class="w-full" />
      </label>
      <button type="button" :aria-expanded="filtersOpen" aria-controls="report-filters" @click="filtersOpen = !filtersOpen">
        Filters<span v-if="filterLabel"> · {{ filterLabel }}</span>
      </button>
      <span class="text-xs text-text-secondary" role="status">{{ resultLabel }}</span>
    </div>
    <div v-if="filtersOpen" id="report-filters" class="flex flex-wrap items-end gap-3 mb-3">
      <SearchSelect label="Model" :options="models" :model-value="model" class="min-w-44" @update:model-value="pushQuery({ model: $event })">
        <template #option="{ option }"><ModelName :model="option" /></template>
      </SearchSelect>
      <label class="grid gap-1 text-xs text-text-secondary">
        Reporter
        <select :value="reporter" @change="pushQuery({ reporter: ($event.target as HTMLSelectElement).value })">
          <option value="">All</option>
          <option v-for="name in reporters" :key="name">{{ name }}</option>
        </select>
      </label>
      <button v-if="filterLabel" class="text-sm" @click="pushQuery({ model: '', reporter: '' })">Clear filters</button>
    </div>

    <p v-if="data?.truncated" class="mb-4 text-xs text-text-muted">
      Showing the newest {{ reports.length }} reports. Search, filters, and oldest-first order cover only
      these; older reports are in the report buckets.
    </p>

    <div v-if="error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2 mb-4">
      {{ error }}
    </div>

    <div v-if="loading && !data" class="text-sm text-text-muted py-12 text-center">Loading…</div>

    <EmptyState v-else-if="shown.length === 0" icon="🔍" message="No reports match these filters." />

    <div v-else class="report-list overflow-x-auto rounded-lg border border-surface-border bg-surface">
      <table class="w-full border-collapse text-sm" aria-label="Playground reports">
        <thead>
          <tr class="border-b border-surface-border bg-surface-raised text-xs font-semibold uppercase tracking-wider text-text-secondary">
            <th class="px-3 py-2 text-left" :aria-sort="oldestFirst ? 'ascending' : 'descending'">
              <button
                class="inline-flex items-center gap-1 uppercase tracking-wider hover:text-text"
                :title="oldestFirst ? 'Showing oldest first; click for newest first' : 'Showing newest first; click for oldest first'"
                @click="oldestFirst = !oldestFirst"
              >Reported {{ oldestFirst ? '↑' : '↓' }}</button>
            </th>
            <th class="px-3 py-2 text-left">Model</th>
            <th class="px-3 py-2 text-left">Reporter</th>
            <th class="px-3 py-2 text-left">What went wrong</th>
            <th class="px-3 py-2 text-left">Chat</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in shown"
            :key="row.report_id"
            class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
          >
            <td class="reported-at px-3 py-2 whitespace-nowrap text-text-secondary">{{ formatTimestamp(row.created_at) }}</td>
            <td class="reported-model px-3 py-2">{{ row.model }}</td>
            <td class="reporter px-3 py-2 text-text-secondary"><span class="mobile-label">Reporter </span>{{ row.reporter || 'anonymous' }}</td>
            <td class="description px-3 py-2">
              <RouterLink
                :to="{ name: 'report', params: { reportId: row.report_id }, query: route.query }"
                class="text-accent hover:underline"
              >{{ row.description || 'Open report' }}</RouterLink>
            </td>
            <td class="chat px-3 py-2 text-text-secondary"><span class="mobile-label">Chat </span>{{ row.has_conversation ? 'included' : 'omitted' }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>

<style scoped>
.report-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem .75rem; margin-bottom: .75rem; }
.report-toolbar input, .report-toolbar button { min-height: 2.15rem; font-size: .875rem; }
.report-toolbar button { max-width: 100%; overflow-wrap: anywhere; }
.description { max-width: 50ch; }
.description a {
  display: -webkit-box;
  overflow: hidden;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow-wrap: anywhere;
}
.reported-model { overflow-wrap: anywhere; }
.mobile-label { display: none; }
@media (max-width: 42rem) {
  .report-list table, .report-list tbody { display: block; }
  .report-list thead { display: block; }
  .report-list thead tr { display: flex; padding: .25rem .75rem; }
  .report-list th { border: 0; padding: 0; }
  .report-list th:not(:first-child) { position: absolute; clip-path: inset(50%); height: 1px; width: 1px; overflow: hidden; }
  .report-list tr { display: grid; grid-template-columns: 1fr auto; gap: .2rem .5rem; padding: .55rem .75rem; }
  .report-list td { border: 0; padding: 0; }
  .description { grid-area: 1 / 1 / 2 / -1; font-weight: 600; }
  .reported-model { grid-area: 2 / 1 / 3 / -1; }
  .reported-at, .reporter, .chat { font-size: .75rem; }
  .reported-at { grid-row: 3; }
  .mobile-label { display: inline; }
  .chat { grid-column: 1 / -1; }
}
</style>
