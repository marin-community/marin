<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatTimestamp } from '@/utils/formatting'
import type { ReportListing, ReportSummary } from '@/types/api'
import EmptyState from '@/components/shared/EmptyState.vue'
import FilterBar, { type Facet } from '@/components/shared/FilterBar.vue'

const route = useRoute()
const router = useRouter()

const FACET_KEYS = ['model', 'reporter'] as const

function queryValue(key: string): string {
  const value = route.query[key]
  return typeof value === 'string' ? value : ''
}

// Filters, search, and sort order all live in the URL. Typing replaces the history entry, so one
// Back undoes a whole search rather than each keystroke.
function setQuery(values: Record<string, string>, navigate: 'push' | 'replace') {
  const query = { ...route.query }
  for (const [key, value] of Object.entries(values)) {
    if (value) query[key] = value
    else delete query[key]
  }
  router[navigate]({ path: '/reports', query })
}

const selected = computed<Record<string, string>>({
  get: () => Object.fromEntries(FACET_KEYS.map((key) => [key, queryValue(key)])),
  set: (values) => setQuery(Object.fromEntries(FACET_KEYS.map((key) => [key, values[key] ?? ''])), 'push'),
})

const { data, loading, error, refresh } = useApi<ReportListing>(() => 'api/playground-reports')
onMounted(refresh)
onViewRefresh(refresh)

const reports = computed(() => data.value?.reports ?? [])

function distinct(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))].sort()
}

const facets = computed<Facet[]>(() => [
  { key: 'model', label: 'Model', searchable: true, options: distinct(reports.value.map((r) => r.model)) },
  { key: 'reporter', label: 'Reporter', options: distinct(reports.value.map((r) => r.reporter)) },
])

const search = computed({
  get: () => queryValue('q'),
  set: (value: string) => setQuery({ q: value }, 'replace'),
})
const oldestFirst = computed({
  get: () => queryValue('order') === 'oldest',
  set: (value: boolean) => setQuery({ order: value ? 'oldest' : '' }, 'replace'),
})

function matchesSearch(r: ReportSummary, terms: string[]): boolean {
  const text = `${r.description}\n${r.model}\n${r.reporter}`.toLowerCase()
  return terms.every((term) => text.includes(term))
}

const shown = computed(() => {
  const terms = search.value.toLowerCase().split(/\s+/).filter(Boolean)
  const matching = reports.value.filter((r) =>
    (!selected.value.model || r.model === selected.value.model) &&
    (!selected.value.reporter || r.reporter === selected.value.reporter) &&
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
      <h2 class="text-lg font-semibold">Playground reports</h2>
    </div>

    <p class="mb-4 text-xs text-text-muted">
      Poor responses reported from the marin-serve dashboard's Report issue button. New reports
      appear after the next ingest pass, within about 10 minutes.
    </p>

    <FilterBar v-model="selected" :facets="facets" :result-label="resultLabel" class="mb-4">
      <template #trailing>
        <label class="flex flex-col text-xs text-text-secondary gap-1 flex-1 basis-64">
          Search
          <input
            :value="search"
            type="search"
            placeholder="Words in the description, model, or reporter"
            class="rounded border border-surface-border bg-surface px-2 py-1 text-sm"
            @input="search = ($event.target as HTMLInputElement).value"
          />
        </label>
      </template>
    </FilterBar>

    <p v-if="data?.truncated" class="mb-4 text-xs text-text-muted">
      Showing the newest {{ reports.length }} reports. Search, filters, and oldest-first order cover only
      these; older reports are in the report buckets.
    </p>

    <div v-if="error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2 mb-4">
      {{ error }}
    </div>

    <div v-if="loading && !data" class="text-sm text-text-muted py-12 text-center">Loading…</div>

    <EmptyState v-else-if="shown.length === 0" icon="🔍" message="No reports match these filters." />

    <div v-else class="overflow-x-auto rounded-lg border border-surface-border">
      <table class="w-full border-collapse text-sm">
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
            <th class="px-3 py-2 text-left"></th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in shown"
            :key="row.report_id"
            class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
          >
            <td class="px-3 py-2 whitespace-nowrap text-text-secondary">{{ formatTimestamp(row.created_at) }}</td>
            <td class="px-3 py-2 font-mono text-[13px] whitespace-nowrap">{{ row.model }}</td>
            <td class="px-3 py-2 whitespace-nowrap text-text-secondary">{{ row.reporter || '—' }}</td>
            <td class="px-3 py-2 max-w-[48ch] truncate" :title="row.description">{{ row.description }}</td>
            <td class="px-3 py-2 text-text-secondary">{{ row.has_conversation ? 'yes' : 'omitted' }}</td>
            <td class="px-3 py-2 text-right">
              <RouterLink
                :to="`/reports/${encodeURIComponent(row.report_id)}`"
                class="text-accent hover:text-accent-hover hover:underline whitespace-nowrap"
              >detail →</RouterLink>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
