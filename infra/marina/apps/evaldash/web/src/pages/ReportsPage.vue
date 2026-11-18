<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatTimestamp } from '@/utils/formatting'
import { REPORT_STATUSES, type ReportSummary } from '@/types/api'
import StatusChip from '@/components/shared/StatusChip.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import FilterBar, { type Facet } from '@/components/shared/FilterBar.vue'

const route = useRoute()
const router = useRouter()

const FACET_KEYS = ['model', 'status', 'reporter'] as const

function queryValue(key: string): string {
  const value = route.query[key]
  return typeof value === 'string' ? value : ''
}

const selected = computed<Record<string, string>>({
  get: () => Object.fromEntries(FACET_KEYS.map((key) => [key, queryValue(key)])),
  set: (values) => {
    const query = { ...route.query }
    for (const key of FACET_KEYS) {
      if (values[key]) query[key] = values[key]
      else delete query[key]
    }
    router.push({ path: '/reports', query })
  },
})

const { data, loading, error, refresh } = useApi<ReportSummary[]>(() => 'api/playground-reports')
onMounted(refresh)
onViewRefresh(refresh)

const reports = computed(() => data.value ?? [])

function distinct(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))].sort()
}

const facets = computed<Facet[]>(() => [
  { key: 'model', label: 'Model', searchable: true, options: distinct(reports.value.map((r) => r.model)) },
  { key: 'status', label: 'Status', options: [...REPORT_STATUSES] },
  { key: 'reporter', label: 'Reporter', options: distinct(reports.value.map((r) => r.reporter)) },
])

const shown = computed(() =>
  reports.value.filter((r) =>
    (!selected.value.model || r.model === selected.value.model) &&
    (!selected.value.status || r.status === selected.value.status) &&
    (!selected.value.reporter || r.reporter === selected.value.reporter),
  ),
)

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

    <FilterBar v-model="selected" :facets="facets" :result-label="resultLabel" class="mb-4" />

    <div v-if="error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2 mb-4">
      {{ error }}
    </div>

    <div v-if="loading && !data" class="text-sm text-text-muted py-12 text-center">Loading…</div>

    <EmptyState v-else-if="shown.length === 0" icon="🔍" message="No reports match these filters." />

    <div v-else class="overflow-x-auto rounded-lg border border-surface-border">
      <table class="w-full border-collapse text-sm">
        <thead>
          <tr class="border-b border-surface-border bg-surface-raised text-xs font-semibold uppercase tracking-wider text-text-secondary">
            <th class="px-3 py-2 text-left">Reported</th>
            <th class="px-3 py-2 text-left">Model</th>
            <th class="px-3 py-2 text-left">Reporter</th>
            <th class="px-3 py-2 text-left">What went wrong</th>
            <th class="px-3 py-2 text-left">Chat</th>
            <th class="px-3 py-2 text-left">Status</th>
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
            <td class="px-3 py-2"><StatusChip :status="row.status" /></td>
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
