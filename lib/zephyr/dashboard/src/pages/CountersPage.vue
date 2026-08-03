<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import ErrorBanner from '@/components/ErrorBanner.vue'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { selectedExecutionId } from '@/composables/usePipelineSelection'
import { useDashboardApi } from '@/composables/useApi'
import type { CounterPage } from '@/types/dashboard'
import { formatNumber } from '@/utils/formatting'

const search = ref('')
const stage = ref('')
const sortField = ref('name')
const sortDescending = ref(false)
const offset = ref(0)
const limit = 50

const api = useDashboardApi<CounterPage>('counters', () => ({
  search: search.value,
  stage: stage.value,
  sort_field: sortField.value,
  sort_descending: sortDescending.value,
  offset: offset.value,
  limit,
  execution_id: selectedExecutionId.value,
}))

const counters = computed(() => api.data.value?.counters ?? [])
const total = computed(() => api.data.value?.total ?? 0)
const stages = computed(() => [...new Set(counters.value.map((counter) => counter.stage).filter(Boolean))].sort())

function sort(field: string) {
  if (sortField.value === field) sortDescending.value = !sortDescending.value
  else {
    sortField.value = field
    sortDescending.value = false
  }
  void api.refresh()
}

watch([search, stage], () => {
  offset.value = 0
  void api.refresh()
})
watch(selectedExecutionId, () => {
  offset.value = 0
  stage.value = ''
  void api.refresh()
})
useAutoRefresh(api.refresh)
onMounted(api.refresh)
</script>

<template>
  <div class="space-y-4">
    <ErrorBanner :message="api.error.value" />
    <section class="card overflow-hidden">
      <div class="flex flex-wrap items-center gap-3 border-b border-surface-border p-4">
        <div>
          <h2 class="text-base font-semibold">Pipeline counters</h2>
          <p class="text-xs text-text-secondary">{{ total }} counter values across completed and active shards</p>
        </div>
        <div class="ml-auto flex gap-2">
          <select v-model="stage" class="rounded-lg border border-surface-border bg-surface-raised px-3 py-2 text-sm">
            <option value="">All stages</option>
            <option v-for="value in stages" :key="value" :value="value">{{ value }}</option>
          </select>
          <input
            v-model="search"
            class="w-56 rounded-lg border border-surface-border bg-surface-raised px-3 py-2 text-sm outline-none focus:border-accent"
            placeholder="Search counters"
          />
        </div>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full">
          <thead class="bg-surface-sunken">
            <tr>
              <th class="table-header"><button @click="sort('name')">Name ↕</button></th>
              <th class="table-header"><button @click="sort('stage')">Stage ↕</button></th>
              <th class="table-header"><button @click="sort('aggregation')">Aggregation ↕</button></th>
              <th class="table-header text-right"><button @click="sort('observations')">Samples ↕</button></th>
              <th class="table-header text-right"><button @click="sort('value')">Value ↕</button></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="counter in counters" :key="`${counter.stage}/${counter.name}/${counter.aggregation}`" class="hover:bg-surface-sunken/60">
              <td class="table-cell font-mono text-xs font-medium">{{ counter.name }}</td>
              <td class="table-cell text-text-secondary">{{ counter.stage || 'pipeline' }}</td>
              <td class="table-cell capitalize text-text-secondary">{{ counter.aggregation }}</td>
              <td class="table-cell text-right tabular-nums text-text-secondary">{{ counter.observations ?? 0 }}</td>
              <td class="table-cell text-right font-mono font-semibold tabular-nums">{{ formatNumber(counter.value, 3) }}</td>
            </tr>
            <tr v-if="!counters.length">
              <td colspan="5" class="p-12 text-center text-sm text-text-muted">No counters match this view.</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="flex items-center justify-between border-t border-surface-border px-4 py-3 text-xs text-text-secondary">
        <span>{{ offset + 1 }}–{{ Math.min(offset + counters.length, total) }} of {{ total }}</span>
        <div class="flex gap-2">
          <button class="rounded border border-surface-border px-3 py-1.5 disabled:opacity-40" :disabled="offset === 0" @click="offset = Math.max(0, offset - limit); api.refresh()">Previous</button>
          <button class="rounded border border-surface-border px-3 py-1.5 disabled:opacity-40" :disabled="offset + limit >= total" @click="offset += limit; api.refresh()">Next</button>
        </div>
      </div>
    </section>
  </div>
</template>
