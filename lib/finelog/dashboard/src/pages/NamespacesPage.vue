<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { statsRpcCall } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import { formatBytes, formatNumber } from '@/utils/formatting'
import type { ProtoSchema } from '@/types/stats'
import InfoCard from '@/components/shared/InfoCard.vue'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'

interface NamespaceInfo {
  namespace: string
  schema?: ProtoSchema
  rowCount?: string | number
  byteSize?: string | number
  segmentCount?: number
}

interface ListNamespacesResponse {
  namespaces?: NamespaceInfo[]
}

interface NamespaceRow {
  namespace: string
  columnCount: number
  rowCount: number
  byteSize: number
  segmentCount: number
}

const rows = ref<NamespaceRow[]>([])
const loading = ref(false)
const error = ref<string | null>(null)

const columns: Column[] = [
  { key: 'namespace', label: 'Namespace', sortable: true, mono: true },
  { key: 'columnCount', label: 'Cols', sortable: true, align: 'right' },
  { key: 'rowCount', label: 'Rows', sortable: true, align: 'right' },
  { key: 'byteSize', label: 'Bytes', sortable: true, align: 'right' },
  { key: 'segmentCount', label: 'Segments', sortable: true, align: 'right' },
]

const sortKey = ref<'namespace' | 'columnCount' | 'rowCount' | 'byteSize' | 'segmentCount'>('namespace')
const sortDir = ref<'asc' | 'desc'>('asc')

const sorted = computed<NamespaceRow[]>(() => {
  const dir = sortDir.value === 'asc' ? 1 : -1
  const key = sortKey.value
  return [...rows.value].sort((a, b) => {
    if (key === 'namespace') return a.namespace.localeCompare(b.namespace) * dir
    const av = (a[key] as number | null) ?? -1
    const bv = (b[key] as number | null) ?? -1
    return (av - bv) * dir
  })
})

async function refresh() {
  loading.value = true
  error.value = null
  try {
    const list = await statsRpcCall<ListNamespacesResponse>('ListNamespaces', {})
    const infos = list.namespaces ?? []
    rows.value = infos.map((info) => ({
      namespace: info.namespace,
      columnCount: info.schema?.columns?.length ?? 0,
      rowCount: Number(info.rowCount ?? 0),
      byteSize: Number(info.byteSize ?? 0),
      segmentCount: info.segmentCount ?? 0,
    }))
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
onMounted(refresh)

function setSort(key: string) {
  const k = key as 'namespace' | 'columnCount' | 'rowCount' | 'byteSize' | 'segmentCount'
  if (sortKey.value === k) {
    sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortKey.value = k
    sortDir.value = k === 'namespace' ? 'asc' : 'desc'
  }
}
</script>

<template>
  <div class="space-y-3">
    <div
      v-if="error"
      class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <InfoCard title="Registered namespaces">
      <DataTable
        :columns="columns"
        :rows="sorted"
        :loading="loading"
        :sort-key="sortKey"
        :sort-dir="sortDir"
        empty-message="No namespaces registered."
        @sort="(k) => setSort(k)"
      >
        <template #cell-namespace="{ value }">
          <RouterLink
            :to="`/ns/${encodeURIComponent(String(value))}`"
            class="text-accent hover:underline font-mono"
          >{{ value }}</RouterLink>
        </template>
        <template #cell-columnCount="{ value }">
          {{ formatNumber(value as number) }}
        </template>
        <template #cell-rowCount="{ value }">
          {{ formatNumber(value as number) }}
        </template>
        <template #cell-byteSize="{ value }">
          {{ formatBytes(value as number) }}
        </template>
        <template #cell-segmentCount="{ value }">
          {{ formatNumber(value as number) }}
        </template>
      </DataTable>
    </InfoCard>
  </div>
</template>
