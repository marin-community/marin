<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { statsRpcCall } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import { decodeArrowIpc } from '@/utils/arrow'
import { formatNumber } from '@/utils/formatting'
import type { ProtoSchema } from '@/types/stats'
import InfoCard from '@/components/shared/InfoCard.vue'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'

interface QueryResponse {
  arrowIpc?: string
  rowCount?: string | number
}

interface NamespaceInfo {
  namespace: string
  schema?: ProtoSchema
}

interface ListNamespacesResponse {
  namespaces?: NamespaceInfo[]
}

interface NamespaceRow {
  namespace: string
  columnCount: number
  rowCount: number | null
}

const rows = ref<NamespaceRow[]>([])
const loading = ref(false)
const error = ref<string | null>(null)

const columns: Column[] = [
  { key: 'namespace', label: 'Namespace', sortable: true, mono: true },
  { key: 'columnCount', label: 'Cols', sortable: true, align: 'right' },
  { key: 'rowCount', label: 'Rows', sortable: true, align: 'right' },
]

const sortKey = ref<'namespace' | 'columnCount' | 'rowCount'>('namespace')
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

async function fetchRowCount(ns: string): Promise<number | null> {
  try {
    const resp = await statsRpcCall<QueryResponse>('Query', {
      sql: `SELECT count(*) AS n FROM "${ns}"`,
    })
    const r = decodeArrowIpc(resp.arrowIpc)
    return Number(r.rows[0]?.n ?? 0)
  } catch {
    return null
  }
}

async function refresh() {
  loading.value = true
  error.value = null
  try {
    const list = await statsRpcCall<ListNamespacesResponse>('ListNamespaces', {})
    const infos = list.namespaces ?? []
    const counts = await Promise.all(infos.map((info) => fetchRowCount(info.namespace)))
    rows.value = infos.map((info, i) => ({
      namespace: info.namespace,
      columnCount: info.schema?.columns?.length ?? 0,
      rowCount: counts[i],
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
  const k = key as 'namespace' | 'columnCount' | 'rowCount'
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
          {{ value === null ? '—' : formatNumber(value as number) }}
        </template>
      </DataTable>
    </InfoCard>
  </div>
</template>
