<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { statsRpcCall } from '@/composables/useRpc'
import { decodeArrowIpc, type ArrowResult } from '@/utils/arrow'
import InfoCard from '@/components/shared/InfoCard.vue'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'

interface QueryResponse {
  arrowIpc?: string
  rowCount?: string | number
}

const route = useRoute()
const sql = ref<string>(typeof route.query.sql === 'string' ? route.query.sql : 'SELECT 1')
const result = ref<ArrowResult>({ columns: [], rows: [] })
const rowCount = ref<number>(0)
const loading = ref(false)
const error = ref<string | null>(null)

const columns = computed<Column[]>(() =>
  result.value.columns.map((c) => ({ key: c, label: c, mono: true })),
)

async function execute() {
  if (!sql.value.trim()) return
  loading.value = true
  error.value = null
  try {
    const resp = await statsRpcCall<QueryResponse>('Query', { sql: sql.value })
    result.value = decodeArrowIpc(resp.arrowIpc)
    rowCount.value = Number(resp.rowCount ?? result.value.rows.length)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    result.value = { columns: [], rows: [] }
    rowCount.value = 0
  } finally {
    loading.value = false
  }
}

function onKeydown(e: KeyboardEvent) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault()
    void execute()
  }
}

onMounted(() => {
  if (typeof route.query.sql === 'string' && route.query.sql.trim()) void execute()
})
</script>

<template>
  <div class="space-y-3">
    <InfoCard title="SQL · Postgres-flavored DuckDB">
      <textarea
        v-model="sql"
        class="w-full font-mono text-sm bg-surface-sunken border border-surface-border rounded p-3 min-h-[120px] focus:outline-none focus:border-accent"
        spellcheck="false"
        @keydown="onKeydown"
      />
      <div class="flex items-center gap-3 mt-2">
        <button
          class="px-3 py-1.5 text-sm rounded bg-accent text-white hover:bg-accent-hover disabled:opacity-50"
          :disabled="loading"
          @click="execute"
        >
          {{ loading ? 'Running…' : 'Execute' }}
        </button>
        <span class="text-xs text-text-muted">⌘/Ctrl-Enter to run</span>
        <span v-if="!loading && !error && result.rows.length > 0" class="text-xs text-text-muted ml-auto">
          {{ rowCount.toLocaleString() }} row{{ rowCount === 1 ? '' : 's' }}
        </span>
      </div>
    </InfoCard>

    <div
      v-if="error"
      class="px-4 py-3 text-sm font-mono text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border whitespace-pre-wrap"
    >{{ error }}</div>

    <InfoCard v-if="!error" title="Result">
      <DataTable
        :columns="columns"
        :rows="result.rows"
        :loading="loading"
        :page-size="50"
        empty-message="No rows."
      />
    </InfoCard>
  </div>
</template>
