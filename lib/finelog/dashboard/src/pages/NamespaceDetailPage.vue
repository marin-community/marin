<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { statsRpcCall } from '@/composables/useRpc'
import { decodeArrowIpc, type ArrowResult } from '@/utils/arrow'
import { shortColumnType, type ProtoSchema } from '@/types/stats'
import InfoCard from '@/components/shared/InfoCard.vue'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'

const props = defineProps<{ name: string }>()
const router = useRouter()

interface QueryResponse {
  arrowIpc?: string
  rowCount?: string | number
}

interface GetTableSchemaResponse {
  schema?: ProtoSchema
}

const schema = ref<ProtoSchema | null>(null)
const sample = ref<ArrowResult>({ columns: [], rows: [] })
const loading = ref(false)
const error = ref<string | null>(null)

const schemaRows = computed(() =>
  (schema.value?.columns ?? []).map((c) => ({
    column_name: c.name,
    column_type: shortColumnType(c.type),
    nullable: c.nullable ? 'YES' : 'NO',
  })),
)

const schemaColumns: Column[] = [
  { key: 'column_name', label: 'Column', mono: true },
  { key: 'column_type', label: 'Type', mono: true },
  { key: 'nullable', label: 'Nullable', align: 'center' },
]

const keyColumn = computed<string | null>(() => {
  const s = schema.value
  if (!s) return null
  if (s.keyColumn) return s.keyColumn
  // Match server-side resolve_key_column fallback: implicit timestamp_ms.
  if (s.columns?.some((c) => c.name === 'timestamp_ms')) return 'timestamp_ms'
  // Privileged log namespace orders by epoch_ms.
  if (s.columns?.some((c) => c.name === 'epoch_ms')) return 'epoch_ms'
  return null
})

async function load() {
  loading.value = true
  error.value = null
  try {
    const ns = props.name
    const schemaResp = await statsRpcCall<GetTableSchemaResponse>('GetTableSchema', { namespace: ns })
    schema.value = schemaResp.schema ?? null

    const orderBy = keyColumn.value ? ` ORDER BY "${keyColumn.value}" DESC` : ''
    const rows = await statsRpcCall<QueryResponse>('Query', {
      sql: `SELECT * FROM "${ns}"${orderBy} LIMIT 100`,
    })
    sample.value = decodeArrowIpc(rows.arrowIpc)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

const sampleColumns = ref<Column[]>([])
watch(sample, (s) => {
  sampleColumns.value = s.columns.map((c) => ({ key: c, label: c, mono: true }))
})

function openInQuery() {
  const sql = `SELECT * FROM "${props.name}" LIMIT 100`
  router.push({ path: '/query', query: { sql } })
}

onMounted(load)
watch(() => props.name, load)
</script>

<template>
  <div class="space-y-3">
    <div class="flex items-center justify-between">
      <div>
        <RouterLink to="/" class="text-xs text-text-muted hover:text-text">← Namespaces</RouterLink>
        <h2 class="text-lg font-mono mt-1">{{ name }}</h2>
        <p v-if="keyColumn" class="text-xs text-text-muted mt-0.5">
          ordered by <span class="font-mono">{{ keyColumn }}</span>
        </p>
      </div>
      <button
        class="text-xs px-3 py-1.5 rounded border border-surface-border hover:bg-surface-raised"
        @click="openInQuery"
      >Open in Query →</button>
    </div>

    <div
      v-if="error"
      class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <InfoCard title="Schema">
      <DataTable
        :columns="schemaColumns"
        :rows="schemaRows"
        :loading="loading && !schema"
        empty-message="No columns."
      />
    </InfoCard>

    <InfoCard :title="`Recent rows · up to 100`">
      <DataTable
        :columns="sampleColumns"
        :rows="sample.rows"
        :loading="loading && sample.rows.length === 0"
        :page-size="25"
        empty-message="No rows."
      />
    </InfoCard>
  </div>
</template>
