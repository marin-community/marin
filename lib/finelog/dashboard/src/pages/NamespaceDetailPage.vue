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

// Recent-rows window. We filter on the implicit ``seq`` column (always
// present, monotonically increasing on insert) before sorting, so only the
// latest segment(s) are read — a SELECT * ORDER BY <ts> DESC LIMIT 100 on a
// multi-GB namespace would otherwise scan every segment (reading the full row
// payload) to compute the top-N, which OOMs the server on the billion-row
// ``log`` namespace. ``RECENT_SEQ_WINDOW`` is the rolling number of newest
// rows considered; 10x the visible page so concurrent writers can't shrink the
// post-filter set below LIMIT.
//
// The bound MUST be a literal: the seq floor is resolved with a separate
// ``max(seq)`` query and inlined. A scalar subquery (``seq > (SELECT max(seq))
// - W``) does NOT prune in DataFusion — the bound isn't constant at plan time,
// so the parquet row-group pruning predicate never forms and the engine
// full-scans every segment. A literal lets the pruning_predicate
// (``seq_max > floor``) drop all but the newest segments.
const RECENT_SEQ_WINDOW = 1000

async function load() {
  loading.value = true
  error.value = null
  try {
    const ns = props.name
    const schemaResp = await statsRpcCall<GetTableSchemaResponse>('GetTableSchema', { namespace: ns })
    schema.value = schemaResp.schema ?? null

    // Resolve the seq floor as a literal (see RECENT_SEQ_WINDOW note) so the
    // sample query prunes to the newest segments instead of full-scanning.
    const maxSeqResp = await statsRpcCall<QueryResponse>('Query', {
      sql: `SELECT max("seq") AS m FROM "${ns}"`,
    })
    const maxSeqCell = (decodeArrowIpc(maxSeqResp.arrowIpc).rows[0] as { m?: number | string } | undefined)?.m
    const maxSeq = maxSeqCell == null ? null : Number(maxSeqCell)
    const seqFloor = maxSeq == null ? 0 : maxSeq - RECENT_SEQ_WINDOW

    const orderBy = keyColumn.value ? `"${keyColumn.value}" DESC` : '"seq" DESC'
    const rows = await statsRpcCall<QueryResponse>('Query', {
      sql: `SELECT * FROM "${ns}" WHERE "seq" > ${seqFloor} ORDER BY ${orderBy} LIMIT 100`,
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
