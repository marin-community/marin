<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { statsRpcCall } from '@/composables/useRpc'
import { timeZoneMode } from '@/composables/useDisplayPrefs'
import { decodeArrowIpc, type ArrowResult } from '@/utils/arrow'
import { classifyColumns, type ColumnKind } from '@/utils/columnKind'
import { formatMetric, formatTimestampMs } from '@/utils/formatting'
import InfoCard from '@/components/shared/InfoCard.vue'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import CellDetailPanel from '@/components/shared/CellDetailPanel.vue'
import ResultChart, { type ChartMark } from '@/components/shared/ResultChart.vue'

interface QueryResponse {
  arrowIpc?: string
  rowCount?: string | number
}

/** Above this many distinct values a column stops being a useful series split. */
const MAX_SERIES_CARDINALITY = 12

const route = useRoute()
const sql = ref<string>(typeof route.query.sql === 'string' ? route.query.sql : 'SELECT 1')
const result = ref<ArrowResult>({ columns: [], types: {}, rows: [] })
const rowCount = ref<number>(0)
const elapsedMs = ref<number | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

const view = ref<'table' | 'chart'>('table')
const chartX = ref('')
const chartY = ref('')
const chartSeries = ref('')
const chartMark = ref<ChartMark>('line')
const detail = ref<{ column: string; row: Record<string, unknown> } | null>(null)

const kinds = computed<Record<string, ColumnKind>>(() =>
  classifyColumns(result.value.columns, result.value.types, result.value.rows),
)

const numericColumns = computed(() =>
  result.value.columns.filter((c) => kinds.value[c] === 'number'),
)

/** Text columns with few enough distinct values to read as a legend. */
const seriesCandidates = computed(() =>
  result.value.columns.filter((c) => {
    if (kinds.value[c] !== 'text') return false
    const distinct = new Set(result.value.rows.map((r) => String(r[c] ?? '')))
    return distinct.size > 1 && distinct.size <= MAX_SERIES_CARDINALITY
  }),
)

const columns = computed<Column[]>(() =>
  result.value.columns.map((c) => ({
    key: c,
    label: c,
    // Mono for data that is scanned character by character — identifiers,
    // numbers, JSON. Prose stays in the body face, where it is easier to read.
    mono: kinds.value[c] !== 'text',
    numeric: kinds.value[c] === 'number',
  })),
)

/** Rows with timestamps rendered for display; the chart reads the raw values. */
const displayRows = computed(() => {
  const timestampColumns = result.value.columns.filter((c) => kinds.value[c] === 'timestamp')
  if (timestampColumns.length === 0 && !result.value.columns.some((c) => kinds.value[c] === 'number')) {
    return result.value.rows
  }
  return result.value.rows.map((row) => {
    const out: Record<string, unknown> = { ...row }
    for (const c of timestampColumns) {
      if (typeof row[c] === 'number') out[c] = formatTimestampMs(row[c] as number, timeZoneMode.value)
    }
    for (const c of result.value.columns) {
      if (kinds.value[c] === 'number' && typeof row[c] === 'number') out[c] = formatMetric(row[c] as number)
    }
    return out
  })
})

/**
 * Seed the chart axes from the result's own shape.
 *
 * y is always a measurement, so it is claimed first; x is then whatever the
 * result offers to plot it against, in the order those columns actually carry
 * meaning — an instant, then a label, then another number.
 */
function pickChartDefaults() {
  const cols = result.value.columns
  chartY.value = numericColumns.value[0] ?? ''

  const timestamp = cols.find((c) => kinds.value[c] === 'timestamp')
  const label = seriesCandidates.value[0] ?? cols.find((c) => kinds.value[c] === 'text')
  const otherNumber = numericColumns.value.find((c) => c !== chartY.value)
  chartX.value = timestamp ?? label ?? otherNumber ?? cols[0] ?? ''

  // A column already spent on x cannot also split the series.
  chartSeries.value = seriesCandidates.value.find((c) => c !== chartX.value) ?? ''
  chartMark.value = kinds.value[chartX.value] === 'timestamp' ? 'line' : 'bar'
}

const canChart = computed(() => numericColumns.value.length > 0 && result.value.rows.length > 0)

async function execute() {
  if (!sql.value.trim()) return
  loading.value = true
  error.value = null
  detail.value = null
  const started = performance.now()
  try {
    const resp = await statsRpcCall<QueryResponse>('Query', { sql: sql.value })
    result.value = decodeArrowIpc(resp.arrowIpc)
    rowCount.value = Number(resp.rowCount ?? result.value.rows.length)
    elapsedMs.value = performance.now() - started
    pickChartDefaults()
    if (!canChart.value) view.value = 'table'
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    result.value = { columns: [], types: {}, rows: [] }
    rowCount.value = 0
    elapsedMs.value = null
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

watch(chartX, () => {
  if (chartY.value === chartX.value) {
    chartY.value = numericColumns.value.find((c) => c !== chartX.value) ?? ''
  }
})

onMounted(() => {
  if (typeof route.query.sql === 'string' && route.query.sql.trim()) void execute()
})
</script>

<template>
  <div class="space-y-3">
    <InfoCard title="SQL · DataFusion">
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
        <span v-if="!loading && !error && elapsedMs !== null" class="text-xs text-text-muted ml-auto tabular-nums">
          {{ rowCount.toLocaleString() }} row{{ rowCount === 1 ? '' : 's' }} · {{ Math.round(elapsedMs) }} ms
        </span>
      </div>
    </InfoCard>

    <div
      v-if="error"
      class="px-4 py-3 text-sm font-mono text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border whitespace-pre-wrap"
    >{{ error }}</div>

    <InfoCard v-if="!error" title="Result">
      <template #default>
        <div class="flex flex-wrap items-center gap-2 pb-2 -mt-1">
          <div class="inline-flex rounded border border-surface-border overflow-hidden">
            <button
              v-for="mode in (['table', 'chart'] as const)"
              :key="mode"
              class="px-2.5 py-1 text-xs capitalize disabled:opacity-40 disabled:cursor-not-allowed"
              :class="view === mode ? 'bg-accent text-white' : 'hover:bg-surface-raised'"
              :disabled="mode === 'chart' && !canChart"
              :title="mode === 'chart' && !canChart ? 'Charting needs a numeric column' : undefined"
              @click="view = mode"
            >{{ mode }}</button>
          </div>

          <template v-if="view === 'chart'">
            <label class="text-xs text-text-secondary">x
              <select v-model="chartX" class="ml-1 text-xs bg-surface-sunken border border-surface-border rounded px-1.5 py-1 font-mono">
                <option v-for="c in result.columns" :key="c" :value="c">{{ c }}</option>
              </select>
            </label>
            <label class="text-xs text-text-secondary">y
              <select v-model="chartY" class="ml-1 text-xs bg-surface-sunken border border-surface-border rounded px-1.5 py-1 font-mono">
                <option v-for="c in numericColumns" :key="c" :value="c">{{ c }}</option>
              </select>
            </label>
            <label class="text-xs text-text-secondary">series
              <select v-model="chartSeries" class="ml-1 text-xs bg-surface-sunken border border-surface-border rounded px-1.5 py-1 font-mono">
                <option value="">none</option>
                <option v-for="c in seriesCandidates" :key="c" :value="c">{{ c }}</option>
              </select>
            </label>
            <div class="inline-flex rounded border border-surface-border overflow-hidden ml-auto">
              <button
                v-for="m in (['line', 'bar', 'scatter'] as const)"
                :key="m"
                class="px-2.5 py-1 text-xs capitalize"
                :class="chartMark === m ? 'bg-accent text-white' : 'hover:bg-surface-raised'"
                @click="chartMark = m"
              >{{ m }}</button>
            </div>
          </template>
          <span v-else-if="result.rows.length" class="text-xs text-text-muted ml-auto">
            Click a cell for its full value
          </span>
        </div>

        <ResultChart
          v-if="view === 'chart'"
          :rows="result.rows"
          :kinds="kinds"
          :x="chartX"
          :y="chartY"
          :series="chartSeries || undefined"
          :mark="chartMark"
        />
        <DataTable
          v-else
          :columns="columns"
          :rows="displayRows"
          :loading="loading"
          :page-size="50"
          inspectable
          empty-message="No rows."
          @inspect="(column, row) => (detail = { column, row: result.rows[displayRows.indexOf(row)] ?? row })"
        />
      </template>
    </InfoCard>

    <CellDetailPanel
      v-if="detail"
      :column="detail.column"
      :row="detail.row"
      :kinds="kinds"
      @close="detail = null"
    />
  </div>
</template>
