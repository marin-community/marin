<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { apiGet, statsRpcCall } from '@/composables/useRpc'
import { timeZoneMode } from '@/composables/useDisplayPrefs'
import {
  shortColumnType,
  type ListNamespacesResponse,
  type NamespaceInfo,
  type ProtoSchema,
} from '@/types/stats'
import type {
  ForwardingResponse,
  ForwardingTargetInfo,
  SegmentInfo,
  SegmentsResponse,
} from '@/types/introspection'
import { formatBytes, formatNumber, formatTimestampMs } from '@/utils/formatting'
import { segmentIndexSummary } from '@/utils/segmentIndexes'
import InfoCard from '@/components/shared/InfoCard.vue'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'

const props = defineProps<{ name: string }>()
const router = useRouter()

interface TableSpec {
  version?: string | number
  operatingPolicy?: {
    l0Mode?: string
  }
}

interface TableMigrationStatus {
  fromVersion?: string | number
  toVersion?: string | number
  phase?: string
  rowsTotal?: string | number
  rowsCompleted?: string | number
}

interface GetTableStatusResponse {
  activeTableSpec?: TableSpec
  desiredTableSpec?: TableSpec
  migration?: TableMigrationStatus
  catalogGeneration?: string | number
  migrationBlocked?: boolean
  migrationError?: string
}

const schema = ref<ProtoSchema | null>(null)
const table = ref<NamespaceInfo | null>(null)
const status = ref<GetTableStatusResponse | null>(null)
const forwarding = ref<ForwardingResponse | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

const schemaRows = computed(() =>
  (schema.value?.columns ?? []).map((c) => ({
    column_name: c.name,
    column_type: shortColumnType(c.type),
    nullable: c.nullable ? 'YES' : 'NO',
    indexes: [
      c.index?.trigram ? 'trigram' : '',
      c.index?.valueCounts ? 'value counts' : '',
      c.index?.exactValues?.length ? `exact (${c.index.exactValues.length})` : '',
    ].filter(Boolean).join(', ') || '—',
  })),
)

const schemaColumns: Column[] = [
  { key: 'column_name', label: 'Column', mono: true },
  { key: 'column_type', label: 'Type', mono: true },
  { key: 'nullable', label: 'Nullable', align: 'center' },
  { key: 'indexes', label: 'Indexes' },
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
    const [list, tableStatus, forwardingStatus] = await Promise.all([
      statsRpcCall<ListNamespacesResponse>('ListNamespaces', {}),
      statsRpcCall<GetTableStatusResponse>('GetTableStatus', { namespace: ns }),
      apiGet<ForwardingResponse>('forwarding', { namespace: ns }),
    ])
    table.value = (list.namespaces ?? []).find((entry) => entry.namespace === ns) ?? null
    if (!table.value) throw new Error(`Table ${ns} is not registered`)
    schema.value = table.value.schema ?? null
    status.value = tableStatus
    forwarding.value = forwardingStatus
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

const tableFacts = computed(() => {
  const info = table.value
  const current = status.value
  if (!info || !current) return []
  const minSeq = Number(info.minSeq ?? 0)
  const maxSeq = Number(info.maxSeq ?? 0)
  const activeVersion = Number(current.activeTableSpec?.version ?? 0)
  const desiredVersion = Number(current.desiredTableSpec?.version ?? 0)
  const migration = current.migration
  let migrationValue = 'none'
  if (migration?.phase) {
    migrationValue = migration.phase.replace('MIGRATION_PHASE_', '').toLowerCase().replace(/_/g, ' ')
    const completed = Number(migration.rowsCompleted ?? 0)
    const total = Number(migration.rowsTotal ?? 0)
    if (total) migrationValue += ` · ${formatNumber(completed)} of ${formatNumber(total)} rows`
  }
  return [
    { label: 'Visible rows', value: formatNumber(Number(info.rowCount ?? 0)) },
    { label: 'Stored bytes', value: formatBytes(Number(info.byteSize ?? 0)) },
    { label: 'Sequence range', value: minSeq || maxSeq ? `${formatNumber(minSeq)}–${formatNumber(maxSeq)}` : 'empty' },
    { label: 'Segments', value: formatNumber(info.segmentCount ?? 0) },
    { label: 'Catalog generation', value: formatNumber(Number(current.catalogGeneration ?? 0)) },
    { label: 'Active spec', value: activeVersion ? `v${activeVersion}` : '—' },
    { label: 'Desired spec', value: desiredVersion ? `v${desiredVersion}` : '—' },
    { label: 'Migration', value: current.migrationBlocked ? 'blocked' : migrationValue, danger: current.migrationBlocked },
  ]
})

const policyFacts = computed(() => {
  const policy = table.value?.storagePolicy
  const s = schema.value
  const l0Mode = status.value?.activeTableSpec?.operatingPolicy?.l0Mode
  return [
    { label: 'Event key', value: keyColumn.value ?? '—', mono: true },
    { label: 'Sort order', value: s?.sortColumns?.join(', ') || keyColumn.value || '—', mono: true },
    { label: 'L0 durability', value: l0Mode?.replace('L0_MODE_', '').toLowerCase().replace(/_/g, ' ') || '—' },
    { label: 'Segment limit', value: policy?.maxSegments ? formatNumber(policy.maxSegments) : 'server default' },
    { label: 'Byte limit', value: Number(policy?.maxBytes ?? 0) ? formatBytes(Number(policy?.maxBytes)) : 'server default' },
    { label: 'Age limit', value: Number(policy?.maxAgeSeconds ?? 0) ? `${formatNumber(Number(policy?.maxAgeSeconds))}s` : 'disabled / server default' },
  ]
})

function targetState(target: ForwardingTargetInfo): string {
  if (target.settledCursor === null) return 'not seeded'
  if (target.forwardingLagSeqPositions === 0) return 'settled through published state'
  return `${formatNumber(target.forwardingLagSeqPositions ?? 0)} positions behind published state`
}

// Segments load separately from the table metadata: asking for physical
// detail reads a parquet footer per segment, which on a multi-GB namespace is
// hundreds of tail reads, and the rest of the page should not wait behind it.
const segments = ref<SegmentInfo[]>([])
const segmentsLoading = ref(false)
const segmentsError = ref<string | null>(null)

async function loadSegments() {
  segmentsLoading.value = true
  segmentsError.value = null
  try {
    const resp = await apiGet<SegmentsResponse>('segments', { namespace: props.name, physical: 'true' })
    segments.value = resp.segments
  } catch (e) {
    segmentsError.value = e instanceof Error ? e.message : String(e)
    segments.value = []
  } finally {
    segmentsLoading.value = false
  }
}

const segmentColumns: Column[] = [
  { key: 'segment', label: 'Segment', mono: true },
  { key: 'level', label: 'Level', align: 'center' },
  { key: 'rows', label: 'Rows', numeric: true },
  { key: 'size', label: 'Size', numeric: true },
  { key: 'row_groups', label: 'Row groups', numeric: true },
  { key: 'footer', label: 'Footer', numeric: true },
  { key: 'layout', label: 'Layout', align: 'center' },
  { key: 'indexes', label: 'Indexes' },
  { key: 'index_size', label: 'Index size', numeric: true },
  { key: 'location', label: 'Location', align: 'center' },
  { key: 'created', label: 'Created' },
]

const segmentRows = computed(() =>
  segments.value.map((s) => ({
    segment: s.path.replace(/\.parquet$/, ''),
    level: `L${s.level}`,
    rows: formatNumber(s.rowCount),
    size: formatBytes(s.byteSize),
    row_groups: s.physical ? formatNumber(s.physical.rowGroups) : '—',
    footer: s.physical ? formatBytes(s.physical.footerBytes) : '—',
    layout: layoutLabel(s),
    indexes:
      s.physical?.indexBundle?.sections
        .map((section) => {
          const columns = section.columns.length ? ` [${section.columns.join(', ')}]` : ''
          return section.id + columns + (section.available ? '' : ' (missing)')
        })
        .join(', ') || '—',
    index_size: s.physical?.indexBundle
      ? formatBytes(s.physical.indexBundle.bytes + s.physical.indexBundle.externalBytes)
      : '—',
    location: s.location.toLowerCase(),
    created: formatTimestampMs(s.createdAtMs, timeZoneMode.value),
  })),
)

/** `v1`, or the stale revision with what it is waiting for. */
function layoutLabel(s: SegmentInfo): string {
  if (!s.physical) return '—'
  const version = s.physical.layoutVersion
  if (s.physical.layoutCurrent) return `v${version}`
  return version === undefined ? 'unstamped' : `v${version}`
}

/** One line per level: how many segments it holds and how big they are. */
const levelSummary = computed(() => {
  const byLevel = new Map<number, { segments: number; rows: number; bytes: number }>()
  for (const s of segments.value) {
    const entry = byLevel.get(s.level) ?? { segments: 0, rows: 0, bytes: 0 }
    entry.segments += 1
    entry.rows += s.rowCount
    entry.bytes += s.byteSize
    byLevel.set(s.level, entry)
  }
  return [...byLevel.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([level, e]) => ({
      level: `L${level}`,
      detail: `${formatNumber(e.segments)} segments · ${formatNumber(e.rows)} rows · ${formatBytes(e.bytes)}`,
    }))
})

/** Physical-layout and planner-facing index backfill progress. */
const backfillSummary = computed(() => {
  const inspected = segments.value.filter((s) => s.physical)
  if (!inspected.length) return null
  const current = inspected.filter((s) => s.physical?.layoutCurrent).length
  const indexes = segmentIndexSummary(segments.value, schema.value)
  const footerBytes = inspected.reduce((total, s) => total + (s.physical?.footerBytes ?? 0), 0)
  const bytes = inspected.reduce((total, s) => total + s.byteSize, 0)
  return {
    layout: `${formatNumber(current)} of ${formatNumber(inspected.length)}`,
    indexes,
    footer: `${formatBytes(footerBytes)} of ${formatBytes(bytes)}`,
  }
})

function openInQuery() {
  const sql = `SELECT * FROM "${props.name}" LIMIT 100`
  router.push({ path: '/query', query: { sql } })
}

function loadAll() {
  load()
  loadSegments()
}

onMounted(loadAll)
watch(() => props.name, loadAll)
</script>

<template>
  <div class="space-y-3">
    <div class="flex items-center justify-between">
      <div>
        <RouterLink to="/" class="text-xs text-text-muted hover:text-text">← Tables</RouterLink>
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

    <InfoCard title="Table status">
      <div class="grid gap-x-8 gap-y-3 sm:grid-cols-2 lg:grid-cols-4">
        <div v-for="fact in tableFacts" :key="fact.label">
          <div class="text-[10px] uppercase tracking-wider text-text-muted">{{ fact.label }}</div>
          <div
            class="mt-0.5 text-sm tabular-nums"
            :class="fact.danger ? 'text-status-danger' : 'text-text'"
          >{{ fact.value }}</div>
        </div>
      </div>
      <p
        v-if="status?.migrationBlocked && status.migrationError"
        class="text-xs text-status-danger bg-status-danger-bg border border-status-danger-border rounded px-2.5 py-2"
      >{{ status.migrationError }}</p>
    </InfoCard>

    <InfoCard title="Storage policy">
      <dl class="grid gap-x-8 gap-y-2 sm:grid-cols-2 lg:grid-cols-3">
        <div v-for="fact in policyFacts" :key="fact.label" class="flex gap-3 text-sm items-baseline">
          <dt class="text-text-muted shrink-0">{{ fact.label }}</dt>
          <dd class="min-w-0 break-all" :class="fact.mono ? 'font-mono text-xs' : ''">{{ fact.value }}</dd>
        </div>
      </dl>
    </InfoCard>

    <InfoCard title="Forwarding">
      <div v-if="!forwarding" class="text-sm text-text-muted">
        {{ loading ? 'Loading…' : 'Unavailable.' }}
      </div>
      <div v-else-if="!forwarding.configured" class="text-sm text-text-muted">
        Forwarding is not configured on this server.
      </div>
      <template v-else>
        <div class="grid gap-x-8 gap-y-3 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <div class="text-[10px] uppercase tracking-wider text-text-muted">Source cluster</div>
            <div class="mt-0.5 text-sm font-mono">{{ forwarding.cluster || '—' }}</div>
          </div>
          <div>
            <div class="text-[10px] uppercase tracking-wider text-text-muted">Visible high-water</div>
            <div class="mt-0.5 text-sm tabular-nums">{{ formatNumber(forwarding.visibleHighWater) }}</div>
          </div>
          <div>
            <div class="text-[10px] uppercase tracking-wider text-text-muted">Published high-water</div>
            <div class="mt-0.5 text-sm tabular-nums">{{ formatNumber(forwarding.publishedHighWater) }}</div>
          </div>
          <div>
            <div class="text-[10px] uppercase tracking-wider text-text-muted">Publication lag</div>
            <div
              class="mt-0.5 text-sm tabular-nums"
              :class="forwarding.publicationLagSeqPositions ? 'text-status-warning' : 'text-status-success'"
            >{{ formatNumber(forwarding.publicationLagSeqPositions) }} seq positions</div>
          </div>
        </div>

        <div
          v-if="forwarding.target"
          class="border-t border-surface-border-subtle"
        >
          <div
            class="grid gap-2 py-2.5 text-sm md:grid-cols-[minmax(0,1fr)_10rem_minmax(13rem,auto)] md:items-baseline"
          >
            <div class="font-mono text-xs break-all">{{ forwarding.target.target }}</div>
            <div class="tabular-nums text-text-secondary">
              settled {{ forwarding.target.settledCursor === null ? '—' : formatNumber(forwarding.target.settledCursor) }}
            </div>
            <div
              :class="forwarding.target.forwardingLagSeqPositions ? 'text-status-warning' : 'text-text-secondary'"
            >{{ targetState(forwarding.target) }}</div>
          </div>
        </div>
      </template>
    </InfoCard>

    <InfoCard title="Schema">
      <DataTable
        :columns="schemaColumns"
        :rows="schemaRows"
        :loading="loading && !schema"
        empty-message="No columns."
      />
    </InfoCard>

    <InfoCard title="Segments">
      <div
        v-if="segmentsError"
        class="text-sm text-status-danger"
      >{{ segmentsError }}</div>

      <div v-if="levelSummary.length" class="flex flex-wrap gap-x-6 gap-y-1 text-xs text-text-muted pb-1">
        <span v-for="l in levelSummary" :key="l.level">
          <span class="font-mono text-text">{{ l.level }}</span> {{ l.detail }}
        </span>
      </div>
      <div v-if="backfillSummary" class="flex flex-wrap gap-x-6 gap-y-1 text-xs text-text-muted pb-2">
        <span>current layout <span class="text-text">{{ backfillSummary.layout }}</span></span>
        <span v-if="backfillSummary.indexes?.methods.length">
          local L1+ full index policy
          <span class="text-text">
            {{ formatNumber(backfillSummary.indexes.stableIndexed) }} of
            {{ formatNumber(backfillSummary.indexes.stableEligible) }}
          </span>
        </span>
        <span v-if="backfillSummary.indexes?.l0Unindexed">
          L0 unindexed by design
          <span class="text-text">{{ formatNumber(backfillSummary.indexes.l0Unindexed) }}</span>
        </span>
        <span v-if="backfillSummary.indexes">
          index storage <span class="text-text">{{ formatBytes(backfillSummary.indexes.bytes) }}</span>
        </span>
        <span>footer <span class="text-text">{{ backfillSummary.footer }}</span></span>
      </div>
      <div
        v-if="backfillSummary?.indexes"
        class="flex flex-wrap gap-x-6 gap-y-1 text-xs text-text-muted pb-2"
      >
        <span v-for="method in backfillSummary.indexes.methods" :key="method.id">
          <span class="font-mono text-text">{{ method.id }}</span>
          {{ formatNumber(method.indexed) }} of {{ formatNumber(method.eligible) }}
        </span>
      </div>
      <div
        v-if="backfillSummary?.indexes?.adaptiveMethods.length"
        class="flex flex-wrap gap-x-6 gap-y-1 text-xs text-text-muted pb-2"
      >
        <span>adaptive summaries</span>
        <span v-for="method in backfillSummary.indexes.adaptiveMethods" :key="method.id">
          <span class="font-mono text-text">{{ method.id }}</span>
          built on {{ formatNumber(method.indexed) }} of {{ formatNumber(method.eligible) }} local L1+
        </span>
      </div>
      <div
        v-if="backfillSummary?.indexes?.countColumns.length"
        class="flex flex-wrap gap-x-6 gap-y-1 text-xs text-text-muted pb-2"
      >
        <span>count summary columns</span>
        <span v-for="column in backfillSummary.indexes.countColumns" :key="column.id">
          <span class="font-mono text-text">{{ column.id }}</span>
          {{ formatNumber(column.indexed) }} of {{ formatNumber(column.eligible) }}
        </span>
      </div>

      <DataTable
        :columns="segmentColumns"
        :rows="segmentRows"
        :loading="segmentsLoading"
        :page-size="25"
        empty-message="No segments — every row is still in the write buffer."
      />
    </InfoCard>

  </div>
</template>
