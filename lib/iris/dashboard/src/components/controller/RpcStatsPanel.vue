<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useLogServerStatsRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import { formatRelativeTime, formatBytes } from '@/utils/formatting'
import { decodeArrowIpc } from '@/utils/arrow'
import InfoCard from '@/components/shared/InfoCard.vue'

interface TelltaleRow {
  name?: string
  value?: number
  ts?: string
  service?: string
  method?: string
  upstream?: string
  status?: string
  le?: string
  scope?: string
  endpoint?: string
  route_kind?: string
}

const METRIC_NAMES = {
  requests: 'iris_rpc_requests_total',
  responses: 'iris_rpc_responses_total',
  inFlight: 'iris_rpc_in_flight',
  durationBucket: 'iris_rpc_duration_seconds_bucket',
  durationSum: 'iris_rpc_duration_seconds_sum',
  durationCount: 'iris_rpc_duration_seconds_count',
} as const

const PROXY_METRIC_NAMES = {
  requests: 'iris_proxy_requests_total',
  responses: 'iris_proxy_responses_total',
  inFlight: 'iris_proxy_in_flight',
  durationBucket: 'iris_proxy_duration_seconds_bucket',
  durationSum: 'iris_proxy_duration_seconds_sum',
  durationCount: 'iris_proxy_duration_seconds_count',
  requestBytes: 'iris_proxy_request_bytes_total',
  responseBytes: 'iris_proxy_response_bytes_total',
} as const

// Trailing window: only series written in the last five minutes are shown, so a retired
// method or a gone endpoint (its last row now older than the window) drops out
// instead of lingering forever as the latest row for its label set. DataFusion
// folds now() to a timestamp literal, which exposes the direct ts predicate to
// Parquet row-group pruning.
const RECENT_STATS_FILTER = "ts >= now() - INTERVAL '5 minutes'"

// Scope to this cluster's own rows. A hub finelog also holds child rows stamped
// with their origin cluster; local writes leave `cluster` empty/NULL, so this
// filter shows only the cluster whose dashboard is open (no cross-cluster mixing).
const LOCAL_CLUSTER_FILTER = "(cluster IS NULL OR cluster = '')"

function statsSql(): string {
  const names = Object.values(METRIC_NAMES).map((name) => `'${name}'`).join(', ')
  return `
SELECT name, value, ts,
       json_get(labels, 'service') AS service,
       json_get(labels, 'method') AS method,
       json_get(labels, 'upstream') AS upstream,
       json_get(labels, 'status') AS status,
       json_get(labels, 'le') AS le
FROM telltale
WHERE source = 'iris' AND name IN (${names})
  AND ${RECENT_STATS_FILTER}
  AND ${LOCAL_CLUSTER_FILTER}
QUALIFY row_number() OVER (
  PARTITION BY name, json_get(labels, 'service'), json_get(labels, 'method'),
               json_get(labels, 'upstream'), json_get(labels, 'status'), json_get(labels, 'le')
  ORDER BY ts DESC
) = 1`.trim()
}

function proxySql(): string {
  const names = Object.values(PROXY_METRIC_NAMES).map((name) => `'${name}'`).join(', ')
  return `
SELECT name, value, ts,
       json_get(labels, 'scope') AS scope,
       json_get(labels, 'endpoint') AS endpoint,
       json_get(labels, 'method') AS method,
       json_get(labels, 'route_kind') AS route_kind,
       json_get(labels, 'status') AS status,
       json_get(labels, 'le') AS le
FROM telltale
WHERE source = 'iris' AND name IN (${names})
  AND ${RECENT_STATS_FILTER}
  AND ${LOCAL_CLUSTER_FILTER}
QUALIFY row_number() OVER (
  PARTITION BY name, json_get(labels, 'scope'), json_get(labels, 'endpoint'),
               json_get(labels, 'method'), json_get(labels, 'route_kind'),
               json_get(labels, 'status'), json_get(labels, 'le')
  ORDER BY ts DESC
) = 1`.trim()
}

const { data, loading, error, refresh } = useLogServerStatsRpc<{ arrowIpc?: string }>('Query', () => ({ sql: statsSql() }))
const { data: proxyData, refresh: refreshProxy } = useLogServerStatsRpc<{ arrowIpc?: string }>('Query', () => ({ sql: proxySql() }))
function refreshAll() {
  refresh()
  refreshProxy()
}
useAutoRefresh(refreshAll, DEFAULT_REFRESH_MS)
onMounted(refreshAll)

type SortKey = 'method' | 'count' | 'errorCount' | 'inFlight' | 'p50' | 'p95' | 'p99' | 'last'
const sortKey = ref<SortKey>('p95')
const sortDir = ref<'asc' | 'desc'>('desc')
const expanded = ref<Set<string>>(new Set())

interface MethodRow {
  method: string
  count: number
  errorCount: number
  inFlight: number
  p50: number
  p95: number
  p99: number
  last: number
  buckets: number[]
  bounds: number[]
  totalDurationMs: number
  durationCount: number
  bucketSamples: Array<{ bound: number; cumulative: number }>
}

// Proxy transport rows share the histogram machinery (finalizeHistogram, sparkBars)
// and add endpoint/route-kind identity plus request/response byte volume.
interface ProxyRow extends MethodRow {
  endpoint: string
  routeKind: string
  requestBytes: number
  responseBytes: number
}

function percentileBound(samples: MethodRow['bucketSamples'], count: number, quantile: number): number {
  if (!count) return 0
  const target = count * quantile
  for (const sample of samples) {
    if (sample.cumulative >= target) return sample.bound
  }
  return samples.at(-1)?.bound ?? 0
}

function finalizeHistogram(row: MethodRow): void {
  row.bucketSamples.sort((a, b) => {
    if (!a.bound) return 1
    if (!b.bound) return -1
    return a.bound - b.bound
  })
  let previous = 0
  for (const sample of row.bucketSamples) {
    row.bounds.push(sample.bound)
    row.buckets.push(Math.max(0, sample.cumulative - previous))
    previous = sample.cumulative
  }
  row.p50 = percentileBound(row.bucketSamples, row.durationCount, 0.5)
  row.p95 = percentileBound(row.bucketSamples, row.durationCount, 0.95)
  row.p99 = percentileBound(row.bucketSamples, row.durationCount, 0.99)
}

const rows = computed<MethodRow[]>(() => {
  const metrics = decodeArrowIpc(data.value?.arrowIpc).rows as TelltaleRow[]
  const byMethod = new Map<string, MethodRow>()
  for (const metric of metrics) {
    if (!metric.service || !metric.method || !metric.upstream) continue
    const method = `${metric.service}/${metric.method} (${metric.upstream})`
    const row = byMethod.get(method) ?? {
      method, count: 0, errorCount: 0, inFlight: 0, p50: 0, p95: 0, p99: 0, last: 0,
      buckets: [], bounds: [], totalDurationMs: 0, durationCount: 0, bucketSamples: [],
    }
    row.last = Math.max(row.last, new Date(metric.ts ?? 0).getTime())
    const value = Number(metric.value ?? 0)
    if (metric.name === METRIC_NAMES.requests) row.count = value
    if (metric.name === METRIC_NAMES.responses && metric.status !== '200') row.errorCount += value
    if (metric.name === METRIC_NAMES.inFlight) row.inFlight = value
    if (metric.name === METRIC_NAMES.durationSum) row.totalDurationMs = value * 1000
    if (metric.name === METRIC_NAMES.durationCount) row.durationCount = value
    if (metric.name === METRIC_NAMES.durationBucket) {
      row.bucketSamples.push({
        bound: metric.le === '+Inf' ? 0 : Number(metric.le ?? 0) * 1000,
        cumulative: value,
      })
    }
    byMethod.set(method, row)
  }
  const list = [...byMethod.values()]
  for (const row of list) finalizeHistogram(row)
  const dir = sortDir.value === 'asc' ? 1 : -1
  const key = sortKey.value
  return [...list].sort((a, b) => {
    if (key === 'method') return a.method.localeCompare(b.method) * dir
    return (a[key] - b[key]) * dir
  })
})

const TOP_PROXY_ENDPOINTS = 5

function emptyProxyRow(key: string, endpoint: string, routeKind: string): ProxyRow {
  return {
    method: key, endpoint, routeKind,
    count: 0, errorCount: 0, inFlight: 0, p50: 0, p95: 0, p99: 0, last: 0,
    buckets: [], bounds: [], totalDurationMs: 0, durationCount: 0, bucketSamples: [],
    requestBytes: 0, responseBytes: 0,
  }
}

function applyProxySample(row: ProxyRow, metric: TelltaleRow): void {
  row.last = Math.max(row.last, new Date(metric.ts ?? 0).getTime())
  const value = Number(metric.value ?? 0)
  if (metric.name === PROXY_METRIC_NAMES.requests) row.count = value
  if (metric.name === PROXY_METRIC_NAMES.responses && metric.status !== '200') row.errorCount += value
  if (metric.name === PROXY_METRIC_NAMES.inFlight) row.inFlight = value
  if (metric.name === PROXY_METRIC_NAMES.durationSum) row.totalDurationMs = value * 1000
  if (metric.name === PROXY_METRIC_NAMES.durationCount) row.durationCount = value
  if (metric.name === PROXY_METRIC_NAMES.requestBytes) row.requestBytes = value
  if (metric.name === PROXY_METRIC_NAMES.responseBytes) row.responseBytes = value
  if (metric.name === PROXY_METRIC_NAMES.durationBucket) {
    row.bucketSamples.push({
      bound: metric.le === '+Inf' ? 0 : Number(metric.le ?? 0) * 1000,
      cumulative: value,
    })
  }
}

// scope=total is the exact aggregate over all proxied traffic; scope=endpoint is
// the bounded per-endpoint breakdown the native proxy emits.
const proxyParsed = computed<{ total: ProxyRow | null; endpoints: ProxyRow[] }>(() => {
  const metrics = decodeArrowIpc(proxyData.value?.arrowIpc).rows as TelltaleRow[]
  let total: ProxyRow | null = null
  const byEndpoint = new Map<string, ProxyRow>()
  for (const metric of metrics) {
    if (metric.scope === 'total') {
      total = total ?? emptyProxyRow('total', '', '')
      applyProxySample(total, metric)
    } else if (metric.scope === 'endpoint' && metric.endpoint) {
      const routeKind = metric.route_kind ?? ''
      const key = `${metric.endpoint} ${metric.method ?? ''} ${routeKind}`
      const row = byEndpoint.get(key) ?? emptyProxyRow(key, metric.endpoint, routeKind)
      applyProxySample(row, metric)
      byEndpoint.set(key, row)
    }
  }
  const endpoints = [...byEndpoint.values()]
  for (const row of endpoints) finalizeHistogram(row)
  if (total) finalizeHistogram(total)
  endpoints.sort((a, b) => b.count - a.count)
  return { total, endpoints }
})

const proxyTotal = computed(() => proxyParsed.value.total)
const proxyEndpoints = computed(() => proxyParsed.value.endpoints.slice(0, TOP_PROXY_ENDPOINTS))
const proxyEndpointCount = computed(() => proxyParsed.value.endpoints.length)

const totalCount = computed(() => rows.value.reduce((a, r) => a + r.count, 0))
const totalErrors = computed(() => rows.value.reduce((a, r) => a + r.errorCount, 0))
const totalInFlight = computed(() => rows.value.reduce((a, r) => a + r.inFlight, 0))
const lastUpdatedAt = computed(() => rows.value.reduce((latest, row) => Math.max(latest, row.last), 0))

function toggleExpand(method: string) {
  const next = new Set(expanded.value)
  if (next.has(method)) next.delete(method)
  else next.add(method)
  expanded.value = next
}

function setSort(key: SortKey) {
  if (sortKey.value === key) {
    sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortKey.value = key
    sortDir.value = key === 'method' ? 'asc' : 'desc'
  }
}

function fmtMs(value: number): string {
  if (!value) return '—'
  if (value < 10) return value.toFixed(1)
  if (value < 1000) return Math.round(value).toString()
  return (value / 1000).toFixed(value < 10000 ? 2 : 1) + 's'
}

// Reuse the shared byte formatter; keep the table's em-dash for zero/absent.
function fmtBytes(value: number): string {
  return value ? formatBytes(value) : '—'
}

function fmtBound(ms: number): string {
  if (!ms) return '+∞'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(ms % 1000 === 0 ? 0 : 1)}s`
}

function fmtSince(epochMs: number): string {
  if (!epochMs) return '—'
  return formatRelativeTime(epochMs)
}

function methodLabel(method: string): string {
  const slash = method.lastIndexOf('/')
  return slash >= 0 ? method.slice(slash + 1) : method
}

function methodService(method: string): string {
  const slash = method.lastIndexOf('/')
  return slash >= 0 ? method.slice(0, slash) : ''
}

function sortIndicator(key: SortKey): string {
  if (sortKey.value !== key) return ''
  return sortDir.value === 'asc' ? '▲' : '▼'
}

interface Sparkbar {
  height: number
  count: number
  upper: number
  isInf: boolean
}

// Log-shade very small counts so a single tail sample remains visible
// against a large mode bucket; pure linear scaling renders p99 tails as
// invisible slivers.
function sparkBars(row: MethodRow): Sparkbar[] {
  const max = Math.max(...row.buckets, 1)
  return row.buckets.map((c, i) => ({
    height: c === 0 ? 0 : Math.max(6, Math.round((Math.log1p(c) / Math.log1p(max)) * 100)),
    count: c,
    upper: row.bounds[i] ?? 0,
    isInf: (row.bounds[i] ?? 0) === 0,
  }))
}

// Locate the bucket whose right edge first crosses `pct` ms, so we can
// draw a p50/p95/p99 tick over the histogram.
function percentileBucket(row: MethodRow, pct: number): number | null {
  if (!pct) return null
  for (let i = 0; i < row.bounds.length; i++) {
    const upper = row.bounds[i]
    if (upper === 0) return i
    if (upper >= pct) return i
  }
  return row.bounds.length - 1
}

function markerLeft(row: MethodRow, pct: number): string | null {
  const idx = percentileBucket(row, pct)
  if (idx === null || !row.buckets.length) return null
  return `${((idx + 0.5) / row.buckets.length) * 100}%`
}

function errorRate(row: MethodRow): string {
  if (!row.count) return '0%'
  const pct = (row.errorCount / row.count) * 100
  if (pct === 0) return '0%'
  if (pct < 0.1) return '<0.1%'
  return pct.toFixed(pct < 10 ? 1 : 0) + '%'
}

function avgMs(row: MethodRow): number {
  return row.durationCount ? row.totalDurationMs / row.durationCount : 0
}
</script>

<template>
  <div class="space-y-4">
    <div
      v-if="error"
      class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <!-- Summary strip -->
    <div class="grid grid-cols-2 md:grid-cols-5 gap-2">
      <div class="rounded-lg border border-surface-border bg-surface px-3 py-2">
        <div class="text-[10px] uppercase tracking-wider text-text-muted">Methods</div>
        <div class="font-mono text-lg text-text">{{ rows.length }}</div>
      </div>
      <div class="rounded-lg border border-surface-border bg-surface px-3 py-2">
        <div class="text-[10px] uppercase tracking-wider text-text-muted">Calls</div>
        <div class="font-mono text-lg text-text">{{ totalCount.toLocaleString() }}</div>
      </div>
      <div class="rounded-lg border border-surface-border bg-surface px-3 py-2">
        <div class="text-[10px] uppercase tracking-wider text-text-muted">In flight</div>
        <div class="font-mono text-lg text-text">{{ totalInFlight.toLocaleString() }}</div>
      </div>
      <div class="rounded-lg border border-surface-border bg-surface px-3 py-2">
        <div class="text-[10px] uppercase tracking-wider text-text-muted">Errors</div>
        <div
          class="font-mono text-lg"
          :class="totalErrors > 0 ? 'text-status-danger' : 'text-text'"
        >
          {{ totalErrors.toLocaleString() }}
        </div>
      </div>
      <div class="rounded-lg border border-surface-border bg-surface px-3 py-2">
        <div class="text-[10px] uppercase tracking-wider text-text-muted">Updated</div>
        <div class="font-mono text-sm text-text pt-1">{{ fmtSince(lastUpdatedAt) }}</div>
      </div>
    </div>

    <InfoCard title="RPC Methods">
      <p class="text-xs text-text-muted mb-3">
        Native-proxy counters, statuses, in-flight requests, and latency histograms,
        queried from Telltale snapshots in the controller log server. Counters are
        lifetime totals for the current native-proxy process and reset when it restarts.
      </p>

      <!-- Header -->
      <div
        class="grid items-center gap-x-3 px-2 py-1.5 text-[10px] uppercase tracking-wider text-text-muted border-b border-surface-border"
        style="grid-template-columns: minmax(0,1fr) 72px 64px 90px 64px 64px 64px 200px 72px"
      >
        <button class="text-left cursor-pointer hover:text-text" @click="setSort('method')">
          Method <span class="text-accent">{{ sortIndicator('method') }}</span>
        </button>
        <button class="text-right cursor-pointer hover:text-text" @click="setSort('count')">
          Count <span class="text-accent">{{ sortIndicator('count') }}</span>
        </button>
        <button class="text-right cursor-pointer hover:text-text" @click="setSort('inFlight')">
          Live <span class="text-accent">{{ sortIndicator('inFlight') }}</span>
        </button>
        <button class="text-right cursor-pointer hover:text-text" @click="setSort('errorCount')">
          Err <span class="text-accent">{{ sortIndicator('errorCount') }}</span>
        </button>
        <button class="text-right cursor-pointer hover:text-text" @click="setSort('p50')">
          p50 <span class="text-accent">{{ sortIndicator('p50') }}</span>
        </button>
        <button class="text-right cursor-pointer hover:text-text" @click="setSort('p95')">
          p95 <span class="text-accent">{{ sortIndicator('p95') }}</span>
        </button>
        <button class="text-right cursor-pointer hover:text-text" @click="setSort('p99')">
          p99 <span class="text-accent">{{ sortIndicator('p99') }}</span>
        </button>
        <div class="text-center">Distribution</div>
        <button class="text-right cursor-pointer hover:text-text" @click="setSort('last')">
          Last <span class="text-accent">{{ sortIndicator('last') }}</span>
        </button>
      </div>

      <div v-if="loading && !data" class="text-sm text-text-muted py-4 text-center">Loading…</div>
      <div v-else-if="!rows.length" class="text-sm text-text-muted py-4 text-center">
        No RPCs recorded yet.
      </div>

      <div v-else class="divide-y divide-surface-border">
        <div v-for="row in rows" :key="row.method">
          <button
            class="w-full grid items-center gap-x-3 px-2 py-2 text-sm text-left hover:bg-surface-raised transition-colors cursor-pointer"
            style="grid-template-columns: minmax(0,1fr) 72px 64px 90px 64px 64px 64px 200px 72px"
            @click="toggleExpand(row.method)"
          >
            <div class="flex items-center gap-1.5 min-w-0">
              <span
                class="text-text-muted text-xs transition-transform inline-block w-3"
                :class="{ 'rotate-90': expanded.has(row.method) }"
              >▶</span>
              <div class="min-w-0 truncate">
                <span class="font-mono text-text">{{ methodLabel(row.method) }}</span>
                <span
                  v-if="methodService(row.method)"
                  class="font-mono text-[10px] text-text-muted ml-1.5"
                >{{ methodService(row.method) }}</span>
              </div>
            </div>
            <div class="text-right font-mono text-text">{{ row.count.toLocaleString() }}</div>
            <div class="text-right font-mono text-text-secondary">{{ row.inFlight.toLocaleString() }}</div>
            <div
              class="text-right font-mono"
              :class="row.errorCount > 0 ? 'text-status-danger' : 'text-text-muted'"
            >
              <template v-if="row.errorCount > 0">{{ row.errorCount }} · {{ errorRate(row) }}</template>
              <template v-else>—</template>
            </div>
            <div class="text-right font-mono text-text-secondary">{{ fmtMs(row.p50) }}</div>
            <div class="text-right font-mono text-text">{{ fmtMs(row.p95) }}</div>
            <div
              class="text-right font-mono"
              :class="row.p99 > 1000 ? 'text-status-warning' : 'text-text-secondary'"
            >{{ fmtMs(row.p99) }}</div>
            <div class="flex items-end h-8 gap-px bg-surface-sunken rounded px-0.5 relative overflow-hidden">
              <div
                v-for="(bar, i) in sparkBars(row)"
                :key="i"
                class="flex-1 min-w-[2px]"
                :style="{ height: bar.height + '%' }"
                :class="bar.count === 0
                  ? 'bg-transparent'
                  : (bar.isInf ? 'bg-status-danger' : 'bg-accent')"
                :title="`≤ ${fmtBound(bar.upper)}: ${bar.count}`"
              ></div>
              <div
                v-if="markerLeft(row, row.p50)"
                class="absolute top-0 bottom-0 w-px bg-text-muted/60 pointer-events-none"
                :style="{ left: markerLeft(row, row.p50)! }"
                title="p50"
              ></div>
              <div
                v-if="markerLeft(row, row.p95)"
                class="absolute top-0 bottom-0 w-px bg-status-warning/80 pointer-events-none"
                :style="{ left: markerLeft(row, row.p95)! }"
                title="p95"
              ></div>
              <div
                v-if="markerLeft(row, row.p99)"
                class="absolute top-0 bottom-0 w-px bg-status-danger/80 pointer-events-none"
                :style="{ left: markerLeft(row, row.p99)! }"
                title="p99"
              ></div>
            </div>
            <div class="text-right font-mono text-text-muted text-xs">{{ fmtSince(row.last) }}</div>
          </button>

          <div
            v-if="expanded.has(row.method)"
            class="px-4 py-3 bg-surface-sunken/60 border-t border-surface-border space-y-4"
          >
            <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div class="border border-surface-border-subtle rounded px-2 py-1.5 bg-surface">
                <div class="text-[10px] uppercase text-text-muted tracking-wider">avg</div>
                <div class="font-mono">{{ fmtMs(avgMs(row)) }}</div>
              </div>
              <div class="border border-surface-border-subtle rounded px-2 py-1.5 bg-surface">
                <div class="text-[10px] uppercase text-text-muted tracking-wider">in flight</div>
                <div class="font-mono">{{ row.inFlight.toLocaleString() }}</div>
              </div>
              <div class="border border-surface-border-subtle rounded px-2 py-1.5 bg-surface">
                <div class="text-[10px] uppercase text-text-muted tracking-wider">total time</div>
                <div class="font-mono">{{ fmtMs(row.totalDurationMs) }}</div>
              </div>
              <div class="border border-surface-border-subtle rounded px-2 py-1.5 bg-surface">
                <div class="text-[10px] uppercase text-text-muted tracking-wider">error rate</div>
                <div
                  class="font-mono"
                  :class="row.errorCount > 0 ? 'text-status-danger' : ''"
                >{{ errorRate(row) }}</div>
              </div>
            </div>

            <div>
              <div class="text-[10px] uppercase tracking-wider text-text-muted mb-1">
                Latency histogram · counts per bucket upper bound
              </div>
              <div class="flex items-end h-24 gap-px bg-surface rounded border border-surface-border-subtle p-1 relative">
                <div
                  v-for="(bar, i) in sparkBars(row)"
                  :key="i"
                  class="flex-1 min-w-[3px] relative group/bar"
                  :style="{ height: bar.height + '%' }"
                  :class="bar.count === 0
                    ? 'bg-transparent border-b border-surface-border-subtle/40'
                    : (bar.isInf ? 'bg-status-danger' : 'bg-accent')"
                >
                  <span
                    v-if="bar.count > 0"
                    class="absolute left-1/2 -translate-x-1/2 -top-6 text-[10px] font-mono bg-surface border border-surface-border rounded px-1 py-0.5 opacity-0 group-hover/bar:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10"
                  >≤ {{ fmtBound(bar.upper) }} · {{ bar.count }}</span>
                </div>
              </div>
              <div class="flex justify-between text-[9px] font-mono text-text-muted mt-1 px-1">
                <span>{{ fmtBound(row.bounds[0] ?? 0) }}</span>
                <span>{{ fmtBound(row.bounds[Math.floor(row.bounds.length / 2)] ?? 0) }}</span>
                <span>+∞</span>
              </div>
              <div class="flex flex-wrap gap-3 text-[10px] font-mono text-text-muted mt-2">
                <span class="flex items-center gap-1"><span class="inline-block w-2 h-2 bg-accent"></span>count</span>
                <span class="flex items-center gap-1"><span class="inline-block w-2 h-2 bg-status-danger"></span>+∞ overflow</span>
                <span class="flex items-center gap-1"><span class="inline-block w-px h-3 bg-text-muted/60"></span>p50</span>
                <span class="flex items-center gap-1"><span class="inline-block w-px h-3 bg-status-warning/80"></span>p95</span>
                <span class="flex items-center gap-1"><span class="inline-block w-px h-3 bg-status-danger/80"></span>p99</span>
              </div>
            </div>

          </div>
        </div>
      </div>
    </InfoCard>

    <InfoCard title="Proxy transport">
      <p class="text-xs text-text-muted mb-3">
        Every request the native proxy forwards upstream, not just Iris Connect RPCs.
        A proxied Connect call is counted here and in RPC Methods above; the two panels
        measure different things and must not be summed. Totals are exact; the per-endpoint
        breakdown is bounded to the busiest endpoints in each refresh window.
      </p>

      <div v-if="!proxyTotal" class="text-sm text-text-muted py-4 text-center">
        No proxied traffic in the last hour.
      </div>

      <template v-else>
        <div class="grid grid-cols-2 md:grid-cols-5 gap-2 mb-4">
          <div class="rounded-lg border border-surface-border bg-surface px-3 py-2">
            <div class="text-[10px] uppercase tracking-wider text-text-muted">Requests</div>
            <div class="font-mono text-lg text-text">{{ proxyTotal.count.toLocaleString() }}</div>
          </div>
          <div class="rounded-lg border border-surface-border bg-surface px-3 py-2">
            <div class="text-[10px] uppercase tracking-wider text-text-muted">In flight</div>
            <div class="font-mono text-lg text-text">{{ proxyTotal.inFlight.toLocaleString() }}</div>
          </div>
          <div class="rounded-lg border border-surface-border bg-surface px-3 py-2">
            <div class="text-[10px] uppercase tracking-wider text-text-muted">Req bytes</div>
            <div class="font-mono text-lg text-text">{{ fmtBytes(proxyTotal.requestBytes) }}</div>
          </div>
          <div class="rounded-lg border border-surface-border bg-surface px-3 py-2">
            <div class="text-[10px] uppercase tracking-wider text-text-muted">Resp bytes</div>
            <div class="font-mono text-lg text-text">{{ fmtBytes(proxyTotal.responseBytes) }}</div>
          </div>
          <div class="rounded-lg border border-surface-border bg-surface px-3 py-2">
            <div class="text-[10px] uppercase tracking-wider text-text-muted">p95</div>
            <div class="font-mono text-lg text-text">{{ fmtMs(proxyTotal.p95) }}</div>
          </div>
        </div>

        <div class="text-[10px] uppercase tracking-wider text-text-muted mb-1">
          Top endpoints
          <span v-if="proxyEndpointCount > proxyEndpoints.length" class="normal-case tracking-normal">
            ({{ proxyEndpoints.length }} of {{ proxyEndpointCount }} by request count)
          </span>
        </div>

        <div v-if="!proxyEndpoints.length" class="text-sm text-text-muted py-3 text-center">
          No per-endpoint breakdown available.
        </div>

        <template v-else>
          <div
            class="grid items-center gap-x-3 px-2 py-1.5 text-[10px] uppercase tracking-wider text-text-muted border-b border-surface-border"
            style="grid-template-columns: minmax(0,1fr) 88px 72px 56px 72px 72px 64px 72px"
          >
            <div>Endpoint</div>
            <div class="text-right">Count</div>
            <div class="text-right">Live</div>
            <div class="text-right">Err</div>
            <div class="text-right">Req</div>
            <div class="text-right">Resp</div>
            <div class="text-right">p95</div>
            <div class="text-right">Last</div>
          </div>
          <div class="divide-y divide-surface-border">
            <div
              v-for="row in proxyEndpoints"
              :key="row.method"
              class="grid items-center gap-x-3 px-2 py-2 text-sm"
              style="grid-template-columns: minmax(0,1fr) 88px 72px 56px 72px 72px 64px 72px"
            >
              <div class="min-w-0 truncate">
                <span class="font-mono text-text">{{ row.endpoint }}</span>
                <span v-if="row.routeKind" class="font-mono text-[10px] text-text-muted ml-1.5">{{ row.routeKind }}</span>
              </div>
              <div class="text-right font-mono text-text">{{ row.count.toLocaleString() }}</div>
              <div class="text-right font-mono text-text-secondary">{{ row.inFlight.toLocaleString() }}</div>
              <div
                class="text-right font-mono"
                :class="row.errorCount > 0 ? 'text-status-danger' : 'text-text-muted'"
              >{{ row.errorCount > 0 ? row.errorCount : '—' }}</div>
              <div class="text-right font-mono text-text-secondary">{{ fmtBytes(row.requestBytes) }}</div>
              <div class="text-right font-mono text-text-secondary">{{ fmtBytes(row.responseBytes) }}</div>
              <div class="text-right font-mono text-text">{{ fmtMs(row.p95) }}</div>
              <div class="text-right font-mono text-text-muted text-xs">{{ fmtSince(row.last) }}</div>
            </div>
          </div>
        </template>
      </template>
    </InfoCard>
  </div>
</template>
