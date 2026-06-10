<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useStatsRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import { formatRelativeTime, timestampMs } from '@/utils/formatting'
import InfoCard from '@/components/shared/InfoCard.vue'
import type { GetRpcStatsResponse, RpcCallSample, RpcMethodStats } from '@/types/rpc'

const { data, loading, error, refresh } = useStatsRpc<GetRpcStatsResponse>('GetRpcStats')
useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
onMounted(refresh)

type SortKey = 'method' | 'count' | 'errorCount' | 'p50' | 'p95' | 'p99' | 'max' | 'last'
const sortKey = ref<SortKey>('p95')
const sortDir = ref<'asc' | 'desc'>('desc')
const expanded = ref<Set<string>>(new Set())
const sampleTab = ref<Record<string, 'recent' | 'slow'>>({})

function toNum(value: string | number | undefined): number {
  if (typeof value === 'number') return value
  if (typeof value === 'string') return parseInt(value, 10) || 0
  return 0
}

interface MethodRow {
  method: string
  count: number
  errorCount: number
  p50: number
  p95: number
  p99: number
  max: number
  last: number
  buckets: number[]
  bounds: number[]
  totalDurationMs: number
}

function toRow(m: RpcMethodStats): MethodRow {
  return {
    method: m.method,
    count: toNum(m.count),
    errorCount: toNum(m.errorCount),
    p50: m.p50Ms ?? 0,
    p95: m.p95Ms ?? 0,
    p99: m.p99Ms ?? 0,
    max: m.maxDurationMs ?? 0,
    last: timestampMs(m.lastCall),
    buckets: (m.bucketCounts ?? []).map(toNum),
    bounds: (m.bucketUpperBoundsMs ?? []).map(toNum),
    totalDurationMs: m.totalDurationMs ?? 0,
  }
}

const rows = computed<MethodRow[]>(() => {
  const list = (data.value?.methods ?? []).map(toRow)
  const dir = sortDir.value === 'asc' ? 1 : -1
  const key = sortKey.value
  return [...list].sort((a, b) => {
    if (key === 'method') return a.method.localeCompare(b.method) * dir
    return (a[key] - b[key]) * dir
  })
})

const slowByMethod = computed<Record<string, RpcCallSample[]>>(() => {
  const out: Record<string, RpcCallSample[]> = {}
  for (const s of data.value?.slowSamples ?? []) {
    (out[s.method] ||= []).push(s)
  }
  for (const k of Object.keys(out)) out[k].reverse()
  return out
})

const discoveryByMethod = computed<Record<string, RpcCallSample[]>>(() => {
  const out: Record<string, RpcCallSample[]> = {}
  for (const s of data.value?.discoverySamples ?? []) {
    (out[s.method] ||= []).push(s)
  }
  for (const k of Object.keys(out)) out[k].reverse()
  return out
})

const totalCount = computed(() => rows.value.reduce((a, r) => a + r.count, 0))
const totalErrors = computed(() => rows.value.reduce((a, r) => a + r.errorCount, 0))
const collectorStartedAt = computed(() => timestampMs(data.value?.collectorStartedAt))

function toggleExpand(method: string) {
  const next = new Set(expanded.value)
  if (next.has(method)) next.delete(method)
  else next.add(method)
  expanded.value = next
  if (!sampleTab.value[method]) sampleTab.value[method] = 'recent'
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

function fmtBound(ms: number): string {
  if (!ms) return '+∞'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(ms % 1000 === 0 ? 0 : 1)}s`
}

function fmtSince(epochMs: number): string {
  if (!epochMs) return '—'
  return formatRelativeTime(epochMs)
}

function prettyPreview(raw: string | undefined): string {
  if (!raw) return ''
  try {
    return JSON.stringify(JSON.parse(raw), null, 2)
  } catch {
    return raw
  }
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
  return row.count ? row.totalDurationMs / row.count : 0
}

function currentSamples(method: string, tab: 'recent' | 'slow'): RpcCallSample[] {
  return tab === 'recent' ? (discoveryByMethod.value[method] ?? []) : (slowByMethod.value[method] ?? [])
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
    <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
      <div class="rounded-lg border border-surface-border bg-surface px-3 py-2">
        <div class="text-[10px] uppercase tracking-wider text-text-muted">Methods</div>
        <div class="font-mono text-lg text-text">{{ rows.length }}</div>
      </div>
      <div class="rounded-lg border border-surface-border bg-surface px-3 py-2">
        <div class="text-[10px] uppercase tracking-wider text-text-muted">Calls</div>
        <div class="font-mono text-lg text-text">{{ totalCount.toLocaleString() }}</div>
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
        <div class="text-[10px] uppercase tracking-wider text-text-muted">Since</div>
        <div class="font-mono text-sm text-text pt-1">{{ fmtSince(collectorStartedAt) }}</div>
      </div>
    </div>

    <InfoCard title="RPC Methods">
      <p class="text-xs text-text-muted mb-3">
        Per-method counters, latency percentiles, and an inline histogram on a log scale
        (3 buckets per octave, ≈1 ms → 60 s). Click a row to expand samples.
        Request previews are redacted server-side for keys matching
        <code class="font-mono">KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL</code>.
      </p>

      <!-- Header -->
      <div
        class="grid items-center gap-x-3 px-2 py-1.5 text-[10px] uppercase tracking-wider text-text-muted border-b border-surface-border"
        style="grid-template-columns: minmax(0,1fr) 72px 90px 64px 64px 64px 200px 72px"
      >
        <button class="text-left cursor-pointer hover:text-text" @click="setSort('method')">
          Method <span class="text-accent">{{ sortIndicator('method') }}</span>
        </button>
        <button class="text-right cursor-pointer hover:text-text" @click="setSort('count')">
          Count <span class="text-accent">{{ sortIndicator('count') }}</span>
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
            style="grid-template-columns: minmax(0,1fr) 72px 90px 64px 64px 64px 200px 72px"
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
                <div class="text-[10px] uppercase text-text-muted tracking-wider">max</div>
                <div class="font-mono">{{ fmtMs(row.max) }}</div>
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

            <div>
              <div class="flex items-center gap-1 mb-2 text-xs">
                <button
                  class="px-2 py-1 border-b-2 font-mono uppercase text-[10px] tracking-wider cursor-pointer"
                  :class="(sampleTab[row.method] ?? 'recent') === 'recent'
                    ? 'border-accent text-text'
                    : 'border-transparent text-text-muted hover:text-text'"
                  @click="sampleTab[row.method] = 'recent'"
                >Recent · {{ (discoveryByMethod[row.method] ?? []).length }}</button>
                <button
                  class="px-2 py-1 border-b-2 font-mono uppercase text-[10px] tracking-wider cursor-pointer"
                  :class="sampleTab[row.method] === 'slow'
                    ? 'border-accent text-text'
                    : 'border-transparent text-text-muted hover:text-text'"
                  @click="sampleTab[row.method] = 'slow'"
                >Slow &amp; errors · {{ (slowByMethod[row.method] ?? []).length }}</button>
              </div>

              <div
                v-if="!currentSamples(row.method, sampleTab[row.method] ?? 'recent').length"
                class="text-xs text-text-muted italic py-2"
              >
                {{ (sampleTab[row.method] ?? 'recent') === 'slow'
                  ? 'No slow calls or errors captured for this method.'
                  : 'No samples captured yet.' }}
              </div>
              <ul v-else class="space-y-1.5">
                <li
                  v-for="(s, i) in currentSamples(row.method, sampleTab[row.method] ?? 'recent')"
                  :key="i"
                  class="border border-surface-border-subtle rounded text-xs bg-surface"
                >
                  <div class="flex flex-wrap items-baseline gap-x-3 gap-y-1 px-2 py-1.5">
                    <span
                      class="font-mono"
                      :class="s.errorCode
                        ? 'text-status-danger'
                        : ((s.durationMs ?? 0) > 1000 ? 'text-status-warning' : 'text-text')"
                    >{{ fmtMs(s.durationMs ?? 0) }}</span>
                    <span v-if="s.errorCode" class="font-mono text-status-danger text-[10px] uppercase">{{ s.errorCode }}</span>
                    <span class="text-text-muted">{{ fmtSince(timestampMs(s.timestamp)) }}</span>
                    <span v-if="s.caller" class="text-text-muted">
                      <span class="text-[10px] uppercase tracking-wider">caller</span>
                      <span class="font-mono ml-1">{{ s.caller }}</span>
                    </span>
                    <span v-if="s.peer" class="text-text-muted">
                      <span class="text-[10px] uppercase tracking-wider">peer</span>
                      <span class="font-mono ml-1">{{ s.peer }}</span>
                    </span>
                  </div>
                  <div v-if="s.errorMessage" class="px-2 pb-1.5 font-mono text-status-danger break-all">
                    {{ s.errorMessage }}
                  </div>
                  <details v-if="s.requestPreview" class="px-2 pb-1.5">
                    <summary class="cursor-pointer text-text-muted text-[10px] uppercase tracking-wider hover:text-text">
                      request · <span class="normal-case tracking-normal">redacted preview</span>
                    </summary>
                    <pre class="mt-1 font-mono text-[11px] whitespace-pre-wrap break-all bg-surface-sunken rounded p-2 border border-surface-border-subtle">{{ prettyPreview(s.requestPreview) }}</pre>
                    <div v-if="s.userAgent" class="mt-1 text-[10px] text-text-muted">
                      <span class="uppercase tracking-wider">ua</span>
                      <span class="font-mono ml-1">{{ s.userAgent }}</span>
                    </div>
                  </details>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </InfoCard>
  </div>
</template>
