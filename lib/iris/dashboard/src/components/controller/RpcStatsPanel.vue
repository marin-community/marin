<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useStatsRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import { formatRelativeTime, timestampMs } from '@/utils/formatting'
import InfoCard from '@/components/shared/InfoCard.vue'
import DataTable, { type Column } from '@/components/shared/DataTable.vue'
import type { GetRpcStatsResponse, RpcCallSample, RpcMethodStats } from '@/types/rpc'

const { data, loading, error, refresh } = useStatsRpc<GetRpcStatsResponse>('GetRpcStats')

useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
onMounted(refresh)

type SortKey = 'method' | 'count' | 'errorCount' | 'p50' | 'p95' | 'p99' | 'max' | 'last'
const sortKey = ref<SortKey>('p95')
const sortDir = ref<'asc' | 'desc'>('desc')

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
  }
}

const methodRows = computed<MethodRow[]>(() => {
  const rows = (data.value?.methods ?? []).map(toRow)
  const dir = sortDir.value === 'asc' ? 1 : -1
  const key = sortKey.value
  return [...rows].sort((a, b) => {
    if (key === 'method') return a.method.localeCompare(b.method) * dir
    return (a[key] - b[key]) * dir
  })
})

function onSort(key: string, dir: 'asc' | 'desc') {
  sortKey.value = key as SortKey
  sortDir.value = dir
}

const columns: Column[] = [
  { key: 'method', label: 'Method', sortable: true, mono: true },
  { key: 'count', label: 'Count', sortable: true, align: 'right', mono: true },
  { key: 'errorCount', label: 'Errors', sortable: true, align: 'right', mono: true },
  { key: 'p50', label: 'p50 (ms)', sortable: true, align: 'right', mono: true },
  { key: 'p95', label: 'p95 (ms)', sortable: true, align: 'right', mono: true },
  { key: 'p99', label: 'p99 (ms)', sortable: true, align: 'right', mono: true },
  { key: 'max', label: 'Max (ms)', sortable: true, align: 'right', mono: true },
  { key: 'last', label: 'Last', sortable: true, align: 'right' },
]

function fmtMs(value: number): string {
  if (!value) return '-'
  if (value < 10) return value.toFixed(1)
  return Math.round(value).toString()
}

function fmtSince(epochMs: number): string {
  if (!epochMs) return '-'
  return formatRelativeTime(epochMs)
}

const slowSamples = computed<RpcCallSample[]>(() => [...(data.value?.slowSamples ?? [])].reverse())
const discoverySamples = computed<RpcCallSample[]>(() => [...(data.value?.discoverySamples ?? [])].reverse())

function prettyPreview(raw: string | undefined): string {
  if (!raw) return ''
  try {
    return JSON.stringify(JSON.parse(raw), null, 2)
  } catch {
    return raw
  }
}
</script>

<template>
  <div class="space-y-4">
    <div v-if="error" class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border">
      {{ error }}
    </div>

    <InfoCard title="RPC Methods">
      <p class="text-xs text-text-muted mb-3">
        Aggregate counters and latency percentiles per ControllerService RPC, since controller start.
        Histogram buckets are coarse (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000 ms, +inf).
      </p>
      <DataTable
        :columns="columns"
        :rows="methodRows"
        :loading="loading && !data"
        :sort-key="sortKey"
        :sort-dir="sortDir"
        :page-size="50"
        empty-message="No RPCs recorded yet"
        @sort="onSort"
      >
        <template #cell-p50="{ row }">{{ fmtMs((row as MethodRow).p50) }}</template>
        <template #cell-p95="{ row }">{{ fmtMs((row as MethodRow).p95) }}</template>
        <template #cell-p99="{ row }">{{ fmtMs((row as MethodRow).p99) }}</template>
        <template #cell-max="{ row }">{{ fmtMs((row as MethodRow).max) }}</template>
        <template #cell-last="{ row }">{{ fmtSince((row as MethodRow).last) }}</template>
      </DataTable>
    </InfoCard>

    <InfoCard title="Slow RPC samples">
      <p class="text-xs text-text-muted mb-3">
        Most recent calls slower than the 1000ms threshold (or any failure). Newest first.
      </p>
      <div v-if="!slowSamples.length" class="text-sm text-text-muted py-4">No slow calls recorded.</div>
      <ul v-else class="space-y-3">
        <li
          v-for="(s, i) in slowSamples"
          :key="i"
          class="border border-surface-border rounded-md p-3 text-xs"
        >
          <div class="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span class="font-mono font-semibold">{{ s.method }}</span>
            <span class="font-mono text-status-warning">{{ fmtMs(s.durationMs ?? 0) }}ms</span>
            <span v-if="s.errorCode" class="font-mono text-status-danger">{{ s.errorCode }}</span>
            <span class="text-text-muted">{{ fmtSince(timestampMs(s.timestamp)) }}</span>
          </div>
          <div class="text-text-muted mt-1 flex flex-wrap gap-x-3 gap-y-0.5">
            <span v-if="s.caller"><strong>caller:</strong> <span class="font-mono">{{ s.caller }}</span></span>
            <span v-if="s.peer"><strong>peer:</strong> <span class="font-mono">{{ s.peer }}</span></span>
            <span v-if="s.userAgent"><strong>ua:</strong> <span class="font-mono">{{ s.userAgent }}</span></span>
          </div>
          <div v-if="s.errorMessage" class="mt-1 text-status-danger font-mono">{{ s.errorMessage }}</div>
          <details v-if="s.requestPreview" class="mt-2">
            <summary class="cursor-pointer text-text-muted">request preview</summary>
            <pre class="mt-1 font-mono text-[11px] whitespace-pre-wrap break-all bg-surface-muted rounded p-2">{{ prettyPreview(s.requestPreview) }}</pre>
          </details>
        </li>
      </ul>
    </InfoCard>

    <InfoCard title="Sampled calls (discovery)">
      <p class="text-xs text-text-muted mb-3">
        One sample per method captured at most every 30 seconds regardless of latency, so you can see what a typical call looks like. Newest first.
      </p>
      <div v-if="!discoverySamples.length" class="text-sm text-text-muted py-4">No samples recorded yet.</div>
      <ul v-else class="space-y-3">
        <li
          v-for="(s, i) in discoverySamples"
          :key="i"
          class="border border-surface-border rounded-md p-3 text-xs"
        >
          <div class="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span class="font-mono font-semibold">{{ s.method }}</span>
            <span class="font-mono">{{ fmtMs(s.durationMs ?? 0) }}ms</span>
            <span class="text-text-muted">{{ fmtSince(timestampMs(s.timestamp)) }}</span>
          </div>
          <div class="text-text-muted mt-1 flex flex-wrap gap-x-3 gap-y-0.5">
            <span v-if="s.caller"><strong>caller:</strong> <span class="font-mono">{{ s.caller }}</span></span>
            <span v-if="s.peer"><strong>peer:</strong> <span class="font-mono">{{ s.peer }}</span></span>
            <span v-if="s.userAgent"><strong>ua:</strong> <span class="font-mono">{{ s.userAgent }}</span></span>
          </div>
          <details v-if="s.requestPreview" class="mt-2">
            <summary class="cursor-pointer text-text-muted">request preview</summary>
            <pre class="mt-1 font-mono text-[11px] whitespace-pre-wrap break-all bg-surface-muted rounded p-2">{{ prettyPreview(s.requestPreview) }}</pre>
          </details>
        </li>
      </ul>
    </InfoCard>
  </div>
</template>
