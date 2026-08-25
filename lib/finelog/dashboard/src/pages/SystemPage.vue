<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { apiGet } from '@/composables/useRpc'
import { timeZoneMode } from '@/composables/useDisplayPrefs'
import { formatBytes, formatNumber, formatTimestampMs } from '@/utils/formatting'
import type { ServerInfo } from '@/types/introspection'
import InfoCard from '@/components/shared/InfoCard.vue'

const info = ref<ServerInfo | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

async function load() {
  loading.value = true
  error.value = null
  try {
    info.value = await apiGet<ServerInfo>('server')
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

/** `4d 3h 12m`, dropping units that are zero from the left. */
function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '—'
  const d = Math.floor(seconds / 86400)
  const h = Math.floor((seconds % 86400) / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (d) return `${d}d ${h}h ${m}m`
  if (h) return `${h}h ${m}m`
  return `${m}m ${Math.floor(seconds % 60)}s`
}

function formatUnix(seconds: number): string {
  if (!seconds) return 'unknown'
  return formatTimestampMs(seconds * 1000, timeZoneMode.value)
}

interface Field {
  label: string
  value: string
  mono?: boolean
  /** Rendered as a muted badge rather than a value, for state worth flagging. */
  note?: string
}

const build = computed<Field[]>(() => {
  const b = info.value?.build
  if (!b) return []
  return [
    { label: 'Version', value: b.version, mono: true },
    { label: 'Commit', value: b.commit || 'unknown', mono: true, note: b.dirty ? 'dirty tree' : undefined },
    { label: 'Tree hash', value: b.tree || 'unknown', mono: true },
    { label: 'Built', value: formatUnix(b.builtAtUnix) },
    { label: 'Profile', value: b.profile, mono: true },
    { label: 'Compiler', value: b.rustc || 'unknown', mono: true },
  ]
})

const process = computed<Field[]>(() => {
  const p = info.value?.process
  if (!p) return []
  return [
    { label: 'Host', value: p.hostname || 'unknown', mono: true },
    { label: 'PID', value: String(p.pid), mono: true },
    { label: 'Started', value: formatUnix(p.startedAtUnix) },
    { label: 'Uptime', value: formatDuration(p.uptimeSeconds) },
    { label: 'Resident memory', value: formatBytes(p.rssBytes) },
    { label: 'Virtual memory', value: formatBytes(p.vmSizeBytes) },
  ]
})

const store = computed<Field[]>(() => {
  const s = info.value?.store
  if (!s) return []
  return [
    { label: 'Data directory', value: s.dataDir || 'in-memory only', mono: true },
    { label: 'Archive', value: s.remoteLogDir || 'offload disabled', mono: true },
    { label: 'Namespaces', value: formatNumber(s.namespaces) },
    { label: 'Write buffers', value: `${formatBytes(s.ramBufferBytes)} in ${formatNumber(s.ramChunks)} chunks` },
  ]
})

/** One row per namespace this server registers for itself; anything but
 * `registered` is rejecting every write to it. */
const ingest = computed<Field[]>(() => {
  const namespaces = info.value?.ingest
  if (!namespaces?.length) return []
  return namespaces.map((n) => ({
    label: n.namespace,
    value: n.state === 'failed' ? (n.error ?? 'registration failed') : n.state,
    note:
      n.state === 'failed'
        ? `failing since ${formatUnix(n.sinceUnix ?? 0)}, ${formatNumber(n.attempts ?? 0)} attempts`
        : undefined,
  }))
})

const cache = computed<Field[]>(() => {
  const c = info.value?.metadataCache
  if (!c) return []
  const pct = c.limitBytes ? Math.round((c.sizeBytes / c.limitBytes) * 100) : 0
  return [
    { label: 'Parquet metadata', value: `${formatBytes(c.sizeBytes)} of ${formatBytes(c.limitBytes)}`, note: `${pct}% full` },
    { label: 'Cached footers', value: formatNumber(c.entries) },
    { label: 'Hits', value: formatNumber(c.hits) },
  ]
})

const indexCache = computed<Field[]>(() => {
  const c = info.value?.indexCache
  if (!c) return []
  const corrupt = c.corruptBundles + c.corruptSections
  const fallbacks = c.exactAggregateFallbacks
  return [
    { label: 'Corrupt bundles', value: formatNumber(c.corruptBundles) },
    { label: 'Corrupt sections', value: formatNumber(c.corruptSections) },
    { label: 'Indexed aggregates', value: formatNumber(c.exactAggregateFull) },
    { label: 'Partial aggregates', value: formatNumber(c.exactAggregatePartial) },
    { label: 'Declined aggregates', value: formatNumber(c.exactAggregateDeclined) },
    { label: 'Aggregate fallbacks', value: formatNumber(c.exactAggregateFallbacks) },
    { label: 'Fallback status', value: corrupt || fallbacks ? 'source fallback observed' : 'healthy' },
  ]
})

const format = computed<Field[]>(() => {
  const f = info.value?.format
  if (!f) return []
  return [
    { label: 'Layout revision', value: String(f.layoutVersion), mono: true },
    { label: 'Row group target', value: formatBytes(f.targetRowGroupBytes) },
    { label: 'Row group ceiling', value: `${formatNumber(f.maxRowGroupRows)} rows` },
    { label: 'Trigram span', value: `${formatNumber(f.sidecarSpanRows)} rows` },
  ]
})

onMounted(load)
</script>

<template>
  <div class="space-y-3">
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-lg">System</h2>
        <p class="text-xs text-text-muted mt-0.5">What this finelog is running and what it is holding.</p>
      </div>
      <button
        class="text-xs px-3 py-1.5 rounded border border-surface-border hover:bg-surface-raised"
        :disabled="loading"
        @click="load"
      >{{ loading ? 'Loading…' : 'Refresh' }}</button>
    </div>

    <div
      v-if="error"
      class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <div class="grid gap-3 md:grid-cols-2">
      <InfoCard
        v-for="card in [
          { title: 'Build', fields: build },
          { title: 'Process', fields: process },
          { title: 'Store', fields: store },
          { title: 'Ingest', fields: ingest },
          { title: 'Query cache', fields: cache },
          { title: 'Index cache', fields: indexCache },
          { title: 'Storage format', fields: format },
        ]"
        :key="card.title"
        :title="card.title"
      >
        <p v-if="!card.fields.length" class="text-sm text-text-muted">
          {{ loading ? 'Loading…' : 'Unavailable.' }}
        </p>
        <dl v-else class="space-y-1.5">
          <div v-for="f in card.fields" :key="f.label" class="flex gap-3 text-sm items-baseline">
            <dt class="text-text-muted shrink-0 w-40">{{ f.label }}</dt>
            <dd class="min-w-0 break-all" :class="f.mono ? 'font-mono text-xs' : ''">
              {{ f.value }}
              <span
                v-if="f.note"
                class="ml-1.5 px-1.5 py-0.5 rounded text-[0.65rem] font-sans bg-surface-raised text-text-muted"
              >{{ f.note }}</span>
            </dd>
          </div>
        </dl>
      </InfoCard>
    </div>
  </div>
</template>
