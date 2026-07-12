<script setup lang="ts">
import { computed, ref } from 'vue'
import { logRpcCall } from '@/composables/useRpc'
import InfoCard from '@/components/shared/InfoCard.vue'

interface LogEntry {
  timestamp?: { epochMs?: string }
  source?: string
  data?: string
  attemptId?: number
  level?: string
  key?: string
}

interface FetchLogsResponse {
  entries?: LogEntry[]
  cursor?: string
}

const sourcePattern = ref('.*')
const substring = ref('')
const minLevel = ref<'' | 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'>('')
const maxLines = ref(500)
const tail = ref(true)

const entries = ref<LogEntry[]>([])
const loading = ref(false)
const error = ref<string | null>(null)

async function fetchLogs() {
  loading.value = true
  error.value = null
  try {
    const body: Record<string, unknown> = {
      source: sourcePattern.value,
      maxLines: maxLines.value,
      tail: tail.value,
    }
    if (substring.value) body.substring = substring.value
    if (minLevel.value) body.minLevel = minLevel.value
    const resp = await logRpcCall<FetchLogsResponse>('FetchLogs', body)
    entries.value = resp.entries ?? []
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    entries.value = []
  } finally {
    loading.value = false
  }
}

function levelClass(level: string | undefined): string {
  switch (level) {
    case 'LOG_LEVEL_ERROR':
    case 'LOG_LEVEL_CRITICAL':
      return 'text-status-danger'
    case 'LOG_LEVEL_WARNING':
      return 'text-status-warning'
    case 'LOG_LEVEL_DEBUG':
      return 'text-text-muted'
    default:
      return 'text-text-secondary'
  }
}

function shortLevel(level: string | undefined): string {
  if (!level) return ''
  return level.replace('LOG_LEVEL_', '').slice(0, 4)
}

function fmtTime(ts: LogEntry['timestamp']): string {
  const ms = Number(ts?.epochMs ?? 0)
  if (!ms) return ''
  return new Date(ms).toISOString().slice(11, 23)
}

const sortedEntries = computed(() => entries.value.slice())
</script>

<template>
  <div class="space-y-3">
    <InfoCard title="FetchLogs">
      <div class="grid grid-cols-2 md:grid-cols-5 gap-2 text-sm">
        <label class="flex flex-col gap-1">
          <span class="text-[10px] uppercase tracking-wider text-text-muted">Source / regex</span>
          <input
            v-model="sourcePattern"
            class="font-mono bg-surface-sunken border border-surface-border rounded px-2 py-1"
            placeholder=".*"
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-[10px] uppercase tracking-wider text-text-muted">Substring</span>
          <input
            v-model="substring"
            class="font-mono bg-surface-sunken border border-surface-border rounded px-2 py-1"
            placeholder="(empty)"
          />
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-[10px] uppercase tracking-wider text-text-muted">Min level</span>
          <select
            v-model="minLevel"
            class="bg-surface-sunken border border-surface-border rounded px-2 py-1"
          >
            <option value="">(any)</option>
            <option value="DEBUG">DEBUG</option>
            <option value="INFO">INFO</option>
            <option value="WARNING">WARNING</option>
            <option value="ERROR">ERROR</option>
            <option value="CRITICAL">CRITICAL</option>
          </select>
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-[10px] uppercase tracking-wider text-text-muted">Max lines</span>
          <input
            v-model.number="maxLines"
            type="number"
            min="1"
            class="font-mono bg-surface-sunken border border-surface-border rounded px-2 py-1"
          />
        </label>
        <label class="flex items-end gap-2">
          <input v-model="tail" type="checkbox" />
          <span class="text-sm">Tail (newest first)</span>
        </label>
      </div>
      <div class="flex items-center gap-3 mt-3">
        <button
          class="px-3 py-1.5 text-sm rounded bg-accent text-white hover:bg-accent-hover disabled:opacity-50"
          :disabled="loading"
          @click="fetchLogs"
        >{{ loading ? 'Fetching…' : 'Fetch' }}</button>
        <span class="text-xs text-text-muted ml-auto">{{ entries.length }} entries</span>
      </div>
    </InfoCard>

    <div
      v-if="error"
      class="px-4 py-3 text-sm font-mono text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border whitespace-pre-wrap"
    >{{ error }}</div>

    <InfoCard v-if="!error" title="Entries">
      <div
        v-if="!sortedEntries.length"
        class="text-sm text-text-muted py-6 text-center"
      >No entries.</div>
      <div v-else class="font-mono text-xs space-y-px max-h-[70vh] overflow-y-auto">
        <div
          v-for="(e, i) in sortedEntries"
          :key="i"
          class="grid gap-2 px-2 py-1 hover:bg-surface-raised rounded"
          style="grid-template-columns: 90px 50px minmax(0, 200px) 1fr"
        >
          <span class="text-text-muted">{{ fmtTime(e.timestamp) }}</span>
          <span :class="levelClass(e.level)">{{ shortLevel(e.level) }}</span>
          <span class="text-text-muted truncate" :title="e.key || e.source">
            {{ e.key || e.source }}
          </span>
          <span class="whitespace-pre-wrap break-all">{{ e.data }}</span>
        </div>
      </div>
    </InfoCard>
  </div>
</template>
