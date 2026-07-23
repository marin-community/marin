<script setup lang="ts">
/**
 * Live finelog logs for a run's jobs. Collapsed by default; on expand it fetches
 * /api/runs/{id}/logs for the active role with a tail size and optional substring filter. When
 * finelog is unreachable (local dev / outside the VPC) it falls back to the log tails recorded
 * on the run's record.
 */
import { computed, ref, watch } from 'vue'
import { useApi } from '@/composables/useApi'
import { formatMillis, protoTimestampMillis } from '@/utils/formatting'
import type { LogsResponse } from '@/types/api'

const props = defineProps<{ runId: string; roles: string[]; logTails: Record<string, string[]> }>()

const TAIL_OPTIONS = [50, 200, 1000, 5000]

const expanded = ref(false)
const role = ref(props.roles[0] ?? '')
const tail = ref(200)
const substring = ref('')

function logsPath(): string {
  const params = new URLSearchParams({ role: role.value, tail: String(tail.value) })
  if (substring.value) params.set('substring', substring.value)
  return `api/runs/${props.runId}/logs?${params.toString()}`
}

const { data, loading, error, refresh } = useApi<LogsResponse>(logsPath)

function expand() {
  expanded.value = true
  if (!data.value) refresh()
}

// Refetch when a control changes while open.
watch([role, tail, substring], () => {
  if (expanded.value) refresh()
})

const recordedFallback = computed<string[]>(() => props.logTails[role.value] ?? [])
</script>

<template>
  <div v-if="roles.length">
    <div class="flex items-center justify-between gap-3 mb-2">
      <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Logs</h3>
      <button
        class="text-xs px-2 py-1 rounded border border-surface-border hover:bg-surface-raised"
        @click="expanded ? (expanded = false) : expand()"
      >
        {{ expanded ? 'Hide' : 'Show live logs' }}
      </button>
    </div>

    <div v-if="expanded" class="rounded-lg border border-surface-border bg-surface p-4 space-y-3">
      <div class="flex flex-wrap items-end gap-3">
        <div class="flex gap-1">
          <button
            v-for="r in roles"
            :key="r"
            class="px-2 py-1 text-xs rounded border"
            :class="r === role
              ? 'border-accent-border bg-accent-subtle text-accent'
              : 'border-surface-border text-text-muted hover:bg-surface-raised'"
            @click="role = r"
          >{{ r }}</button>
        </div>
        <label class="flex flex-col text-xs text-text-secondary gap-1">
          Tail
          <select v-model.number="tail" class="rounded border border-surface-border bg-surface px-2 py-1 text-sm">
            <option v-for="n in TAIL_OPTIONS" :key="n" :value="n">{{ n }}</option>
          </select>
        </label>
        <label class="flex flex-col text-xs text-text-secondary gap-1 flex-1 min-w-[12rem]">
          Substring
          <input
            v-model.lazy="substring"
            type="text"
            placeholder="filter log text…"
            class="rounded border border-surface-border bg-surface px-2 py-1 text-sm"
          />
        </label>
        <button
          class="text-xs px-2 py-1.5 rounded border border-surface-border hover:bg-surface-raised"
          :disabled="loading"
          @click="refresh"
        >
          {{ loading ? 'Loading…' : 'Refresh' }}
        </button>
      </div>

      <div v-if="error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-xs px-2 py-1">
        {{ error }}
      </div>

      <!-- Live logs -->
      <template v-if="data && data.reachable">
        <pre
          v-if="data.entries.length"
          class="rounded border border-surface-border bg-surface-sunken p-3 text-[12px] font-mono overflow-auto max-h-96 whitespace-pre-wrap"
        >{{ data.entries.map(e => `[${formatMillis(protoTimestampMillis(e.timestamp))}] ${e.data}`).join('\n') }}</pre>
        <p v-else class="text-xs text-text-muted">No matching log lines.</p>
      </template>

      <!-- Unreachable: fall back to the recorded log tail -->
      <template v-else-if="data && !data.reachable">
        <p class="text-xs text-status-warning" :title="data.error ?? ''">
          Live finelog unreachable — showing the log tail recorded with the run.
        </p>
        <pre
          v-if="recordedFallback.length"
          class="rounded border border-surface-border bg-surface-sunken p-3 text-[12px] font-mono overflow-auto max-h-96 whitespace-pre-wrap"
        >{{ recordedFallback.join('\n') }}</pre>
        <p v-else class="text-xs text-text-muted">No recorded log tail for this role.</p>
      </template>
    </div>
  </div>
</template>
