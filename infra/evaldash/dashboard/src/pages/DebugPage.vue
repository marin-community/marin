<script setup lang="ts">
import { onMounted } from 'vue'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatRelativeAge, formatTimestamp } from '@/utils/formatting'
import type { Status } from '@/types/api'

const { data, loading, error, refresh } = useApi<Status>(() => 'api/status')

onMounted(refresh)
onViewRefresh(refresh)
</script>

<template>
  <section>
    <div class="mb-4">
      <h2 class="text-lg font-semibold">Debug</h2>
      <p class="text-xs text-text-muted mt-0.5">
        Where records are scanned, the active data source, and any errors seen while ingesting. Use the
        header Refresh to run an ingest pass now.
      </p>
    </div>

    <div v-if="error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2 mb-4">
      {{ error }}
    </div>

    <div v-if="loading && !data" class="text-sm text-text-muted py-12 text-center">Loading…</div>

    <div v-else-if="data" class="space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="rounded-lg border border-surface-border bg-surface p-4">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Store</h3>
          <dl class="text-sm space-y-1">
            <div class="flex gap-2"><dt class="text-text-muted w-24">backend</dt><dd>{{ data.store.backend }}</dd></div>
            <div v-if="data.store.instance" class="flex gap-2">
              <dt class="text-text-muted w-24">instance</dt><dd class="font-mono break-all">{{ data.store.instance }}</dd>
            </div>
            <div v-if="data.store.database" class="flex gap-2">
              <dt class="text-text-muted w-24">database</dt><dd class="font-mono">{{ data.store.database }}</dd>
            </div>
          </dl>
        </div>
        <div class="rounded-lg border border-surface-border bg-surface p-4">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Ingest</h3>
          <dl class="text-sm space-y-1">
            <div class="flex gap-2"><dt class="text-text-muted w-24">interval</dt><dd>{{ data.ingest.interval_seconds }}s</dd></div>
            <div class="flex gap-2">
              <dt class="text-text-muted w-24">last pass</dt>
              <dd :title="formatTimestamp(data.ingest.last_pass_time)">{{ formatRelativeAge(data.ingest.last_pass_time) }}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div>
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Prefixes scanned</h3>
        <div class="space-y-2">
          <div
            v-for="p in data.ingest.prefixes"
            :key="p.prefix"
            class="rounded-lg border border-surface-border bg-surface p-3"
          >
            <div class="flex items-center justify-between gap-3 flex-wrap">
              <code class="font-mono text-[13px] break-all">{{ p.prefix }}</code>
              <div class="flex items-center gap-4 text-xs text-text-secondary whitespace-nowrap">
                <span :title="formatTimestamp(p.last_probe_time)">probed {{ formatRelativeAge(p.last_probe_time) }}</span>
                <span>{{ p.record_count ?? '—' }} records</span>
                <span v-if="p.parse_failures.length" class="text-status-warning">
                  {{ p.parse_failures.length }} skipped
                </span>
                <span v-else-if="!p.error" class="text-status-success">ok</span>
              </div>
            </div>
            <div
              v-if="p.error"
              class="mt-2 rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-xs px-2 py-1 break-all"
              :title="`last succeeded ${formatRelativeAge(p.last_success_time)}`"
            >
              {{ p.error }}
            </div>
            <!-- Records found under this prefix that failed to parse: dropped from the snapshot, listed
                 here so a schema drift on old records is visible rather than silently swallowed. -->
            <ul v-if="p.parse_failures.length" class="mt-2 space-y-1">
              <li
                v-for="f in p.parse_failures"
                :key="f.path"
                class="rounded border border-status-warning-border bg-status-warning-bg text-status-warning text-xs px-2 py-1"
              >
                <code class="font-mono break-all">{{ f.path }}</code>
                <div class="text-text-secondary break-all">{{ f.error }}</div>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
