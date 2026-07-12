<script setup lang="ts">
import { computed, ref } from 'vue'
import type { DedupCluster, Overview } from '../types'
import { useQuery } from '../composables/useQuery'
import { DEFAULT_SEED, fmt, pct } from '../utils/formatting'
import SeedControl from './SeedControl.vue'
import SourcePicker from './SourcePicker.vue'
import StatRow from './StatRow.vue'

const props = defineProps<{ ov: Overview }>()

const source = ref('')
const seed = ref(DEFAULT_SEED)
const loaded = ref(false)

const storeStats = computed(() => {
  const counters = props.ov.counters || {}
  return [
    { value: fmt(counters['datakit_store/records_in']), label: 'records in' },
    { value: fmt(counters['datakit_store/dedup_noncanonical_dropped']), label: 'dropped as non-canonical' },
    { value: fmt(counters['datakit_store/records_out']), label: 'kept (canonical/singleton)' },
  ]
})

// Sources with a dedup drop rate in the baked-in summary — the ones worth inspecting.
const dedupSources = computed(() =>
  (props.ov.source_summary || [])
    .filter((row) => row.drop_rate != null)
    .map((row) => row.source)
    .sort(),
)

const summaryRow = computed(() => (props.ov.source_summary || []).find((row) => row.source === source.value.trim()))

const sourceStats = computed(() => {
  const s = summaryRow.value
  return [
    { value: pct(s?.drop_rate), label: 'docs dropped as dup' },
    { value: pct(s?.dup_prevalence), label: 'in a dup cluster' },
    { value: fmt(s?.dup_largest), label: 'largest cluster (sampled)' },
    { value: String(s?.dup_avg_size ?? '—'), label: 'avg cluster size' },
    { value: fmt(s?.docs_est), label: 'docs (est)' },
  ]
})

const clusters = useQuery<DedupCluster[]>()

function load() {
  const src = source.value.trim()
  if (!src) return
  loaded.value = true
  void clusters.run('dedup_examples', { source: src, n_clusters: 6, seed: seed.value })
}
</script>

<template>
  <div class="mb-3.5 rounded-lg border border-surface-border bg-surface p-4">
    <h3 class="mb-2 font-semibold">Deduplication (store-wide)</h3>
    <StatRow :stats="storeStats" />
    <p class="mt-2 text-sm text-text-muted">
      fuzzy-dup dataset:
      <code class="rounded bg-surface-sunken px-1 font-mono">{{ ov.dedup ?? 'unresolved' }}</code>
    </p>
  </div>

  <div class="rounded-lg border border-surface-border bg-surface p-4">
    <div class="mb-3 flex flex-wrap items-center gap-2.5">
      <SourcePicker v-model="source" :sources="dedupSources" />
      <SeedControl v-model="seed" @reroll="load" />
      <button
        class="rounded-md bg-accent px-3 py-1.5 text-sm font-semibold text-white hover:bg-accent-hover"
        @click="load"
      >
        load duplicate clusters
      </button>
    </div>

    <p v-if="!loaded" class="text-sm text-text-muted">
      type a source to inspect its duplicate clusters (canonical kept vs dropped)
    </p>
    <p v-else-if="clusters.running.value" class="text-sm text-accent">running query…</p>
    <p v-else-if="clusters.error.value" class="text-sm text-status-danger">error: {{ clusters.error.value }}</p>
    <template v-else-if="clusters.result.value">
      <StatRow :stats="sourceStats" />
      <h4 class="mb-2 mt-4 font-semibold">
        example duplicate clusters
        <span class="font-normal text-text-muted">(densest clusters in a sampled window; KEPT = canonical)</span>
      </h4>
      <p v-if="!clusters.result.value.length" class="text-sm text-text-muted">
        no multi-member clusters found in the sampled window
      </p>
      <div
        v-for="cluster in clusters.result.value"
        :key="cluster.cluster_id"
        class="mb-3 rounded-lg border border-surface-border"
      >
        <div class="flex items-center justify-between bg-surface-sunken px-2.5 py-1.5 text-sm text-text-secondary">
          <span>cluster {{ String(cluster.cluster_id).slice(0, 14) }}…</span>
          <span>~{{ fmt(cluster.sampled_size) }} members</span>
        </div>
        <div
          v-for="(member, i) in cluster.members"
          :key="i"
          class="border-t border-surface-border px-2.5 py-1.5"
        >
          <span
            class="rounded-full px-2 py-0.5 text-[11px] font-semibold"
            :class="
              member.canonical ? 'bg-status-success/15 text-status-success' : 'bg-status-warning/15 text-status-warning'
            "
          >
            {{ member.canonical ? 'KEPT' : 'dropped' }}
          </span>
          <pre class="mt-1 max-h-60 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-[1.45]">{{
            member.text
          }}</pre>
        </div>
      </div>
    </template>
  </div>
</template>
