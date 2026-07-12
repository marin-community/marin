<script setup lang="ts">
import { computed, ref } from 'vue'
import type { HistBucket, Overview } from '../types'
import { useQuery } from '../composables/useQuery'
import { DEFAULT_SEED, cnum, fmt } from '../utils/formatting'
import BarChart from './BarChart.vue'
import SampleTable from './SampleTable.vue'
import SeedControl from './SeedControl.vue'
import SourcePicker from './SourcePicker.vue'
import StatRow from './StatRow.vue'

interface NormalizedStats {
  docs: number | null
  avg_chars: number | null
  median_chars: number | null
  max_chars: number | null
  sampled_rows: number | null
}

const props = defineProps<{ ov: Overview }>()

const source = ref('')
const search = ref('')
const seed = ref(DEFAULT_SEED)
const message = ref('type a source name and load')
const loaded = ref(false)

const resolved = computed(() => new Set(props.ov.resolved.normalize))

const stats = useQuery<NormalizedStats>()
const hist = useQuery<HistBucket[]>()
const samples = useQuery<Record<string, unknown>[]>()

const statRow = computed(() => {
  const s = stats.result.value
  if (!s) return []
  return [
    { value: fmt(s.docs), label: `docs${s.docs == null ? ' (n/a)' : ''}` },
    { value: fmt(s.avg_chars), label: `avg chars (over ${cnum(s.sampled_rows)} sample)` },
    { value: fmt(s.median_chars), label: 'median chars' },
    { value: fmt(s.max_chars), label: 'max chars (in sample)' },
  ]
})

const histBars = computed(() => (hist.result.value ?? []).map((b) => ({ label: `${fmt(b.lo)}–${fmt(b.hi)}`, count: b.docs })))
const histTotal = computed(() => (hist.result.value ?? []).reduce((a, b) => a + (b.docs || 0), 0))

function load() {
  const src = source.value.trim()
  if (!src) {
    message.value = 'type a source name'
    loaded.value = false
    return
  }
  if (!resolved.value.has(src)) {
    message.value = `normalized not resolved for ${src} (unknown or unresolved source)`
    loaded.value = false
    return
  }
  loaded.value = true
  void stats.run('normalized_stats', { source: src })
  void hist.run('normalized_hist', { source: src })
  void samples.run('normalized_samples', { source: src, n: 20, search: search.value, seed: seed.value })
}
</script>

<template>
  <div class="rounded-lg border border-surface-border bg-surface p-4">
    <div class="mb-3 flex flex-wrap items-center gap-2.5">
      <SourcePicker v-model="source" :sources="ov.sources" />
      <input
        v-model="search"
        type="search"
        placeholder="search text…"
        class="w-[220px] rounded border border-surface-border bg-surface px-2 py-1 text-[13px] text-text"
      />
      <SeedControl v-model="seed" @reroll="load" />
      <button
        class="rounded-md bg-accent px-3 py-1.5 text-sm font-semibold text-white hover:bg-accent-hover"
        @click="load"
      >
        load
      </button>
    </div>

    <p v-if="!loaded" class="text-sm text-text-muted">{{ message }}</p>
    <template v-else>
      <p v-if="stats.running.value" class="text-sm text-accent">stats…</p>
      <p v-else-if="stats.error.value" class="text-sm text-status-danger">error: {{ stats.error.value }}</p>
      <StatRow v-else-if="stats.result.value" :stats="statRow" />

      <h4 class="mb-1 mt-4 font-semibold">char-length distribution</h4>
      <p v-if="hist.running.value" class="text-sm text-accent">loading…</p>
      <p v-else-if="hist.error.value" class="text-sm text-status-danger">error: {{ hist.error.value }}</p>
      <template v-else-if="hist.result.value">
        <p class="mb-1 text-sm text-text-muted">
          sampled over {{ fmt(histTotal) }} documents (first rows of the source; id is a content hash, so unbiased)
        </p>
        <BarChart :bars="histBars" />
      </template>

      <h4 class="mb-1 mt-4 font-semibold">samples</h4>
      <p v-if="samples.running.value" class="text-sm text-accent">loading…</p>
      <p v-else-if="samples.error.value" class="text-sm text-status-danger">error: {{ samples.error.value }}</p>
      <SampleTable v-else-if="samples.result.value" :rows="samples.result.value" :columns="['id', 'chars', 'text']" />
    </template>
  </div>
</template>
