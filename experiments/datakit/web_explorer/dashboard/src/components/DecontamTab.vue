<script setup lang="ts">
import { computed, ref } from 'vue'
import type { Overview } from '../types'
import { useQuery } from '../composables/useQuery'
import { DEFAULT_SEED, fmt } from '../utils/formatting'
import SampleTable from './SampleTable.vue'
import SeedControl from './SeedControl.vue'
import SourcePicker from './SourcePicker.vue'
import StatRow from './StatRow.vue'

interface DecontamStats {
  docs: number | null
  contaminated_pct: number | string
  avg_overlap: number | string
  max_overlap: number | string
}

const props = defineProps<{ ov: Overview }>()

const source = ref('')
const seed = ref(DEFAULT_SEED)
const message = ref('type a source name and load')
const loaded = ref(false)

const resolved = computed(() => new Set(props.ov.resolved.decontam))

const stats = useQuery<DecontamStats>()
const samples = useQuery<Record<string, unknown>[]>()

const statRow = computed(() => {
  const s = stats.result.value
  if (!s) return []
  return [
    { value: fmt(s.docs), label: `docs${s.docs == null ? ' (n/a)' : ''}` },
    { value: `${s.contaminated_pct}%`, label: 'contaminated (sampled)' },
    { value: String(s.avg_overlap), label: 'avg overlap (sampled)' },
    { value: String(s.max_overlap), label: 'max overlap (sampled)' },
  ]
})

function load() {
  const src = source.value.trim()
  if (!src) {
    message.value = 'type a source name'
    loaded.value = false
    return
  }
  if (!resolved.value.has(src)) {
    message.value = `decontam not resolved for ${src} (unknown or unresolved source)`
    loaded.value = false
    return
  }
  loaded.value = true
  void stats.run('decontam_stats', { source: src })
  void samples.run('decontam_samples', { source: src, n: 20, seed: seed.value })
}
</script>

<template>
  <div class="rounded-lg border border-surface-border bg-surface p-4">
    <div class="mb-3 flex flex-wrap items-center gap-2.5">
      <SourcePicker v-model="source" :sources="ov.sources" />
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

      <h4 class="mb-1 mt-4 font-semibold">most-contaminated samples</h4>
      <p v-if="samples.running.value" class="text-sm text-accent">loading…</p>
      <p v-else-if="samples.error.value" class="text-sm text-status-danger">error: {{ samples.error.value }}</p>
      <SampleTable
        v-else-if="samples.result.value"
        :rows="samples.result.value"
        :columns="['id', 'max_overlap', 'text']"
      />
    </template>
  </div>
</template>
