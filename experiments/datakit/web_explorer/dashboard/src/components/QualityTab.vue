<script setup lang="ts">
import { computed, ref } from 'vue'
import type { HistBucket, Overview } from '../types'
import { useQuery } from '../composables/useQuery'
import { DEFAULT_SEED, fmt } from '../utils/formatting'
import BarChart from './BarChart.vue'
import SampleTable from './SampleTable.vue'
import SeedControl from './SeedControl.vue'
import SourcePicker from './SourcePicker.vue'

const props = defineProps<{ ov: Overview }>()

const source = ref('')
const seed = ref(DEFAULT_SEED)
const message = ref('type a source name and load')
const loaded = ref(false)
const selectedBar = ref<number | null>(null)

const resolved = computed(() => new Set(props.ov.resolved.quality))

const hist = useQuery<HistBucket[]>()
const samples = useQuery<Record<string, unknown>[]>()

const histBars = computed(() => (hist.result.value ?? []).map((b) => ({ label: `${b.lo}–${b.hi}`, count: b.docs })))
const histTotal = computed(() => (hist.result.value ?? []).reduce((a, b) => a + (b.docs || 0), 0))
const selectedRange = computed(() => {
  if (selectedBar.value == null) return null
  return hist.result.value?.[selectedBar.value] ?? null
})

function load() {
  const src = source.value.trim()
  if (!src) {
    message.value = 'type a source name'
    loaded.value = false
    return
  }
  if (!resolved.value.has(src)) {
    message.value = `quality not resolved for ${src} (unknown or unresolved source)`
    loaded.value = false
    return
  }
  loaded.value = true
  selectedBar.value = null
  samples.result.value = null
  void hist.run('quality_hist', { source: src })
}

function sampleRange(index: number) {
  const bucket = hist.result.value?.[index]
  if (!bucket) return
  selectedBar.value = index
  void samples.run('quality_samples', {
    source: source.value.trim(),
    lo: bucket.lo,
    hi: bucket.hi + 0.0001,
    n: 20,
    seed: seed.value,
  })
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
      <p v-if="hist.running.value" class="text-sm text-accent">running query…</p>
      <p v-else-if="hist.error.value" class="text-sm text-status-danger">error: {{ hist.error.value }}</p>
      <template v-else-if="hist.result.value">
        <h4 class="mb-1 font-semibold">
          quality score distribution
          <span class="font-normal text-text-muted">
            (sampled over {{ fmt(histTotal) }} docs; click a bar to sample that score range)
          </span>
        </h4>
        <BarChart :bars="histBars" clickable :selected="selectedBar" @select="sampleRange" />

        <div class="mt-3">
          <p v-if="selectedBar == null" class="text-sm text-text-muted">
            click a score range above to load samples
          </p>
          <p v-else-if="samples.running.value" class="text-sm text-accent">running query…</p>
          <p v-else-if="samples.error.value" class="text-sm text-status-danger">error: {{ samples.error.value }}</p>
          <template v-else-if="samples.result.value && selectedRange">
            <h4 class="mb-1 font-semibold">samples with score in {{ selectedRange.lo }}–{{ selectedRange.hi }}</h4>
            <SampleTable :rows="samples.result.value" :columns="['score', 'text']" />
          </template>
        </div>
      </template>
    </template>
  </div>
</template>
