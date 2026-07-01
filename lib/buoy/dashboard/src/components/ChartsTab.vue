<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import MetricChart from './MetricChart.vue'
import { useMetrics } from '../composables/useMetrics'
import type { RunRef } from '../composables/useRun'

const props = defineProps<{ runRef: RunRef; columns: string[]; lastStep: number | null }>()

const DEFAULTS = [
  'train/cross_entropy_loss',
  'train/loss',
  'optim/learning_rate',
  'throughput/hook_time',
  'throughput/loading_time',
  'run_progress',
]

const { selected, data, reset, ensure, add, remove } = useMetrics()
const search = ref('')

function seed() {
  const def = DEFAULTS.filter((k) => props.columns.includes(k))
  reset(def.length ? def : props.columns.slice(0, 6))
  ensure(props.runRef)
}

onMounted(seed)
watch(() => props.runRef, seed) // switching runs reseeds default charts
watch(selected, () => ensure(props.runRef), { deep: true })

// A running run grew (live refresh bumped last_step): if metrics just appeared,
// seed the defaults; otherwise drop the cached series so ensure refetches them.
watch(
  () => props.lastStep,
  () => {
    if (!selected.value.length && props.columns.length) {
      seed()
      return
    }
    for (const k of selected.value) delete data.value[k]
    ensure(props.runRef)
  },
)

const options = computed(() => {
  const q = search.value.toLowerCase()
  return props.columns.filter((c) => c.toLowerCase().includes(q) && !selected.value.includes(c)).slice(0, 30)
})

const fullscreenKey = ref<string | null>(null)
const fullscreenHeight = ref(600)
function openFullscreen(key: string) {
  fullscreenHeight.value = Math.round(window.innerHeight * 0.85)
  fullscreenKey.value = key
}
</script>

<template>
  <div>
    <div class="mb-2 flex flex-wrap gap-1.5">
      <span
        v-for="k in selected"
        :key="k"
        class="flex items-center gap-1 rounded-full bg-accent-subtle px-2 py-0.5 font-mono text-xs text-accent"
      >
        {{ k }}
        <button class="text-accent/60 hover:text-status-danger" @click="remove(k)">×</button>
      </span>
    </div>

    <div class="relative mb-4 max-w-md">
      <input
        v-model="search"
        placeholder="add a metric to plot…"
        class="w-full rounded border border-surface-border bg-surface-raised px-3 py-1.5 text-sm"
      />
      <div
        v-if="search && options.length"
        class="absolute z-10 mt-1 max-h-64 w-full overflow-auto rounded border border-surface-border bg-surface-raised shadow"
      >
        <button
          v-for="o in options"
          :key="o"
          class="block w-full truncate px-3 py-1 text-left font-mono text-xs hover:bg-accent-subtle"
          @click="((add(o)), (search = ''))"
        >
          {{ o }}
        </button>
      </div>
    </div>

    <p v-if="!selected.length && !columns.length" class="text-text-muted">
      no data yet — this run is still starting; charts appear automatically as metrics arrive.
    </p>
    <p v-else-if="!selected.length" class="text-text-muted">add a metric above to plot</p>
    <div v-else class="grid gap-4" style="grid-template-columns: repeat(auto-fill, minmax(560px, 1fr))">
      <MetricChart
        v-for="k in selected"
        :key="k"
        :metric-key="k"
        :series="data[k] || { x: [], y: [] }"
        @close="remove(k)"
        @fullscreen="openFullscreen(k)"
      />
    </div>

    <div v-if="fullscreenKey" class="fixed inset-0 z-[300] flex flex-col bg-surface-raised p-4">
      <button
        class="mb-1 self-end rounded border border-surface-border px-2 py-1 text-lg leading-none hover:border-status-danger hover:text-status-danger"
        title="close (Esc)"
        @click="fullscreenKey = null"
      >
        ×
      </button>
      <MetricChart
        :metric-key="fullscreenKey"
        :series="data[fullscreenKey] || { x: [], y: [] }"
        :height="fullscreenHeight"
        @close="fullscreenKey = null"
        @fullscreen="fullscreenKey = null"
      />
    </div>
  </div>
</template>
