<script setup lang="ts">
/**
 * Full-screen, one-sample-at-a-time viewer for a run's per-sample exports. Replaces the old
 * slide-over with a dedicated route so a sample is directly linkable: task, filter, and the
 * 0-based row index all live in the URL query (?task=&filter=&i=) and stay in sync via
 * router.replace as the viewer state changes. MCQ rows render as a ranked choice list with
 * loglikelihood bars; generative rows render as a prompt/output pair with fenced code blocks.
 */
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { useSamplePager, type SampleFilter } from '@/composables/useSamplePager'
import type { SampleRow, SampleTasksResponse } from '@/types/api'
import McqSample from '@/components/samples/McqSample.vue'
import GenerativeSample from '@/components/samples/GenerativeSample.vue'

const props = defineProps<{ runId: string }>()
const route = useRoute()
const router = useRouter()

const FILTERS: SampleFilter[] = ['all', 'correct', 'incorrect']

function queryTask(): string {
  return typeof route.query.task === 'string' ? route.query.task : ''
}
function queryFilter(): SampleFilter {
  const f = route.query.filter
  return f === 'correct' || f === 'incorrect' ? f : 'all'
}
function queryIndex(): number {
  const i = Number(route.query.i)
  return Number.isFinite(i) && i >= 0 ? Math.floor(i) : 0
}

const task = ref(queryTask())
const filter = ref<SampleFilter>(queryFilter())
const index = ref(queryIndex())

const { data: tasksData, error: tasksError, refresh: refreshTasks } = useApi<SampleTasksResponse>(
  () => `api/runs/${props.runId}/samples/tasks`,
)
onMounted(refreshTasks)

// Default to the first discovered task when the URL didn't pin one.
watch(tasksData, (tasks) => {
  if (!task.value && tasks?.tasks.length) task.value = tasks.tasks[0].task
})

const { total, counts, primaryMetric, loading, error, sample: sampleAt, ensure } = useSamplePager(
  props.runId,
  task,
  filter,
)

watch([task, filter], () => {
  index.value = 0
})

// Fetch (and keep the URL pointed at) whichever row is current; runs once immediately for deep
// links, then again on every task/filter/index change.
watch(
  [task, filter, index],
  () => {
    ensure(index.value)
    if (!task.value) return
    router.replace({ query: { task: task.value, filter: filter.value, i: String(index.value) } })
  },
  { immediate: true },
)

const row = computed<SampleRow | null>(() => sampleAt(index.value))

const positionLabel = computed(() => (total.value === 0 ? '0 of 0' : `${index.value + 1} of ${total.value}`))

function prev() {
  if (index.value > 0) index.value -= 1
}
function next() {
  if (index.value + 1 < total.value) index.value += 1
}
function random() {
  if (total.value > 1) index.value = Math.floor(Math.random() * total.value)
}

function isTypingTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName)
}

function onKeydown(e: KeyboardEvent) {
  if (isTypingTarget(e.target)) return
  if (e.key === 'ArrowLeft') prev()
  else if (e.key === 'ArrowRight') next()
  else if (e.key === 'r') random()
  else if (e.key === 'Escape') router.push(`/runs/${props.runId}`)
}

onMounted(() => window.addEventListener('keydown', onKeydown))
onUnmounted(() => window.removeEventListener('keydown', onKeydown))

function metricEntries(row: SampleRow): [string, number][] {
  return Object.entries(row.metrics)
}
</script>

<template>
  <div class="-mx-6 -mt-4">
    <div class="sticky top-0 z-10 border-b border-surface-border bg-surface px-4 py-2 flex items-center gap-3 flex-wrap text-sm">
      <RouterLink :to="`/runs/${runId}`" class="text-accent hover:text-accent-hover hover:underline whitespace-nowrap">← run</RouterLink>
      <span class="font-mono text-xs text-text-muted">{{ runId }}</span>

      <select v-model="task" class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[10rem]">
        <option v-for="t in tasksData?.tasks ?? []" :key="t.task" :value="t.task">{{ t.task }}</option>
      </select>

      <div class="flex gap-1">
        <button
          v-for="f in FILTERS"
          :key="f"
          class="px-2 py-1 text-xs rounded border capitalize"
          :class="f === filter
            ? 'border-accent-border bg-accent-subtle text-accent'
            : 'border-surface-border text-text-muted hover:bg-surface-raised'"
          @click="filter = f"
        >{{ f }}<span v-if="counts" class="ml-1 tabular-nums">({{ counts[f] }})</span></button>
      </div>

      <span class="text-xs text-text-secondary tabular-nums whitespace-nowrap">{{ positionLabel }}</span>

      <div class="flex gap-1">
        <button class="px-2 py-1 text-xs rounded border border-surface-border hover:bg-surface-raised disabled:opacity-40" :disabled="index === 0" @click="prev">Prev</button>
        <button class="px-2 py-1 text-xs rounded border border-surface-border hover:bg-surface-raised disabled:opacity-40" :disabled="total <= 1" @click="random">Random</button>
        <button class="px-2 py-1 text-xs rounded border border-surface-border hover:bg-surface-raised disabled:opacity-40" :disabled="index + 1 >= total" @click="next">Next</button>
      </div>

      <span class="text-xs text-text-muted ml-auto whitespace-nowrap">← → navigate · r random · esc back</span>
    </div>

    <div class="max-w-4xl mx-auto px-6 py-6">
      <p v-if="tasksError" class="text-sm text-status-danger">{{ tasksError }}</p>
      <p v-else-if="error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2">{{ error }}</p>
      <p v-else-if="loading && !row" class="text-sm text-text-muted py-12 text-center">Loading samples…</p>
      <p v-else-if="!task" class="text-sm text-text-muted py-12 text-center">No task selected.</p>
      <p v-else-if="total === 0" class="text-sm text-text-muted py-12 text-center">No samples for this task/filter.</p>
      <template v-else-if="row">
        <div class="flex items-center gap-2 flex-wrap mb-6">
          <span class="font-mono text-sm text-text-secondary">doc {{ row.doc_id }}</span>
          <span
            class="inline-block rounded px-1.5 py-0.5 text-xs border font-medium"
            :class="row.correct
              ? 'bg-status-success-bg text-status-success border-status-success-border'
              : 'bg-status-danger-bg text-status-danger border-status-danger-border'"
          >{{ row.correct ? 'correct' : 'incorrect' }}</span>
          <span
            v-for="[name, value] in metricEntries(row)"
            :key="name"
            class="inline-block rounded px-1.5 py-0.5 text-xs border font-mono"
            :class="name === primaryMetric
              ? 'border-accent-border bg-accent-subtle text-accent font-semibold'
              : 'border-surface-border text-text-secondary'"
          >{{ name }} {{ value ?? '—' }}</span>
        </div>

        <McqSample v-if="row.kind === 'multiple_choice'" :sample="row" />
        <GenerativeSample v-else :sample="row" />
      </template>
    </div>
  </div>
</template>
