<script setup lang="ts">
/**
 * Per-sample browser for a succeeded run. Lists the tasks with exported sample parquets, then
 * pages their rows filtered by correctness (per-sample primary metric == 1). A row click opens a
 * side panel with the full prompt arguments and the raw + filtered model responses.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useApi } from '@/composables/useApi'
import type { SampleRow, SamplesResponse, SampleTasksResponse } from '@/types/api'

const props = defineProps<{ runId: string }>()

const LIMIT = 50
type Correct = 'all' | 'correct' | 'incorrect'

const selectedTask = ref('')
const correct = ref<Correct>('all')
const offset = ref(0)
const selectedRow = ref<SampleRow | null>(null)

const { data: tasksData, error: tasksError, refresh: refreshTasks } = useApi<SampleTasksResponse>(
  () => `api/runs/${props.runId}/samples/tasks`,
)

function samplesPath(): string {
  const params = new URLSearchParams({
    task: selectedTask.value,
    offset: String(offset.value),
    limit: String(LIMIT),
    correct: correct.value,
  })
  return `api/runs/${props.runId}/samples?${params.toString()}`
}

const { data, loading, error, refresh } = useApi<SamplesResponse>(samplesPath)

onMounted(refreshTasks)

// Default to the first discovered task once the list loads.
watch(tasksData, (tasks) => {
  if (!selectedTask.value && tasks?.tasks.length) selectedTask.value = tasks.tasks[0].task
})

watch([selectedTask, correct], () => {
  offset.value = 0
  if (selectedTask.value) refresh()
})

function toText(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  return JSON.stringify(value)
}

function pretty(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  return JSON.stringify(value, null, 2)
}

function truncate(text: string, n = 140): string {
  return text.length > n ? `${text.slice(0, n)}…` : text
}

const total = computed(() => data.value?.total ?? 0)
const shownFrom = computed(() => (total.value === 0 ? 0 : offset.value + 1))
const shownTo = computed(() => Math.min(offset.value + LIMIT, total.value))

function nextPage() {
  if (offset.value + LIMIT < total.value) {
    offset.value += LIMIT
    refresh()
  }
}
function prevPage() {
  if (offset.value > 0) {
    offset.value = Math.max(0, offset.value - LIMIT)
    refresh()
  }
}
</script>

<template>
  <div>
    <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Samples</h3>

    <p v-if="tasksError" class="text-sm text-status-danger">{{ tasksError }}</p>
    <p v-else-if="tasksData && !tasksData.available" class="text-sm text-text-muted">
      No per-sample exports available{{ tasksData.error ? ` — ${tasksData.error}` : '' }}.
    </p>
    <p v-else-if="tasksData && tasksData.tasks.length === 0" class="text-sm text-text-muted">No sample files found.</p>

    <div v-else-if="tasksData" class="rounded-lg border border-surface-border bg-surface p-4 space-y-3">
      <!-- Controls -->
      <div class="flex flex-wrap items-end gap-3">
        <label class="flex flex-col text-xs text-text-secondary gap-1">
          Task
          <select v-model="selectedTask" class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[12rem]">
            <option v-for="t in tasksData.tasks" :key="t.task" :value="t.task">{{ t.task }}</option>
          </select>
        </label>
        <div class="flex gap-1">
          <button
            v-for="opt in (['all', 'correct', 'incorrect'] as Correct[])"
            :key="opt"
            class="px-2 py-1 text-xs rounded border capitalize"
            :class="opt === correct
              ? 'border-accent-border bg-accent-subtle text-accent'
              : 'border-surface-border text-text-muted hover:bg-surface-raised'"
            @click="correct = opt"
          >{{ opt }}</button>
        </div>
        <span v-if="data" class="text-xs text-text-muted ml-auto">
          primary: <span class="font-mono">{{ data.primary_metric ?? '—' }}</span>
        </span>
      </div>

      <div v-if="error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-xs px-2 py-1">
        {{ error }}
      </div>
      <div v-if="loading && !data" class="text-sm text-text-muted py-6 text-center">Loading samples…</div>

      <template v-else-if="data && data.available">
        <div class="overflow-x-auto rounded border border-surface-border">
          <table class="w-full border-collapse text-xs">
            <thead>
              <tr class="border-b border-surface-border bg-surface-raised text-text-secondary">
                <th class="px-2 py-1.5 text-left w-16">Doc</th>
                <th class="px-2 py-1.5 text-left w-16">Metric</th>
                <th class="px-2 py-1.5 text-left">Gold target</th>
                <th class="px-2 py-1.5 text-left">Model answer</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(row, i) in data.rows"
                :key="`${row.doc_id}-${i}`"
                class="border-b border-surface-border-subtle hover:bg-surface-raised cursor-pointer"
                @click="selectedRow = row"
              >
                <td class="px-2 py-1.5 font-mono text-text-secondary">{{ row.doc_id }}</td>
                <td class="px-2 py-1.5">
                  <span
                    class="inline-block rounded px-1 py-0.5 border font-medium"
                    :class="row.correct
                      ? 'bg-status-success-bg text-status-success border-status-success-border'
                      : 'bg-status-danger-bg text-status-danger border-status-danger-border'"
                  >{{ row.primary_value ?? (row.correct ? '✓' : '✗') }}</span>
                </td>
                <td class="px-2 py-1.5 font-mono max-w-[24ch] truncate">{{ truncate(toText(row.target), 60) }}</td>
                <td class="px-2 py-1.5 font-mono max-w-[40ch] truncate">{{ truncate(toText(row.filtered_responses)) }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="flex items-center justify-between text-xs text-text-muted">
          <span>{{ shownFrom }}–{{ shownTo }} of {{ total }}</span>
          <div class="flex gap-2">
            <button
              class="px-2 py-1 rounded border border-surface-border hover:bg-surface-raised disabled:opacity-40"
              :disabled="offset === 0"
              @click="prevPage"
            >Prev</button>
            <button
              class="px-2 py-1 rounded border border-surface-border hover:bg-surface-raised disabled:opacity-40"
              :disabled="offset + LIMIT >= total"
              @click="nextPage"
            >Next</button>
          </div>
        </div>
      </template>
      <p v-else-if="data" class="text-sm text-text-muted">{{ data.error ?? 'No samples.' }}</p>
    </div>

    <!-- Detail side panel -->
    <div v-if="selectedRow" class="fixed inset-0 z-50 flex justify-end bg-black/40" @click.self="selectedRow = null">
      <div class="w-full max-w-xl h-full overflow-auto bg-surface border-l border-surface-border p-5 space-y-4">
        <div class="flex items-center justify-between gap-3">
          <h4 class="text-sm font-semibold">
            Sample <span class="font-mono">{{ selectedRow.doc_id }}</span>
            <span
              class="ml-2 inline-block rounded px-1.5 py-0.5 text-xs border"
              :class="selectedRow.correct
                ? 'bg-status-success-bg text-status-success border-status-success-border'
                : 'bg-status-danger-bg text-status-danger border-status-danger-border'"
            >{{ selectedRow.correct ? 'correct' : 'incorrect' }}</span>
          </h4>
          <button class="text-xs px-2 py-1 rounded border border-surface-border hover:bg-surface-raised" @click="selectedRow = null">Close</button>
        </div>

        <div>
          <h5 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1">Prompt arguments</h5>
          <pre class="rounded border border-surface-border bg-surface-sunken p-3 text-[12px] font-mono overflow-auto max-h-72 whitespace-pre-wrap">{{ pretty(selectedRow.arguments) }}</pre>
        </div>
        <div>
          <h5 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1">Gold target</h5>
          <pre class="rounded border border-surface-border bg-surface-sunken p-3 text-[12px] font-mono overflow-auto max-h-40 whitespace-pre-wrap">{{ pretty(selectedRow.target) }}</pre>
        </div>
        <div>
          <h5 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1">Filtered response</h5>
          <pre class="rounded border border-surface-border bg-surface-sunken p-3 text-[12px] font-mono overflow-auto max-h-40 whitespace-pre-wrap">{{ pretty(selectedRow.filtered_responses) }}</pre>
        </div>
        <div>
          <h5 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1">Raw response</h5>
          <pre class="rounded border border-surface-border bg-surface-sunken p-3 text-[12px] font-mono overflow-auto max-h-56 whitespace-pre-wrap">{{ pretty(selectedRow.responses) }}</pre>
        </div>
        <div>
          <h5 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1">Metrics</h5>
          <dl class="text-xs grid grid-cols-2 gap-1">
            <template v-for="(v, k) in selectedRow.metrics" :key="k">
              <dt class="text-text-muted font-mono">{{ k }}</dt>
              <dd class="tabular-nums">{{ v }}</dd>
            </template>
          </dl>
        </div>
      </div>
    </div>
  </div>
</template>
