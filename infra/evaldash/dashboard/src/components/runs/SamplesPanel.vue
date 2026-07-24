<script setup lang="ts">
/**
 * Per-sample browser for a succeeded run. Lists the tasks with exported sample parquets, then
 * pages their rows filtered by correctness (per-sample primary metric == 1). A row click opens
 * the full-screen sample viewer at that row; "Open viewer" jumps straight in at the first row.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import type { SampleRow, SamplesResponse, SampleTasksResponse } from '@/types/api'

const props = defineProps<{ runId: string }>()
const router = useRouter()

const LIMIT = 50
type Correct = 'all' | 'correct' | 'incorrect'

const selectedTask = ref('')
const correct = ref<Correct>('all')
const offset = ref(0)

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

/** A short "answer" summary for a row: the picked choice, the extracted generation, or a trajectory hint. */
function answerSummary(row: SampleRow): string {
  if (row.kind === 'multiple_choice') {
    const choice = row.model_choice !== null ? (row.choices?.[row.model_choice] ?? null) : null
    return choice ? `${choice.label}: ${choice.text.trim()}` : '—'
  }
  if (row.kind === 'agentic') {
    return row.trajectory_uri ? 'agentic trajectory' : '—'
  }
  return row.extracted ?? ''
}

/** The row's primary-metric value, looked up by the page's primary metric name. */
function primaryValue(row: SampleRow): number | null {
  const metric = data.value?.primary_metric
  return metric ? (row.metrics[metric] ?? null) : null
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

function openViewer(rowIndex: number) {
  router.push({
    path: `/runs/${props.runId}/samples`,
    query: { task: selectedTask.value, filter: correct.value, i: String(offset.value + rowIndex) },
  })
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
        <button
          class="px-2 py-1 text-xs rounded border border-surface-border hover:bg-surface-raised"
          @click="openViewer(0)"
        >Open viewer ⛶</button>
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
                <th class="px-2 py-1.5 text-left w-28">Grading</th>
                <th class="px-2 py-1.5 text-left">Gold target</th>
                <th class="px-2 py-1.5 text-left">Model answer</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(row, i) in data.rows"
                :key="`${row.doc_id}-${i}`"
                class="border-b border-surface-border-subtle hover:bg-surface-raised cursor-pointer"
                @click="openViewer(i)"
              >
                <td class="px-2 py-1.5 font-mono text-text-secondary">{{ row.doc_id }}</td>
                <td class="px-2 py-1.5">
                  <span
                    class="inline-block rounded px-1 py-0.5 border font-medium"
                    :class="row.correct
                      ? 'bg-status-success-bg text-status-success border-status-success-border'
                      : 'bg-status-danger-bg text-status-danger border-status-danger-border'"
                  >{{ primaryValue(row) ?? (row.correct ? '✓' : '✗') }}</span>
                </td>
                <td class="px-2 py-1.5 font-mono text-text-muted truncate">{{ row.grading?.method ?? '—' }}</td>
                <td class="px-2 py-1.5 font-mono max-w-[24ch] truncate">{{ truncate(row.target_text ?? '', 60) }}</td>
                <td class="px-2 py-1.5 font-mono max-w-[40ch] truncate">{{ truncate(answerSummary(row)) }}</td>
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
  </div>
</template>
