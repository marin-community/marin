<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatDelta, formatScore, formatStderr } from '@/utils/formatting'
import type { LeaderboardEntry, Matrix, MatrixCell } from '@/types/api'
import EmptyState from '@/components/shared/EmptyState.vue'
import ModelCompareChart from '@/components/charts/ModelCompareChart.vue'
import EvalProfileChart from '@/components/charts/EvalProfileChart.vue'
import SparkStrip from '@/components/charts/SparkStrip.vue'
import HistoryModal from '@/components/charts/HistoryModal.vue'

const router = useRouter()
const { data, loading, error, refresh } = useApi<Matrix>(() => 'api/matrix')

onMounted(refresh)
onViewRefresh(refresh)

// --- Model comparison selection (2–4 models -> grouped bar chart) ---
const MAX_COMPARE = 4
const selected = ref<string[]>([])

function toggleModel(model: string) {
  const at = selected.value.indexOf(model)
  if (at >= 0) selected.value.splice(at, 1)
  else if (selected.value.length < MAX_COMPARE) selected.value.push(model)
}

function canSelect(model: string): boolean {
  return selected.value.includes(model) || selected.value.length < MAX_COMPARE
}

const comparing = computed(() => selected.value.length >= 2)

// --- Δ best: gap from each model's mean score to the leader's, empty for the leader ---
const topScore = computed<number | null>(() => data.value?.leaderboard.find((e) => e.score !== null)?.score ?? null)

function deltaBest(entry: LeaderboardEntry): number | null {
  if (entry.score === null || topScore.value === null || entry.score === topScore.value) return null
  return entry.score - topScore.value
}

// --- Score-over-time modal target ---
const historyTarget = ref<{ model: string; task: string } | null>(null)

// A translucent accent tint scaled by the (accuracy) score: legible over both themes.
function heatStyle(cell: MatrixCell): Record<string, string> {
  if (cell.value === null) return {}
  const v = Math.max(0, Math.min(1, cell.value))
  return { backgroundColor: `rgba(56, 142, 255, ${(0.1 + 0.55 * v).toFixed(3)})` }
}

// Outline class for a non-succeeded cell: amber for a failed eval, red for an infra failure.
function outlineClass(status: string): string {
  if (status === 'infra_failed') return 'ring-1 ring-inset ring-status-danger-border text-status-danger'
  if (status === 'failed') return 'ring-1 ring-inset ring-status-warning-border text-status-warning'
  return ''
}

function cellFor(row: { cells: Record<string, MatrixCell> }, task: string): MatrixCell | undefined {
  return row.cells[task]
}

function openHistory(model: string, task: string) {
  historyTarget.value = { model, task }
}

function goToRun(runId: string) {
  router.push(`/runs/${runId}`)
}
</script>

<template>
  <section>
    <div class="flex items-baseline justify-between mb-4">
      <div>
        <h2 class="text-lg font-semibold">Leaderboard</h2>
        <p class="text-xs text-text-muted mt-0.5">
          Mean of per-task primary-metric scores (each benchmark equal weight, mmlu subtasks rolled up).
          Cell colour scales with the score; click a score for its history, a failure to open the run.
        </p>
      </div>
    </div>

    <div v-if="error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2 mb-4">
      {{ error }}
    </div>

    <div v-if="loading && !data" class="text-sm text-text-muted py-12 text-center">Loading…</div>

    <EmptyState
      v-else-if="data && data.rows.length === 0"
      icon="🏁"
      message="No runs yet."
    />

    <div v-else-if="data" class="space-y-8">
      <!-- Ranking: mean of per-task scores per model, with compare checkboxes -->
      <div>
        <div class="flex items-baseline justify-between mb-2">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Ranking</h3>
          <span class="text-xs text-text-muted">Tick 2–{{ MAX_COMPARE }} models to compare</span>
        </div>
        <div class="overflow-x-auto rounded-lg border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead>
              <tr class="border-b border-surface-border bg-surface-raised text-xs font-semibold uppercase tracking-wider text-text-secondary">
                <th class="px-3 py-2 text-left w-8"></th>
                <th class="px-3 py-2 text-left w-8">#</th>
                <th class="px-3 py-2 text-left">Model</th>
                <th class="px-3 py-2 text-right">Mean score</th>
                <th class="px-3 py-2 text-right">Δ best</th>
                <th class="px-3 py-2 text-right">Coverage</th>
                <th class="px-3 py-2 text-left">Profile</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(entry, i) in data.leaderboard"
                :key="entry.model"
                class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
              >
                <td class="px-3 py-2">
                  <input
                    type="checkbox"
                    class="align-middle accent-accent"
                    :checked="selected.includes(entry.model)"
                    :disabled="!canSelect(entry.model)"
                    @change="toggleModel(entry.model)"
                  />
                </td>
                <td class="px-3 py-2 text-text-muted tabular-nums">{{ i + 1 }}</td>
                <td class="px-3 py-2 font-mono text-[13px] whitespace-nowrap">{{ entry.model }}</td>
                <td class="px-3 py-2 text-right tabular-nums font-medium whitespace-nowrap">
                  <template v-if="entry.score !== null">
                    {{ formatScore(entry.score) }}
                    <span class="text-text-muted text-xs">{{ formatStderr(entry.score, entry.stderr) }}</span>
                  </template>
                  <span v-else class="text-text-muted">—</span>
                </td>
                <td class="px-3 py-2 text-right tabular-nums text-text-muted whitespace-nowrap">
                  {{ formatDelta(deltaBest(entry)) }}
                </td>
                <td class="px-3 py-2 text-right tabular-nums text-text-secondary">{{ entry.covered }}/{{ entry.total }} tasks</td>
                <td class="px-3 py-2">
                  <SparkStrip :model="entry.model" :tasks="data.tasks" :matrix="data" />
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Eval profile: every model's score on every task, snowball highlighted -->
      <div>
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Eval profile</h3>
        <p class="text-xs text-text-muted mb-2">
          each row: one eval; dots: models; highlighted: snowball; line: gap to fleet best
        </p>
        <EvalProfileChart :matrix="data" />
      </div>

      <!-- Model comparison chart -->
      <div v-if="comparing">
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">
          Comparing {{ selected.length }} models
        </h3>
        <ModelCompareChart :matrix="data" :models="selected" />
      </div>

      <!-- Per-task heatmap -->
      <div>
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Per-task scores</h3>
        <div class="overflow-x-auto rounded-lg border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead>
              <tr class="border-b border-surface-border bg-surface-raised">
                <th class="sticky left-0 z-10 bg-surface-raised px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
                  Model
                </th>
                <th
                  v-for="task in data.tasks"
                  :key="task"
                  class="px-3 py-2 text-center text-xs font-semibold uppercase tracking-wider text-text-secondary whitespace-nowrap"
                >
                  {{ task }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="row in data.rows"
                :key="row.model"
                class="border-b border-surface-border-subtle"
              >
                <td class="sticky left-0 z-10 bg-surface px-3 py-2 font-mono text-[13px] whitespace-nowrap">
                  {{ row.model }}
                </td>
                <td
                  v-for="task in data.tasks"
                  :key="task"
                  class="p-1 text-center align-middle"
                >
                  <template v-if="cellFor(row, task)">
                    <!-- Succeeded: heatmap score cell, click -> history -->
                    <button
                      v-if="cellFor(row, task)!.value !== null"
                      class="w-full rounded px-2 py-1.5 leading-tight hover:ring-1 hover:ring-accent-border cursor-pointer"
                      :style="heatStyle(cellFor(row, task)!)"
                      :title="`${cellFor(row, task)!.metric} — click for history`"
                      @click="openHistory(row.model, task)"
                    >
                      <span class="tabular-nums font-medium">{{ formatScore(cellFor(row, task)!.value) }}</span>
                      <span class="block text-[10px] text-text-muted tabular-nums leading-none min-h-[0.75rem]">
                        {{ formatStderr(cellFor(row, task)!.value, cellFor(row, task)!.stderr) }}
                      </span>
                    </button>
                    <!-- Failure: outlined, click -> run detail -->
                    <button
                      v-else
                      class="w-full rounded px-2 py-1.5 text-[11px] leading-tight cursor-pointer hover:bg-surface-raised"
                      :class="outlineClass(cellFor(row, task)!.status)"
                      :title="`${cellFor(row, task)!.status} — open run ${cellFor(row, task)!.run_id}`"
                      @click="goToRun(cellFor(row, task)!.run_id)"
                    >
                      {{ cellFor(row, task)!.status === 'infra_failed' ? 'infra' : 'failed' }}
                    </button>
                  </template>
                  <span v-else class="text-text-muted">—</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <HistoryModal
      v-if="historyTarget"
      :model="historyTarget.model"
      :task="historyTarget.task"
      @close="historyTarget = null"
    />
  </section>
</template>
