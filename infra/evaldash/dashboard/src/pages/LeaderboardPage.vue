<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { apiPost, useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatDelta, formatScore, formatStderr } from '@/utils/formatting'
import { scoreTint } from '@/utils/score'
import { cellsByModel, fleetBest } from '@/utils/matrix'
import { MAX_COMPARE } from '@/constants'
import type { LeaderboardEntry, Matrix, MatrixCell, MatrixRow, Meta } from '@/types/api'
import EmptyState from '@/components/shared/EmptyState.vue'
import EvalRail from '@/components/charts/EvalRail.vue'
import HistoryModal from '@/components/charts/HistoryModal.vue'

const router = useRouter()

const showArchived = ref(false)
const { data, loading, error, refresh } = useApi<Matrix>(() =>
  showArchived.value ? 'api/matrix?include_archived=1' : 'api/matrix',
)
const { data: meta, refresh: refreshMeta } = useApi<Meta>(() => 'api/meta')

onMounted(() => {
  refresh()
  refreshMeta()
})
watch(showArchived, refresh)
onViewRefresh(() => {
  refresh()
  refreshMeta()
})

// --- Model comparison selection (2–4 models) -> the Compare surface ---
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

// Benchmarks every selected model has a score on — what Compare will actually score.
const sharedTasks = computed<string[]>(() => {
  if (!comparing.value) return []
  return visibleTasks.value.filter((task) =>
    selected.value.every((model) => modelCells.value[model]?.[task]?.value != null),
  )
})

function goCompare() {
  router.push({ path: '/compare', query: { models: selected.value.join(',') } })
}

// --- Δ best: gap from each model's mean score to the leader's ---
const topEntry = computed<LeaderboardEntry | null>(() => data.value?.leaderboard.find((e) => e.score !== null) ?? null)
const topScore = computed<number | null>(() => topEntry.value?.score ?? null)

function deltaBest(entry: LeaderboardEntry): number | null {
  if (entry.score === null || topScore.value === null || entry.score === topScore.value) return null
  return entry.score - topScore.value
}

// --- Archived models sort last ---
const rankedLeaderboard = computed<LeaderboardEntry[]>(() =>
  [...(data.value?.leaderboard ?? [])].sort((a, b) => Number(a.archived) - Number(b.archived)),
)

const modelCells = computed(() => cellsByModel(data.value?.rows ?? []))

async function toggleArchive(model: string, archived: boolean) {
  await apiPost(`api/models/${encodeURIComponent(model)}/archive`, { archived: !archived })
  await Promise.all([refresh(), refreshMeta()])
}

// --- Eval column selection: a suite tree, persisted so column choices survive reloads ---
const SELECTED_KEY = 'evaldash.selectedEvals'
const KNOWN_KEY = 'evaldash.knownEvals'
const columnsOpen = ref(false)
const selectedEvals = reactive(new Set<string>())

function readStored(key: string): string[] | null {
  try {
    const raw = localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as string[]) : null
  } catch {
    return null
  }
}
function persistSelection(present: string[]) {
  localStorage.setItem(SELECTED_KEY, JSON.stringify([...selectedEvals]))
  localStorage.setItem(KNOWN_KEY, JSON.stringify(present))
}
function syncSelection(present: string[]) {
  const stored = readStored(SELECTED_KEY)
  const known = new Set(readStored(KNOWN_KEY) ?? [])
  selectedEvals.clear()
  for (const name of present) {
    if (stored === null || stored.includes(name) || !known.has(name)) selectedEvals.add(name)
  }
  persistSelection(present)
}
watch(
  () => data.value?.tasks,
  (tasks) => {
    if (tasks) syncSelection(tasks)
  },
  { immediate: true },
)

interface SuiteNode {
  suite: string
  evals: string[]
}
const presentTasks = computed(() => new Set(data.value?.tasks ?? []))
const suiteTree = computed<SuiteNode[]>(() =>
  (meta.value?.suites ?? [])
    .map((s) => ({ suite: s.suite, evals: s.evals.filter((e) => presentTasks.value.has(e)) }))
    .filter((s) => s.evals.length > 0),
)
function suiteState(node: SuiteNode): 'all' | 'none' | 'some' {
  const on = node.evals.filter((e) => selectedEvals.has(e)).length
  if (on === 0) return 'none'
  if (on === node.evals.length) return 'all'
  return 'some'
}
function toggleSuite(node: SuiteNode) {
  const enable = suiteState(node) !== 'all'
  for (const e of node.evals) {
    if (enable) selectedEvals.add(e)
    else selectedEvals.delete(e)
  }
  persistSelection(data.value?.tasks ?? [])
}
function toggleEval(name: string) {
  if (selectedEvals.has(name)) selectedEvals.delete(name)
  else selectedEvals.add(name)
  persistSelection(data.value?.tasks ?? [])
}
const visibleTasks = computed(() => (data.value?.tasks ?? []).filter((t) => selectedEvals.has(t)))

// --- Fleet best per benchmark (the rail caret and matrix column marker) ---
const best = computed(() => fleetBest(data.value?.rows ?? [], visibleTasks.value))

// --- Fleet readout ---
// Coverage counts scored cells over the *visible* columns only, so deselecting benchmarks keeps the
// numerator and denominator on the same set (a full-set numerator would inflate past 100%).
const readout = computed(() => {
  const rows = data.value?.rows ?? []
  const tasks = visibleTasks.value
  const covered = rows.reduce((n, row) => n + tasks.filter((t) => row.cells[t]?.value != null).length, 0)
  const cells = rows.length * tasks.length
  return {
    models: rows.length,
    benchmarks: tasks.length,
    coverage: cells ? Math.round((covered / cells) * 100) : 0,
  }
})

// --- Per-benchmark matrix (opt-in), sortable by mean or a benchmark column ---
const showMatrix = ref(false)
const sortKey = ref<string>('rank')
function sortBy(key: string) {
  sortKey.value = sortKey.value === key ? 'rank' : key
}
const matrixRows = computed<MatrixRow[]>(() => {
  const order = new Map(rankedLeaderboard.value.map((e, i) => [e.model, i]))
  const rows = [...(data.value?.rows ?? [])]
  if (sortKey.value === 'rank') {
    return rows.sort((a, b) => (order.get(a.model) ?? 0) - (order.get(b.model) ?? 0))
  }
  const val = (r: MatrixRow) => r.cells[sortKey.value]?.value ?? null
  return rows.sort((a, b) => {
    const av = val(a)
    const bv = val(b)
    if (av === null && bv === null) return (order.get(a.model) ?? 0) - (order.get(b.model) ?? 0)
    if (av === null) return 1
    if (bv === null) return -1
    return bv - av
  })
})

function heatStyle(cell: MatrixCell): Record<string, string> {
  if (cell.value === null) return {}
  return { backgroundColor: scoreTint(cell.value) }
}
function isColumnBest(model: string, task: string): boolean {
  return best.value[task]?.model === model
}
function cellFor(row: MatrixRow, task: string): MatrixCell | undefined {
  return row.cells[task]
}

// --- Score-over-time modal ---
const historyTarget = ref<{ model: string; task: string } | null>(null)
function openHistory(model: string, task: string) {
  historyTarget.value = { model, task }
}
function goToRun(runId: string) {
  router.push(`/runs/${runId}`)
}
function goToModel(model: string) {
  router.push(`/models/${encodeURIComponent(model)}`)
}
</script>

<template>
  <section>
    <div class="mb-4">
      <h2 class="text-lg font-semibold">Leaderboard</h2>
      <p class="text-xs text-text-muted mt-0.5">
        Mean of per-benchmark primary-metric scores — each benchmark equal weight, mmlu subtasks rolled up.
      </p>
    </div>

    <!-- Fleet readout -->
    <div v-if="data" class="flex rounded-lg border border-surface-border bg-surface overflow-hidden mb-5">
      <div class="px-5 py-3 border-r border-surface-border-subtle">
        <div class="font-mono text-[10px] uppercase tracking-widest text-text-muted">Models</div>
        <div class="font-mono text-2xl font-semibold tabular-nums">{{ readout.models }}</div>
      </div>
      <div class="px-5 py-3 border-r border-surface-border-subtle">
        <div class="font-mono text-[10px] uppercase tracking-widest text-text-muted">Benchmarks</div>
        <div class="font-mono text-2xl font-semibold tabular-nums">{{ readout.benchmarks }}</div>
      </div>
      <div class="px-5 py-3">
        <div class="font-mono text-[10px] uppercase tracking-widest text-text-muted">Fleet coverage</div>
        <div class="font-mono text-2xl font-semibold tabular-nums">
          {{ readout.coverage }}<span class="text-sm text-text-muted">%</span>
        </div>
      </div>
    </div>

    <div class="flex flex-wrap items-center gap-4 mb-4">
      <label class="flex items-center gap-2 text-sm text-text-secondary">
        <input v-model="showArchived" type="checkbox" class="accent-accent" />
        Show archived
      </label>
      <button
        class="text-sm px-3 py-1 rounded border border-surface-border hover:bg-surface-raised"
        @click="columnsOpen = !columnsOpen"
      >
        Columns ({{ visibleTasks.length }}/{{ data?.tasks.length ?? 0 }})
      </button>
    </div>

    <div v-if="columnsOpen && suiteTree.length" class="rounded-lg border border-surface-border bg-surface p-4 mb-4">
      <div class="flex flex-wrap gap-x-8 gap-y-4">
        <div v-for="node in suiteTree" :key="node.suite" class="min-w-[10rem]">
          <label class="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1.5">
            <input
              type="checkbox"
              class="accent-accent"
              :checked="suiteState(node) === 'all'"
              :indeterminate.prop="suiteState(node) === 'some'"
              @change="toggleSuite(node)"
            />
            {{ node.suite }}
          </label>
          <label
            v-for="e in node.evals"
            :key="e"
            class="flex items-center gap-2 text-sm text-text-secondary pl-1 py-0.5"
          >
            <input type="checkbox" class="accent-accent" :checked="selectedEvals.has(e)" @change="toggleEval(e)" />
            <span class="font-mono text-[13px]">{{ e }}</span>
          </label>
        </div>
      </div>
    </div>

    <div v-if="error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2 mb-4">
      {{ error }}
    </div>

    <div v-if="loading && !data" class="text-sm text-text-muted py-12 text-center">Loading…</div>

    <EmptyState v-else-if="data && data.rows.length === 0" icon="🏁" message="No runs yet." />

    <div v-else-if="data" class="space-y-6">
      <!-- Ranking: mean score + the eval rail as each row's measurement profile -->
      <div>
        <div class="flex items-baseline justify-between mb-2">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Ranking</h3>
          <span class="text-xs text-text-muted">
            Tick 2–{{ MAX_COMPARE }} to compare · one gauge per benchmark, height = score, whisker = ±stderr,
            <span class="font-mono" style="color: var(--c-best)">▬</span> = fleet best
          </span>
        </div>
        <div class="overflow-x-auto rounded-lg border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead>
              <tr class="border-b border-surface-border bg-surface-raised text-xs font-semibold uppercase tracking-wider text-text-secondary">
                <th class="px-3 py-2 text-left w-8"></th>
                <th class="px-3 py-2 text-left w-8">#</th>
                <th class="px-3 py-2 text-left">Model</th>
                <th class="px-3 py-2 text-right">Mean</th>
                <th class="px-3 py-2 text-right">Δ best</th>
                <th class="px-3 py-2 text-left">Coverage</th>
                <th class="px-3 py-2 text-left">Profile</th>
                <th class="px-3 py-2 text-right"></th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(entry, i) in rankedLeaderboard"
                :key="entry.model"
                class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors cursor-pointer"
                :class="{ 'opacity-50': entry.archived }"
                @click="goToModel(entry.model)"
              >
                <td class="px-3 py-2" @click.stop>
                  <input
                    type="checkbox"
                    class="align-middle accent-accent"
                    :checked="selected.includes(entry.model)"
                    :disabled="!canSelect(entry.model)"
                    @change="toggleModel(entry.model)"
                  />
                </td>
                <td class="px-3 py-2 text-text-muted tabular-nums">{{ i + 1 }}</td>
                <td class="px-3 py-2 whitespace-nowrap">
                  <span class="font-mono font-semibold text-[13px] text-accent">{{ entry.model }}</span>
                  <span
                    v-if="entry.version"
                    class="ml-1 rounded bg-surface-sunken px-1.5 py-0.5 text-[11px] font-mono text-text-muted"
                  >{{ entry.version }}</span>
                </td>
                <td class="px-3 py-2 text-right tabular-nums font-semibold font-mono whitespace-nowrap">
                  <template v-if="entry.score !== null">
                    {{ formatScore(entry.score) }}
                    <span class="text-text-muted text-xs font-normal">{{ formatStderr(entry.score, entry.stderr) }}</span>
                  </template>
                  <span v-else class="text-text-muted">—</span>
                </td>
                <td class="px-3 py-2 text-right tabular-nums font-mono text-xs text-text-muted whitespace-nowrap">
                  {{ formatDelta(deltaBest(entry)) }}
                </td>
                <td class="px-3 py-2">
                  <div class="flex items-center gap-2">
                    <span class="w-[52px] h-1.5 rounded-full bg-surface-sunken overflow-hidden">
                      <span class="block h-full rounded-full bg-accent" :style="{ width: `${(entry.covered / (entry.total || 1)) * 100}%` }" />
                    </span>
                    <span
                      class="font-mono text-[11px] tabular-nums"
                      :class="entry.covered <= 1 ? 'text-status-warning' : 'text-text-secondary'"
                    >{{ entry.covered }}/{{ entry.total }}<span v-if="entry.covered <= 1"> ⚠</span></span>
                  </div>
                </td>
                <td class="px-3 py-2" @click.stop>
                  <EvalRail
                    :tasks="visibleTasks"
                    :cells="modelCells[entry.model] ?? {}"
                    :best="best"
                    :model="entry.model"
                    size="sm"
                    @pick="(task) => openHistory(entry.model, task)"
                  />
                </td>
                <td class="px-3 py-2 text-right" @click.stop>
                  <button
                    class="text-[11px] text-text-muted hover:text-accent whitespace-nowrap"
                    :title="entry.archived ? 'Unarchive model' : 'Archive model'"
                    @click="toggleArchive(entry.model, entry.archived)"
                  >{{ entry.archived ? 'unarchive' : 'archive' }}</button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p class="text-xs text-text-muted mt-2 leading-relaxed">
          Coverage is honest: the mean is over the benchmarks a model has actually run, so a row at 1/{{ readout.benchmarks }}
          is not comparable to one at {{ readout.benchmarks }}/{{ readout.benchmarks }} — low-coverage rows are flagged.
          Use Compare to score any set of models on their shared benchmarks only.
        </p>
      </div>

      <!-- Per-benchmark matrix: opt-in, coverage-aware, sortable -->
      <div>
        <button
          class="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2 hover:text-text"
          @click="showMatrix = !showMatrix"
        >
          <span class="text-text-muted">{{ showMatrix ? '▾' : '▸' }}</span>
          Per-benchmark matrix
          <span class="font-normal normal-case text-text-muted">({{ matrixRows.length }} models × {{ visibleTasks.length }} benchmarks · click a header to sort · a cell for history)</span>
        </button>
        <div v-if="showMatrix" class="overflow-x-auto rounded-lg border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead>
              <tr class="border-b border-surface-border bg-surface-raised">
                <th
                  class="sticky left-0 z-10 bg-surface-raised px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider cursor-pointer"
                  :class="sortKey === 'rank' ? 'text-accent' : 'text-text-secondary'"
                  @click="sortBy('rank')"
                >Model</th>
                <th
                  v-for="task in visibleTasks"
                  :key="task"
                  class="px-3 py-2 text-center text-xs font-semibold uppercase tracking-wider whitespace-nowrap cursor-pointer"
                  :class="sortKey === task ? 'text-accent' : 'text-text-secondary'"
                  @click="sortBy(task)"
                >
                  {{ task }}
                  <span v-if="best[task]" class="block font-normal normal-case font-mono text-[10px]" style="color: var(--c-best)">▲ {{ formatScore(best[task].value) }}</span>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="row in matrixRows"
                :key="row.model"
                class="border-b border-surface-border-subtle"
                :class="{ 'opacity-50': row.archived }"
              >
                <td class="sticky left-0 z-10 bg-surface px-3 py-2 whitespace-nowrap">
                  <button class="font-mono text-[13px] text-accent hover:underline" @click="goToModel(row.model)">{{ row.model }}</button>
                  <span
                    v-if="row.version"
                    class="ml-1 rounded bg-surface-sunken px-1.5 py-0.5 text-[11px] font-mono text-text-muted"
                  >{{ row.version }}</span>
                </td>
                <td v-for="task in visibleTasks" :key="task" class="p-1 text-center align-middle">
                  <template v-if="cellFor(row, task)">
                    <button
                      v-if="cellFor(row, task)!.value !== null"
                      class="w-full rounded px-2 py-1.5 leading-tight cursor-pointer hover:ring-1 hover:ring-accent-border"
                      :class="isColumnBest(row.model, task) ? 'ring-2 ring-inset' : ''"
                      :style="{ ...heatStyle(cellFor(row, task)!), ...(isColumnBest(row.model, task) ? { '--tw-ring-color': 'var(--c-best)' } : {}) }"
                      :title="`${cellFor(row, task)!.metric} — click for history`"
                      @click="openHistory(row.model, task)"
                    >
                      <span class="tabular-nums font-mono font-medium">{{ formatScore(cellFor(row, task)!.value) }}</span>
                      <span class="block text-[10px] text-text-muted tabular-nums font-mono leading-none min-h-[0.75rem]">
                        {{ formatStderr(cellFor(row, task)!.value, cellFor(row, task)!.stderr) }}
                      </span>
                    </button>
                    <button
                      v-else
                      class="w-full rounded px-2 py-1.5 text-[11px] font-mono font-semibold leading-tight cursor-pointer"
                      :class="cellFor(row, task)!.status === 'infra_failed' ? 'text-status-warning bg-status-warning-bg' : 'text-status-danger bg-status-danger-bg'"
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

    <!-- Compare bar -->
    <div
      v-if="comparing"
      class="sticky bottom-0 mt-5 flex flex-wrap items-center gap-4 rounded-lg border border-surface-border bg-surface px-4 py-3 shadow-lg"
    >
      <span class="font-mono text-[10px] uppercase tracking-widest text-text-muted">Compare</span>
      <span
        v-for="m in selected"
        :key="m"
        class="inline-flex items-center gap-2 font-mono text-xs px-2.5 py-1 rounded-full border border-surface-border"
      >{{ m }}</span>
      <span class="text-xs text-text-muted">
        shared benchmarks:
        <span class="font-mono text-text-secondary">{{ sharedTasks.length ? sharedTasks.join(' · ') : 'none' }}</span>
      </span>
      <div class="flex-1"></div>
      <button class="px-4 py-2 rounded-lg bg-accent text-surface text-sm font-medium hover:bg-accent-hover" @click="goCompare">
        Compare {{ selected.length }} models →
      </button>
    </div>

    <HistoryModal
      v-if="historyTarget"
      :model="historyTarget.model"
      :task="historyTarget.task"
      @close="historyTarget = null"
    />
  </section>
</template>
