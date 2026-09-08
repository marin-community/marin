<script setup lang="ts">
/**
 * The panel: models by benchmark, every cell a score with the 95% interval behind it.
 *
 * There is no headline cross-benchmark mean. A mean over benchmarks has no interpretation without a
 * declared panel, a per-benchmark metric, and a rule for the benchmarks a model never ran, so it is
 * opt-in and renders those three things with it. Per-benchmark comparison is the primary surface,
 * and every ordering uses the interval's lower bound: a run that lost items should not outrank one
 * that graded them.
 */
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { apiPost, useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatCoverage, formatDelta, formatInterval, formatScore } from '@/utils/formatting'
import { scoreTint } from '@/utils/score'
import { cellsByModel, fleetBest, isPartialCoverage } from '@/utils/panel'
import { MAX_COMPARE, isSmokeEval } from '@/constants'
import {
  FLAG_NOTES,
  INTERVAL_KIND,
  RESULT_FLAG,
  type Meta,
  type MissingCell,
  type Panel,
  type PanelCell,
  type PanelRow,
} from '@/types/api'
import EmptyState from '@/components/shared/EmptyState.vue'
import EvalRail from '@/components/charts/EvalRail.vue'
import HistoryModal from '@/components/charts/HistoryModal.vue'

const router = useRouter()

// --- Request state. Everything the server needs to resolve a panel goes in the query, so a panel is
// a shareable URL and the filters that produced a number travel with it. ---
const showArchived = ref(false)
const showFlagged = ref(false)
const completeOnly = ref(false)
const cohort = ref('')
const modelQuery = ref('')
const facetValues = reactive<Record<string, string>>({})
const aggregatePolicy = ref('')

const SELECTED_KEY = 'evaldash.selectedEvals'
const KNOWN_KEY = 'evaldash.knownEvals'
const selectedEvals = reactive(new Set<string>())
const knownEvals = ref<string[]>([])

// The selection every panel-backed endpoint shares. Compare takes the same one, so a head-to-head
// launched from a narrowed panel scores the benchmarks and cohort the reader was looking at.
const selection = computed<Record<string, string>>(() => {
  const params: Record<string, string> = {}
  if (completeOnly.value) params.complete = '1'
  if (showFlagged.value) params.include_flagged = '1'
  if (cohort.value) params.cohort = cohort.value
  if (selectedEvals.size && selectedEvals.size !== knownEvals.value.length) {
    params.benchmarks = [...selectedEvals].join(',')
  }
  for (const [facet, value] of Object.entries(facetValues)) if (value) params[facet] = value
  return params
})

const query = computed(() => {
  const params = new URLSearchParams(selection.value)
  if (showArchived.value) params.set('include_archived', '1')
  if (modelQuery.value.trim()) params.set('model', modelQuery.value.trim())
  if (aggregatePolicy.value) params.set('aggregate', aggregatePolicy.value)
  const suffix = params.toString()
  return suffix ? `api/panel?${suffix}` : 'api/panel'
})

const { data, loading, error, refresh } = useApi<Panel>(() => query.value)
const { data: meta, refresh: refreshMeta } = useApi<Meta>(() => 'api/meta')

onMounted(() => {
  refresh()
  refreshMeta()
})
watch(query, refresh)
onViewRefresh(() => {
  refresh()
  refreshMeta()
})

// Every filter narrowing the panel, shown as removable chips: a number on screen should never be the
// product of a filter the reader cannot see.
const activeFilters = computed(() => {
  const active: { label: string; value: string; clear: () => void }[] = []
  for (const [facet, value] of Object.entries(facetValues)) {
    if (value) active.push({ label: facet, value, clear: () => (facetValues[facet] = '') })
  }
  if (cohort.value) active.push({ label: 'cohort', value: cohort.value, clear: () => (cohort.value = '') })
  if (modelQuery.value.trim()) {
    active.push({ label: 'model', value: modelQuery.value.trim(), clear: () => (modelQuery.value = '') })
  }
  if (completeOnly.value) {
    active.push({ label: 'coverage', value: 'complete panel only', clear: () => (completeOnly.value = false) })
  }
  if (showArchived.value) {
    active.push({ label: 'archived', value: 'shown', clear: () => (showArchived.value = false) })
  }
  if (showFlagged.value) {
    active.push({ label: 'flagged', value: 'admitted', clear: () => (showFlagged.value = false) })
  }
  return active
})

function clearFilters() {
  for (const facet of Object.keys(facetValues)) facetValues[facet] = ''
  cohort.value = ''
  modelQuery.value = ''
  completeOnly.value = false
  showArchived.value = false
  showFlagged.value = false
}

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

// Benchmarks every selected model has a cell on — what Compare will actually score.
const sharedTasks = computed<string[]>(() => {
  if (!comparing.value) return []
  return visibleTasks.value.filter((task) => selected.value.every((model) => modelCells.value[model]?.[task]))
})

// Carry the panel's selection into the compare route, so the comparison answers the same question
// the panel was showing rather than falling back to every benchmark at the newest cohort.
function goCompare() {
  router.push({ path: '/compare', query: { ...selection.value, models: selected.value.join(',') } })
}

const modelCells = computed(() => cellsByModel(data.value?.rows ?? []))

async function toggleArchive(model: string, archived: boolean) {
  await apiPost(`api/models/${encodeURIComponent(model)}/archive`, { archived: !archived })
  await Promise.all([refresh(), refreshMeta()])
}

// --- Benchmark column selection: a suite tree, persisted so column choices survive reloads. The
// selection is sent to the server, so it also fixes the panel any aggregate is computed over. ---
const columnsOpen = ref(false)

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
  knownEvals.value = present
  persistSelection(present)
}
// Driven by meta rather than the panel: the panel reflects the current selection, so syncing off it
// would let a narrowed selection permanently forget the columns it dropped.
watch(
  () => meta.value?.evals,
  (evals) => {
    if (evals) syncSelection(evals.filter((name) => !isSmokeEval(name)))
  },
  { immediate: true },
)

interface SuiteNode {
  suite: string
  evals: string[]
}
const suiteTree = computed<SuiteNode[]>(() => {
  const present = new Set(knownEvals.value)
  return (meta.value?.suites ?? [])
    .map((s) => ({ suite: s.suite, evals: s.evals.filter((e) => present.has(e)) }))
    .filter((s) => s.evals.length > 0)
})
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
  persistSelection(knownEvals.value)
}
function toggleEval(name: string) {
  if (selectedEvals.has(name)) selectedEvals.delete(name)
  else selectedEvals.add(name)
  persistSelection(knownEvals.value)
}
const visibleTasks = computed(() => data.value?.panel ?? [])

// --- Fleet best per benchmark (the rail caret and the column marker) ---
const best = computed(() => fleetBest(data.value?.rows ?? [], visibleTasks.value))

// --- Ordering: by a benchmark column's interval lower bound, else by panel coverage. Sorting on the
// lower bound is the point — a partly-graded run cannot buy rank with the items it kept. ---
const COVERAGE_SORT = 'coverage'
const sortKey = ref<string>(COVERAGE_SORT)
function sortBy(key: string) {
  sortKey.value = sortKey.value === key ? COVERAGE_SORT : key
}
const rows = computed<PanelRow[]>(() => {
  const all = [...(data.value?.rows ?? [])]
  return all.sort((a, b) => {
    if (a.archived !== b.archived) return Number(a.archived) - Number(b.archived)
    if (sortKey.value === COVERAGE_SORT) {
      if (a.covered !== b.covered) return b.covered - a.covered
      return a.model.localeCompare(b.model)
    }
    const av = a.cells[sortKey.value]?.low ?? null
    const bv = b.cells[sortKey.value]?.low ?? null
    if (av === null && bv === null) return a.model.localeCompare(b.model)
    if (av === null) return 1
    if (bv === null) return -1
    return bv - av
  })
})

// Δ best is per benchmark, where a difference between two measurements of the same thing is defined.
// There is no cross-benchmark Δ.
function deltaBest(row: PanelRow, task: string): number | null {
  const cell = row.cells[task]
  const leader = best.value[task]
  if (!cell || !leader || leader.model === row.model) return null
  return cell.value - leader.value
}

const readout = computed(() => {
  const panelRows = data.value?.rows ?? []
  const tasks = visibleTasks.value
  const covered = panelRows.reduce((n, row) => n + tasks.filter((t) => row.cells[t]).length, 0)
  const cells = panelRows.length * tasks.length
  return {
    models: panelRows.length,
    benchmarks: tasks.length,
    coverage: cells ? Math.round((covered / cells) * 100) : 0,
  }
})

// Harness versions the shown aggregates span. More than one means the same benchmark name was
// defined by more than one harness across the models being averaged, which is worth reading before
// the number is.
const aggregateRuntimes = computed(() => {
  const runtimes = new Set<string>()
  for (const row of data.value?.rows ?? []) for (const runtime of row.aggregate?.runtimes ?? []) runtimes.add(runtime)
  return [...runtimes].sort()
})

function heatStyle(cell: PanelCell): Record<string, string> {
  return { backgroundColor: scoreTint(cell.value) }
}
function isColumnBest(model: string, task: string): boolean {
  return best.value[task]?.model === model
}
function cellFor(row: PanelRow, task: string): PanelCell | undefined {
  return row.cells[task]
}
function gapFor(row: PanelRow, task: string): MissingCell | undefined {
  return row.missing[task]
}
// A cell names the run behind it, the cohort it came from, and the harness that defined the
// benchmark. Cells in one column can come from different cohorts -- that is the point of merging the
// newest valid result per benchmark -- so the row heading cannot carry this and the cell must.
function cellTitle(cell: PanelCell): string {
  const scope =
    cell.interval_kind === INTERVAL_KIND.IDENTIFIED
      ? formatCoverage(cell.coverage)
      : 'attempted count not reported, so completeness is unknown'
  return [
    `${cell.metric} · ${cell.n_scored} items graded`,
    `95% ${formatInterval(cell.low, cell.high)} · ${scope}`,
    ...cell.flags.filter((flag) => flag in FLAG_NOTES).map((flag) => FLAG_NOTES[flag]),
    `run ${cell.run_id}`,
    `cohort ${cell.version ?? 'unversioned'} · ${cell.eval_runtime}`,
    'click for history',
  ].join('\n')
}
// A flagged cell is still a real measurement, so it is marked and explained rather than withheld.
function isSuspect(cell: PanelCell): boolean {
  return cell.flags.includes(RESULT_FLAG.NO_ANSWERS)
}
function gapLabel(gap: MissingCell): string {
  if (gap.reason.startsWith('coverage')) return 'under-covered'
  if (gap.reason.startsWith('flagged')) return 'flagged'
  return 'no result'
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
      <h2 class="text-lg font-semibold">Panel</h2>
      <p class="text-xs text-text-muted mt-0.5">
        One row per model, one column per benchmark, each cell the newest valid result. A score is the rate over the
        items a run graded; its 95% interval covers sampling error and widens by whatever share of the attempted items
        the run never graded. Ordering everywhere uses the interval's lower bound.
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
        <div class="font-mono text-[10px] uppercase tracking-widest text-text-muted">Panel coverage</div>
        <div class="font-mono text-2xl font-semibold tabular-nums">
          {{ readout.coverage }}<span class="text-sm text-text-muted">%</span>
        </div>
      </div>
    </div>

    <!-- Filters -->
    <div class="flex flex-wrap items-end gap-3 mb-3">
      <label class="flex flex-col text-xs text-text-secondary gap-1">
        Model
        <input
          v-model="modelQuery"
          type="search"
          placeholder="name contains…"
          class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[11rem]"
        />
      </label>
      <label
        v-for="(values, facet) in meta?.facets ?? {}"
        :key="facet"
        class="flex flex-col text-xs text-text-secondary gap-1"
      >
        {{ facet }}
        <select
          v-model="facetValues[facet]"
          class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[8rem]"
        >
          <option value="">All</option>
          <option v-for="value in values" :key="value" :value="value">{{ value }}</option>
        </select>
      </label>
      <label class="flex flex-col text-xs text-text-secondary gap-1">
        Cohort
        <select v-model="cohort" class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[9rem]">
          <option value="">Newest per benchmark</option>
          <option v-for="version in meta?.versions ?? []" :key="version" :value="version">{{ version }}</option>
        </select>
      </label>
      <label class="flex flex-col text-xs text-text-secondary gap-1">
        Aggregate
        <select
          v-model="aggregatePolicy"
          class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[10rem]"
        >
          <option value="">off</option>
          <option value="require_complete">complete panels only</option>
          <option value="bound">bound the gaps</option>
        </select>
      </label>
      <button
        class="text-sm px-3 py-1.5 rounded border border-surface-border hover:bg-surface-raised"
        @click="columnsOpen = !columnsOpen"
      >
        Columns ({{ visibleTasks.length }}/{{ knownEvals.length }})
      </button>
    </div>

    <div class="flex flex-wrap items-center gap-4 mb-4">
      <label class="flex items-center gap-2 text-sm text-text-secondary">
        <input v-model="completeOnly" type="checkbox" class="accent-accent" />
        Only models with every selected benchmark
      </label>
      <label class="flex items-center gap-2 text-sm text-text-secondary">
        <input v-model="showArchived" type="checkbox" class="accent-accent" />
        Show archived
      </label>
      <label
        class="flex items-center gap-2 text-sm text-text-secondary"
        title="Readmit results the engine flags as suspect — a benchmark no graded item yielded an extractable answer for scores zero on the strength of nothing"
      >
        <input v-model="showFlagged" type="checkbox" class="accent-accent" />
        Show flagged results
      </label>
    </div>

    <div v-if="activeFilters.length" class="flex flex-wrap items-center gap-2 mb-4 text-xs">
      <span class="font-mono text-[10px] uppercase tracking-widest text-text-muted">Filtered by</span>
      <button
        v-for="filter in activeFilters"
        :key="filter.label"
        class="inline-flex items-center gap-1.5 rounded-full border border-surface-border px-2.5 py-1 font-mono hover:bg-surface-raised"
        :title="`Remove the ${filter.label} filter`"
        @click="filter.clear()"
      >
        {{ filter.label }}: {{ filter.value }} <span class="text-text-muted">×</span>
      </button>
      <button class="px-2 py-1 rounded border border-surface-border hover:bg-surface-raised" @click="clearFilters">
        Clear all
      </button>
    </div>

    <div v-if="columnsOpen && suiteTree.length" class="rounded-lg border border-surface-border bg-surface p-4 mb-4">
      <div class="flex flex-wrap gap-x-8 gap-y-4">
        <div v-for="node in suiteTree" :key="node.suite" class="min-w-[10rem]">
          <label
            class="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1.5"
          >
            <input
              type="checkbox"
              class="accent-accent"
              :checked="suiteState(node) === 'all'"
              :indeterminate.prop="suiteState(node) === 'some'"
              @change="toggleSuite(node)"
            />
            {{ node.suite }}
          </label>
          <label v-for="e in node.evals" :key="e" class="flex items-center gap-2 text-sm text-text-secondary pl-1 py-0.5">
            <input type="checkbox" class="accent-accent" :checked="selectedEvals.has(e)" @change="toggleEval(e)" />
            <span class="font-mono text-[13px]">{{ e }}</span>
          </label>
        </div>
      </div>
    </div>

    <div
      v-if="error"
      class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2 mb-4"
    >
      {{ error }}
    </div>

    <div v-if="loading && !data" class="text-sm text-text-muted py-12 text-center">Loading…</div>

    <EmptyState v-else-if="data && data.rows.length === 0" icon="🏁" message="No models match these filters." />

    <div v-else-if="data" class="space-y-6">
      <!-- Models: the eval rail as each row's measurement profile, plus the opt-in aggregate -->
      <div>
        <div class="flex items-baseline justify-between mb-2">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Models</h3>
          <span class="text-xs text-text-muted">
            Tick 2–{{ MAX_COMPARE }} to compare · one gauge per benchmark, height = score, whisker = 95% interval,
            <span class="font-mono" style="color: var(--c-best)">▬</span> = fleet best
          </span>
        </div>
        <div class="overflow-x-auto rounded-lg border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead>
              <tr
                class="border-b border-surface-border bg-surface-raised text-xs font-semibold uppercase tracking-wider text-text-secondary"
              >
                <th class="px-3 py-2 text-left w-8"></th>
                <th class="px-3 py-2 text-left">Model</th>
                <th class="px-3 py-2 text-left">Coverage</th>
                <th v-if="aggregatePolicy" class="px-3 py-2 text-right">Panel aggregate</th>
                <th class="px-3 py-2 text-left">Profile</th>
                <th class="px-3 py-2 text-right"></th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="row in rows"
                :key="row.model"
                class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors cursor-pointer"
                :class="{ 'opacity-50': row.archived }"
                @click="goToModel(row.model)"
              >
                <td class="px-3 py-2" @click.stop>
                  <input
                    type="checkbox"
                    class="align-middle accent-accent"
                    :checked="selected.includes(row.model)"
                    :disabled="!canSelect(row.model)"
                    @change="toggleModel(row.model)"
                  />
                </td>
                <td class="px-3 py-2 whitespace-nowrap">
                  <span class="font-mono font-semibold text-[13px] text-accent">{{ row.model }}</span>
                </td>
                <td class="px-3 py-2">
                  <div class="flex items-center gap-2">
                    <span class="w-[52px] h-1.5 rounded-full bg-surface-sunken overflow-hidden">
                      <span
                        class="block h-full rounded-full bg-accent"
                        :style="{ width: `${(row.covered / (visibleTasks.length || 1)) * 100}%` }"
                      />
                    </span>
                    <span
                      class="font-mono text-[11px] tabular-nums"
                      :class="row.covered <= 1 ? 'text-status-warning' : 'text-text-secondary'"
                      >{{ row.covered }}/{{ visibleTasks.length }}<span v-if="row.covered <= 1"> ⚠</span></span
                    >
                  </div>
                </td>
                <td v-if="aggregatePolicy" class="px-3 py-2 text-right whitespace-nowrap">
                  <template v-if="row.aggregate">
                    <span class="font-mono font-semibold tabular-nums">{{ formatScore(row.aggregate.value) }}</span>
                    <span class="block font-mono text-[10px] text-text-muted tabular-nums leading-none">
                      {{ formatInterval(row.aggregate.low, row.aggregate.high) }}
                    </span>
                  </template>
                  <span
                    v-else
                    class="text-text-muted text-xs"
                    title="This model is missing at least one benchmark on the panel, and the chosen policy does not aggregate over gaps."
                    >incomplete</span
                  >
                </td>
                <td class="px-3 py-2" @click.stop>
                  <EvalRail
                    :tasks="visibleTasks"
                    :cells="row.cells"
                    :missing="row.missing"
                    :best="best"
                    :model="row.model"
                    size="sm"
                    @pick="(task) => openHistory(row.model, task)"
                  />
                </td>
                <td class="px-3 py-2 text-right" @click.stop>
                  <button
                    class="text-[11px] text-text-muted hover:text-accent whitespace-nowrap"
                    :title="row.archived ? 'Unarchive model' : 'Archive model'"
                    @click="toggleArchive(row.model, row.archived)"
                  >
                    {{ row.archived ? 'unarchive' : 'archive' }}
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p v-if="aggregatePolicy" class="text-xs text-text-muted mt-2 leading-relaxed">
          Aggregate: unweighted mean over {{ visibleTasks.length }} benchmarks ({{ visibleTasks.join(' · ') }}), each at
          its own primary metric,
          {{
            aggregatePolicy === 'require_complete'
              ? 'omitted for any model missing one of them'
              : 'with a missing benchmark bounded to the full [0, 1] range it could have taken'
          }}<template v-if="aggregateRuntimes.length">, over
          <span class="font-mono">{{ aggregateRuntimes.join(', ') }}</span></template>. The interval covers item
          sampling and ungraded items. It does not cover run-to-run variation, and it is marginal per model, so it
          makes no simultaneous claim across the table.
        </p>
        <p v-else class="text-xs text-text-muted mt-2 leading-relaxed">
          No cross-benchmark mean is shown by default: it has no interpretation without a fixed benchmark panel, the
          metric each benchmark contributes, and a rule for the benchmarks a model never ran. Pick an Aggregate policy
          to choose that rule and see all three alongside the number.
        </p>
      </div>

      <!-- Per-benchmark panel -->
      <div>
        <div class="flex items-baseline justify-between mb-2">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Per-benchmark
            <span class="font-normal normal-case text-text-muted">
              ({{ rows.length }} models × {{ visibleTasks.length }} benchmarks · click a header to sort by its lower
              bound · a cell for history)
            </span>
          </h3>
        </div>
        <div class="overflow-x-auto rounded-lg border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead>
              <tr class="border-b border-surface-border bg-surface-raised">
                <th
                  class="sticky left-0 z-10 bg-surface-raised px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider cursor-pointer"
                  :class="sortKey === COVERAGE_SORT ? 'text-accent' : 'text-text-secondary'"
                  title="Sort by panel coverage"
                  @click="sortBy(COVERAGE_SORT)"
                >
                  Model
                </th>
                <th
                  v-for="task in visibleTasks"
                  :key="task"
                  class="px-3 py-2 text-center text-xs font-semibold uppercase tracking-wider whitespace-nowrap cursor-pointer"
                  :class="sortKey === task ? 'text-accent' : 'text-text-secondary'"
                  @click="sortBy(task)"
                >
                  {{ task }}
                  <span
                    v-if="best[task]"
                    class="block font-normal normal-case font-mono text-[10px]"
                    style="color: var(--c-best)"
                    >▲ {{ formatScore(best[task].value) }}</span
                  >
                </th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="row in rows"
                :key="row.model"
                class="border-b border-surface-border-subtle"
                :class="{ 'opacity-50': row.archived }"
              >
                <td class="sticky left-0 z-10 bg-surface px-3 py-2 whitespace-nowrap">
                  <button class="font-mono text-[13px] text-accent hover:underline" @click="goToModel(row.model)">
                    {{ row.model }}
                  </button>
                </td>
                <td v-for="task in visibleTasks" :key="task" class="p-1 text-center align-middle">
                  <button
                    v-if="cellFor(row, task)"
                    class="w-full rounded px-2 py-1.5 leading-tight cursor-pointer hover:ring-1 hover:ring-accent-border"
                    :class="isColumnBest(row.model, task) ? 'ring-2 ring-inset' : ''"
                    :style="{
                      ...heatStyle(cellFor(row, task)!),
                      ...(isColumnBest(row.model, task) ? { '--tw-ring-color': 'var(--c-best)' } : {}),
                    }"
                    :title="cellTitle(cellFor(row, task)!)"
                    @click="openHistory(row.model, task)"
                  >
                    <span class="tabular-nums font-mono font-medium">
                      {{ formatScore(cellFor(row, task)!.value)
                      }}<span v-if="isSuspect(cellFor(row, task)!)" class="text-status-warning">*</span>
                    </span>
                    <span class="block text-[10px] text-text-muted tabular-nums font-mono leading-none">
                      {{ formatInterval(cellFor(row, task)!.low, cellFor(row, task)!.high) }}
                    </span>
                    <span
                      v-if="isPartialCoverage(cellFor(row, task)!)"
                      class="block text-[9px] font-mono leading-none text-status-warning"
                      >{{ formatCoverage(cellFor(row, task)!.coverage) }}</span
                    >
                    <span
                      v-else-if="deltaBest(row, task) !== null"
                      class="block text-[9px] font-mono leading-none text-text-muted"
                      >{{ formatDelta(deltaBest(row, task)) }}</span
                    >
                  </button>
                  <button
                    v-else-if="gapFor(row, task)"
                    class="w-full rounded px-2 py-1.5 text-[11px] font-mono leading-tight cursor-pointer text-status-warning bg-status-warning-bg"
                    :title="`${gapFor(row, task)!.reason} — open run ${gapFor(row, task)!.run_id}`"
                    @click="goToRun(gapFor(row, task)!.run_id)"
                  >
                    {{ gapLabel(gapFor(row, task)!) }}
                  </button>
                  <span v-else class="text-text-muted" title="Never run on this benchmark">—</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p class="text-xs text-text-muted mt-2 leading-relaxed">
          An empty cell says which: <span class="text-status-warning">under-covered</span> or
          <span class="text-status-warning">no result</span> links the run that failed the panel's admission rule, and
          — means the model never ran that benchmark. A result the engine flags as suspect is held out of the
          panel as <span class="text-status-warning">flagged</span> rather than standing as a model's newest score;
          "Show flagged results" admits it, marked with a <span class="text-status-warning">*</span>.
        </p>
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
        >{{ m }}</span
      >
      <span class="text-xs text-text-muted">
        shared benchmarks:
        <span class="font-mono text-text-secondary">{{ sharedTasks.length ? sharedTasks.join(' · ') : 'none' }}</span>
      </span>
      <div class="flex-1"></div>
      <button
        class="px-4 py-2 rounded-lg bg-accent text-surface text-sm font-medium hover:bg-accent-hover"
        @click="goCompare"
      >
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
