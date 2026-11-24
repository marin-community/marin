<script setup lang="ts">
/**
 * The panel: models by benchmark, every cell a score with the 95% interval behind it.
 *
 * There is no headline cross-benchmark score mean. A score mean over benchmarks has no interpretation without a
 * declared panel, a per-benchmark metric, and a rule for the benchmarks a model never ran, so it is
 * opt-in and renders those three things with it. Per-benchmark comparison is the primary surface.
 * A benchmark column sorts on the score, and the fleet-best marker and delta-best pick the leader
 * by the same rule. The interval under each score stays visible, and Compare ranks on it.
 */
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { apiPost, useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatDelta, formatInterval, formatScore, formatTimestamp } from '@/utils/formatting'
import { scoreTint } from '@/utils/score'
import { UNVERSIONED_COHORT, cellCoverageLabel, cellsByModel, cohortWarning, fleetBest, isPartialCoverage, scoredItemsLabel, withVariant } from '@/utils/panel'
import { compareCells, meanReciprocalRanks } from '@/utils/ranking'
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
import PolicyRejections from '@/components/shared/PolicyRejections.vue'
import SearchSelect from '@/components/shared/SearchSelect.vue'
import HistoryModal from '@/components/charts/HistoryModal.vue'

const router = useRouter()
const route = useRoute()

// --- Request state. Everything the server needs to resolve a panel goes in the query, so a panel is
// a shareable URL and the filters that produced a number travel with it. ---
const showArchived = ref(false)
const showFlagged = ref(false)
const completeOnly = ref(false)
const cohort = computed({
  get: () => {
    const raw = route.query.cohort
    return (Array.isArray(raw) ? raw[0] : raw) || meta.value?.default_cohort || 'all'
  },
  set: (value: string) => {
    void router.replace({ query: { ...route.query, cohort: value || meta.value?.default_cohort } })
  },
})
const modelQuery = ref('')
const facetValues = reactive<Record<string, string>>({})
const aggregatePolicy = ref('')

const SELECTED_KEY = 'evaldash.selectedEvals'
const KNOWN_KEY = 'evaldash.knownEvals'
const selectedEvals = reactive(new Set<string>())
const knownEvals = ref<string[]>([])
const selectionReady = ref(false)

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
// Pin the resolved default so shared URLs retain their cohort after the default changes.
watch(
  () => meta.value?.default_cohort,
  (value) => {
    if (value && !route.query.cohort) cohort.value = value
  },
)
const comparabilityWarning = computed(() =>
  meta.value ? cohortWarning(cohort.value || meta.value.default_cohort, meta.value) : null,
)

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
watch(selection, () => {
  selected.value = []
})

// Benchmarks every selected model has a cell on — what Compare will actually score.
const sharedTasks = computed<string[]>(() => {
  if (!comparing.value) return []
  return visibleTasks.value.filter((task) => selected.value.every((model) => modelCells.value[model]?.[task]))
})

// Pin Compare to the exact variants shown in the panel.
function goCompare() {
  const benchmarks = columns.value.map((column) => column.task).join(',')
  const query: Record<string, string> = { ...selection.value, models: selected.value.join(',') }
  if (benchmarks) query.benchmarks = benchmarks
  router.push({ path: '/compare', query })
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
function routeBenchmarks(): string[] | null {
  const value = route.query.benchmarks
  if (value == null) return null
  const values = Array.isArray(value) ? value : [value]
  return values.flatMap((entry) => entry?.split(',') ?? []).filter(Boolean)
}
function persistSelection(present: string[], updateRoute = true) {
  localStorage.setItem(SELECTED_KEY, JSON.stringify([...selectedEvals]))
  localStorage.setItem(KNOWN_KEY, JSON.stringify(present))
  if (!updateRoute) return
  const selected = [...selectedEvals]
  const benchmarks = selected.length > 0 && selected.length !== present.length ? selected.join(',') : undefined
  void router.replace({ query: { ...route.query, cohort: cohort.value, benchmarks } })
}
function syncSelection(present: string[]) {
  const fromRoute = routeBenchmarks()
  const stored = readStored(SELECTED_KEY)
  const known = new Set(readStored(KNOWN_KEY) ?? [])
  selectedEvals.clear()
  for (const name of present) {
    if (fromRoute ? fromRoute.includes(name) : stored === null || stored.includes(name) || !known.has(name)) {
      selectedEvals.add(name)
    }
  }
  knownEvals.value = present
  selectionReady.value = true
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
watch(
  () => route.query.benchmarks,
  () => {
    if (!selectionReady.value) return
    const requested = routeBenchmarks()
    selectedEvals.clear()
    for (const name of knownEvals.value) {
      if (requested === null || requested.includes(name)) selectedEvals.add(name)
    }
    persistSelection(knownEvals.value, false)
  },
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

const best = computed(() => fleetBest(data.value?.rows ?? [], visibleTasks.value))

// --- Ordering. Benchmark columns use compareCells: the score, then the lower bound. ---
const MODEL_SORT = 'model'
const MRR_SORT = 'mrr'
const COVERAGE_SORT = 'coverage'
const UPDATED_SORT = 'last_updated'
type SortDirection = 'asc' | 'desc'

const sortKey = ref<string>(MRR_SORT)
const sortDirection = ref<SortDirection>('desc')
const meanRanks = computed(() => meanReciprocalRanks(data.value?.rows ?? [], visibleTasks.value))

function defaultDirection(key: string): SortDirection {
  return key === MODEL_SORT ? 'asc' : 'desc'
}
function sortBy(key: string) {
  if (sortKey.value === key) {
    sortDirection.value = sortDirection.value === 'asc' ? 'desc' : 'asc'
    return
  }
  sortKey.value = key
  sortDirection.value = defaultDirection(key)
}
function sortGlyph(key: string): string {
  if (sortKey.value !== key) return ''
  return sortDirection.value === 'asc' ? '▲' : '▼'
}
function headerClass(key: string): string {
  return sortKey.value === key ? 'text-accent' : 'text-text-secondary'
}

// A benchmark column orders its cells by compareCells, the rule the fleet-best marker uses. The model,
// MRR, coverage and last-updated keys order on that field alone. Missing values sort last in both directions.
function compareRows(a: PanelRow, b: PanelRow, direction: number): number {
  if (sortKey.value === MODEL_SORT) return a.model.localeCompare(b.model) * direction
  if (sortKey.value === MRR_SORT) {
    const rankA = meanRanks.value.get(a.model)
    const rankB = meanRanks.value.get(b.model)
    if (rankA === undefined || rankB === undefined) return Number(rankA === undefined) - Number(rankB === undefined)
    return (rankA - rankB) * direction
  }
  if (sortKey.value === COVERAGE_SORT) return (a.covered - b.covered) * direction
  if (sortKey.value === UPDATED_SORT) {
    if (a.last_updated === null || b.last_updated === null) {
      return Number(a.last_updated === null) - Number(b.last_updated === null)
    }
    return a.last_updated.localeCompare(b.last_updated) * direction
  }
  const cellA = a.cells[sortKey.value]
  const cellB = b.cells[sortKey.value]
  if (!cellA || !cellB) return Number(!cellA) - Number(!cellB)
  return -compareCells(cellA, cellB) * direction
}

const rows = computed<PanelRow[]>(() => {
  const all = [...(data.value?.rows ?? [])]
  const direction = sortDirection.value === 'asc' ? 1 : -1
  return all.sort((a, b) => {
    if (a.archived !== b.archived) return Number(a.archived) - Number(b.archived)
    return compareRows(a, b, direction) || a.model.localeCompare(b.model)
  })
})

interface FamilyColumn {
  family: string
  task: string
  variants: string[]
}

// Meta includes siblings omitted by a narrowed panel, so the picker can switch back.
const knownVariants = computed<Record<string, string[]>>(() => {
  const known = new Set(knownEvals.value)
  const out: Record<string, string[]> = {}
  for (const entry of meta.value?.families ?? []) {
    const present = entry.variants.filter((name) => known.has(name))
    if (present.length) out[entry.family] = present
  }
  return out
})

const columns = computed<FamilyColumn[]>(() => {
  const shown = new Set(visibleTasks.value)
  const coverage = new Map<string, number>()
  for (const row of data.value?.rows ?? []) {
    for (const name of Object.keys(row.cells)) coverage.set(name, (coverage.get(name) ?? 0) + 1)
  }
  return (data.value?.families ?? []).map((entry) => ({
    family: entry.family,
    task: entry.variants.find((name) => shown.has(name)) ?? entry.default,
    variants: knownVariants.value[entry.family] ?? entry.variants,
  })).sort((a, b) => (coverage.get(b.task) ?? 0) - (coverage.get(a.task) ?? 0) || a.family.localeCompare(b.family))
})

function pickVariant(column: FamilyColumn, variant: string) {
  const next = withVariant(selectedEvals, column.variants, variant)
  selectedEvals.clear()
  for (const name of next) selectedEvals.add(name)
  if (sortKey.value === column.task) sortKey.value = variant
  persistSelection(knownEvals.value)
}

// Δ best is per benchmark, where a difference between two measurements of the same thing is defined.
// There is no cross-benchmark Δ.
function deltaBest(row: PanelRow, task: string): number | null {
  const cell = row.cells[task]
  const leader = best.value[task]
  if (!cell || !leader || leader.model === row.model) return null
  return cell.value - leader.value
}

const MODEL_PAGE_SIZE = 20
const BENCHMARK_PAGE_SIZE = 6
const modelPage = ref(0)
const benchmarkPage = ref(0)
const pageRows = computed(() => rows.value.slice(modelPage.value * MODEL_PAGE_SIZE, (modelPage.value + 1) * MODEL_PAGE_SIZE))
const pageColumns = computed(() => columns.value.slice(benchmarkPage.value * BENCHMARK_PAGE_SIZE, (benchmarkPage.value + 1) * BENCHMARK_PAGE_SIZE))
watch([query, sortKey, sortDirection], () => { modelPage.value = 0 })
watch(selection, () => { benchmarkPage.value = 0 })
watch(() => rows.value.length, count => { modelPage.value = Math.min(modelPage.value, Math.max(0, Math.ceil(count / MODEL_PAGE_SIZE) - 1)) })
watch(() => columns.value.length, count => { benchmarkPage.value = Math.min(benchmarkPage.value, Math.max(0, Math.ceil(count / BENCHMARK_PAGE_SIZE) - 1)) })
function jumpToFamily(family: string) {
  const index = columns.value.findIndex(column => column.family === family)
  if (index >= 0) benchmarkPage.value = Math.floor(index / BENCHMARK_PAGE_SIZE)
}

// Harness versions the shown aggregates span. More than one means the same benchmark name was
// defined by more than one harness across the models being averaged, which is worth reading before
// the number is.
const aggregateRuntimes = computed(() => {
  const runtimes = new Set<string>()
  for (const row of data.value?.rows ?? []) for (const runtime of row.aggregate?.runtimes ?? []) runtimes.add(runtime)
  return [...runtimes].sort()
})

function heatStyle(cell: PanelCell): Record<string, string> {
  return { backgroundColor: `color-mix(in srgb, ${scoreTint(cell.value)} 28%, var(--c-surface))` }
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
function protocolLabel(task: string): string {
  const protocol = data.value?.protocols[task]
  return protocol ? `${protocol.metric} · ${protocol.kind}` : 'legacy metric'
}
// A cell names the run behind it, the cohort it came from, and the harness that defined the
// benchmark. Cells in one column can come from different cohorts -- that is the point of merging the
// newest valid result per benchmark -- so the row heading cannot carry this and the cell must.
function cellTitle(cell: PanelCell): string {
  const scope =
    cell.interval_kind === INTERVAL_KIND.IDENTIFIED
      ? cellCoverageLabel(cell)
      : 'attempted count not reported, so completeness is unknown'
  const shotSetting = cell.num_fewshot === null ? 'default shots' : `${cell.num_fewshot}-shot`
  return [
    `${cell.metric} · ${shotSetting} · ${scoredItemsLabel(cell)}`,
    cell.repeat_aggregation?.coverage_basis === 'source'
      ? 'source coverage retained; item count uncertain'
      : cell.n_benchmark === null
        ? 'benchmark size unreported'
        : `${cell.n_attempted ?? 'unknown'} of ${cell.n_benchmark} benchmark items attempted`,
    ...(cell.repeat_aggregation?.coverage_basis === 'item_lower_bound'
      ? [`at least ${cell.n_scored} items scored`]
      : []),
    `95% ${formatInterval(cell.low, cell.high)} · ${scope}`,
    ...cell.flags.filter((flag) => flag in FLAG_NOTES).map((flag) => FLAG_NOTES[flag]),
    `run ${cell.run_id} · ${formatTimestamp(cell.created_at)}`,
    `cohort ${cell.version ?? UNVERSIONED_COHORT}${cell.eval_runtime ? ` · ${cell.eval_runtime}` : ''}`,
    'click for history',
  ].join('\n')
}
// A flagged cell is still a real measurement, so it is marked and explained rather than withheld.
function isSuspect(cell: PanelCell): boolean {
  return cell.flags.includes(RESULT_FLAG.NO_ANSWERS)
}
function gapLabel(gap: MissingCell): string {
  if (gap.reason.includes('coverage')) return 'under-covered'
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
  router.push({
    path: `/models/${encodeURIComponent(model)}`,
    query: { cohort: cohort.value || data.value?.request.cohort_version },
  })
}
</script>

<template>
  <section class="panel">
    <div class="mb-4">
      <RouterLink :to="{ path: '/benchmarks', query: { cohort } }" class="text-sm text-accent hover:underline">← Benchmarks</RouterLink>
      <h1 class="mt-2">Fleet coverage</h1>
      <p class="mt-1 text-sm text-text-secondary">Rank models across the selected benchmarks, or open one benchmark for its scores and source runs.</p>
      <p v-if="comparabilityWarning" class="text-sm text-text-secondary mt-2">{{ comparabilityWarning }}</p>
    </div>

    <div class="panel-controls flex flex-wrap items-end gap-3 mb-3">
      <label class="flex flex-col text-xs text-text-secondary gap-1">
        Model
        <input
          v-model="modelQuery"
          type="search"
          placeholder="name contains…"
          class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[11rem]"
        />
      </label>
      <label class="flex flex-col text-xs text-text-secondary gap-1">
        Cohort
        <select v-model="cohort" class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[9rem]">
          <option value="all">Newest per benchmark (all cohorts)*</option>
          <option v-for="version in meta?.versions ?? []" :key="version" :value="version">
            {{ version }}{{ meta?.verified_cohorts.includes(version) ? '' : '*' }}
          </option>
        </select>
      </label>
      <SearchSelect label="Find benchmark" placeholder="Search benchmarks…" :options="columns.map(column => column.family)" model-value="" class="w-64" @update:model-value="jumpToFamily" />
    </div>

    <details class="mb-4">
      <summary class="text-sm cursor-pointer text-accent">Panel settings</summary>
      <div class="flex flex-wrap items-end gap-3 my-3">
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

    </details>

    <div
      v-if="error"
      class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2 mb-4"
    >
      {{ error }}
    </div>

    <PolicyRejections v-if="data && !loading && !error" :rejections="data.policy_rejections" scope="this cohort" class="mb-4" />

    <div v-if="loading" class="text-sm text-text-muted py-12 text-center">Loading…</div>

    <EmptyState v-else-if="data && data.rows.length === 0" icon="🏁" message="No models match these filters." />

    <div v-else-if="data && !error" class="space-y-6">
      <div>
        <div class="matrix-pages">
          <div>
            <button aria-label="Previous models" :disabled="modelPage === 0" @click="modelPage--">‹</button>
            <span>Models {{ modelPage * MODEL_PAGE_SIZE + 1 }}–{{ modelPage * MODEL_PAGE_SIZE + pageRows.length }} of {{ rows.length }}</span>
            <button aria-label="Next models" :disabled="(modelPage + 1) * MODEL_PAGE_SIZE >= rows.length" @click="modelPage++">›</button>
          </div>
          <div v-if="columns.length">
            <button aria-label="Previous benchmarks" :disabled="benchmarkPage === 0" @click="benchmarkPage--">‹</button>
            <span>Benchmarks {{ benchmarkPage * BENCHMARK_PAGE_SIZE + 1 }}–{{ benchmarkPage * BENCHMARK_PAGE_SIZE + pageColumns.length }} of {{ columns.length }}</span>
            <button aria-label="Next benchmarks" :disabled="(benchmarkPage + 1) * BENCHMARK_PAGE_SIZE >= columns.length" @click="benchmarkPage++">›</button>
          </div>
        </div>
        <div class="overflow-x-auto rounded-lg border border-surface-border">
          <table aria-label="Per-benchmark results" class="matrix w-full border-collapse text-sm">
            <thead>
              <tr class="border-b border-surface-border bg-surface-raised">
                <th
                  class="sticky left-0 z-10 bg-surface-raised px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider cursor-pointer"
                  :class="headerClass(MODEL_SORT)"
                  title="Sort by model name"
                >
                  <button class="column-name" @click="sortBy(MODEL_SORT)">Model {{ sortGlyph(MODEL_SORT) }}</button>
                  <button class="block text-xs text-accent" @click="sortBy(COVERAGE_SORT)">Coverage {{ sortGlyph(COVERAGE_SORT) }}</button>
                </th>
                <th
                  class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider whitespace-nowrap cursor-pointer"
                  :class="headerClass(MRR_SORT)"
                  title="Mean reciprocal rank across completed results on the selected benchmarks; missing results are omitted"
                >
                  <button class="column-name" @click="sortBy(MRR_SORT)">MRR {{ sortGlyph(MRR_SORT) }}</button>
                </th>
                <th v-if="aggregatePolicy">Aggregate</th>
                <th
                  v-for="column in pageColumns"
                  :key="column.family"
                  class="px-3 py-2 text-center text-xs font-semibold uppercase tracking-wider whitespace-nowrap cursor-pointer"
                  :class="headerClass(column.task)"
                >
                  <button class="column-name" @click.stop="sortBy(column.task)">{{ column.family }} {{ sortGlyph(column.task) }}</button>
                  <RouterLink class="block text-xs normal-case text-accent" :to="{ name: 'benchmarks', query: { benchmark: column.task, cohort } }" @click.stop>Open benchmark ↗</RouterLink>
                  <select
                    v-if="column.variants.length > 1"
                    class="block mx-auto mt-0.5 rounded border border-surface-border bg-surface px-1 py-0 font-mono text-[10px] font-normal normal-case text-text-secondary"
                    title="Which setting of this benchmark the column shows. Compare uses the selected setting."
                    :value="column.task"
                    @click.stop
                    @change="pickVariant(column, ($event.target as HTMLSelectElement).value)"
                  >
                    <option v-for="variant in column.variants" :key="variant" :value="variant">{{ variant }}</option>
                  </select>
                  <span class="block font-normal normal-case font-mono text-[10px] text-text-muted">
                    {{ protocolLabel(column.task) }}
                  </span>
                  <span
                    v-if="best[column.task]"
                    class="block font-normal normal-case font-mono text-[10px]"
                    style="color: var(--c-best)"
                    >▲ {{ formatScore(best[column.task].value) }}</span
                  >
                </th>
                <th
                  class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider whitespace-nowrap cursor-pointer"
                  :class="headerClass(UPDATED_SORT)"
                  title="Latest result included in this row"
                >
                  <button class="column-name" @click="sortBy(UPDATED_SORT)">Last updated {{ sortGlyph(UPDATED_SORT) }}</button>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="row in pageRows"
                :key="row.model"
                class="border-b border-surface-border-subtle"
                :class="{ 'opacity-50': row.archived }"
              >
                <td class="model-column sticky left-0 z-10 bg-surface px-3 py-2">
                  <input type="checkbox" :aria-label="'Compare ' + row.model" :checked="selected.includes(row.model)" :disabled="!canSelect(row.model)" @change="toggleModel(row.model)" />
                  <button class="font-mono text-[13px] text-accent hover:underline" @click="goToModel(row.model)">
                    {{ row.model }}
                  </button>
                  <small class="block text-text-secondary">{{ row.covered }}/{{ visibleTasks.length }} benchmarks ·
                    <button class="text-xs" @click="toggleArchive(row.model, row.archived)">{{ row.archived ? 'Unarchive' : 'Archive' }}</button>
                  </small>
                </td>
                <td class="px-3 py-2 text-right font-mono text-xs tabular-nums" :title="`${row.covered} completed selected benchmarks`">
                  {{ meanRanks.get(row.model)?.toFixed(3) ?? '—' }}
                </td>
                <td v-if="aggregatePolicy" class="text-center">
                  <template v-if="row.aggregate">{{ formatScore(row.aggregate.value) }}<small class="block">{{ formatInterval(row.aggregate.low, row.aggregate.high) }}</small></template>
                  <span v-else>Incomplete</span>
                </td>
                <td v-for="{ task } in pageColumns" :key="task" class="p-1 text-center align-middle">
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
                      >{{ cellCoverageLabel(cellFor(row, task)!) }}</span
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
                <td
                  class="px-3 py-2 text-right whitespace-nowrap font-mono text-[11px] tabular-nums text-text-muted"
                  title="Latest result included in this row"
                >
                  {{ formatTimestamp(row.last_updated) }}
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
        <p class="mt-2 text-xs text-text-secondary">MRR is the mean reciprocal rank over a model’s scored benchmarks. Select a score for its history or an excluded result for its source run.</p>
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
        <span class="font-mono text-text-secondary">{{ sharedTasks.length }}</span>
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

<style scoped>
.panel-controls > *, .panel label { min-width: 0; max-width: 100%; }
.panel select { max-width: 100%; }
.matrix-pages, .matrix-pages > div { display: flex; flex-wrap: wrap; align-items: center; gap: .55rem; }
.matrix-pages { justify-content: space-between; margin: .8rem 0; font-size: .85rem; }
.matrix-pages button { padding: .1rem .6rem; }
.column-name { display: block; max-width: 14rem; margin: 0 auto; overflow-wrap: anywhere; white-space: normal; font-weight: 600; }
.matrix select { max-width: 12rem; }
.model-column { min-width: 9rem; max-width: 15rem; overflow-wrap: anywhere; }
.model-column button, .column-name { background: none; border: 0; padding: .2rem; text-align: inherit; }
@media (max-width: 40rem) { .panel-controls > * { width: 100%; } .model-column { max-width: 8rem; min-width: 6rem; } }
</style>
