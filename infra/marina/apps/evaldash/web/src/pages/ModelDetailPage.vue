<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatInterval, formatScore, formatTimestamp, objectStoreUrl } from '@/utils/formatting'
import { ALL_COHORTS, UNVERSIONED_COHORT, cellCoverageLabel, fleetBest, isPartialCoverage } from '@/utils/panel'
import { isSmokeEval } from '@/constants'
import type { Meta, MissingCell, ModelDetail, ModelRun, Panel, PanelCell } from '@/types/api'
import EvalRail from '@/components/charts/EvalRail.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import StatusChip from '@/components/shared/StatusChip.vue'

const props = defineProps<{ model: string }>()
const router = useRouter()
const route = useRoute()

const { data: meta, refresh: refreshMeta } = useApi<Meta>(() => 'api/meta')
const selectedCohort = computed(() => {
  const selected = route.query.cohort
  return (typeof selected === 'string' && selected) || meta.value?.default_cohort || ALL_COHORTS
})
const modelPath = computed(() =>
  `api/models/${encodeURIComponent(props.model)}?cohort=${encodeURIComponent(selectedCohort.value)}`,
)
const fleetPath = computed(() => `api/panel?include_archived=1&cohort=${encodeURIComponent(selectedCohort.value)}`)
const { data, loading, error, refresh } = useApi<ModelDetail>(() => modelPath.value)
const { data: panel, refresh: refreshPanel } = useApi<Panel>(() => fleetPath.value)

onMounted(refreshMeta)
watch(modelPath, () => {
  data.value = null
  void refresh()
}, { immediate: true })
watch(fleetPath, () => {
  panel.value = null
  void refreshPanel()
}, { immediate: true })
onViewRefresh(() => {
  refresh()
  refreshPanel()
  refreshMeta()
})


watch(
  [() => data.value?.model, () => meta.value?.default_cohort, () => route.query.cohort, () => props.model],
  () => {
    if (data.value?.model !== props.model || !meta.value) return
    if (typeof route.query.cohort === 'string' && route.query.cohort) return
    router.replace({ query: { ...route.query, cohort: meta.value.default_cohort } })
  },
)

function selectCohort(cohort: string) {
  void router.push({ query: { ...route.query, cohort } })
}

// Benchmarks this model has runs for, in the panel's column order where known (no smoke).
const tasks = computed<string[]>(() => {
  const present = new Set((data.value?.runs ?? []).filter((r) => !isSmokeEval(r.eval_name)).map((r) => r.eval_name))
  const ordered = (panel.value?.panel ?? []).filter((t) => present.has(t))
  const extra = [...present].filter((t) => !ordered.includes(t)).sort()
  return [...ordered, ...extra]
})

// Use the selected cohort's panel for the fleet reference.
const best = computed(() => fleetBest(panel.value?.rows ?? [], tasks.value))

// Every run in the selected cohort (smoke included), for the Runs list.
const scopedRuns = computed<ModelRun[]>(() =>
  (data.value?.runs ?? []).filter((r) =>
    selectedCohort.value === ALL_COHORTS || (r.version ?? UNVERSIONED_COHORT) === selectedCohort.value,
  ),
)

// Per-benchmark result within the cohort: the newest run that produced a score, and — where none did
// — the newest run that did not, so the rail shows a reason instead of a hole.
const scopedResults = computed<{ cells: Record<string, PanelCell>; missing: Record<string, MissingCell> }>(() => {
  const scored: Record<string, ModelRun> = {}
  const latest: Record<string, ModelRun> = {}
  for (const run of scopedRuns.value) {
    if (isSmokeEval(run.eval_name)) continue
    const key = run.eval_name
    if (!latest[key] || (run.created_at ?? '') > (latest[key].created_at ?? '')) latest[key] = run
    if (run.headline && (!scored[key] || (run.created_at ?? '') > (scored[key].created_at ?? ''))) scored[key] = run
  }
  const cells: Record<string, PanelCell> = {}
  const missing: Record<string, MissingCell> = {}
  for (const [key, run] of Object.entries(latest)) {
    const win = scored[key]
    if (win?.headline) {
      cells[key] = win.headline
      continue
    }
    missing[key] = {
      reason: run.gap_reason ?? `status ${run.status}`,
      run_id: run.run_id,
      status: run.status,
      created_at: run.created_at ?? '',
    }
  }
  return { cells, missing }
})

const runIdByTask = computed<Record<string, string>>(() => {
  const out: Record<string, string> = {}
  for (const [task, cell] of Object.entries(scopedResults.value.cells)) out[task] = cell.run_id
  for (const [task, gap] of Object.entries(scopedResults.value.missing)) out[task] = gap.run_id
  return out
})

function onPick(task: string) {
  const runId = runIdByTask.value[task]
  if (runId) router.push(`/runs/${runId}/samples`)
}

const cohortLabel = computed(() => selectedCohort.value === ALL_COHORTS ? 'all runs' : selectedCohort.value)
const location = computed(() => data.value?.location ?? null)

function coverageNote(cell: PanelCell): string {
  return isPartialCoverage(cell) ? cellCoverageLabel(cell) : ''
}
</script>

<template>
  <section>
    <RouterLink :to="{ path: '/benchmarks', query: { cohort: route.query.cohort } }" class="text-sm text-accent hover:underline">← Benchmarks</RouterLink>

    <div
      v-if="error"
      class="mt-4 rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2"
    >
      {{ error }}
    </div>
    <div v-else-if="loading && !data" class="text-sm text-text-muted py-12 text-center">Loading…</div>

    <div v-else-if="data" class="mt-4 space-y-6">
      <!-- header -->
      <div class="flex flex-wrap items-start gap-4">
        <div>
          <h1 class="break-words">{{ data.model }}</h1>
          <div class="mt-1.5 text-sm text-text-secondary flex flex-wrap items-center gap-x-2 gap-y-1">
            <a
              v-if="objectStoreUrl(location)"
              :href="objectStoreUrl(location)!"
              target="_blank"
              class="text-accent hover:underline"
              >{{ location }} ↗</a
            >
            <span v-else-if="location">{{ location }}</span>
            <span v-if="data.backend" class="text-text-muted">· {{ data.backend }}</span>
            <span v-if="data.user" class="text-text-muted">· {{ data.user }}</span>
          </div>
        </div>
        <div class="ml-auto flex items-center gap-1.5">
          <span class="font-mono text-[10px] uppercase tracking-widest text-text-muted self-center mr-1">Cohort</span>
          <button
            v-for="c in data.cohorts"
            :key="c.version ?? 'none'"
            class="font-mono text-xs px-2.5 py-1 rounded border"
            :class="
              selectedCohort === (c.version ?? UNVERSIONED_COHORT)
                ? 'border-accent text-text bg-accent-subtle'
                : 'border-surface-border text-text-secondary hover:bg-surface-raised'
            "
            @click="selectCohort(c.version ?? UNVERSIONED_COHORT)"
          >
            {{ c.version ?? UNVERSIONED_COHORT }}
          </button>
          <button
            class="font-mono text-xs px-2.5 py-1 rounded border"
            :class="
              selectedCohort === ALL_COHORTS
                ? 'border-accent text-text bg-accent-subtle'
                : 'border-surface-border text-text-secondary hover:bg-surface-raised'
            "
            @click="selectCohort(ALL_COHORTS)"
          >
            all runs
          </button>
        </div>
      </div>

      <!-- measurement profile: the large eval rail -->
      <div>
        <div class="flex items-baseline justify-between mb-2">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Measurement profile</h3>
          <span class="text-xs text-text-muted">
            cohort {{ cohortLabel }} · newest scored run per benchmark · whisker = 95% interval · click a gauge for
            samples
          </span>
        </div>
        <div class="rounded-lg border border-surface-border bg-surface p-6">
          <EmptyState v-if="!tasks.length" icon="📉" message="No benchmarks scored for this cohort yet." />
          <div v-else class="flex gap-3">
            <div class="flex flex-col justify-between h-[150px] font-mono text-[10px] text-text-muted text-right pr-1">
              <span>1.0</span><span>0.8</span><span>0.6</span><span>0.4</span><span>0.2</span><span>0</span>
            </div>
            <div class="overflow-x-auto pb-1">
              <EvalRail
                :tasks="tasks"
                :cells="scopedResults.cells"
                :missing="scopedResults.missing"
                :best="best"
                :model="data.model"
                size="lg"
                :history="data.history"
                @pick="onPick"
              />
            </div>
          </div>
          <div class="mt-4 flex flex-wrap gap-x-5 gap-y-1 font-mono text-[11px] text-text-secondary">
            <span class="inline-flex items-center gap-2"
              ><span class="inline-block w-4 h-[3px] rounded-full" style="background: var(--c-best)"></span> fleet best
              (this benchmark)</span
            >
            <span>Sparklines show admitted scores across cohorts.</span>
          </div>
        </div>
        <p class="text-xs text-text-muted mt-2 leading-relaxed">
          How runs merge: a model runs each benchmark many times across versions. The profile shows the newest run that
          produced a score within the selected cohort. The selected cohort controls which runs the sparklines admit.
          Each whisker is the run's 95% interval,
          so a benchmark whose run lost items reads as less determined rather than simply lower.
        </p>
      </div>

      <!-- runs in scope -->
      <div>
        <div class="flex items-baseline justify-between mb-2">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Runs</h3>
          <span class="text-xs text-text-muted"
            >{{ scopedRuns.length }} run{{ scopedRuns.length === 1 ? '' : 's' }} · cohort {{ cohortLabel }}</span
          >
        </div>
        <div class="rounded-lg border border-surface-border overflow-hidden bg-surface">
          <button
            v-for="run in scopedRuns"
            :key="run.run_id"
            class="w-full flex items-center gap-4 px-4 py-2.5 border-b border-surface-border-subtle last:border-b-0 hover:bg-surface-raised text-left"
            @click="router.push(`/runs/${run.run_id}`)"
          >
            <span class="text-sm font-semibold w-40 truncate">{{ run.eval_name }}</span>
            <StatusChip :status="run.status" />
            <span class="font-mono text-[11px] text-text-muted w-40">{{ formatTimestamp(run.created_at) }}</span>
            <span v-if="run.version" class="font-mono text-[11px] text-text-muted">{{ run.version }}</span>
            <span
              v-if="run.headline && coverageNote(run.headline)"
              class="font-mono text-[11px] text-status-warning"
              >{{ coverageNote(run.headline) }}</span
            >
            <span class="ml-auto font-mono text-[13px] font-semibold tabular-nums text-right">
              <template v-if="run.headline">
                {{ formatScore(run.headline) }}
                <span class="block text-[10px] text-text-muted font-normal leading-none">
                  {{ formatInterval(run.headline) }}
                </span>
              </template>
              <span v-else class="text-text-muted">—</span>
            </span>
          </button>
        </div>
      </div>
    </div>
  </section>
</template>
