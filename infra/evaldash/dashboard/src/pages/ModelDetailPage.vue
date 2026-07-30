<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatScore, formatStderr, formatTimestamp, objectStoreUrl } from '@/utils/formatting'
import type { Matrix, MatrixCell, ModelDetail, ModelRun } from '@/types/api'
import type { BestCell } from '@/components/charts/EvalRail.vue'
import EvalRail from '@/components/charts/EvalRail.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import StatusChip from '@/components/shared/StatusChip.vue'

const props = defineProps<{ model: string }>()
const router = useRouter()

const { data, loading, error, refresh } = useApi<ModelDetail>(() => `api/models/${encodeURIComponent(props.model)}`)
const { data: matrix, refresh: refreshMatrix } = useApi<Matrix>(() => 'api/matrix?include_archived=1')

onMounted(() => {
  refresh()
  refreshMatrix()
})
watch(() => props.model, () => {
  selectedVersion.value = undefined
  refresh()
})
onViewRefresh(() => {
  refresh()
  refreshMatrix()
})

// Cohort scope: a specific version, or 'all' for the union across every run. Defaults to the
// current (newest) version once the record loads.
const ALL = '__all__'
const selectedVersion = ref<string | undefined>(undefined)
const scope = computed<string>(() => selectedVersion.value ?? data.value?.current_version ?? ALL)

watch(
  () => data.value?.current_version,
  (v) => {
    if (selectedVersion.value === undefined) selectedVersion.value = v ?? ALL
  },
)

// Benchmarks this model has ever run, in the matrix's column order where known.
const tasks = computed<string[]>(() => {
  const present = new Set((data.value?.runs ?? []).map((r) => r.eval_name))
  const ordered = (matrix.value?.tasks ?? []).filter((t) => present.has(t))
  const extra = [...present].filter((t) => !ordered.includes(t)).sort()
  return [...ordered, ...extra]
})

// Fleet best per benchmark (for the rail caret), from the matrix across all models.
const best = computed<Record<string, BestCell>>(() => {
  const out: Record<string, BestCell> = {}
  for (const task of tasks.value) {
    let bv = -Infinity
    let bm = ''
    for (const row of matrix.value?.rows ?? []) {
      const c = row.cells[task]
      if (c && c.value !== null && c.value > bv) {
        bv = c.value
        bm = row.model
      }
    }
    if (bm) out[task] = { value: bv, model: bm }
  }
  return out
})

// The runs in scope, and the per-benchmark cell derived from them (latest succeeded per benchmark,
// else the latest run's failure) — the same union rule the server applies to the current cohort.
const scopedRuns = computed<ModelRun[]>(() =>
  (data.value?.runs ?? []).filter((r) => scope.value === ALL || r.version === scope.value),
)

const displayCells = computed<Record<string, MatrixCell>>(() => {
  const succeeded: Record<string, ModelRun> = {}
  const latest: Record<string, ModelRun> = {}
  for (const run of scopedRuns.value) {
    const key = run.eval_name
    if (!latest[key] || (run.created_at ?? '') > (latest[key].created_at ?? '')) latest[key] = run
    if (run.value !== null && (!succeeded[key] || (run.created_at ?? '') > (succeeded[key].created_at ?? ''))) {
      succeeded[key] = run
    }
  }
  const out: Record<string, MatrixCell> = {}
  for (const key of Object.keys(latest)) {
    const win = succeeded[key]
    const src = win ?? latest[key]
    out[key] = {
      status: win ? 'succeeded' : latest[key].status,
      value: win ? win.value : null,
      stderr: win ? win.stderr : null,
      metric: win ? win.metric : null,
      run_id: src.run_id,
      created_at: src.created_at ?? '',
    }
  }
  return out
})

const runIdByTask = computed<Record<string, string>>(() => {
  const out: Record<string, string> = {}
  for (const [task, cell] of Object.entries(displayCells.value)) out[task] = cell.run_id
  return out
})

function onPick(task: string) {
  const runId = runIdByTask.value[task]
  if (runId) router.push(`/runs/${runId}/samples`)
}

const cohortLabel = computed(() => (scope.value === ALL ? 'all runs' : scope.value ?? 'unversioned'))
const location = computed(() => data.value?.location ?? null)
</script>

<template>
  <section>
    <RouterLink to="/" class="text-sm text-accent hover:underline">← Leaderboard</RouterLink>

    <div v-if="error" class="mt-4 rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2">
      {{ error }}
    </div>
    <div v-else-if="loading && !data" class="text-sm text-text-muted py-12 text-center">Loading…</div>

    <div v-else-if="data" class="mt-4 space-y-6">
      <!-- header -->
      <div class="flex flex-wrap items-start gap-4">
        <div>
          <h1 class="font-mono text-2xl font-semibold tracking-tight">{{ data.model }}</h1>
          <div class="mt-1.5 text-sm text-text-secondary font-mono flex flex-wrap items-center gap-x-2 gap-y-1">
            <a v-if="objectStoreUrl(location)" :href="objectStoreUrl(location)!" target="_blank" class="text-accent hover:underline">{{ location }} ↗</a>
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
            :class="scope === (c.version ?? ALL) ? 'border-accent text-text bg-accent-subtle' : 'border-surface-border text-text-secondary hover:bg-surface-raised'"
            @click="selectedVersion = c.version ?? ALL"
          >{{ c.version ?? 'unversioned' }}</button>
          <button
            class="font-mono text-xs px-2.5 py-1 rounded border"
            :class="scope === ALL ? 'border-accent text-text bg-accent-subtle' : 'border-surface-border text-text-secondary hover:bg-surface-raised'"
            @click="selectedVersion = ALL"
          >all runs</button>
        </div>
      </div>

      <!-- measurement profile: the large eval rail -->
      <div>
        <div class="flex items-baseline justify-between mb-2">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Measurement profile</h3>
          <span class="text-xs text-text-muted">
            cohort {{ cohortLabel }} · latest succeeded run per benchmark · whisker = ±stderr · click a gauge for samples
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
                :cells="displayCells"
                :best="best"
                :model="data.model"
                size="lg"
                :history="data.history"
                @pick="onPick"
              />
            </div>
          </div>
          <div class="mt-4 flex flex-wrap gap-x-5 gap-y-1 font-mono text-[11px] text-text-secondary">
            <span class="inline-flex items-center gap-2"><span class="inline-block w-4 h-[3px] rounded-full" style="background: var(--c-best)"></span> fleet best (this benchmark)</span>
            <span>sparkline under each gauge = this model's score on that benchmark across its runs</span>
          </div>
        </div>
        <p class="text-xs text-text-muted mt-2 leading-relaxed">
          How runs merge: a model runs each benchmark many times across versions. The profile shows the latest
          succeeded run within the selected cohort — the default union. Switch cohorts to pin an older model state;
          the sparkline under each gauge keeps every run visible so a regression is one glance away.
        </p>
      </div>

      <!-- runs in scope -->
      <div>
        <div class="flex items-baseline justify-between mb-2">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Runs</h3>
          <span class="text-xs text-text-muted">{{ scopedRuns.length }} run{{ scopedRuns.length === 1 ? '' : 's' }} · cohort {{ cohortLabel }}</span>
        </div>
        <div class="rounded-lg border border-surface-border overflow-hidden bg-surface">
          <button
            v-for="run in scopedRuns"
            :key="run.run_id"
            class="w-full flex items-center gap-4 px-4 py-2.5 border-b border-surface-border-subtle last:border-b-0 hover:bg-surface-raised text-left"
            @click="router.push(`/runs/${run.run_id}`)"
          >
            <span class="font-mono text-[13px] font-semibold w-40 truncate">{{ run.eval_name }}</span>
            <StatusChip :status="run.status" />
            <span class="font-mono text-[11px] text-text-muted w-40">{{ formatTimestamp(run.created_at) }}</span>
            <span v-if="run.version" class="font-mono text-[11px] text-text-muted">{{ run.version }}</span>
            <span class="ml-auto font-mono text-[13px] font-semibold tabular-nums">
              <template v-if="run.value !== null">{{ formatScore(run.value) }}<span class="text-text-muted text-[11px] font-normal"> {{ formatStderr(run.value, run.stderr) }}</span></template>
              <span v-else class="text-text-muted">—</span>
            </span>
          </button>
        </div>
      </div>
    </div>
  </section>
</template>
