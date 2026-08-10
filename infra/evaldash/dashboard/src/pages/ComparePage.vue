<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatScore, formatStderr, formatDelta } from '@/utils/formatting'
import { scoreTint } from '@/utils/score'
import { cellsByModel } from '@/utils/matrix'
import { MAX_COMPARE } from '@/constants'
import type { Matrix, MatrixCell, Meta } from '@/types/api'
import EmptyState from '@/components/shared/EmptyState.vue'
import ModelCompareChart from '@/components/charts/ModelCompareChart.vue'

const route = useRoute()
const router = useRouter()

const { data: matrix, refresh } = useApi<Matrix>(() => 'api/matrix?include_archived=1')
const { data: meta, refresh: refreshMeta } = useApi<Meta>(() => 'api/meta')

const selected = ref<string[]>([])

function fromQuery(): string[] {
  const raw = route.query.models
  const csv = Array.isArray(raw) ? raw[0] : raw
  return (csv ?? '').split(',').map((s) => s.trim()).filter(Boolean).slice(0, MAX_COMPARE)
}

onMounted(() => {
  selected.value = fromQuery()
  refresh()
  refreshMeta()
})
watch(() => route.query.models, () => (selected.value = fromQuery()))
onViewRefresh(() => {
  refresh()
  refreshMeta()
})

function syncQuery() {
  router.replace({ path: '/compare', query: selected.value.length ? { models: selected.value.join(',') } : {} })
}
function toggle(model: string) {
  const at = selected.value.indexOf(model)
  if (at >= 0) selected.value.splice(at, 1)
  else if (selected.value.length < MAX_COMPARE) selected.value.push(model)
  syncQuery()
}

const rowByModel = computed(() => cellsByModel(matrix.value?.rows ?? []))

// Benchmarks any selected model covers (delta-table rows), and the subset all cover (scored set).
const unionTasks = computed<string[]>(() =>
  (matrix.value?.tasks ?? []).filter((t) => selected.value.some((m) => rowByModel.value[m]?.[t]?.value != null)),
)
const sharedTasks = computed<string[]>(() =>
  unionTasks.value.filter((t) => selected.value.every((m) => rowByModel.value[m]?.[t]?.value != null)),
)

// Mean over the shared benchmarks only — an apples-to-apples score, ranked.
const sharedScores = computed(() =>
  selected.value
    .map((model) => {
      const vals = sharedTasks.value.map((t) => rowByModel.value[model]?.[t]?.value ?? 0)
      const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null
      return { model, mean }
    })
    .sort((a, b) => (b.mean ?? -1) - (a.mean ?? -1)),
)
const leaderMean = computed<number | null>(() => sharedScores.value[0]?.mean ?? null)

function bestInRow(task: string): number | null {
  let bv: number | null = null
  for (const m of selected.value) {
    const v = rowByModel.value[m]?.[task]?.value ?? null
    if (v !== null && (bv === null || v > bv)) bv = v
  }
  return bv
}
function cell(model: string, task: string): MatrixCell | undefined {
  return rowByModel.value[model]?.[task]
}

// Matrix restricted to the compared models + union benchmarks, for the grouped bar chart.
const compareMatrix = computed<Matrix>(() => ({
  tasks: unionTasks.value,
  leaderboard: [],
  rows: (matrix.value?.rows ?? []).filter((r) => selected.value.includes(r.model)),
}))

const comparing = computed(() => selected.value.length >= 2)
</script>

<template>
  <section>
    <div class="mb-4">
      <h2 class="text-lg font-semibold">Compare</h2>
      <p class="text-xs text-text-muted mt-0.5">
        Pick 2–{{ MAX_COMPARE }} models. The ranking scores them on their shared benchmarks only, so a coverage gap never
        flatters a model.
      </p>
    </div>

    <!-- model picker -->
    <div class="rounded-lg border border-surface-border bg-surface p-4 mb-5">
      <div class="font-mono text-[10px] uppercase tracking-widest text-text-muted mb-2">Models ({{ selected.length }}/{{ MAX_COMPARE }})</div>
      <div class="flex flex-wrap gap-2">
        <button
          v-for="m in meta?.models ?? []"
          :key="m"
          class="font-mono text-xs px-2.5 py-1 rounded-full border"
          :class="selected.includes(m)
            ? 'border-accent bg-accent-subtle text-text'
            : selected.length >= MAX_COMPARE
              ? 'border-surface-border-subtle text-text-muted opacity-50 cursor-not-allowed'
              : 'border-surface-border text-text-secondary hover:bg-surface-raised'"
          :disabled="!selected.includes(m) && selected.length >= MAX_COMPARE"
          @click="toggle(m)"
        >{{ m }}</button>
      </div>
    </div>

    <EmptyState v-if="!comparing" icon="⚖" message="Pick at least two models to compare." />

    <div v-else class="space-y-6">
      <!-- shared-benchmark ranking -->
      <div>
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">
          Shared-benchmark ranking
          <span class="font-normal normal-case text-text-muted">
            — {{ sharedTasks.length }} shared: <span class="font-mono">{{ sharedTasks.join(' · ') || 'none' }}</span>
          </span>
        </h3>
        <div v-if="!sharedTasks.length" class="text-sm text-text-muted rounded-lg border border-surface-border bg-surface p-4">
          These models have no benchmark in common. The per-benchmark table below still shows where each one has run.
        </div>
        <div v-else class="rounded-lg border border-surface-border overflow-hidden bg-surface">
          <div
            v-for="(s, i) in sharedScores"
            :key="s.model"
            class="flex items-center gap-3 px-4 py-2.5 border-b border-surface-border-subtle last:border-b-0"
          >
            <span class="font-mono text-text-muted tabular-nums w-5">{{ i + 1 }}</span>
            <button class="font-mono text-[13px] font-semibold text-accent hover:underline" @click="router.push(`/models/${encodeURIComponent(s.model)}`)">{{ s.model }}</button>
            <span class="ml-auto font-mono text-base font-semibold tabular-nums">{{ formatScore(s.mean) }}</span>
            <span class="font-mono text-xs text-text-muted tabular-nums w-16 text-right">
              {{ i === 0 || s.mean === null || leaderMean === null ? '' : formatDelta(s.mean - leaderMean) }}
            </span>
          </div>
        </div>
      </div>

      <!-- per-benchmark delta table -->
      <div>
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Per-benchmark</h3>
        <div class="overflow-x-auto rounded-lg border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead>
              <tr class="border-b border-surface-border bg-surface-raised text-xs font-semibold uppercase tracking-wider text-text-secondary">
                <th class="px-3 py-2 text-left">Benchmark</th>
                <th v-for="m in selected" :key="m" class="px-3 py-2 text-center font-mono normal-case">{{ m }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="task in unionTasks" :key="task" class="border-b border-surface-border-subtle">
                <td class="px-3 py-2 font-mono text-[13px] whitespace-nowrap">
                  {{ task }}
                  <span v-if="sharedTasks.includes(task)" class="ml-1 text-[10px] font-sans text-text-muted">shared</span>
                </td>
                <td v-for="m in selected" :key="m" class="p-1 text-center">
                  <div
                    v-if="cell(m, task) && cell(m, task)!.value !== null"
                    class="rounded px-2 py-1.5 leading-tight"
                    :class="cell(m, task)!.value === bestInRow(task) ? 'ring-2 ring-inset' : ''"
                    :style="{ backgroundColor: scoreTint(cell(m, task)!.value!), ...(cell(m, task)!.value === bestInRow(task) ? { '--tw-ring-color': 'var(--c-best)' } : {}) }"
                  >
                    <span class="font-mono font-medium tabular-nums">{{ formatScore(cell(m, task)!.value) }}</span>
                    <span class="block font-mono text-[10px] text-text-muted tabular-nums leading-none">{{ formatStderr(cell(m, task)!.value, cell(m, task)!.stderr) }}</span>
                  </div>
                  <span v-else class="text-text-muted">—</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- grouped bars -->
      <div>
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Per-benchmark bars</h3>
        <ModelCompareChart :matrix="compareMatrix" :models="selected" />
      </div>
    </div>
  </section>
</template>
