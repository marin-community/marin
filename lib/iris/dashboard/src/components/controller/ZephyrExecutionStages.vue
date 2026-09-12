<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { useLogServerStatsRpc } from '@/composables/useRpc'
import { DEFAULT_REFRESH_MS, useAutoRefresh } from '@/composables/useAutoRefresh'
import { decodeArrowIpc } from '@/utils/arrow'
import { formatBytes } from '@/utils/formatting'
import ZephyrExecutionGraph from './ZephyrExecutionGraph.vue'
import {
  stageStatsSql, reducerStatsSql, reducerTaskStatsSql, shuffleSummarySql, relativePayload, REDUCER_PAGE_SIZE,
  stageView, joinStep, STAGE_VIEWS, STAGE_VIEW_STORAGE_KEY, type StageView,
  type ExecutionPlan, type ExecutionStage, type StageStat, type ReducerStat, type ShuffleSummary,
} from '@/utils/zephyr'

const props = defineProps<{ execution: ExecutionPlan; startMs: number; namespaces: string[] }>()
const selectedStage = ref('')
const page = ref(0)
const savedView = ref<string | null>(null)
try {
  savedView.value = localStorage.getItem(STAGE_VIEW_STORAGE_KEY)
} catch (error) {
  console.warn('Cannot read the saved Zephyr stage view; using the plan default.', error)
}
const plan = computed(() => {
  try {
    const nodes = JSON.parse(props.execution.stages_json) as ExecutionStage[]
    if (!Array.isArray(nodes)) throw new Error('Expected a stage list')
    return { nodes, error: '' }
  } catch (error) {
    return { nodes: [] as ExecutionStage[], error: `Cannot read execution plan: ${String(error)}` }
  }
})
const selected = computed(() => plan.value.nodes.find(node => node.stage_name === selectedStage.value))
const view = computed(() => stageView(plan.value.nodes, savedView.value))
const stageStatuses = computed(() => Object.fromEntries(plan.value.nodes.map(stage => [stage.stage_name, stageStatus(stage)])))
const stages = useLogServerStatsRpc<{ arrowIpc?: string }>('Query', () => ({
  sql: stageStatsSql(props.execution.execution_id, props.startMs),
}))
const reducers = useLogServerStatsRpc<{ arrowIpc?: string }>('Query', () => ({
  sql: (props.namespaces.includes('zephyr.worker') ? reducerTaskStatsSql : reducerStatsSql)(
    props.execution.execution_id, selectedStage.value, props.startMs, page.value,
  ),
}))
const summary = useLogServerStatsRpc<{ arrowIpc?: string }>('Query', () => ({
  sql: shuffleSummarySql(props.execution.execution_id, selectedStage.value, props.startMs),
}))
const stageStats = computed(() => decodeArrowIpc(stages.data.value?.arrowIpc).rows as unknown as StageStat[])
const selectedStat = computed(() => stageStats.value.find(item => item.stage_name === selectedStage.value))
const targets = computed(() => decodeArrowIpc(reducers.data.value?.arrowIpc).rows as unknown as ReducerStat[])
const totals = computed(() => (decodeArrowIpc(summary.data.value?.arrowIpc).rows as unknown as ShuffleSummary[])[0])
const error = computed(() => plan.value.error || stages.error.value || reducers.error.value || summary.error.value)
const coordinatorLink = computed(() => `/job/${encodeURIComponent(props.execution.coordinator_job_id)}/task/${encodeURIComponent(props.execution.coordinator_job_id + '/0')}`)

function stageStatus(stage: ExecutionStage): string {
  if (stage.stage_type === 'reshard') return 'Reference-only · no worker task'
  const stat = stageStats.value.find(item => item.stage_name === stage.stage_name)
  if (stat?.status === 'END') return 'Completed'
  if (stat?.status === 'FAILED') return 'Failed'
  return 'No completion report'
}

function branchLabel(stage: ExecutionStage): string {
  const branch = joinStep(stage, plan.value.nodes)
  return branch ? `Right input of stage${branch.parentStage} · step ${branch.step} of ${branch.total}` : ''
}

function selectView(next: StageView) {
  savedView.value = next
  try {
    localStorage.setItem(STAGE_VIEW_STORAGE_KEY, next)
  } catch (error) {
    console.warn('Cannot save the Zephyr stage view; keeping it for this execution.', error)
  }
}

function handleTabKey(event: KeyboardEvent) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
  event.preventDefault()
  const next = event.key === 'Home' ? 'List' : event.key === 'End' ? 'Graph' : view.value === 'List' ? 'Graph' : 'List'
  selectView(next)
  document.getElementById(`${props.execution.execution_id}-${next}-tab`)?.focus()
}

async function refreshReducers() {
  if (!selected.value?.has_reduce || !props.namespaces.includes('zephyr.shuffle')) return
  await Promise.all([reducers.refresh(), summary.refresh()])
}

async function refresh() {
  if (stages.loading.value || reducers.loading.value || summary.loading.value) return
  await Promise.all([
    props.namespaces.includes('zephyr.stage') ? stages.refresh() : Promise.resolve(),
    refreshReducers(),
  ])
}

watch([selectedStage, page], () => {
  reducers.data.value = null
  summary.data.value = null
  reducers.error.value = null
  summary.error.value = null
  void refreshReducers()
})

function selectStage(stage: ExecutionStage) {
  page.value = 0
  selectedStage.value = stage.stage_name
}

selectedStage.value = plan.value.nodes.find(stage => stage.stage_type !== 'reshard')?.stage_name ?? ''
onMounted(refresh)
useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
</script>

<template>
  <div class="mt-3 text-sm">
    <div role="tablist" aria-label="Stage view" class="mb-3 flex gap-1 border-b border-surface-border" @keydown="handleTabKey">
      <button v-for="mode in STAGE_VIEWS" :key="mode" :id="`${execution.execution_id}-${mode}-tab`"
        role="tab" :aria-selected="view === mode" :tabindex="view === mode ? 0 : -1"
        :aria-controls="`${execution.execution_id}-stage-picker`" @click="selectView(mode)"
        class="border-b-2 px-4 py-2 font-medium focus-visible:outline-2 focus-visible:outline-accent"
        :class="view === mode ? 'border-accent text-accent' : 'border-transparent text-text-secondary hover:text-text'">
        {{ mode }}
      </button>
    </div>
    <div class="flex flex-wrap items-center justify-between gap-2 text-text-muted">
      <p v-if="view === 'List'">Stages in dependency order. Each indented block is one chain feeding the right input of a join.</p>
      <p v-else>Arrows show data dependencies. Reshard stages can change task counts; the graph does not show scheduling concurrency.</p>
      <RouterLink v-if="execution.coordinator_job_id" :to="coordinatorLink" class="text-accent hover:underline">Coordinator task / live progress →</RouterLink>
    </div>
    <p v-if="error" role="alert" class="mt-3 text-red-500">{{ error }}</p>
    <div class="mt-4 grid items-start gap-5" :class="view === 'List' ? 'xl:grid-cols-[260px_minmax(0,1fr)]' : ''">
      <div role="tabpanel" :id="`${execution.execution_id}-stage-picker`" :aria-labelledby="`${execution.execution_id}-${view}-tab`" class="min-w-0">
      <ZephyrExecutionGraph v-if="view === 'Graph'" :stages="plan.nodes" :selected-stage="selectedStage" :statuses="stageStatuses" @select="selectStage" />
      <ol v-else class="space-y-2" aria-label="Execution stages">
        <li v-for="node in plan.nodes" :key="node.stage_name" :class="branchLabel(node) ? 'ml-5 border-l-2 border-surface-border pl-3' : ''">
          <div v-if="node.stage_type === 'reshard'" class="border-y border-dashed border-surface-border px-3 py-2 text-xs text-text-muted">
            {{ node.stage_name }} · references only
          </div>
          <button v-else :title="node.stage_name" :aria-pressed="selectedStage === node.stage_name" @click="selectStage(node)"
            class="flex w-full flex-col gap-1 rounded-lg border bg-surface px-3 py-3 text-left focus-visible:outline-2 focus-visible:outline-accent"
            :class="selectedStage === node.stage_name ? 'border-accent bg-accent-subtle ring-1 ring-accent' : 'border-surface-border hover:border-text-muted'">
            <span v-if="branchLabel(node)" class="text-xs text-text-secondary">{{ branchLabel(node) }}</span>
            <span class="w-full truncate text-xs text-text-muted">{{ node.stage_name }}</span>
            <span class="font-semibold text-text">{{ node.label }}</span>
            <span class="text-xs" :class="stageStatus(node) === 'Failed' ? 'text-red-500' : 'text-text-secondary'">{{ stageStatus(node) }}</span>
          </button>
        </li>
      </ol>
      </div>
      <p v-if="!plan.nodes.length" class="text-text-muted">This execution has no planned stages.</p>
      <div v-if="selected" class="min-w-0">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <h4 class="font-mono font-semibold text-text">{{ selected.stage_name }}</h4>
          <button class="text-accent hover:underline" :disabled="stages.loading.value || reducers.loading.value" @click="refresh">Refresh stage</button>
        </div>
        <p v-if="selectedStat" class="mt-2 text-text-secondary">
          {{ selectedStat.elapsed.toFixed(2) }} s · {{ selectedStat.total_shards }} tasks · {{ selectedStat.items.toLocaleString() }} output items · {{ formatBytes(selectedStat.mem_peak_bytes_max) }} peak task RAM
        </p>
        <p v-else class="mt-2 text-text-muted">{{ stageStatus(selected) }}. Completion telemetry can be delayed or unavailable.</p>
        <template v-if="selected.has_reduce">
          <p class="mt-3 text-text-muted">Task status is the latest worker report. Sizes arrive independently; UNREPORTED means no size measurement and zero means an observed empty target. Worker status has no retry-attempt identifier.</p>
          <p v-if="summary.loading.value && !totals" class="mt-3 text-text-muted">Loading reducer measurements…</p>
          <template v-if="totals && totals.expected_targets !== null">
            <div class="my-3 flex flex-wrap gap-x-8 gap-y-2 text-text-secondary">
              <span><strong class="text-text">{{ totals.observed_targets }} / {{ totals.expected_targets }}</strong> measured</span>
              <span><strong class="text-text">{{ totals.expected_targets - totals.observed_targets }}</strong> UNREPORTED</span>
              <span>Median / max rows: <strong class="text-text">{{ totals.median_rows?.toLocaleString() ?? '—' }} / {{ totals.max_rows?.toLocaleString() ?? '—' }}</strong></span>
              <span>Median / max payload: <strong class="text-text">{{ totals.median_bytes === null ? '—' : formatBytes(totals.median_bytes) }} / {{ totals.max_bytes === null ? '—' : formatBytes(totals.max_bytes) }}</strong></span>
            </div>
            <p class="mb-2 text-xs text-text-muted">Largest encoded payloads first. Only persisted targets appear below; undelivered placeholders are included in UNREPORTED above.</p>
            <div class="overflow-x-auto">
              <table class="w-full text-left text-sm">
                <thead class="sticky top-0 border-b border-surface-border bg-surface text-text-muted"><tr><th class="py-2 pr-3">Reducer</th><th class="pr-3">Task status</th><th class="pr-3">Input rows</th><th class="pr-3">Encoded payload</th><th class="pr-3">Payload / median</th><th class="pr-3">Mapper outputs</th><th>Size attempt</th></tr></thead>
                <tbody>
                  <tr v-for="target in targets" :key="target.target_shard" class="border-b border-surface-border text-text-secondary">
                    <td class="py-2 font-mono">{{ target.target_shard }}</td>
                    <td class="pr-3" :class="target.task_status === 'FAILED' ? 'text-red-500' : ''">{{ target.task_status ?? 'No report' }}</td>
                    <td class="py-2 pr-4">{{ target.input_rows?.toLocaleString() ?? 'UNREPORTED' }}</td>
                    <td class="py-2 pr-4">{{ target.payload_bytes === null ? 'UNREPORTED' : formatBytes(target.payload_bytes) }}</td>
                    <td class="pr-3 font-mono font-semibold text-text">{{ relativePayload(target.payload_bytes, totals.median_bytes) }}</td>
                    <td>{{ target.num_sources ?? '—' }}</td><td>{{ target.attempt }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div class="mt-3 flex items-center gap-4 text-text-secondary">
              <button :disabled="page === 0 || reducers.loading.value" class="text-accent disabled:opacity-40" @click="page--">← Previous</button>
              <span>{{ page * REDUCER_PAGE_SIZE + 1 }}–{{ Math.min((page + 1) * REDUCER_PAGE_SIZE, totals.persisted_targets) }} of {{ totals.persisted_targets }} persisted targets</span>
              <button :disabled="(page + 1) * REDUCER_PAGE_SIZE >= totals.persisted_targets || reducers.loading.value" class="text-accent disabled:opacity-40" @click="page++">Next →</button>
            </div>
          </template>
          <p v-else-if="!summary.loading.value" class="mt-3 text-text-muted">No reducer reports yet. Target count is unavailable until a placeholder or measurement arrives.</p>
        </template>
        <p v-else class="mt-3 text-text-muted">This stage has no reducer inputs.</p>
      </div>
    </div>
  </div>
</template>
