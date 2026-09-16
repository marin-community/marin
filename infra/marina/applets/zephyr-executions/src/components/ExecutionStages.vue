<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { encodeSegment, useJson } from "../api.js";
import {
  formatBytes,
  irisTaskUrl,
  joinStep,
  relativePayload,
  REDUCER_PAGE_SIZE,
  stageView,
  STAGE_VIEW_STORAGE_KEY,
  STAGE_VIEWS,
} from "../zephyr.js";
import ExecutionGraph from "./ExecutionGraph.vue";

const props = defineProps({
  execution: { type: Object, required: true },
  namespaces: { type: Array, required: true },
  refreshMs: { type: Number, default: 30000 },
});

const selectedStage = ref("");
const page = ref(0);
const savedView = ref(null);
try {
  savedView.value = localStorage.getItem(STAGE_VIEW_STORAGE_KEY);
} catch (error) {
  console.warn("Cannot read the saved stage view; using the plan default.", error);
}

const nodes = computed(() => (Array.isArray(props.execution.stages) ? props.execution.stages : []));
const selected = computed(() => nodes.value.find((node) => node.stage_name === selectedStage.value));
const view = computed(() => stageView(nodes.value, savedView.value));
const base = computed(() => `api/executions/${encodeSegment(props.execution.execution_id)}`);
const stagePath = computed(() => `${base.value}/stages/${encodeSegment(selectedStage.value)}`);
const hasShuffle = computed(() => props.namespaces.includes("zephyr.shuffle"));

const stages = useJson(() => (props.namespaces.includes("zephyr.stage") ? `${base.value}/stages` : null));
const summary = useJson(() => (selected.value?.has_reduce && hasShuffle.value ? `${stagePath.value}/summary` : null));
const reducers = useJson(() =>
  selected.value?.has_reduce && hasShuffle.value ? `${stagePath.value}/reducers?page=${page.value}` : null,
);

const stageStats = computed(() => stages.data.value ?? []);
const selectedStat = computed(() => stageStats.value.find((item) => item.stage_name === selectedStage.value));
const targets = computed(() => reducers.data.value ?? []);
const totals = computed(() => summary.data.value);
const error = computed(() => props.execution.plan_error || stages.error.value || reducers.error.value || summary.error.value);
const stageStatuses = computed(() => Object.fromEntries(nodes.value.map((stage) => [stage.stage_name, stageStatus(stage)])));

function stageStatus(stage) {
  if (stage.stage_type === "reshard") return "Reference-only · no worker task";
  const stat = stageStats.value.find((item) => item.stage_name === stage.stage_name);
  if (stat?.status === "END") return "Completed";
  if (stat?.status === "FAILED") return "Failed";
  return "No completion report";
}

function branchLabel(stage) {
  const branch = joinStep(stage, nodes.value);
  return branch ? `Right input of stage${branch.parentStage} · step ${branch.step} of ${branch.total}` : "";
}

function selectView(next) {
  savedView.value = next;
  try {
    localStorage.setItem(STAGE_VIEW_STORAGE_KEY, next);
  } catch (error) {
    console.warn("Cannot save the stage view; keeping it for this execution.", error);
  }
}

function handleTabKey(event) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  const next = event.key === "Home" ? "List" : event.key === "End" ? "Graph" : view.value === "List" ? "Graph" : "List";
  selectView(next);
  document.getElementById(`${props.execution.execution_id}-${next}-tab`)?.focus();
}

async function refreshReducers() {
  await Promise.all([reducers.refresh(), summary.refresh()]);
}

async function refresh() {
  if (stages.loading.value || reducers.loading.value || summary.loading.value) return;
  await Promise.all([stages.refresh(), refreshReducers()]);
}

watch([selectedStage, page], () => {
  reducers.reset();
  summary.reset();
  void refreshReducers();
});

function selectStage(stage) {
  page.value = 0;
  selectedStage.value = stage.stage_name;
}

selectedStage.value = nodes.value.find((stage) => stage.stage_type !== "reshard")?.stage_name ?? "";
let timer = null;
onMounted(() => {
  void refresh();
  timer = setInterval(() => {
    if (document.visibilityState === "visible") void refresh();
  }, props.refreshMs);
});
onBeforeUnmount(() => {
  if (timer) clearInterval(timer);
});
</script>

<template>
  <div>
    <div class="tabs" role="tablist" aria-label="Stage view" @keydown="handleTabKey">
      <button
        v-for="mode in STAGE_VIEWS"
        :id="`${execution.execution_id}-${mode}-tab`"
        :key="mode"
        role="tab"
        :aria-selected="view === mode"
        :tabindex="view === mode ? 0 : -1"
        :aria-controls="`${execution.execution_id}-stage-picker`"
        @click="selectView(mode)"
      >
        {{ mode }}
      </button>
    </div>
    <div class="row muted">
      <p v-if="view === 'List'" style="margin: 0">Stages in dependency order. Each indented block is one chain feeding the right input of a join.</p>
      <p v-else style="margin: 0">Arrows show data dependencies. Reshard stages can change task counts; the graph does not show scheduling concurrency.</p>
      <a v-if="execution.coordinator_job_id" :href="irisTaskUrl(execution.coordinator_job_id)" target="_blank" rel="noopener">Coordinator task in Iris →</a>
    </div>
    <p v-if="error" role="alert" class="danger">{{ error }}</p>
    <div class="layout" :class="{ list: view === 'List' }">
      <div :id="`${execution.execution_id}-stage-picker`" role="tabpanel" :aria-labelledby="`${execution.execution_id}-${view}-tab`">
        <ExecutionGraph v-if="view === 'Graph'" :stages="nodes" :selected-stage="selectedStage" :statuses="stageStatuses" @select="selectStage" />
        <ol v-else class="stages" aria-label="Execution stages">
          <li v-for="node in nodes" :key="node.stage_name" :class="{ branch: branchLabel(node) }">
            <div v-if="node.stage_type === 'reshard'" class="separator">{{ node.stage_name }} · references only</div>
            <button v-else class="stage" :title="node.stage_name" :aria-pressed="selectedStage === node.stage_name" @click="selectStage(node)">
              <span v-if="branchLabel(node)" class="faint">{{ branchLabel(node) }}</span>
              <span class="name">{{ node.stage_name }}</span>
              <span class="label">{{ node.label }}</span>
              <span class="status" :class="{ failed: stageStatus(node) === 'Failed' }">{{ stageStatus(node) }}</span>
            </button>
          </li>
        </ol>
      </div>
      <p v-if="!nodes.length" class="muted">This execution has no planned stages.</p>
      <div v-if="selected">
        <div class="row">
          <h4>{{ selected.stage_name }}</h4>
          <button class="link" :disabled="stages.loading.value || reducers.loading.value" @click="refresh">Refresh stage</button>
        </div>
        <p v-if="selectedStat" class="muted">
          {{ selectedStat.elapsed.toFixed(2) }} s · {{ selectedStat.total_shards }} tasks · {{ selectedStat.items.toLocaleString() }} output items ·
          {{ formatBytes(selectedStat.mem_peak_bytes_max) }} peak task RAM
        </p>
        <p v-else class="muted">{{ stageStatus(selected) }}. Completion telemetry can be delayed or unavailable.</p>
        <template v-if="selected.has_reduce">
          <p class="muted">
            Task status is the latest worker report for the target. Sizes arrive independently; UNREPORTED means no size measurement and zero means an
            observed empty target.
          </p>
          <p v-if="summary.loading.value && !totals" class="muted">Loading reducer measurements…</p>
          <template v-if="totals && totals.expected_targets !== null">
            <div class="summary">
              <span><strong>{{ totals.observed_targets }} / {{ totals.expected_targets }}</strong> measured</span>
              <span><strong>{{ totals.expected_targets - totals.observed_targets }}</strong> UNREPORTED</span>
              <span>Median / max rows: <strong>{{ totals.median_rows?.toLocaleString() ?? "—" }} / {{ totals.max_rows?.toLocaleString() ?? "—" }}</strong></span>
              <span>Median / max payload: <strong>{{ totals.median_bytes === null ? "—" : formatBytes(totals.median_bytes) }} / {{ totals.max_bytes === null ? "—" : formatBytes(totals.max_bytes) }}</strong></span>
            </div>
            <p class="faint">Largest encoded payloads first. Only persisted targets appear below; undelivered placeholders are included in UNREPORTED above.</p>
            <div class="tablewrap">
              <table>
                <thead>
                  <tr><th>Reducer</th><th>Task status</th><th>Input rows</th><th>Encoded payload</th><th>Payload / median</th><th>Mapper outputs</th><th>Size attempt</th></tr>
                </thead>
                <tbody>
                  <tr v-for="target in targets" :key="target.target_shard">
                    <td class="mono">{{ target.target_shard }}</td>
                    <td :class="{ failed: target.task_status === 'FAILED' }">{{ target.task_status ?? "No report" }}</td>
                    <td>{{ target.input_rows === null ? "UNREPORTED" : target.input_rows.toLocaleString() }}</td>
                    <td>{{ target.payload_bytes === null ? "UNREPORTED" : formatBytes(target.payload_bytes) }}</td>
                    <td class="ratio">{{ relativePayload(target.payload_bytes, totals.median_bytes) }}</td>
                    <td>{{ target.num_sources ?? "—" }}</td>
                    <td>{{ target.attempt }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div class="pager">
              <button class="link" :disabled="page === 0 || reducers.loading.value" @click="page--">← Previous</button>
              <span>{{ page * REDUCER_PAGE_SIZE + 1 }}–{{ Math.min((page + 1) * REDUCER_PAGE_SIZE, totals.persisted_targets) }} of {{ totals.persisted_targets }} persisted targets</span>
              <button class="link" :disabled="(page + 1) * REDUCER_PAGE_SIZE >= totals.persisted_targets || reducers.loading.value" @click="page++">Next →</button>
            </div>
          </template>
          <p v-else-if="!summary.loading.value" class="muted">No reducer reports yet. Target count is unavailable until a placeholder or measurement arrives.</p>
        </template>
        <p v-else class="muted">This stage has no reducer inputs.</p>
      </div>
    </div>
  </div>
</template>

