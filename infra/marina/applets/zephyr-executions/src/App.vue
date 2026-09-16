<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useJson } from "./api.js";
import { EXECUTION_LIMIT, irisJobUrl } from "./zephyr.js";
import ExecutionStages from "./components/ExecutionStages.vue";

const DAYS = 14;
const rootJobFilter = ref("");
const selectedExecution = ref("");

function readHash() {
  const match = location.hash.match(/^#\/execution\/([^/?]+)(?:\?job=([^&]*))?/);
  if (!match) return;
  selectedExecution.value = decodeURIComponent(match[1]);
  if (match[2]) rootJobFilter.value = decodeURIComponent(match[2]);
}

const health = useJson(() => "api/health");
const executions = useJson(() => {
  const params = new URLSearchParams({ days: String(DAYS), limit: String(EXECUTION_LIMIT) });
  if (rootJobFilter.value.trim()) params.set("root_job", rootJobFilter.value.trim());
  return `api/executions?${params}`;
});
// A deep-linked execution can be older than the listing window; it is fetched on its own.
const linked = useJson(() => (selectedExecution.value ? `api/executions/${encodeURIComponent(selectedExecution.value)}` : null));

const listed = computed(() => executions.data.value ?? []);
const list = computed(() => {
  const extra = linked.data.value;
  return extra && !listed.value.some((item) => item.execution_id === extra.execution_id) ? [extra, ...listed.value] : listed.value;
});
const execution = computed(() => list.value.find((item) => item.execution_id === selectedExecution.value));
const namespaces = computed(() => health.data.value?.namespaces ?? []);
const error = computed(() => health.error.value || executions.error.value);
// A requested execution that cannot be shown keeps its id and its own error.
const lookupError = computed(() => (selectedExecution.value && !execution.value ? linked.error.value : null));

async function refresh() {
  await Promise.all([health.refresh(), executions.refresh()]);
  if (!selectedExecution.value) {
    selectedExecution.value = list.value[0]?.execution_id ?? "";
    return;
  }
  if (!execution.value && !linked.loading.value) await linked.refresh();
}

watch(selectedExecution, (value) => {
  if (value && listed.value.some((item) => item.execution_id === value)) linked.reset();
  else if (value && !list.value.some((item) => item.execution_id === value)) void linked.refresh();
  const suffix = rootJobFilter.value.trim() ? `?job=${encodeURIComponent(rootJobFilter.value.trim())}` : "";
  const next = value ? `#/execution/${encodeURIComponent(value)}${suffix}` : "";
  if (location.hash !== next) history.replaceState(null, "", next || location.pathname);
});

let filterTimer = null;
watch(rootJobFilter, () => {
  clearTimeout(filterTimer);
  filterTimer = setTimeout(async () => {
    await executions.refresh();
    if (!execution.value) selectedExecution.value = list.value[0]?.execution_id ?? "";
  }, 400);
});

let timer = null;
onMounted(() => {
  readHash();
  window.addEventListener("hashchange", readHash);
  void refresh();
  // Health is refreshed too, so tables created after page load are discovered.
  timer = setInterval(() => {
    if (document.visibilityState === "visible") void refresh();
  }, 60000);
});
onBeforeUnmount(() => {
  window.removeEventListener("hashchange", readHash);
  if (timer) clearInterval(timer);
});
</script>

<template>
  <main>
    <h1>Zephyr</h1>
    <div class="row">
      <h2>Execution Data</h2>
      <button class="link" :disabled="executions.loading.value" @click="refresh">Refresh</button>
    </div>
    <div class="controls">
      <label>
        Root job
        <input v-model="rootJobFilter" class="mono" type="text" placeholder="/user/job-name (optional)" size="34" />
      </label>
      <label>
        Execution
        <select v-model="selectedExecution" class="mono">
          <option v-if="selectedExecution && !execution" :value="selectedExecution">{{ selectedExecution }} · requested</option>
          <option v-for="item in list" :key="item.execution_id" :value="item.execution_id">
            {{ item.execution_id }} · {{ item.root_job_id || "local" }}
          </option>
        </select>
      </label>
      <span v-if="list.length === EXECUTION_LIMIT" class="faint">Newest {{ EXECUTION_LIMIT }} executions in the last {{ DAYS }} days</span>
    </div>
    <p v-if="error" role="alert" class="danger">Zephyr telemetry unavailable: {{ error }}</p>
    <p v-if="lookupError" role="alert" class="danger">Execution {{ selectedExecution }}: {{ lookupError }}</p>
    <p v-else-if="health.data.value && health.data.value.plan_records === false" class="muted">
      This Finelog has no execution records yet.
    </p>
    <p v-else-if="!executions.loading.value && !list.length" class="muted">
      No executions found in the last {{ DAYS }} days.
    </p>
    <section v-if="execution" class="panel" aria-label="Execution">
      <div class="row">
        <div>
          <span class="faint">Execution</span>
          <div class="mono">{{ execution.execution_id }}</div>
        </div>
        <div v-if="execution.root_job_id">
          <span class="faint">Iris job</span>
          <div><a :href="irisJobUrl(execution.root_job_id)" target="_blank" rel="noopener" class="mono">{{ execution.root_job_id }}</a></div>
        </div>
        <div>
          <span class="faint">Started</span>
          <div>{{ new Date(execution.ts).toLocaleString() }}</div>
        </div>
      </div>
      <ExecutionStages :key="execution.execution_id" :execution="execution" :namespaces="namespaces" />
    </section>
  </main>
</template>
