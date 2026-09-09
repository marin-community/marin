<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";

import { chartApi } from "../api.js";
import ChartLibrary from "../components/ChartLibrary.vue";
import GanttChart from "../components/GanttChart.vue";
import JsonEditor from "../components/JsonEditor.vue";
import {
  EXAMPLE_PLAN,
  clonePlan,
  decodeImport,
  emptyPlan,
  formatDate,
  resolvePlan,
} from "../model.js";

const route = useRoute();
const router = useRouter();

const charts = ref([]);
const record = ref(null);
const plan = ref(null);
const selected = ref("");
const busy = ref(false);
const loading = ref(false);
const dirty = ref(false);
const saving = ref(false);
const conflict = ref(false);
const message = ref("");
const errorMessage = ref("");
const editorError = ref("");
const editorOpen = ref(true);
const showToday = ref(true);
const showDependencies = ref(true);
const importOpen = ref(false);
const importText = ref("");
const importError = ref("");

let suppressPlanWatch = false;
let savedDocument = "";
let saveTimer = null;
let loadSequence = 0;

const currentId = computed(() => (typeof route.params.id === "string" ? route.params.id : ""));
const statusText = computed(() => {
  if (saving.value) return "Saving…";
  if (conflict.value) return "Save conflict";
  if (dirty.value) return "Unsaved";
  if (record.value) return `Saved · revision ${record.value.revision}`;
  return "";
});

async function refreshCharts() {
  charts.value = await chartApi.list();
}

async function loadChart(id) {
  const sequence = ++loadSequence;
  loading.value = true;
  errorMessage.value = "";
  try {
    const loaded = await chartApi.get(id);
    if (sequence !== loadSequence) return;
    suppressPlanWatch = true;
    record.value = loaded;
    plan.value = clonePlan(loaded.document);
    savedDocument = JSON.stringify(loaded.document);
    dirty.value = false;
    conflict.value = false;
    selected.value = "";
    editorError.value = "";
    await nextTick();
    suppressPlanWatch = false;
  } catch (error) {
    if (sequence !== loadSequence) return;
    record.value = null;
    plan.value = null;
    errorMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    if (sequence === loadSequence) loading.value = false;
  }
}

watch(
  currentId,
  (id) => {
    clearTimeout(saveTimer);
    if (id) loadChart(id);
    else {
      loadSequence += 1;
      record.value = null;
      plan.value = null;
      loading.value = false;
      errorMessage.value = "";
    }
  },
  { immediate: true },
);

watch(
  plan,
  () => {
    if (suppressPlanWatch || !record.value || !plan.value) return;
    dirty.value = JSON.stringify(plan.value) !== savedDocument;
    if (!dirty.value || conflict.value) return;
    message.value = "";
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => saveChart(), 900);
  },
  { deep: true },
);

onMounted(async () => {
  try {
    await refreshCharts();
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
});

onBeforeUnmount(() => clearTimeout(saveTimer));

async function createChart(document) {
  busy.value = true;
  errorMessage.value = "";
  try {
    resolvePlan(document);
    const created = await chartApi.create(document);
    await refreshCharts();
    await router.push(`/charts/${created.id}`);
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    busy.value = false;
  }
}

async function saveChart() {
  if (!record.value || !plan.value || !dirty.value || saving.value || conflict.value) return !conflict.value;
  clearTimeout(saveTimer);
  try {
    resolvePlan(plan.value);
  } catch (error) {
    editorError.value = error instanceof Error ? error.message : String(error);
    return false;
  }

  saving.value = true;
  editorError.value = "";
  const document = clonePlan(plan.value);
  const serialized = JSON.stringify(document);
  try {
    const updated = await chartApi.update(record.value.id, document, record.value.revision);
    record.value = updated;
    savedDocument = serialized;
    dirty.value = JSON.stringify(plan.value) !== savedDocument;
    const index = charts.value.findIndex((chart) => chart.id === updated.id);
    const summary = { ...updated };
    delete summary.document;
    if (index >= 0) charts.value.splice(index, 1);
    charts.value.unshift(summary);
    message.value = "Saved";
    if (dirty.value) saveTimer = setTimeout(() => saveChart(), 300);
    return true;
  } catch (error) {
    if (error && error.status === 409) conflict.value = true;
    errorMessage.value = error instanceof Error ? error.message : String(error);
    return false;
  } finally {
    saving.value = false;
  }
}

function replacePlan(updated) {
  resolvePlan(updated);
  plan.value = clonePlan(updated);
  selected.value = "";
}

function mutatePlan(mutation) {
  const updated = clonePlan(plan.value);
  mutation(updated);
  replacePlan(updated);
}

function allNames(document) {
  return document.workstreams.flatMap((workstream) => [
    ...workstream.tasks.map((task) => task.name),
    ...(workstream.milestones || []).map((milestone) => milestone.name),
  ]);
}

function uniqueName(prefix, names) {
  let index = 1;
  while (names.includes(`${prefix} ${index}`)) index += 1;
  return `${prefix} ${index}`;
}

function quickAdd(kind) {
  const today = formatDate(new Date());
  mutatePlan((document) => {
    if (kind === "workstream") {
      document.workstreams.push({
        name: uniqueName("Workstream", document.workstreams.map((workstream) => workstream.name)),
        tasks: [],
      });
      return;
    }
    if (kind === "capacity") {
      document.capacity ||= [];
      document.capacity.push({
        name: uniqueName("Cluster", document.capacity.map((capacity) => capacity.name)),
        chip: "H100",
        chips: 128,
        from: today,
        color: "#647d92",
      });
      return;
    }
    if (!document.workstreams.length) document.workstreams.push({ name: "Workstream 1", tasks: [] });
    const workstream = document.workstreams[0];
    const names = allNames(document);
    if (kind === "task") {
      workstream.tasks.push({ name: uniqueName("Task", names), start: ["date", today], end: ["weeks", 2] });
    } else {
      workstream.milestones ||= [];
      workstream.milestones.push({ name: uniqueName("Milestone", names), date: today });
    }
  });
  editorOpen.value = true;
}

async function copyShareLink() {
  if (!(await saveChart())) return;
  const url = new URL(`/plantt/charts/${record.value.id}`, window.location.origin).href;
  await navigator.clipboard.writeText(url);
  message.value = "Link copied";
}

function downloadJson() {
  const blob = new Blob([`${JSON.stringify(plan.value, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${plan.value.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "plan"}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

async function importChart() {
  try {
    const imported = decodeImport(importText.value);
    importOpen.value = false;
    importText.value = "";
    importError.value = "";
    await createChart(imported);
  } catch (error) {
    importError.value = error instanceof Error ? error.message : String(error);
  }
}

async function deleteChart() {
  if (!record.value || !window.confirm(`Delete “${record.value.title}”? This cannot be undone.`)) return;
  busy.value = true;
  try {
    await chartApi.remove(record.value.id, record.value.revision);
    await refreshCharts();
    await router.push("/");
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    busy.value = false;
  }
}

async function reloadLatest() {
  if (record.value) await loadChart(record.value.id);
}

async function saveCopy() {
  if (!plan.value) return;
  const copy = clonePlan(plan.value);
  copy.title = `${copy.title} (copy)`;
  conflict.value = false;
  await createChart(copy);
}
</script>

<template>
  <div class="workspace">
    <ChartLibrary
      :charts="charts"
      :current-id="currentId"
      :busy="busy"
      @open="router.push(`/charts/${$event}`)"
      @create-blank="createChart(emptyPlan())"
      @create-example="createChart(clonePlan(EXAMPLE_PLAN))"
      @import="importOpen = true"
    />

    <section class="canvas-area">
      <div v-if="loading" class="empty-state"><p>Loading plan…</p></div>
      <div v-else-if="!plan" class="empty-state">
        <span class="eyebrow">Database-backed plans</span>
        <h2>Make the schedule legible.</h2>
        <p>Keep project work, dependencies, milestones, and accelerator capacity in one shared chart.</p>
        <div class="empty-actions">
          <button type="button" @click="createChart(clonePlan(EXAMPLE_PLAN))">Open an example</button>
          <button type="button" class="quiet-button" @click="importOpen = true">Import a Plantt link</button>
        </div>
        <p v-if="errorMessage" class="inline-error">{{ errorMessage }}</p>
      </div>

      <template v-else>
        <header class="plan-header">
          <div class="plan-heading">
            <input v-model="plan.title" class="title-input" aria-label="Plan title" />
            <input v-model="plan.note" class="note-input" aria-label="Plan note" placeholder="Add a short note…" />
          </div>
          <div class="plan-status" :class="{ warning: conflict }">{{ statusText }}</div>
          <div class="plan-actions">
            <button type="button" :disabled="saving || !dirty" @click="saveChart">Save</button>
            <button type="button" @click="copyShareLink">Share</button>
            <button type="button" class="icon-button" title="Export JSON" @click="downloadJson">↓</button>
            <button type="button" class="icon-button danger-button" title="Delete plan" @click="deleteChart">×</button>
          </div>
        </header>

        <div v-if="conflict" class="notice conflict-notice">
          <span>Someone else saved this plan while you were editing.</span>
          <button type="button" @click="reloadLatest">Load latest</button>
          <button type="button" @click="saveCopy">Save my version as a copy</button>
        </div>
        <div v-if="errorMessage || editorError" class="notice error-notice">
          {{ editorError || errorMessage }}
          <button type="button" aria-label="Dismiss error" @click="errorMessage = ''; editorError = ''">×</button>
        </div>

        <div class="tool-row">
          <div class="add-group" aria-label="Add to plan">
            <span>Add</span>
            <button type="button" @click="quickAdd('workstream')">Workstream</button>
            <button type="button" @click="quickAdd('task')">Task</button>
            <button type="button" @click="quickAdd('milestone')">Milestone</button>
            <button type="button" @click="quickAdd('capacity')">Compute</button>
          </div>
          <label><input v-model="showDependencies" type="checkbox" /> Dependencies</label>
          <label><input v-model="showToday" type="checkbox" /> Today</label>
          <button type="button" class="quiet-button editor-toggle" :aria-expanded="editorOpen" @click="editorOpen = !editorOpen">
            {{ editorOpen ? "Hide JSON" : "Edit JSON" }}
          </button>
        </div>

        <div class="plan-body" :class="{ 'editor-closed': !editorOpen }">
          <section class="chart-panel">
            <GanttChart
              :plan="plan"
              :selected="selected"
              :show-today="showToday"
              :show-dependencies="showDependencies"
              @select="selected = $event.name"
            />
          </section>
          <JsonEditor v-if="editorOpen" :plan="plan" @apply="replacePlan" @error="editorError = $event" />
        </div>
        <p v-if="message" class="toast" role="status">{{ message }}</p>
      </template>
    </section>

    <div v-if="importOpen" class="modal-backdrop" @click.self="importOpen = false">
      <section class="modal" role="dialog" aria-modal="true" aria-labelledby="import-title">
        <span class="eyebrow">One-time migration</span>
        <h2 id="import-title">Import a Plantt plan</h2>
        <p>Paste an existing compressed Plantt URL or a plan JSON document. The imported plan will get a short database-backed link.</p>
        <textarea v-model="importText" autofocus aria-label="Plantt URL or JSON" placeholder="https://openathena.ai/plantt/?N4… or { … }" />
        <p v-if="importError" class="inline-error">{{ importError }}</p>
        <div class="modal-actions">
          <button type="button" class="quiet-button" @click="importOpen = false">Cancel</button>
          <button type="button" :disabled="!importText.trim()" @click="importChart">Import</button>
        </div>
      </section>
    </div>
  </div>
</template>
