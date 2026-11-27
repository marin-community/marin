<script setup>
import { computed, ref, watch } from "vue";

import { api, encodeId, filtersOf, go, href, label, num, verdictClass, verdictText } from "../lib.js";
import FilterBar from "./FilterBar.vue";
import Pager from "./Pager.vue";

const props = defineProps({ query: Object, meta: Object });
const data = ref(null);
const error = ref("");
const loading = ref(false);
const COLUMNS = [
  { key: "capability", text: "Capability · title" },
  { key: "domain", text: "Domain" },
  { key: "slot", text: "Slot", num: true },
  { key: "kind", text: "Kind" },
  { key: "environment", text: "Environment" },
  { key: "verdict", text: "Verdict" },
  { key: "completion_tokens", text: "Completion", num: true },
  { key: "reasoning_tokens", text: "Reasoning", num: true },
  { key: "prompt_tokens", text: "Prompt", num: true },
];
const sort = computed(() => props.query.sort || "capability");
const dir = computed(() => props.query.dir || "asc");
const domainName = computed(() => Object.fromEntries(props.meta.domains.map((d) => [d.subject_id, d.domain])));

let serial = 0;
watch(() => props.query, async (query) => {
  const mine = ++serial;
  loading.value = true;
  try {
    const result = await api("proposals", { ...filtersOf(query), sort: sort.value, dir: dir.value, page: query.page || 1, page_size: query.page_size || 50 });
    if (mine === serial) { data.value = result; error.value = ""; }
  } catch (e) {
    if (mine === serial) error.value = e.message;
  } finally {
    if (mine === serial) loading.value = false;
  }
}, { immediate: true, deep: true });

function sortBy(key) {
  const nextDir = sort.value === key ? (dir.value === "asc" ? "desc" : "asc") : key.endsWith("tokens") ? "desc" : "asc";
  go("/proposals", { ...filtersOf(props.query), sort: key, dir: nextDir, page_size: props.query.page_size });
}
function page(n, size) {
  go("/proposals", { ...filtersOf(props.query), sort: props.query.sort, dir: props.query.dir, page: n, page_size: size });
}
</script>

<template>
  <div class="page-head">
    <div>
      <h1>Proposals</h1>
      <p class="lede">Every planned slot of the <strong>curriculum-v3-n25-20261102</strong> run: one row per capability and slot, with the preflight verdict and the tokens its completions used.</p>
    </div>
  </div>
  <div class="section"><FilterBar :query="query" :meta="meta" path="/proposals" /></div>
  <div class="section">
    <div class="section-head">
      <h2>{{ data ? num(data.total) : "…" }} slots <span v-if="loading" class="muted small">loading…</span></h2>
      <a class="small" :href="href('/stats', filtersOf(query))">Statistics for this selection →</a>
    </div>
    <div v-if="error" class="error">{{ error }}</div>
    <div v-else-if="data" class="card flush" style="overflow-x: auto">
      <table class="items">
        <thead>
          <tr>
            <th v-for="c in COLUMNS" :key="c.key" class="sortable" :class="{ num: c.num, sorted: sort === c.key }" @click="sortBy(c.key)">
              {{ c.text }}<span v-if="sort === c.key">{{ dir === "asc" ? " ↑" : " ↓" }}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in data.rows" :key="`${row.capability_id}/${row.slot}`" @click="go(`/p/${encodeId(row.capability_id)}/${row.slot}`)">
            <td>
              <a class="rowlink" :href="href(`/p/${encodeId(row.capability_id)}/${row.slot}`)" @click.stop>
                <div class="title clip">{{ row.title || "(untitled slot)" }}</div>
                <div class="id clip">{{ row.capability_id }}</div>
              </a>
            </td>
            <td><span class="nowrap" :title="domainName[row.subject_id]">{{ row.subject_id }}</span> <span class="sub clip" style="display: inline-block; max-width: 160px; vertical-align: bottom">{{ domainName[row.subject_id] }}</span></td>
            <td class="num">{{ row.slot }}</td>
            <td class="nowrap"><span class="swatch" :class="`kind-${row.verification}`"></span>{{ label(row.verification) }}</td>
            <td class="nowrap">{{ label(row.environment) }}</td>
            <td><span class="chip" :class="verdictClass(row)" :title="row.reason || ''">{{ verdictText(row) }}</span></td>
            <td class="num">{{ num(row.completion_tokens) }}</td>
            <td class="num">{{ num(row.reasoning_tokens) }}</td>
            <td class="num">{{ num(row.prompt_tokens) }}</td>
          </tr>
          <tr v-if="!data.rows.length"><td :colspan="COLUMNS.length" class="empty">No slots match these filters.</td></tr>
        </tbody>
      </table>
    </div>
    <Pager v-if="data" :total="data.total" :page="data.page" :page-size="data.page_size" @page="page" />
  </div>
</template>
