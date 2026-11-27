<script setup>
import { computed, ref, watch } from "vue";

import { api, compact, duration, encodeId, go, href, num } from "../lib.js";
import Pager from "./Pager.vue";

const props = defineProps({ query: Object, meta: Object });
const data = ref(null);
const error = ref("");
const text = ref(props.query.q || "");
const COLUMNS = [
  { key: "id", text: "Capability" }, { key: "domain", text: "Domain" }, { key: "status", text: "Status" },
  { key: "accepted", text: "Accepted", num: true }, { key: "rejected", text: "Rejected", num: true },
  { key: "cut_off", text: "Cut off", num: true }, { key: "completion_tokens", text: "Completion tokens", num: true },
  { key: "wall_time", text: "Wall time", num: true },
];
const sort = computed(() => props.query.sort || "id");
const dir = computed(() => props.query.dir || "asc");
const state = (extra) => ({ q: props.query.q, domain: props.query.domain, status: props.query.status, sort: props.query.sort, dir: props.query.dir, page_size: props.query.page_size, ...extra });

// Only the latest request may update the table; an earlier one can resolve after it.
let serial = 0;
watch(() => props.query, async (query) => {
  const mine = ++serial;
  try {
    const result = await api("capabilities", { q: query.q, domain: query.domain, status: query.status, sort: sort.value, dir: dir.value, page: query.page || 1, page_size: query.page_size || 50 });
    if (mine === serial) { data.value = result; error.value = ""; }
  } catch (e) { if (mine === serial) error.value = e.message; }
}, { immediate: true, deep: true });

function sortBy(key) {
  go("/capabilities", state({ sort: key, dir: sort.value === key && dir.value === "asc" ? "desc" : sort.value === key ? "asc" : key === "id" || key === "domain" ? "asc" : "desc", page: 1 }));
}
</script>

<template>
  <h1>Capabilities</h1>
  <p class="lede">The 1,999 catalog capabilities, each planned into 25 slots.</p>
  <div class="section card">
    <div class="filters">
      <input v-model="text" class="search" type="search" placeholder="Filter by id, name or outcome" @keydown.enter="go('/capabilities', state({ q: text.trim() || null, page: 1 }))" />
      <select :value="(query.domain || [])[0] || ''" @change="go('/capabilities', state({ domain: $event.target.value ? [$event.target.value] : null, page: 1 }))">
        <option value="">All domains</option>
        <option v-for="d in meta.domains" :key="d.subject_id" :value="d.subject_id">{{ d.subject_id }} {{ d.domain }} ({{ d.capabilities }})</option>
      </select>
      <select :value="(query.status || [])[0] || ''" @change="go('/capabilities', state({ status: $event.target.value ? [$event.target.value] : null, page: 1 }))">
        <option value="">Any status</option>
        <option v-for="s in ['done', 'failed', 'planned', 'pending']" :key="s" :value="s">{{ s }}</option>
      </select>
    </div>
  </div>
  <div class="section">
    <div v-if="error" class="error">{{ error }}</div>
    <template v-else-if="data">
      <h2 style="margin-bottom: 12px">{{ num(data.total) }} capabilities</h2>
      <div class="card flush" style="overflow-x: auto">
        <table class="items">
          <thead><tr><th v-for="c in COLUMNS" :key="c.key" class="sortable" :class="{ num: c.num, sorted: sort === c.key }" @click="sortBy(c.key)">{{ c.text }}<span v-if="sort === c.key">{{ dir === "asc" ? " ↑" : " ↓" }}</span></th></tr></thead>
          <tbody>
            <tr v-for="row in data.rows" :key="row.id" @click="go(`/c/${encodeId(row.id)}`)">
              <td><a class="rowlink" :href="href(`/c/${encodeId(row.id)}`)" @click.stop><div class="title clip">{{ row.name }}</div><div class="id clip">{{ row.id }}</div></a></td>
              <td class="nowrap">{{ row.subject_id }} <span class="sub">{{ row.domain }}</span></td>
              <td><span class="chip" :class="{ done: 'good', failed: 'bad' }[row.status] || 'warn'">{{ row.status }}</span></td>
              <td class="num">{{ num(row.accepted) }}</td><td class="num">{{ num(row.rejected) }}</td><td class="num">{{ num(row.cut_off) }}</td>
              <td class="num">{{ compact(row.completion_tokens) }}</td><td class="num">{{ duration(row.wall_time) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <Pager :total="data.total" :page="data.page" :page-size="data.page_size" @page="(n, size) => go('/capabilities', state({ page: n, page_size: size }))" />
    </template>
  </div>
</template>
