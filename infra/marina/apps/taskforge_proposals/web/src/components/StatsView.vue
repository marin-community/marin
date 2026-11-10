<script setup>
import { computed, ref, watch } from "vue";

import { KIND_ORDER, api, compact, filtersOf, href, label, num, pct } from "../lib.js";
import BarList from "./BarList.vue";
import Crosstab from "./Crosstab.vue";
import FilterBar from "./FilterBar.vue";
import Histogram from "./Histogram.vue";

const props = defineProps({ query: Object, meta: Object });
const data = ref(null);
const error = ref("");
const loading = ref(false);
const crossMetric = ref("share");
const domainSort = ref("domain");
const domains = computed(() => Object.fromEntries(props.meta.domains.map((d) => [d.subject_id, d.domain])));

let serial = 0;
watch(() => props.query, async (query) => {
  const mine = ++serial;
  loading.value = true;
  try {
    const result = await api("stats", filtersOf(query));
    if (mine === serial) { data.value = result; error.value = ""; }
  } catch (e) {
    if (mine === serial) error.value = e.message;
  } finally {
    if (mine === serial) loading.value = false;
  }
}, { immediate: true, deep: true });

const t = computed(() => data.value?.totals || {});
const decided = (r) => (r.accepted || 0) + (r.rejected || 0);
const tiles = computed(() => [
  ["Slots", num(t.value.n), `${num(t.value.capabilities)} capabilities`],
  ["Accept rate", pct(t.value.accepted, decided(t.value)), `${num(t.value.accepted)} of ${num(decided(t.value))} with a verdict`],
  ["Rejected", num(t.value.rejected), pct(t.value.rejected, t.value.n) + " of slots"],
  ["Cut off", num(t.value.cut_off), pct(t.value.cut_off, t.value.n) + " of slots"],
  ["Repaired", num(t.value.repaired), "more than one completion"],
  ["Median completion", compact(t.value.median_completion_tokens), `${compact(t.value.median_reasoning_tokens)} reasoning`],
  ["Completion tokens", compact(t.value.completion_tokens), `${compact(t.value.prompt_tokens)} prompt`],
]);

function kindRows(groups, metric) {
  const order = (k) => { const i = KIND_ORDER.indexOf(k); return i < 0 ? 99 : i; };
  return [...groups].sort((a, b) => order(a.key) - order(b.key)).map((g) => rowFor(g, metric, `kind-${g.key}`, label(g.key ?? "none")));
}
function rowFor(g, metric, cls, name) {
  if (metric === "accept") {
    const d = decided(g);
    return { key: g.key ?? "none", label: name, value: d ? g.accepted / d : 0, text: pct(g.accepted, d), note: ` n=${num(d)}`, cls,
      tip: [`${num(g.accepted)} accepted, ${num(g.rejected)} rejected`, `${num(g.cut_off)} cut off`] };
  }
  return { key: g.key ?? "none", label: name, value: g.n, text: num(g.n), note: ` ${pct(g.n, t.value.n)}`, cls,
    tip: [`${num(g.n)} slots`, `${pct(g.accepted, decided(g))} accepted`] };
}
const simpleRows = (groups) => groups.map((g) => rowFor(g, "count", null, label(g.key ?? "none")));
const domainRows = computed(() => {
  const rows = [...(data.value?.by_domain || [])];
  if (domainSort.value === "count") rows.sort((a, b) => b.n - a.n);
  if (domainSort.value === "accept") rows.sort((a, b) => a.accepted / Math.max(1, decided(a)) - b.accepted / Math.max(1, decided(b)));
  return rows;
});
const label2 = (g) => `${g.key} ${(domains.value[g.key] || "").slice(0, 32)}`;
</script>

<template>
  <h1>Statistics</h1>
  <p class="lede">Computed in Postgres over every slot that matches the filters; <a :href="href('/proposals', filtersOf(query))">open them in the table</a>.</p>
  <div class="section"><FilterBar :query="query" :meta="meta" path="/stats" /></div>
  <div v-if="error" class="section error">{{ error }}</div>
  <template v-else-if="data">
    <div class="section grid tiles" :style="{ opacity: loading ? 0.6 : 1 }">
      <div v-for="[name, value, note] in tiles" :key="name" class="tile"><div class="label">{{ name }}</div><div class="value">{{ value }}</div><div class="note">{{ note }}</div></div>
    </div>

    <div class="section grid two">
      <div class="card"><div class="card-head"><h3>Slots by verification kind</h3></div><BarList :rows="kindRows(data.by_kind, 'count')" /></div>
      <div class="card"><div class="card-head"><h3>Accept rate by verification kind</h3><span class="muted small">of slots with a verdict</span></div><BarList :rows="kindRows(data.by_kind, 'accept')" :max="1" /></div>
      <div class="card"><div class="card-head"><h3>Slots by environment</h3></div><BarList :rows="simpleRows(data.by_environment)" /></div>
      <div class="card"><div class="card-head"><h3>Accept rate by environment</h3></div><BarList :rows="data.by_environment.map((g) => rowFor(g, 'accept', null, label(g.key ?? 'none')))" :max="1" /></div>
      <div class="card"><div class="card-head"><h3>Slots by verdict</h3></div><BarList :rows="simpleRows(data.by_verdict)" /></div>
      <div class="card"><div class="card-head"><h3>Slots by status</h3><span class="muted small">cut off and failed slots have no document</span></div><BarList :rows="simpleRows(data.by_status)" /></div>
    </div>

    <div class="section">
      <div class="section-head"><h2>Token distributions</h2><span class="sub">per slot, summed over its completions (repairs included)</span></div>
      <div class="grid three">
        <div v-for="name in ['completion_tokens', 'reasoning_tokens', 'prompt_tokens']" :key="name" class="card">
          <div class="card-head"><h3>{{ label(name) }}</h3><span class="muted small" v-if="data.histograms[name].lo != null">{{ num(data.histograms[name].lo) }}–{{ num(data.histograms[name].max) }}; last bin holds the top 1%</span></div>
          <Histogram :hist="data.histograms[name]" :name="label(name)" />
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-head">
        <h2>By domain</h2>
        <span class="row small">Sort <select v-model="domainSort"><option value="domain">by domain</option><option value="count">by slots</option><option value="accept">by accept rate (lowest first)</option></select></span>
      </div>
      <div class="grid two">
        <div class="card"><div class="card-head"><h3>Slots</h3></div><BarList :rows="domainRows.map((g) => rowFor(g, 'count', null, label2(g)))" /></div>
        <div class="card"><div class="card-head"><h3>Accept rate</h3></div><BarList :rows="domainRows.map((g) => rowFor(g, 'accept', null, label2(g)))" :max="1" /></div>
      </div>
    </div>

    <div class="section card">
      <div class="card-head">
        <h3>Verification kind by domain</h3>
        <span class="row small">Shade <select v-model="crossMetric"><option value="share">share of the domain's slots</option><option value="accept">accept rate</option></select></span>
      </div>
      <Crosstab :cells="data.crosstab" :domains="domains" :metric="crossMetric" />
    </div>
  </template>
  <div v-else class="loading">Computing…</div>
</template>
