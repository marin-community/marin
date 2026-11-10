<script setup>
// Search, multi-select facets and token ranges; every change becomes the URL's query.
import { computed, ref, watch } from "vue";

import { LIST_FILTERS, RANGE_FILTERS, filtersOf, go, label } from "../lib.js";

const props = defineProps({ query: Object, meta: Object, path: String });
const text = ref(props.query.q || "");
watch(() => props.query.q, (q) => { text.value = q || ""; });

const facets = computed(() => ({
  domain: props.meta.domains.map((d) => ({ value: d.subject_id, text: `${d.subject_id} ${d.domain}`, n: d.slots })),
  kind: props.meta.kinds.map((k) => ({ value: k.key ?? "none", text: label(k.key ?? "none"), n: k.n })),
  environment: props.meta.environments.map((k) => ({ value: k.key ?? "none", text: label(k.key ?? "none"), n: k.n })),
  verdict: ["accept", "reject", "none"].map((v) => ({ value: v, text: label(v) })),
  status: ["proposed", "cut_off", "failed", "pending"].map((v) => ({ value: v, text: label(v) })),
}));
const TITLES = { domain: "Domain", kind: "Kind", environment: "Environment", verdict: "Verdict", status: "Slot status" };

function update(changes) {
  const next = { ...filtersOf(props.query), ...changes };
  for (const [key, value] of Object.entries(next)) if (value == null || value === "" || (Array.isArray(value) && !value.length)) delete next[key];
  const keep = props.query.sort ? { sort: props.query.sort, dir: props.query.dir } : {};
  if (changes.q && !props.query.sort) Object.assign(keep, { sort: "rank", dir: "desc" });
  go(props.path, { ...next, ...keep, page_size: props.query.page_size });
}

function toggle(facet, value) {
  const current = new Set(props.query[facet] || []);
  current.has(value) ? current.delete(value) : current.add(value);
  update({ [facet]: [...current] });
}

const ranges = ref({});
watch(() => props.query, (query) => {
  ranges.value = Object.fromEntries(RANGE_FILTERS.flatMap((n) => [[`${n}_min`, query[`${n}_min`] || ""], [`${n}_max`, query[`${n}_max`] || ""]]));
}, { immediate: true });

const active = computed(() => {
  const out = [];
  if (props.query.q) out.push({ text: `“${props.query.q}”`, clear: { q: null } });
  if (props.query.capability) out.push({ text: `Capability: ${props.query.capability}`, clear: { capability: null } });
  for (const facet of LIST_FILTERS) {
    for (const value of props.query[facet] || []) {
      const option = facets.value[facet].find((o) => o.value === value);
      out.push({ text: `${TITLES[facet]}: ${option ? option.text : value}`, clear: { [facet]: (props.query[facet] || []).filter((v) => v !== value) } });
    }
  }
  if (props.query.cut_off) out.push({ text: props.query.cut_off === "true" ? "cut off" : "not cut off", clear: { cut_off: null } });
  for (const n of RANGE_FILTERS) {
    const lo = props.query[`${n}_min`], hi = props.query[`${n}_max`];
    if (lo || hi) out.push({ text: `${label(n)} ${lo || "0"}–${hi || "∞"}`, clear: { [`${n}_min`]: null, [`${n}_max`]: null } });
  }
  return out;
});

function clearAll() {
  go(props.path, props.query.page_size ? { page_size: props.query.page_size } : {});
}
</script>

<template>
  <div class="card">
    <div class="filters">
      <input v-model="text" class="search" type="search" placeholder="Search titles, summaries and sections" title="Web search syntax: &quot;exact phrase&quot;, -word, or"
        @keydown.enter="update({ q: text.trim() || null })" @search="update({ q: text.trim() || null })" />
      <details v-for="(options, facet) in facets" :key="facet" class="dropdown">
        <summary>{{ TITLES[facet] }} <span v-if="(query[facet] || []).length" class="count">{{ query[facet].length }}</span> ▾</summary>
        <div class="menu">
          <label v-for="option in options" :key="option.value">
            <input type="checkbox" :checked="(query[facet] || []).includes(option.value)" @change="toggle(facet, option.value)" />
            <span v-if="facet === 'kind'" class="swatch" :class="`kind-${option.value}`"></span>{{ option.text }}
            <span v-if="option.n != null" class="n">{{ option.n.toLocaleString() }}</span>
          </label>
        </div>
      </details>
      <select :value="query.cut_off || ''" @change="update({ cut_off: $event.target.value || null })">
        <option value="">Cut off: any</option>
        <option value="true">Cut off only</option>
        <option value="false">Not cut off</option>
      </select>
      <details class="dropdown">
        <summary>Tokens <span v-if="RANGE_FILTERS.some((n) => query[`${n}_min`] || query[`${n}_max`])" class="count">•</span> ▾</summary>
        <div class="menu">
          <form @submit.prevent="update({ ...Object.fromEntries(Object.entries(ranges).map(([k, v]) => [k, v === '' ? null : v])) })">
            <div v-for="n in RANGE_FILTERS" :key="n" class="range">
              <span>{{ label(n) }}</span>
              <input v-model="ranges[`${n}_min`]" type="number" min="0" placeholder="min" />
              <input v-model="ranges[`${n}_max`]" type="number" min="0" placeholder="max" />
            </div>
            <div class="row" style="justify-content: flex-end; padding: 6px"><button class="primary" type="submit">Apply</button></div>
          </form>
        </div>
      </details>
    </div>
    <div v-if="active.length" class="active-filters">
      <button v-for="item in active" :key="item.text" class="chip info" title="Remove this filter" @click="update(item.clear)">{{ item.text }} ✕</button>
      <button class="ghost small" @click="clearAll">Clear all</button>
    </div>
  </div>
</template>
