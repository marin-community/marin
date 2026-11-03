<script setup>
import { computed, ref } from "vue";

import { api, compact, duration, encodeId, href, label, num, verdictClass, verdictText } from "../lib.js";
import JsonPanel from "./JsonPanel.vue";
import Markdown from "./Markdown.vue";

const props = defineProps({ capabilityId: String });
const data = ref(null);
const error = ref("");
const tab = ref("plan");
const base = `capabilities/${encodeId(props.capabilityId)}`;
api(base).then((value) => { data.value = value; }).catch((e) => { error.value = e.message; });

const c = computed(() => data.value?.capability);
const plan = computed(() => c.value?.plan || {});
const entry = computed(() => c.value?.catalog_entry || {});
const statusClass = computed(() => ({ done: "good", failed: "bad" })[c.value?.status] || "warn");
const raws = computed(() => [
  ["plan", "plan.json", c.value?.plan],
  ["result", "result.json", c.value?.result],
  ["failure", "failure.json", c.value?.failure],
  ["catalog", "catalog entry", c.value?.catalog_entry],
].filter(([, , value]) => value != null));
const tiles = computed(() => {
  const v = c.value;
  return [
    ["Accepted", num(v.accepted), `of ${num(v.slots_planned)} slots`],
    ["Rejected", num(v.rejected), ""],
    ["Cut off", num(v.cut_off), ""],
    ["Requests", num(v.requests), "plan, slots and repairs"],
    ["Completion tokens", compact(v.completion_tokens), `${compact(v.reasoning_tokens)} reasoning`],
    ["Prompt tokens", compact(v.prompt_tokens), `${compact(v.cached_tokens)} cached`],
    ["Wall time", duration(v.wall_time), v.git_sha ? `git ${v.git_sha.slice(0, 10)}` : ""],
  ];
});
</script>

<template>
  <div v-if="error" class="error">{{ error }}</div>
  <div v-else-if="!c" class="loading">Loading the capability…</div>
  <template v-else>
    <nav class="crumbs small" style="margin-bottom: 8px">
      <a :href="href('/capabilities')">Capabilities</a><span class="sep">/</span>
      <a :href="href('/capabilities', { domain: [c.subject_id] })">{{ c.subject_id }} {{ c.domain }}</a><span class="sep">/</span>
      <span class="here">{{ c.id }}</span>
    </nav>
    <div class="page-head">
      <div style="flex: 1; min-width: 0">
        <h1>{{ c.name || c.id }}</h1>
        <div class="id">{{ c.id }} · {{ c.section_path }}</div>
        <p class="lede">{{ c.outcome }}</p>
      </div>
      <div class="chips">
        <span class="chip" :class="statusClass">{{ c.status }}</span>
        <a class="small" :href="href('/proposals', { capability: c.id, sort: 'slot' })">Slots in the table →</a>
        <a class="small" :href="href('/stats', { capability: c.id })">Stats →</a>
      </div>
    </div>

    <div class="section grid tiles">
      <div v-for="[name, value, note] in tiles" :key="name" class="tile"><div class="label">{{ name }}</div><div class="value">{{ value }}</div><div class="note">{{ note }}</div></div>
    </div>

    <div v-if="c.failure && !c.result" class="section error">This capability failed and has no result: {{ c.failure.error || c.failure.kind }}. Its finished slots are listed below.</div>
    <div v-else-if="c.failure" class="section muted small">An earlier attempt failed ({{ c.failure.error || c.failure.kind }}); a rerun produced the result.json shown here, which is authoritative. The old failure.json is kept as history in the failure.json tab below.</div>

    <div class="section">
      <div class="section-head"><h2>Slots</h2><span class="sub">the plan's assignment and what each slot produced</span></div>
      <div class="slot-grid">
        <a v-for="s in data.slots" :key="s.slot" class="card slot-card" :href="href(`/p/${encodeId(c.id)}/${s.slot}`)">
          <div class="top"><span class="muted small">Slot {{ s.slot }}</span><span class="chip" :class="verdictClass(s)">{{ verdictText(s) }}</span></div>
          <div class="t">{{ s.title || "(untitled)" }}</div>
          <div class="chips" style="margin-top: 6px">
            <span class="chip outline"><span class="swatch" :class="`kind-${s.verification}`"></span>{{ label(s.verification) }}</span>
            <span class="chip outline">{{ label(s.environment) }}</span>
          </div>
          <div class="facts">
            <span>{{ compact(s.completion_tokens) }} out</span><span>{{ compact(s.reasoning_tokens) }} reasoning</span>
            <span v-if="s.completions > 1">{{ s.completions - 1 }} repair{{ s.completions > 2 ? "s" : "" }}</span>
          </div>
          <div v-if="s.reason" class="small muted" style="margin-top: 4px">{{ s.reason }}</div>
        </a>
      </div>
    </div>

    <div class="section split">
      <div class="stack">
        <div class="card">
          <div class="card-head"><h3>Coverage rationale</h3><span class="muted small">from plan.json</span></div>
          <Markdown :text="c.coverage_rationale || '_No plan._'" />
        </div>
        <div v-if="(plan.excluded_combinations || []).length" class="card">
          <div class="card-head"><h3>Excluded combinations ({{ plan.excluded_combinations.length }})</h3></div>
          <table class="items">
            <thead><tr><th>Environment</th><th>Verification</th><th>Reason</th></tr></thead>
            <tbody><tr v-for="(x, i) in plan.excluded_combinations" :key="i" style="cursor: default"><td>{{ label(x.environment) }}</td><td>{{ label(x.verification) }}</td><td class="small">{{ x.reason }}</td></tr></tbody>
          </table>
        </div>
        <div v-if="(plan.research_priorities || []).length" class="card">
          <div class="card-head"><h3>Research priorities</h3></div>
          <ul class="prose small"><li v-for="(r, i) in plan.research_priorities" :key="i">{{ typeof r === "string" ? r : JSON.stringify(r) }}</li></ul>
        </div>
        <div class="card">
          <div class="tabs">
            <button v-for="[key, name] in raws" :key="key" :class="{ on: tab === key }" @click="tab = key">{{ name }}</button>
          </div>
          <template v-for="[key, name, value] in raws" :key="key">
            <JsonPanel v-if="tab === key" :value="value" :title="name" :download="`api/${base}/raw/${key}?download=true`" />
          </template>
        </div>
      </div>
      <div class="stack">
        <div class="card">
          <div class="card-head"><h3>Result</h3></div>
          <dl v-if="c.result" class="kv">
            <dt>Requests</dt><dd>{{ num(c.result.requests) }}</dd>
            <template v-for="(v, k) in c.result.usage || {}" :key="k"><dt>{{ String(k).replace("_", " ") }}</dt><dd>{{ num(v) }}</dd></template>
            <dt>Plan call</dt><dd>{{ num(c.plan_completion_tokens) }} out · {{ num(c.plan_reasoning_tokens) }} reasoning</dd>
            <dt>Wall time</dt><dd>{{ duration(c.result.wall_time) }}</dd>
            <template v-for="(v, k) in c.result.provenance || {}" :key="k"><dt>{{ k }}</dt><dd class="mono small" style="overflow-wrap: anywhere">{{ Array.isArray(v) ? v.join(", ") : v }}</dd></template>
          </dl>
          <p v-else class="muted">No result.json yet.</p>
        </div>
        <div class="card">
          <div class="card-head"><h3>Catalog entry</h3></div>
          <dl class="kv">
            <dt>Domain</dt><dd>{{ c.subject_id }} {{ c.domain }}</dd>
            <dt>Area</dt><dd>{{ c.area }}</dd>
          </dl>
          <details v-if="(entry.includes || []).length" class="fold" open style="margin-top: 10px"><summary>Includes</summary><div class="fold-body"><ul class="prose small"><li v-for="x in entry.includes" :key="x">{{ x }}</li></ul></div></details>
          <details v-if="(entry.excludes || []).length" class="fold"><summary>Excludes</summary><div class="fold-body"><ul class="prose small"><li v-for="x in entry.excludes" :key="x">{{ x }}</li></ul></div></details>
          <details v-if="(entry.sample_tasks || []).length" class="fold"><summary>Sample tasks</summary><div class="fold-body"><ul class="prose small"><li v-for="(t, i) in entry.sample_tasks" :key="i"><span class="chip outline">{{ t.kind }}</span> {{ t.instruction }}</li></ul></div></details>
        </div>
        <div class="card small" style="overflow-wrap: anywhere">
          <div class="muted">Plan request: <span class="mono">{{ c.plan_request_path }}</span></div>
          <div class="muted" style="margin-top: 4px">Plan completions: <span class="mono">{{ c.plan_completions_path }}</span></div>
        </div>
      </div>
    </div>
  </template>
</template>
