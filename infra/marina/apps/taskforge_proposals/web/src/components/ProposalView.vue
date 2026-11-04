<script setup>
import { computed, ref } from "vue";

import { api, copyText, duration, encodeId, href, label, num, verdictClass, verdictText } from "../lib.js";
import JsonPanel from "./JsonPanel.vue";
import Markdown from "./Markdown.vue";

const props = defineProps({ capabilityId: String, slotNumber: Number });
const data = ref(null);
const error = ref("");
const tab = ref("document");
const base = `proposals/${encodeId(props.capabilityId)}/${props.slotNumber}`;

api(base).then((value) => { data.value = value; }).catch((e) => { error.value = e.message; });

const p = computed(() => data.value?.proposal);
const header = computed(() => p.value?.proposal?.header || null);
const checks = computed(() => p.value?.verdict_record?.structural || []);
const facts = computed(() => {
  const h = header.value;
  if (!h) return [];
  const out = [
    ["Proposal id", h.id],
    ["Source", h.source ? `${h.source.kind} ${h.source.ref}` : "—"],
    ["Environment", label(h.environment)],
    ["Solver machine", h.solver_machine ?? "—"],
    ["Verification", label(h.verification)],
    ["Grounding", h.grounding],
    ["Research", label(h.research_need)],
    ["Research reason", h.research_reason],
  ];
  if ((h.composite || []).length) out.push(["Composite", h.composite.map((c) => `${label(c.kind)} ×${c.weight}`).join(", ")]);
  if (h.null_reason) out.push(["Null reason", h.null_reason]);
  return out;
});
const planned = computed(() => {
  const r = p.value;
  if (!r) return [];
  return [
    ["Planned kind", label(r.planned_verification), r.planned_verification !== r.verification && r.verification],
    ["Planned environment", label(r.planned_environment), r.planned_environment !== r.environment && r.environment],
    ["Plan status", r.plan_status ?? "—"],
  ];
});
const graderSection = computed(() => (p.value?.sections || []).find((s) => /grader/i.test(s.heading)));
const siblingClass = (s) => (s.verdict || (s.cut_off ? "cut_off" : ""));
</script>

<template>
  <div v-if="error" class="error">{{ error }}</div>
  <div v-else-if="!p" class="loading">Loading the proposal…</div>
  <template v-else>
    <nav class="crumbs small" style="margin-bottom: 8px">
      <a :href="href('/proposals')">Proposals</a><span class="sep">/</span>
      <a :href="href('/proposals', { domain: [p.subject_id] })">{{ p.subject_id }} {{ p.domain }}</a><span class="sep">/</span>
      <a :href="href(`/c/${encodeId(p.capability_id)}`)">{{ p.capability_id }}</a><span class="sep">/</span>
      <span class="here">slot {{ p.slot }}</span>
    </nav>
    <div class="page-head">
      <div style="min-width: 0; flex: 1">
        <h1>{{ p.title || `Slot ${p.slot}` }}</h1>
        <p class="lede" v-if="p.summary">{{ p.summary }}</p>
        <div class="chips" style="margin-top: 10px">
          <span class="chip" :class="verdictClass(p)">{{ verdictText(p) }}</span>
          <span class="chip outline"><span class="swatch" :class="`kind-${p.verification}`"></span>{{ label(p.verification) }}</span>
          <span class="chip outline">{{ label(p.environment) }}</span>
          <span v-if="p.solver_machine" class="chip outline">machine: {{ p.solver_machine }}</span>
          <span v-if="p.superseded" class="chip warn" title="proposal.json predates the slot's last request">superseded proposal</span>
          <span v-if="p.completions > 1" class="chip warn" title="The document was repaired after failing to parse">{{ p.completions - 1 }} repair{{ p.completions > 2 ? "s" : "" }}</span>
        </div>
      </div>
      <a class="small" :href="href(`/c/${encodeId(p.capability_id)}`)">Capability {{ data.capability?.name || p.capability_id }} →</a>
    </div>

    <div class="section">
      <div class="section-head"><h3>Slots of this capability</h3><span class="sub">green accepted, red rejected, amber cut off</span></div>
      <div class="slot-strip">
        <a v-for="s in data.siblings" :key="s.slot" :class="[siblingClass(s), { here: s.slot === p.slot }]" :href="href(`/p/${encodeId(p.capability_id)}/${s.slot}`)"
          :title="`${s.slot}: ${s.title || ''} · ${label(s.verification)} · ${verdictText(s)}`">{{ s.slot }}</a>
      </div>
    </div>

    <div class="section split">
      <div>
        <div class="tabs">
          <button :class="{ on: tab === 'document' }" @click="tab = 'document'">Proposal</button>
          <button :class="{ on: tab === 'proposal' }" :disabled="!p.proposal" @click="tab = 'proposal'">proposal.json</button>
          <button :class="{ on: tab === 'verdict' }" :disabled="!p.verdict_record" @click="tab = 'verdict'">verdict.json</button>
        </div>
        <div v-if="p.superseded" class="error" style="margin-bottom: 12px">
          This slot was rerun and ended <strong>{{ verdictText(p) }}</strong>. The proposal and verdict below are an
          earlier attempt's files, older than the slot's last request and its plan, kept for reference only.
        </div>
        <template v-if="tab === 'document'">
          <div v-if="!p.proposal" class="card">
            <h3>No proposal document</h3>
            <p class="muted">This slot ended as <strong>{{ verdictText(p) }}</strong>{{ p.failure ? `: ${p.failure}` : "." }}</p>
          </div>
          <div v-for="(section, i) in p.sections" :key="i" class="card section-card" :style="{ marginTop: i ? '12px' : 0 }">
            <h2 v-if="section.heading">{{ section.heading }}</h2>
            <Markdown :text="section.text" />
          </div>
        </template>
        <JsonPanel v-else-if="tab === 'proposal'" :value="p.proposal" :title="`slot-${String(p.slot).padStart(2, '0')}.proposal.json`" :download="`api/${base}/raw/proposal?download=true`" />
        <JsonPanel v-else :value="p.verdict_record" :title="`slot-${String(p.slot).padStart(2, '0')}.verdict.json`" :download="`api/${base}/raw/verdict?download=true`" />
      </div>

      <div class="stack">
        <div class="card">
          <div class="card-head"><h3>Verdict</h3></div>
          <div class="decision" v-if="p.verdict">
            <div class="badge" :class="p.verdict">{{ p.verdict === "accept" ? "✓" : "✕" }}</div>
            <div>
              <div class="verdict">{{ p.verdict === "accept" ? "Accepted" : "Rejected" }} by the preflight</div>
              <ul v-if="(p.verdict_record?.reasons || []).length" class="small" style="margin: 4px 0 0; padding-left: 18px">
                <li v-for="r in p.verdict_record.reasons" :key="r">{{ r }}</li>
              </ul>
            </div>
          </div>
          <p v-else class="muted">No verdict: the slot is {{ verdictText(p) }}.</p>
          <ul v-if="checks.length" class="checklist" style="margin-top: 10px">
            <li v-for="c in checks" :key="c.name">
              <span class="mark" :class="c.status">{{ c.status === "pass" ? "✓" : c.status === "fail" ? "✕" : "–" }}</span>
              <div><div class="name">{{ c.name }} <span class="muted small">{{ c.severity }}</span></div><div class="why">{{ c.reason }}</div></div>
            </li>
          </ul>
        </div>

        <div class="card">
          <div class="card-head"><h3>Plan for this slot</h3></div>
          <dl class="kv">
            <template v-if="p.workflow"><dt>Workflow</dt><dd>{{ p.workflow }}</dd></template>
            <template v-if="p.challenge"><dt>Challenge</dt><dd>{{ p.challenge }}</dd></template>
            <template v-for="[k, v, changed] in planned" :key="k">
              <dt>{{ k }}</dt><dd>{{ v }}<span v-if="changed" class="chip warn" style="margin-left: 6px">proposal: {{ label(changed) }}</span></dd>
            </template>
          </dl>
        </div>

        <div v-if="header" class="card">
          <div class="card-head"><h3>Header</h3></div>
          <dl class="kv"><template v-for="[k, v] in facts" :key="k"><dt>{{ k }}</dt><dd>{{ v ?? "—" }}</dd></template></dl>
          <details v-if="(header.build || []).length" class="fold" open style="margin-top: 10px">
            <summary>Build items ({{ header.build.length }})</summary>
            <div class="fold-body"><ul class="prose small"><li v-for="b in header.build" :key="b.name"><b>{{ b.name }}</b>: {{ b.description }}</li></ul></div>
          </details>
          <details v-if="(header.research || []).length" class="fold" style="margin-top: 8px">
            <summary>Research items ({{ header.research.length }})</summary>
            <div class="fold-body"><ul class="prose small"><li v-for="(r, i) in header.research" :key="i"><span class="chip outline">{{ r.kind }}</span> {{ r.purpose }}</li></ul></div>
          </details>
          <div v-if="(header.resources || []).length" class="chips" style="margin-top: 10px">
            <span class="muted small">Resources</span><span v-for="r in header.resources" :key="r" class="chip outline mono">{{ r }}</span>
          </div>
        </div>

        <div v-if="graderSection" class="card">
          <div class="card-head"><h3>Grader</h3><span class="muted small">from “{{ graderSection.heading }}”</span></div>
          <Markdown class="small" :text="graderSection.text.split('\n\n')[0]" />
        </div>

        <div class="card">
          <div class="card-head"><h3>Generation</h3></div>
          <dl class="kv">
            <dt>Completions</dt><dd>{{ num(p.completions) }}</dd>
            <dt>Prompt tokens</dt><dd>{{ num(p.prompt_tokens) }} <span class="muted">({{ num(p.cached_tokens) }} cached)</span></dd>
            <dt>Completion tokens</dt><dd>{{ num(p.completion_tokens) }}</dd>
            <dt>Reasoning tokens</dt><dd>{{ num(p.reasoning_tokens) }}</dd>
            <dt>Finish</dt><dd>{{ (p.finish_reasons || []).join(", ") || "—" }}</dd>
            <dt>Wall time</dt><dd>{{ duration(p.wall_time) }}</dd>
            <dt>Body</dt><dd>{{ num(p.body_chars) }} characters</dd>
          </dl>
          <div class="stack small" style="margin-top: 10px">
            <div v-for="[name, path] in [['Request', p.request_path], ['Completions', p.completions_path]]" :key="name" v-show="path" class="row">
              <span class="muted">{{ name }}</span><span class="mono clip" style="flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" :title="path">{{ path }}</span>
              <button class="ghost copy" @click="copyText(path, 'Copied the path')">⧉</button>
            </div>
            <div v-if="p.completions_bytes" class="muted">completions file {{ (p.completions_bytes / 1e6).toFixed(1) }} MB, not loaded here</div>
          </div>
        </div>
      </div>
    </div>
  </template>
</template>
