<script setup>
// Domain by verification kind: cell shade is the kind's share of the domain's slots (one blue ramp).
import { computed } from "vue";

import { KIND_ORDER, hideTip, label, num, pct, showTip } from "../lib.js";

const props = defineProps({ cells: Array, domains: Object, metric: { type: String, default: "share" } });
const kinds = computed(() => {
  const present = new Set(props.cells.map((c) => c.kind ?? "none"));
  return [...KIND_ORDER.filter((k) => present.has(k)), ...[...present].filter((k) => !KIND_ORDER.includes(k))];
});
const table = computed(() => {
  const rows = {};
  for (const c of props.cells) {
    const row = (rows[c.domain] ||= { domain: c.domain, total: 0, cells: {} });
    row.cells[c.kind ?? "none"] = c;
    row.total += c.n;
  }
  return Object.values(rows).sort((a, b) => a.domain.localeCompare(b.domain));
});
const STEPS = 6;
function shade(row, kind) {
  const c = row.cells[kind];
  if (!c || !c.n) return { cls: "zero", style: {} };
  // Accept rate counts decided slots only, as elsewhere on the page; a cell with none decided is blank.
  if (props.metric === "accept" && !c.decided) return { cls: "zero", style: {} };
  const value = props.metric === "accept" ? c.accepted / c.decided : c.n / row.total;
  const step = Math.min(STEPS, Math.max(1, Math.ceil(value * STEPS)));
  return { cls: step >= 4 ? "dark" : "", style: { background: `var(--seq-${step})` } };
}
function text(row, kind) {
  const c = row.cells[kind];
  if (!c) return "·";
  if (props.metric === "accept") return c.decided ? pct(c.accepted, c.decided, 0) : "–";
  return num(c.n);
}
</script>

<template>
  <div style="overflow-x: auto">
    <table class="heat">
      <thead><tr><th></th><th v-for="k in kinds" :key="k">{{ label(k) }}</th><th>Total</th></tr></thead>
      <tbody>
        <tr v-for="row in table" :key="row.domain">
          <th :title="domains[row.domain]">{{ row.domain }} <span class="muted">{{ (domains[row.domain] || "").slice(0, 34) }}</span></th>
          <td v-for="k in kinds" :key="k" :class="shade(row, k).cls" :style="shade(row, k).style"
            @mousemove="showTip($event, `${row.domain} · ${label(k)}`, row.cells[k] ? [`${num(row.cells[k].n)} slots (${pct(row.cells[k].n, row.total)} of the domain)`, row.cells[k].decided ? `${pct(row.cells[k].accepted, row.cells[k].decided)} of decided slots accepted` : 'no decided slots'] : ['no slots'])" @mouseleave="hideTip">{{ text(row, k) }}</td>
          <td class="zero">{{ num(row.total) }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
