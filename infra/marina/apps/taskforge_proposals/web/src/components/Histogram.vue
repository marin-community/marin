<script setup>
// Column histogram of a token count with a per-bin hover tooltip and a table fallback.
import { computed } from "vue";

import { compact, hideTip, num, showTip } from "../lib.js";

const props = defineProps({ hist: Object, name: String });
const W = 400, H = 170, L = 40, B = 24, T = 8;
const bins = computed(() => props.hist?.bins || []);
const top = computed(() => Math.max(1, ...bins.value));
const ticks = computed(() => {
  const step = Math.pow(10, Math.floor(Math.log10(top.value)));
  const unit = top.value / step > 5 ? step * 2 : step;
  const out = [];
  for (let v = 0; v <= top.value; v += unit) out.push(v);
  return out;
});
const bw = computed(() => (W - L) / Math.max(1, bins.value.length));
const y = (v) => T + (H - T - B) * (1 - v / top.value);
const edge = (i) => props.hist.lo + i * props.hist.width;
const last = (i) => i === bins.value.length - 1;
const binText = (i) => (last(i) ? `${num(edge(i))}–${num(props.hist.max)} ${props.name} (includes the top 1%)` : `${num(edge(i))}–${num(edge(i + 1) - 1)} ${props.name}`);
const xLabels = computed(() => {
  const n = bins.value.length;
  return n ? [0, Math.floor(n / 2), n].map((i) => ({ x: L + i * bw.value, text: i === n ? `${compact(edge(i))}+` : compact(edge(i)) })) : [];
});
</script>

<template>
  <div class="chart">
    <div v-if="!bins.length" class="empty inline">No values in this selection.</div>
    <template v-else>
      <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="`Histogram of ${name}`">
        <g v-for="t in ticks" :key="t">
          <line class="gridline" :x1="L" :x2="W" :y1="y(t)" :y2="y(t)" />
          <text :x="L - 6" :y="y(t) + 4" text-anchor="end">{{ compact(t) }}</text>
        </g>
        <line class="axis-line" :x1="L" :x2="W" :y1="H - B" :y2="H - B" />
        <g v-for="(n, i) in bins" :key="i">
          <path v-if="n" class="col" :d="`M${L + i * bw + 1},${H - B} V${y(n) + 3} q0,-3 3,-3 H${L + (i + 1) * bw - 4} q3,0 3,3 V${H - B} Z`"
            :style="{ fill: 'var(--accent)' }" />
          <rect :x="L + i * bw" :y="T" :width="bw" :height="H - T - B" fill="transparent"
            @mousemove="showTip($event, binText(i), [`${num(n)} slots`])" @mouseleave="hideTip" />
        </g>
        <text v-for="l in xLabels" :key="l.x" :x="l.x" :y="H - 8" text-anchor="middle">{{ l.text }}</text>
      </svg>
      <details class="fold" style="margin-top: 8px">
        <summary>Table</summary>
        <div class="fold-body">
          <table class="items small"><thead><tr><th>From</th><th>To</th><th class="num">Slots</th></tr></thead>
            <tbody><tr v-for="(n, i) in bins" :key="i" style="cursor: default"><td>{{ num(edge(i)) }}</td><td>{{ last(i) ? num(hist.max) : num(edge(i + 1) - 1) }}</td><td class="num">{{ num(n) }}</td></tr></tbody>
          </table>
        </div>
      </details>
    </template>
  </div>
</template>
