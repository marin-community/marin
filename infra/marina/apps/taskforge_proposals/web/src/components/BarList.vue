<script setup>
// Horizontal bars with direct value labels: one series, so one color unless a row names its own class.
import { computed } from "vue";

import { hideTip, showTip } from "../lib.js";

const props = defineProps({
  rows: Array, // [{ key, label, value, text, note, cls, tip: [] }]
  max: { type: Number, default: null },
});
const top = computed(() => props.max ?? Math.max(1, ...props.rows.map((r) => r.value || 0)));
</script>

<template>
  <div class="barlist">
    <div v-for="row in rows" :key="row.key" class="row" @mousemove="showTip($event, row.label, row.tip || [row.text])" @mouseleave="hideTip">
      <span class="label" :title="row.label"><span v-if="row.cls" class="swatch" :class="row.cls"></span>{{ row.label }}</span>
      <span class="track"><i class="bar" :class="row.cls" :style="{ width: `${(100 * (row.value || 0)) / top}%` }"></i></span>
      <span class="val">{{ row.text }}<span v-if="row.note" class="muted">{{ row.note }}</span></span>
    </div>
  </div>
</template>
