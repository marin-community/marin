<script setup>
import { computed } from "vue";

import { fmtEastern, fmtUtc } from "../time.js";

const props = defineProps({
  hover: { type: Object, default: null },
  colorClass: { type: Function, required: true },
});

const cardWidth = 430;
const cardHeight = 290;

const position = computed(() => {
  if (!props.hover) return {};
  const x = Math.min(props.hover.x + 14, window.innerWidth - cardWidth - 8);
  const y = Math.min(props.hover.y + 14, window.innerHeight - cardHeight - 8);
  return { left: `${Math.max(x, 8)}px`, top: `${Math.max(y, 8)}px` };
});

const band = computed(() => {
  const value = props.hover?.holder?.priority_band;
  return value ? value.replace(/^PRIORITY_BAND_/, "").toLowerCase() : "—";
});

const holderRows = computed(() => {
  const holder = props.hover?.holder;
  if (!holder) return null;
  return [
    { label: "Priority", value: band.value },
    { label: "GPUs held", value: `${holder.gpus} on ${props.hover.node}` },
    { label: "GPU type", value: holder.accelerator_type || "—" },
    {
      label: "Started",
      value: fmtEastern(holder.started_ms),
      utc: fmtUtc(holder.started_ms),
    },
    holder.finished_ms == null
      ? { label: "Running", value: "still holding the node" }
      : { label: "Ended", value: fmtEastern(holder.finished_ms), utc: fmtUtc(holder.finished_ms) },
  ];
});
</script>

<template>
  <div v-if="hover" class="hover-card" :style="position" role="status">
    <template v-if="hover.idle">
      <div class="hc-title">Idle</div>
      <div class="hc-row">
        <span>{{ hover.node }} holds no GPUs in this window.</span>
      </div>
    </template>
    <template v-else>
      <div class="hc-title">
        <span class="swatch" :class="colorClass(hover.holder.user_id)"></span>
        <span>{{ hover.holder.user_id }}</span>
      </div>
      <div class="hc-job">{{ hover.holder.root_name || hover.holder.job_id }}</div>
      <div v-for="row in holderRows" :key="row.label" class="hc-row">
        <span class="hc-label">{{ row.label }}</span>
        <span class="hc-value">
          {{ row.value }}
          <span v-if="row.utc" class="hc-utc">{{ row.utc }}</span>
        </span>
      </div>
    </template>
  </div>
</template>
