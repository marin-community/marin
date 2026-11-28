<script setup>
import { computed, onMounted, ref, watch } from "vue";

import { useJson } from "./api.js";
import { assignColors, colorClass } from "./palette.js";
import { clamp, floorToStep, fmtEastern, fmtEasternClock, fmtUtc, pickerToMs, pickerValue } from "./time.js";
import ClusterSection from "./components/ClusterSection.vue";
import HoverCard from "./components/HoverCard.vue";

const HOUR_MS = 60 * 60 * 1000;

const coverage = useJson(() => "api/coverage");
const startMs = ref(null);
const hover = ref(null);

// The slider's right end is "now": the live snapshot read from the controllers.
// Everything to its left is history from the Finelog mirror.
const sliderMax = computed(() => {
  const range = coverage.data.value;
  if (range?.now_ms == null) return 0;
  return floorToStep(range.now_ms, range.step_ms);
});

const isNow = computed(() => startMs.value != null && startMs.value >= sliderMax.value);

const history = useJson(() => (isNow.value || startMs.value == null ? null : `api/occupancy?start_ms=${startMs.value}`));
const liveNow = useJson(() => (isNow.value ? "api/now" : null));

const view = computed(() => (isNow.value ? liveNow : history));

onMounted(async () => {
  await coverage.refresh();
  const range = coverage.data.value;
  if (range?.latest_ms != null) {
    startMs.value = sliderMax.value;
    refreshView();
  }
});

// Slider drags fire faster than the hub should answer; fetch the last position.
let debounce = 0;
watch(startMs, () => {
  if (startMs.value == null) return;
  clearTimeout(debounce);
  debounce = setTimeout(refreshView, 150);
});

function refreshView() {
  if (isNow.value) {
    liveNow.refresh();
  } else {
    history.refresh();
  }
}

const colors = computed(() => assignColors((view.value.data.value?.legend ?? []).map((entry) => entry.user_id)));

const windowLabel = computed(() => {
  if (isNow.value) {
    const asOf = liveNow.data.value?.as_of_ms;
    return asOf == null ? "Now" : `Now — as of ${fmtEasternClock(asOf)}`;
  }
  const body = history.data.value;
  if (!body) return "";
  return `${fmtEasternClock(body.window.start_ms)} – ${fmtEasternClock(body.window.end_ms)}`;
});

const windowUtcTitle = computed(() => {
  if (isNow.value) {
    const asOf = liveNow.data.value?.as_of_ms;
    return asOf == null ? "" : `Live read ${fmtUtc(asOf)}`;
  }
  const body = history.data.value;
  if (!body) return "";
  return `${fmtUtc(body.window.start_ms)} – ${fmtUtc(body.window.end_ms)}`;
});

const freshness = computed(() => {
  if (isNow.value) {
    const asOf = liveNow.data.value?.as_of_ms;
    return asOf == null ? "Live" : `Live from the controllers — ${fmtEastern(asOf)}`;
  }
  const range = coverage.data.value;
  return range?.latest_ms == null ? null : `Data through ${fmtEastern(range.latest_ms)}`;
});

const freshnessTitle = computed(() => {
  if (isNow.value) {
    const asOf = liveNow.data.value?.as_of_ms;
    return asOf == null ? "" : `Live read ${fmtUtc(asOf)}`;
  }
  const range = coverage.data.value;
  return range?.latest_ms == null ? "" : `Latest mirror write ${fmtUtc(range.latest_ms)}`;
});

const picker = computed({
  get: () => (startMs.value == null ? "" : pickerValue(startMs.value)),
  set: (value) => {
    const ms = pickerToMs(value);
    const range = coverage.data.value;
    if (ms == null || range?.latest_ms == null) return;
    const floored = floorToStep(ms, range.step_ms);
    startMs.value = clamp(floored, range.earliest_ms, sliderMax.value);
  },
});

function sliderBounds() {
  const range = coverage.data.value;
  return {
    min: range?.earliest_ms ?? 0,
    max: sliderMax.value,
    step: range?.step_ms ?? 1,
  };
}

const legendUnits = computed(() => (isNow.value ? "GPUs" : "GPU-h"));

const legendTotal = computed(() => {
  const entries = view.value.data.value?.legend ?? [];
  const total = entries.reduce((sum, entry) => sum + (entry.gpu_hours ?? entry.gpus ?? 0), 0);
  const rounded = Math.round(total * 100) / 100;
  return isNow.value ? total : rounded;
});

const legendTotalLabel = computed(() =>
  isNow.value ? `${legendTotal.value} GPUs held across the fleet, right now` : `${legendTotal.value} GPU-h total in the window`,
);
</script>

<template>
  <main>
    <header class="head">
      <h1>Cluster occupancy</h1>
      <span v-if="freshness" class="freshness" :title="freshnessTitle">{{ freshness }}</span>
    </header>

    <div class="controls">
      <label class="picker-label">
        <span>Window start (Eastern)</span>
        <input v-model="picker" type="datetime-local" step="300" aria-label="Window start, US Eastern" />
      </label>
      <div class="slider-block">
        <div class="window-label">
          {{ windowLabel }}
          <span class="window-title" :title="windowUtcTitle">· {{ isNow ? "live" : "window in US Eastern, UTC on hover" }}</span>
        </div>
        <input
          v-if="coverage.data.value"
          v-model.number="startMs"
          type="range"
          class="slider"
          aria-label="Window start time"
          v-bind="sliderBounds()"
        />
        <div v-if="view.loading.value" class="status">Loading…</div>
        <div v-else-if="view.error.value" class="status error">
          {{ view.error.value }}
          <button type="button" @click="refreshView()">Retry</button>
        </div>
      </div>
    </div>

    <div v-if="view.data.value?.legend?.length" class="legend" aria-label="GPU holdings in the visible view">
      <div v-for="entry in view.data.value.legend" :key="entry.user_id" class="legend-item">
        <span class="swatch" :class="colorClass(entry.user_id, colors)"></span>
        <span class="legend-user">{{ entry.user_id }}</span>
        <span class="legend-hours">{{ entry.gpu_hours ?? entry.gpus }} {{ legendUnits }}</span>
      </div>
      <div class="legend-total">{{ legendTotalLabel }}</div>
    </div>

    <ClusterSection
      v-for="cluster in view.data.value?.clusters ?? []"
      :key="cluster.cluster"
      :cluster="cluster"
      :color-class="(user) => colorClass(user, colors)"
      :hover="(value) => (hover = value)"
    />

    <HoverCard :hover="hover" :color-class="(user) => colorClass(user, colors)" />
  </main>
</template>
