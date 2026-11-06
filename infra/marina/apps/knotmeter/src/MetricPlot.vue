<script setup lang="ts">
import { computed } from "vue";
import { type Run, modelName } from "./types";

const props = defineProps<{
  runs: Run[];
  axis: "concurrency" | "ttft" | "date";
  percentile: string;
}>();
const xValue = (r: Run) =>
  props.axis === "concurrency"
    ? r.concurrency
    : props.axis === "ttft"
      ? r.metrics.latency.ttft[props.percentile]!
      : Date.parse(`${r.date}T00:00:00Z`);
const yValue = (r: Run) => r.metrics.throughput.per_gpu.output_tput_tps;
const extent = computed(() => {
  const values = props.runs.map(xValue);
  const min = Math.min(...values),
    max = Math.max(...values);
  const pad =
    max === min
      ? props.axis === "date"
        ? 86400000
        : Math.max(1, min * 0.2)
      : (max - min) * 0.15;
  return {
    min: min - pad,
    span: max - min + pad * 2,
    top: Math.max(...props.runs.map(yValue), 0.01) * 1.25,
  };
});
const x = (r: Run) =>
  72 + ((xValue(r) - extent.value.min) * 490) / extent.value.span;
const y = (value: number) => 225 - (value * 180) / extent.value.top;
const formatX = (value: number) =>
  props.axis === "date"
    ? new Date(value).toISOString().slice(5, 10)
    : Number(value.toPrecision(3)).toString();
const axisLabel = computed(() =>
  props.axis === "concurrency"
    ? "Concurrent requests (C)"
    : props.axis === "ttft"
      ? `${props.percentile} time to first token (seconds)`
      : "Source run date (UTC)",
);
</script>
<template>
  <div class="plot-wrap">
    <svg
      v-if="runs.length"
      role="img"
      :aria-label="`Output throughput by ${axisLabel}`"
      viewBox="0 0 600 280"
      class="metric-plot"
    >
      <g v-for="n in [0, 1, 2, 3, 4]" :key="n">
        <line
          x1="72"
          x2="562"
          :y1="y((extent.top * n) / 4)"
          :y2="y((extent.top * n) / 4)"
          class="grid"
        />
        <text x="62" :y="y((extent.top * n) / 4) + 4" text-anchor="end">
          {{ ((extent.top * n) / 4).toFixed(2) }}
        </text>
      </g>
      <g v-for="n in [0, 1, 2, 3, 4]" :key="`x${n}`">
        <text :x="72 + (490 * n) / 4" y="245" text-anchor="middle">
          {{ formatX(extent.min + (extent.span * n) / 4) }}
        </text>
      </g>
      <text transform="translate(15,140) rotate(-90)" text-anchor="middle">
        Output tokens/s/GPU
      </text>
      <text x="317" y="272" text-anchor="middle">{{ axisLabel }}</text>
      <a
        v-for="run in runs"
        :key="run.key"
        :href="`#/run/${run.key}`"
        :data-run="run.key"
        :data-value="yValue(run)"
      >
        <title>
          {{ modelName(run) }} · {{ run.hardware }} · C{{ run.concurrency }} ·
          {{ run.date }} · {{ yValue(run) }} output tokens/s/GPU
        </title>
        <circle
          :cx="x(run)"
          :cy="y(yValue(run))"
          r="7"
          :fill="run.hardware === 'H100' ? '#258568' : '#a85b20'"
        />
        <text :x="x(run) + 12" :y="y(yValue(run)) - 10">
          {{ run.hardware }} · C{{ run.concurrency }}
        </text>
      </a>
    </svg>
    <p v-else class="empty">No measurements match these filters.</p>
  </div>
</template>
