<script setup lang="ts">
import { onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import Plotly, { CHART_CONFIG, LINE_COLOR } from '../utils/plot'
import { ema } from '../utils/smoothing'
import type { Series } from '../composables/useMetrics'

const props = withDefaults(defineProps<{ metricKey: string; series: Series; height?: number }>(), { height: 400 })
defineEmits<{ close: []; fullscreen: [] }>()

const el = ref<HTMLElement | null>(null)
const st = reactive({ logY: false, logX: false, smoothing: 0 })

function traces() {
  const { x, y } = props.series
  if (st.smoothing > 0 && y.length > 1) {
    // raw faint underneath, smoothed bold on top (wandb style)
    return [
      { x, y, mode: 'lines', line: { width: 1, color: LINE_COLOR }, opacity: 0.22, hoverinfo: 'skip', showlegend: false },
      { x, y: ema(y, st.smoothing), mode: 'lines', line: { width: 2, color: LINE_COLOR }, name: props.metricKey },
    ]
  }
  return [{ x, y, mode: 'lines', line: { width: 1.6, color: LINE_COLOR }, name: props.metricKey }]
}

function layout() {
  const axis = (log: boolean) => ({ type: log ? 'log' : 'linear', showgrid: true, gridcolor: '#eef2f6', zeroline: false })
  return {
    title: { text: props.metricKey, font: { size: 12.5 }, x: 0.02, xanchor: 'left', y: 0.98 },
    height: props.height,
    margin: { l: 52, r: 14, t: 26, b: 32 },
    dragmode: 'pan',
    template: 'plotly_white',
    hovermode: 'x unified',
    showlegend: false,
    xaxis: axis(st.logX),
    yaxis: axis(st.logY),
  }
}

const redraw = () => el.value && Plotly.react(el.value, traces(), layout(), CHART_CONFIG)

onMounted(() => el.value && Plotly.newPlot(el.value, traces(), layout(), CHART_CONFIG))
onBeforeUnmount(() => el.value && Plotly.purge(el.value))
watch(() => props.series, redraw)
watch(() => st.smoothing, redraw)

function toggleLog(which: 'logY' | 'logX') {
  st[which] = !st[which]
  if (el.value) {
    Plotly.relayout(el.value, { [which === 'logY' ? 'yaxis.type' : 'xaxis.type']: st[which] ? 'log' : 'linear' })
  }
}
</script>

<template>
  <div class="chart-card overflow-hidden rounded-lg border border-surface-border bg-surface-raised shadow-sm">
    <div class="chart-bar flex items-center justify-end gap-2 border-b border-surface-border/70 px-2 py-1">
      <button class="cbtn" :class="{ active: st.logY }" @click="toggleLog('logY')">log y</button>
      <button class="cbtn" :class="{ active: st.logX }" @click="toggleLog('logX')">log x</button>
      <input v-model.number="st.smoothing" type="range" min="0" max="0.97" step="0.01" class="w-16" title="smoothing" />
      <button class="cbtn" title="full screen" @click="$emit('fullscreen')">⛶</button>
      <button class="cbtn close" title="remove this chart" @click="$emit('close')">×</button>
    </div>
    <div ref="el"></div>
  </div>
</template>

<style scoped>
.chart-bar {
  opacity: 0.5;
  transition: opacity 0.12s;
}
.chart-card:hover .chart-bar {
  opacity: 1;
}
.cbtn {
  font-size: 10px;
  padding: 1px 6px;
  border: 1px solid var(--c-surface-border);
  border-radius: 4px;
  background: var(--c-surface-raised);
  color: var(--c-text-muted);
  cursor: pointer;
}
.cbtn:hover {
  color: var(--c-text-secondary);
}
.cbtn.active {
  background: var(--c-accent);
  color: #fff;
  border-color: var(--c-accent);
}
.cbtn.close:hover {
  color: var(--c-status-danger);
  border-color: var(--c-status-danger);
}
</style>
