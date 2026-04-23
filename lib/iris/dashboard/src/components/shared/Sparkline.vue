<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  data: number[]
  color?: string
  width?: number
  height?: number
  fillColor?: string
  /** Stretch the SVG to 100% of its container width. */
  fill?: boolean
  /** Show a max-value line and label on the right. */
  showYAxis?: boolean
  /** Label for the max tick. If omitted, the raw max value is shown. */
  yAxisTopLabel?: string
}>(), {
  color: 'var(--color-accent, #2563eb)',
  width: 64,
  height: 20,
})

const PAD = 1
/** Extra top padding reserved for the y-axis label. */
const LABEL_H = 11

const topPad = computed(() => props.showYAxis ? PAD + LABEL_H : PAD)

const dataMax = computed(() => {
  if (!props.data || props.data.length < 1) return 0
  return Math.max(...props.data)
})

const topLabel = computed(() =>
  props.yAxisTopLabel ?? dataMax.value.toFixed(1)
)

/**
 * Build the SVG polyline points string from the data array.
 * Single-value arrays are duplicated to draw a flat line.
 * When all values are zero, draws a flat line at the bottom.
 */
const points = computed(() => {
  if (!props.data || props.data.length < 1) return ''

  const data = props.data.length === 1 ? [props.data[0], props.data[0]] : props.data
  const max = dataMax.value
  const innerW = props.width - 2 * PAD
  const innerH = props.height - topPad.value - PAD

  return data.map((v, i) => {
    const x = PAD + (i / (data.length - 1)) * innerW
    const y = max === 0
      ? topPad.value + innerH
      : topPad.value + innerH - (Math.min(v, max) / max) * innerH
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
})

/** Area fill polygon: the line points plus closing along the bottom edge. */
const areaPoints = computed(() => {
  if (!points.value) return ''
  const innerW = props.width - 2 * PAD
  const innerH = props.height - topPad.value - PAD
  const bottomRight = `${(PAD + innerW).toFixed(1)},${(topPad.value + innerH).toFixed(1)}`
  const bottomLeft = `${PAD.toFixed(1)},${(topPad.value + innerH).toFixed(1)}`
  return `${points.value} ${bottomRight} ${bottomLeft}`
})

const hasData = computed(() => props.data && props.data.length >= 1)
</script>

<template>
  <div
    v-if="hasData"
    class="relative"
    :class="fill ? 'w-full' : 'inline-block'"
    :style="fill ? undefined : { width: `${width}px`, height: `${height}px` }"
  >
    <svg
      class="sparkline block"
      :class="fill ? 'w-full' : ''"
      :width="fill ? undefined : width"
      :height="height"
      :viewBox="`0 0 ${width} ${height}`"
      preserveAspectRatio="none"
    >
      <!-- Y-axis: thin line across the top of the data area -->
      <line
        v-if="showYAxis"
        :x1="PAD" :y1="topPad"
        :x2="width - PAD" :y2="topPad"
        stroke="currentColor"
        stroke-width="0.5"
        opacity="0.25"
      />

      <polygon
        v-if="fillColor"
        :points="areaPoints"
        :fill="fillColor"
      />
      <polyline
        fill="none"
        :stroke="color"
        stroke-width="1.5"
        stroke-linecap="round"
        stroke-linejoin="round"
        :points="points"
      />
    </svg>

    <!-- Label rendered as HTML so it isn't stretched by the SVG scaling -->
    <span
      v-if="showYAxis"
      class="absolute right-1 text-[9px] opacity-60 leading-none pointer-events-none"
      :style="{ top: `${PAD}px` }"
    >{{ topLabel }}</span>
  </div>
</template>
