<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  data: number[]
  color?: string
  height?: number
  fillColor?: string
}>(), {
  color: 'var(--color-accent, #2563eb)',
  height: 20,
})

// Internal viewBox width. The SVG renders at 100% of its container with
// preserveAspectRatio="none", so this only sets coordinate density for
// stroke positioning and does not affect on-screen width.
const VIEWBOX_W = 100
const PAD = 1

/**
 * Build the SVG polyline points string from the data array.
 * Single-value arrays are duplicated to draw a flat line.
 * When all values are zero, draws a flat line at the bottom.
 */
const points = computed(() => {
  if (!props.data || props.data.length < 1) return ''

  const data = props.data.length === 1 ? [props.data[0], props.data[0]] : props.data
  const max = Math.max(...data)

  const innerW = VIEWBOX_W - 2 * PAD
  const innerH = props.height - 2 * PAD

  return data.map((v, i) => {
    const x = PAD + (i / (data.length - 1)) * innerW
    const y = max === 0
      ? PAD + innerH
      : PAD + innerH - (Math.min(v, max) / max) * innerH
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
})

/** Area fill polygon: the line points plus closing along the bottom edge. */
const areaPoints = computed(() => {
  if (!points.value) return ''
  const innerW = VIEWBOX_W - 2 * PAD
  const innerH = props.height - 2 * PAD
  const bottomRight = `${(PAD + innerW).toFixed(1)},${(PAD + innerH).toFixed(1)}`
  const bottomLeft = `${PAD.toFixed(1)},${(PAD + innerH).toFixed(1)}`
  return `${points.value} ${bottomRight} ${bottomLeft}`
})

const hasData = computed(() => props.data && props.data.length >= 1)
</script>

<template>
  <svg
    v-if="hasData"
    class="sparkline"
    width="100%"
    :height="height"
    :viewBox="`0 0 ${VIEWBOX_W} ${height}`"
    preserveAspectRatio="none"
    :style="{ display: 'block' }"
  >
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
</template>
