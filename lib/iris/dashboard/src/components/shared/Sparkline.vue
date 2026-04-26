<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  data: number[]
  color?: string
  /**
   * Rendered SVG width. Pass a number for a fixed pixel width, or a CSS
   * length string (e.g. `'100%'`) to fill the container responsively.
   * Defaults to `'100%'` so the chart stretches to its parent.
   */
  width?: number | string
  height?: number
  /**
   * Width of the SVG viewBox coordinate system used for path calculations.
   * Only used when `width` is a string; with a numeric `width`, the rendered
   * width also serves as the viewBox width. The path is drawn at this resolution
   * and then stretched horizontally to the rendered width via
   * `preserveAspectRatio="none"`.
   */
  viewBoxWidth?: number
  fillColor?: string
}>(), {
  color: 'var(--color-accent, #2563eb)',
  width: '100%',
  height: 20,
  viewBoxWidth: 200,
})

const PAD = 1

const vbWidth = computed(() =>
  typeof props.width === 'number' ? props.width : props.viewBoxWidth
)

const svgWidth = computed(() =>
  typeof props.width === 'number' ? String(props.width) : props.width
)

/**
 * Build the SVG polyline points string from the data array.
 * Single-value arrays are duplicated to draw a flat line.
 * When all values are zero, draws a flat line at the bottom.
 */
const points = computed(() => {
  if (!props.data || props.data.length < 1) return ''

  const data = props.data.length === 1 ? [props.data[0], props.data[0]] : props.data
  const max = Math.max(...data)

  const innerW = vbWidth.value - 2 * PAD
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
  const innerW = vbWidth.value - 2 * PAD
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
    class="sparkline block"
    :width="svgWidth"
    :height="height"
    :viewBox="`0 0 ${vbWidth} ${height}`"
    preserveAspectRatio="none"
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
      vector-effect="non-scaling-stroke"
      :points="points"
    />
  </svg>
</template>
