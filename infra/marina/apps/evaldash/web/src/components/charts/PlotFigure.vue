<script setup lang="ts">
/**
 * Thin Observable Plot wrapper. Renders `Plot.plot(options)` into a container and swaps the
 * SVG whenever `options` changes, cleaning up on unmount. Charts inherit the theme by setting
 * their marks' colors to `currentColor`, which resolves against this container's text color.
 */
import { onBeforeUnmount, ref, watchEffect } from 'vue'
import * as Plot from '@observablehq/plot'

const props = defineProps<{ options: Record<string, unknown> }>()

const container = ref<HTMLDivElement | null>(null)
let figure: HTMLElement | SVGSVGElement | null = null

function clear() {
  if (figure) {
    figure.remove()
    figure = null
  }
}

watchEffect(() => {
  const host = container.value
  // Read options so the effect re-runs when the parent swaps them; read container so it
  // re-runs once mounted.
  const options = props.options
  if (!host) return
  clear()
  figure = Plot.plot(options as Parameters<typeof Plot.plot>[0])
  host.append(figure)
})

onBeforeUnmount(clear)
</script>

<template>
  <div ref="container" class="w-full overflow-x-auto text-text"></div>
</template>
