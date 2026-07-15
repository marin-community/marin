<script setup lang="ts">
import { computed } from 'vue'

/**
 * One device variant's free/total accelerator capacity as a labeled meter.
 * The bar encodes the FREE fraction (this is an availability display, not a
 * utilization display): a full bar means the whole capacity is placeable.
 * `total` is null when the reporting cluster predates totals — the meter then
 * shows the free count alone, with no bar.
 */
const props = withDefaults(
  defineProps<{
    label: string
    free: number
    total: number | null
    /** sm: one-line table cell; md: labeled card row; lg: summary stat tile. */
    size?: 'sm' | 'md' | 'lg'
    /** Render dimmed — the numbers come from a stale observation (e.g. an
     *  unreachable peer's last heartbeat). */
    stale?: boolean
  }>(),
  { size: 'md', stale: false },
)

const freeFraction = computed(() =>
  props.total != null && props.total > 0 ? Math.min(1, props.free / props.total) : 0,
)

// Scarcity is a state: plenty free = success, under ~15% free = warning, none
// free = danger (carried by the value text — an empty bar has no color to read).
const LOW_FREE_FRACTION = 0.15

const scarcity = computed<'ok' | 'low' | 'none'>(() => {
  if (props.total == null || props.total === 0) return 'ok'
  if (props.free === 0) return 'none'
  return freeFraction.value <= LOW_FREE_FRACTION ? 'low' : 'ok'
})

const fillClass = computed(() =>
  scarcity.value === 'low' ? 'bg-status-warning' : 'bg-status-success',
)

const freeTextClass = computed(() => {
  if (scarcity.value === 'none') return 'text-status-danger'
  if (scarcity.value === 'low') return 'text-status-warning'
  return 'text-text'
})
</script>

<template>
  <div :class="stale ? 'opacity-50' : ''">
    <template v-if="size === 'lg'">
      <div class="text-xs font-mono uppercase tracking-wider text-text-secondary">{{ label }}</div>
      <div class="mt-0.5 text-2xl font-semibold font-mono tabular-nums leading-tight">
        <span :class="freeTextClass">{{ free }}</span>
        <span class="text-sm font-normal text-text-muted"> free<template v-if="total != null"> of {{ total }}</template></span>
      </div>
      <div v-if="total != null" class="mt-1.5 h-1.5 w-full rounded-full bg-surface-sunken overflow-hidden">
        <div
          class="h-full rounded-full transition-all duration-300"
          :class="fillClass"
          :style="{ width: (freeFraction * 100).toFixed(1) + '%' }"
        />
      </div>
    </template>

    <template v-else-if="size === 'md'">
      <div class="flex items-baseline justify-between gap-2">
        <span class="text-xs font-mono uppercase tracking-wide text-text-secondary">{{ label }}</span>
        <span class="text-xs font-mono tabular-nums">
          <span :class="freeTextClass" class="font-semibold">{{ free }}</span>
          <span class="text-text-muted"><template v-if="total != null"> / {{ total }}</template> free</span>
        </span>
      </div>
      <div v-if="total != null" class="mt-1 h-1.5 w-full rounded-full bg-surface-sunken overflow-hidden">
        <div
          class="h-full rounded-full transition-all duration-300"
          :class="fillClass"
          :style="{ width: (freeFraction * 100).toFixed(1) + '%' }"
        />
      </div>
    </template>

    <template v-else>
      <span class="inline-flex items-center gap-1.5">
        <span class="text-xs font-mono uppercase tracking-wide text-text-secondary">{{ label }}</span>
        <span v-if="total != null" class="h-1 w-14 rounded-full bg-surface-sunken overflow-hidden">
          <span
            class="block h-full rounded-full"
            :class="fillClass"
            :style="{ width: (freeFraction * 100).toFixed(1) + '%' }"
          />
        </span>
        <span class="text-xs font-mono tabular-nums">
          <span :class="freeTextClass">{{ free }}</span>
          <span v-if="total != null" class="text-text-muted">/{{ total }}</span>
        </span>
      </span>
    </template>
  </div>
</template>
