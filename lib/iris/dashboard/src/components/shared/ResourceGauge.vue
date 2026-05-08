<script setup lang="ts">
import { computed } from 'vue'
import { formatBytes } from '@/utils/formatting'
import { DIVERGING_COLORS } from '@/types/status'

const props = defineProps<{
  label: string
  used: number
  total: number
  unit: string
}>()

const percentage = computed(() =>
  props.total > 0 ? Math.min(100, (props.used / props.total) * 100) : 0
)

/** Map 0-100% utilization to the diverging palette (green → neutral → red).
 *  Low usage is positive (green), high usage is negative (red). */
function divergingColor(pct: number): string {
  // Palette goes negative(0) → neutral(5) → positive(10).
  // We want low utilization = positive (green) and high = negative (red),
  // so invert: 0% → index 10 (greenest), 100% → index 0 (reddest).
  const idx = Math.round((1 - pct / 100) * (DIVERGING_COLORS.length - 1))
  return DIVERGING_COLORS[Math.max(0, Math.min(idx, DIVERGING_COLORS.length - 1))]
}

const barColor = computed(() => divergingColor(percentage.value))
const textColor = computed(() => divergingColor(percentage.value))

function formatValue(value: number, unit: string): string {
  if (unit === 'bytes') return formatBytes(value)
  if (unit === 'cores') return formatCores(value)
  return String(value)
}

function formatCores(cores: number): string {
  if (cores >= 1 && cores === Math.floor(cores)) return cores + 'c'
  return cores.toFixed(1) + 'c'
}

const displayText = computed(() => {
  const usedStr = formatValue(props.used, props.unit)
  const totalStr = formatValue(props.total, props.unit)
  return `${usedStr} / ${totalStr}`
})
</script>

<template>
  <div class="space-y-1">
    <div class="flex items-baseline justify-between">
      <span class="text-xs font-medium text-text-secondary">{{ label }}</span>
      <span class="text-xs font-mono tabular-nums" :style="{ color: textColor }">
        {{ displayText }}
        <span class="text-text-muted ml-1">({{ Math.round(percentage) }}%)</span>
      </span>
    </div>
    <div class="h-1.5 w-full rounded-full bg-surface-sunken overflow-hidden">
      <div
        class="h-full rounded-full transition-all duration-300"
        :style="{ width: percentage.toFixed(1) + '%', backgroundColor: barColor }"
      />
    </div>
  </div>
</template>
