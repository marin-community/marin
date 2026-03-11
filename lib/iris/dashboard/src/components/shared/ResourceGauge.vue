<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  label: string
  used: number
  total: number
  unit: string
}>()

const percentage = computed(() =>
  props.total > 0 ? Math.min(100, (props.used / props.total) * 100) : 0
)

const level = computed<'ok' | 'warning' | 'danger'>(() => {
  const pct = percentage.value
  if (pct >= 90) return 'danger'
  if (pct >= 70) return 'warning'
  return 'ok'
})

const BAR_COLORS: Record<string, string> = {
  ok: 'bg-status-success',
  warning: 'bg-status-warning',
  danger: 'bg-status-danger',
}

const TEXT_COLORS: Record<string, string> = {
  ok: 'text-status-success',
  warning: 'text-status-warning',
  danger: 'text-status-danger',
}

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB']

function formatValue(value: number, unit: string): string {
  if (unit === 'bytes') return formatBytes(value)
  if (unit === 'cores') return formatCores(value)
  return String(value)
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), BYTE_UNITS.length - 1)
  const val = bytes / Math.pow(1024, i)
  return (val >= 100 ? Math.round(val) : val.toFixed(1)) + ' ' + BYTE_UNITS[i]
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
      <span :class="['text-xs font-mono tabular-nums', TEXT_COLORS[level]]">
        {{ displayText }}
        <span class="text-text-muted ml-1">({{ Math.round(percentage) }}%)</span>
      </span>
    </div>
    <div class="h-1.5 w-full rounded-full bg-surface-sunken overflow-hidden">
      <div
        :class="['h-full rounded-full transition-all duration-300', BAR_COLORS[level]]"
        :style="{ width: percentage.toFixed(1) + '%' }"
      />
    </div>
  </div>
</template>
