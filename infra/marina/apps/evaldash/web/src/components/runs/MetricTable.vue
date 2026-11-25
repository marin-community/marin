<script setup lang="ts">
import { formatScore } from '@/utils/formatting'
import type { MetricRow } from '@/utils/runMetrics'

defineProps<{ rows: MetricRow[]; keyPrefix?: string }>()
</script>

<template>
  <table class="w-full border-collapse text-sm">
    <thead>
      <tr class="border-b border-surface-border bg-surface-raised text-xs font-semibold uppercase tracking-wider text-text-secondary">
        <th class="px-3 py-2 text-left">Task</th>
        <th class="px-3 py-2 text-left">Metric</th>
        <th class="px-3 py-2 text-right">Value</th>
        <th class="px-3 py-2 text-right">Score</th>
      </tr>
    </thead>
    <tbody>
      <tr
        v-for="m in rows"
        :key="`${keyPrefix ?? ''}${m.task}/${m.metric}`"
        class="border-b border-surface-border-subtle"
      >
        <td class="px-3 py-2 font-mono text-[13px]">{{ m.task }}</td>
        <td class="px-3 py-2 font-mono text-[13px]">{{ m.metric }}</td>
        <td class="px-3 py-2 text-right tabular-nums text-text-secondary">{{ m.value }}</td>
        <td class="px-3 py-2 text-right tabular-nums">{{ formatScore(m.value) }}</td>
      </tr>
    </tbody>
  </table>
</template>
