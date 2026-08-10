<script setup lang="ts">
import { fmtCell } from '../utils/formatting'

defineProps<{ columns: string[]; rows: unknown[][] }>()
</script>

<template>
  <div class="overflow-auto rounded-lg border border-surface-border">
    <table class="w-full border-collapse text-[13px]">
      <thead>
        <tr>
          <th
            v-for="col in columns"
            :key="col"
            class="sticky top-0 border-b border-surface-border bg-surface-sunken px-3 py-1.5 text-left font-semibold text-text-secondary"
          >
            {{ col }}
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, i) in rows" :key="i" class="even:bg-surface-raised">
          <td
            v-for="(cell, j) in row"
            :key="j"
            class="whitespace-pre border-b border-surface-border px-3 py-1 font-mono"
            :class="cell === null ? 'text-text-muted italic' : 'text-text'"
          >
            {{ fmtCell(cell) }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
