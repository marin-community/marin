<script setup lang="ts">
defineProps<{ rows: Record<string, unknown>[]; columns: string[] }>()

function cell(value: unknown): string {
  return value == null ? '' : String(value)
}
</script>

<template>
  <p v-if="!rows.length" class="text-sm text-text-muted">no rows</p>
  <div v-else class="overflow-auto rounded-lg border border-surface-border">
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
        <tr v-for="(row, i) in rows" :key="i" class="align-top even:bg-surface-raised">
          <td v-for="col in columns" :key="col" class="border-b border-surface-border px-3 py-1">
            <pre
              v-if="col === 'text'"
              class="m-0 max-h-60 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-[1.45]"
              >{{ cell(row[col]) }}</pre
            >
            <span v-else class="font-mono">{{ cell(row[col]) }}</span>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
