<script setup lang="ts">
import { useSlots } from 'vue'

export interface Column {
  key: string
  label: string
  sortable?: boolean
  align?: 'left' | 'center' | 'right'
  width?: string
  mono?: boolean
}

const props = withDefaults(defineProps<{
  columns: Column[]
  rows: unknown[]
  loading?: boolean
  sortKey?: string
  sortDir?: 'asc' | 'desc'
  emptyMessage?: string
}>(), {
  loading: false,
  sortDir: 'asc',
  emptyMessage: 'No data',
})

const emit = defineEmits<{
  sort: [key: string, dir: 'asc' | 'desc']
}>()

const slots = useSlots()

function handleSort(col: Column) {
  if (!col.sortable) return
  const nextDir = props.sortKey === col.key && props.sortDir === 'asc' ? 'desc' : 'asc'
  emit('sort', col.key, nextDir)
}

function cellValue(row: unknown, key: string): unknown {
  return (row as Record<string, unknown>)[key]
}

function alignClass(align?: 'left' | 'center' | 'right'): string {
  if (align === 'center') return 'text-center'
  if (align === 'right') return 'text-right'
  return 'text-left'
}

function hasSlot(name: string): boolean {
  return name in (slots ?? {})
}
</script>

<template>
  <div class="w-full">
    <!-- Loading overlay -->
    <div v-if="loading" class="flex items-center justify-center py-12 text-text-muted text-sm">
      <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
      Loading...
    </div>

    <!-- Table -->
    <div v-else-if="rows.length > 0" class="overflow-x-auto">
      <table class="w-full border-collapse">
        <thead>
          <tr class="border-b border-surface-border">
            <th
              v-for="col in columns"
              :key="col.key"
              :class="[
                'px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary',
                alignClass(col.align),
                col.sortable ? 'cursor-pointer select-none hover:text-text' : '',
              ]"
              :style="col.width ? { width: col.width } : undefined"
              @click="handleSort(col)"
            >
              <span class="inline-flex items-center gap-1">
                {{ col.label }}
                <span v-if="col.sortable && sortKey === col.key" class="text-accent">
                  {{ sortDir === 'asc' ? '↑' : '↓' }}
                </span>
                <span v-else-if="col.sortable" class="text-text-muted/40">
                  ↕
                </span>
              </span>
            </th>
          </tr>
        </thead>
        <tbody>
          <template v-for="(row, rowIdx) in rows" :key="rowIdx">
            <!-- Full row override slot -->
            <tr v-if="hasSlot('row')" class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors">
              <slot name="row" :row="row" :index="rowIdx" />
            </tr>
            <!-- Default cell rendering -->
            <tr v-else class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors">
              <td
                v-for="col in columns"
                :key="col.key"
                :class="[
                  'px-3 py-2 text-[13px]',
                  alignClass(col.align),
                  col.mono ? 'font-mono' : '',
                ]"
              >
                <slot :name="`cell-${col.key}`" :value="cellValue(row, col.key)" :row="row">
                  {{ cellValue(row, col.key) ?? '—' }}
                </slot>
              </td>
            </tr>
            <!-- Expanded row content -->
            <tr v-if="hasSlot('expanded')" class="bg-surface-sunken">
              <td :colspan="columns.length" class="p-0">
                <slot name="expanded" :row="row" :index="rowIdx" />
              </td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>

    <!-- Empty state -->
    <div v-else class="flex items-center justify-center py-12 text-text-muted text-sm">
      {{ emptyMessage }}
    </div>
  </div>
</template>
