<script setup lang="ts">
import { ref, computed, useSlots } from 'vue'

export interface Column {
  key: string
  label: string
  sortable?: boolean
  align?: 'left' | 'center' | 'right'
  width?: string
  mono?: boolean
  /** Right-aligns and applies tabular figures, so digits line up down the column. */
  numeric?: boolean
}

/**
 * Character budget for a cell before it truncates. Matches the `max-w-[44ch]`
 * below closely enough to decide whether to offer the expand affordance.
 */
const TRUNCATE_AT = 44

const props = withDefaults(defineProps<{
  columns: Column[]
  rows: unknown[]
  loading?: boolean
  sortKey?: string
  sortDir?: 'asc' | 'desc'
  pageSize?: number
  emptyMessage?: string
  /**
   * Enables the cell-detail affordance. Cells stay truncated but become
   * clickable, and the table emits `inspect` instead of swallowing the value.
   */
  inspectable?: boolean
}>(), {
  loading: false,
  sortDir: 'asc',
  pageSize: 25,
  emptyMessage: 'No data',
  inspectable: false,
})

const emit = defineEmits<{
  sort: [key: string, dir: 'asc' | 'desc']
  page: [offset: number]
  inspect: [column: string, row: Record<string, unknown>]
}>()

const slots = useSlots()

const currentPage = ref(0)

const totalPages = computed(() =>
  Math.max(1, Math.ceil(props.rows.length / props.pageSize))
)

const paginatedRows = computed(() => {
  const start = currentPage.value * props.pageSize
  return props.rows.slice(start, start + props.pageSize)
})

function handleSort(col: Column) {
  if (!col.sortable) return
  const nextDir = props.sortKey === col.key && props.sortDir === 'asc' ? 'desc' : 'asc'
  emit('sort', col.key, nextDir)
}

function goToPage(page: number) {
  if (page < 0 || page >= totalPages.value) return
  currentPage.value = page
  emit('page', page * props.pageSize)
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

/** True when the rendered text is wider than the column can show. */
function isTruncated(row: unknown, key: string): boolean {
  const text = String(cellValue(row, key) ?? '')
  return text.length > TRUNCATE_AT
}

function inspect(row: unknown, key: string) {
  if (!props.inspectable) return
  emit('inspect', key, row as Record<string, unknown>)
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
          <template v-for="(row, rowIdx) in paginatedRows" :key="rowIdx">
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
                  'px-3 py-2 text-[13px] align-top',
                  col.numeric ? 'text-right tabular-nums' : alignClass(col.align),
                  col.mono ? 'font-mono' : '',
                ]"
              >
                <component
                  :is="inspectable ? 'button' : 'div'"
                  :class="[
                    'max-w-[44ch] truncate block',
                    col.numeric ? 'text-right w-full' : 'text-left',
                    inspectable ? 'hover:text-accent hover:underline decoration-dotted underline-offset-2 cursor-pointer' : '',
                  ]"
                  :title="inspectable
                    ? (isTruncated(row, col.key) ? 'Show full value' : 'Show row')
                    : String(cellValue(row, col.key) ?? '')"
                  @click="inspect(row, col.key)"
                >
                  <slot :name="`cell-${col.key}`" :value="cellValue(row, col.key)" :row="row">
                    {{ cellValue(row, col.key) ?? '—' }}
                  </slot>
                </component>
              </td>
            </tr>
          </template>
        </tbody>
      </table>

      <!-- Pagination -->
      <div v-if="totalPages > 1" class="flex items-center justify-between px-3 py-2 text-xs text-text-secondary border-t border-surface-border">
        <span>
          {{ currentPage * pageSize + 1 }}–{{ Math.min((currentPage + 1) * pageSize, rows.length) }}
          of {{ rows.length }}
        </span>
        <div class="flex items-center gap-1">
          <button
            :disabled="currentPage === 0"
            class="px-2 py-1 rounded hover:bg-surface-raised disabled:opacity-30 disabled:cursor-not-allowed"
            @click="goToPage(currentPage - 1)"
          >
            ← Prev
          </button>
          <span class="px-2 font-mono">{{ currentPage + 1 }} / {{ totalPages }}</span>
          <button
            :disabled="currentPage >= totalPages - 1"
            class="px-2 py-1 rounded hover:bg-surface-raised disabled:opacity-30 disabled:cursor-not-allowed"
            @click="goToPage(currentPage + 1)"
          >
            Next →
          </button>
        </div>
      </div>
    </div>

    <!-- Empty state -->
    <div v-else class="flex items-center justify-center py-12 text-text-muted text-sm">
      {{ emptyMessage }}
    </div>
  </div>
</template>
