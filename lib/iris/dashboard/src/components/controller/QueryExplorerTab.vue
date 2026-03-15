<script setup lang="ts">
import { ref, computed } from 'vue'
import { executeQuery, executeRawQuery, parseRows } from '@/composables/useQuery'
import type { QueryRequest, Query } from '@/composables/useQuery'
import type { ColumnMeta } from '@/types/rpc'

type QueryMode = 'structured' | 'raw'

interface QueryTemplate {
  label: string
  description: string
  mode: QueryMode
  content: string
}

const QUERY_TEMPLATES: QueryTemplate[] = [
  {
    label: 'All Jobs',
    description: 'List all jobs ordered by submission time',
    mode: 'structured',
    content: JSON.stringify({
      from: { name: 'jobs' },
      columns: [
        { name: 'job_id' },
        { name: 'user_id' },
        { name: 'state' },
        { name: 'submitted_at_ms' },
        { name: 'started_at_ms' },
        { name: 'finished_at_ms' },
        { name: 'num_tasks' },
        { name: 'error' },
      ],
      orderBy: [{ column: 'submitted_at_ms', direction: 'SORT_DESC' }],
      limit: 100,
    } satisfies Query, null, 2),
  },
  {
    label: 'Running Tasks',
    description: 'Tasks currently in running state',
    mode: 'structured',
    content: JSON.stringify({
      from: { name: 'tasks' },
      columns: [
        { name: 'task_id' },
        { name: 'job_id' },
        { name: 'state' },
        { name: 'started_at_ms' },
        { name: 'failure_count' },
        { name: 'preemption_count' },
      ],
      where: {
        comparison: {
          column: 'state',
          op: 'CMP_EQ',
          value: { intValue: '3' },
        },
      },
      orderBy: [{ column: 'started_at_ms', direction: 'SORT_DESC' }],
      limit: 100,
    } satisfies Query, null, 2),
  },
  {
    label: 'Worker Status',
    description: 'All workers with health and resource info',
    mode: 'structured',
    content: JSON.stringify({
      from: { name: 'workers' },
      columns: [
        { name: 'worker_id' },
        { name: 'address' },
        { name: 'healthy' },
        { name: 'active' },
        { name: 'consecutive_failures' },
        { name: 'last_heartbeat_ms' },
        { name: 'committed_gpu' },
        { name: 'committed_tpu' },
      ],
      orderBy: [{ column: 'last_heartbeat_ms', direction: 'SORT_DESC' }],
    } satisfies Query, null, 2),
  },
  {
    label: 'Jobs per User',
    description: 'Job count grouped by user and state',
    mode: 'structured',
    content: JSON.stringify({
      from: { name: 'jobs', alias: 'j' },
      columns: [
        { name: 'user_id' },
        { name: 'state' },
        { name: 'job_id', func: 'AGG_COUNT', alias: 'job_count' },
      ],
      groupBy: { columns: [{ name: 'user_id' }, { name: 'state' }] },
      orderBy: [{ column: 'user_id', direction: 'SORT_ASC' }],
    } satisfies Query, null, 2),
  },
]

const mode = ref<QueryMode>('structured')
const queryInput = ref(QUERY_TEMPLATES[0].content)
const rawSqlInput = ref('SELECT job_id, state, user_id FROM jobs ORDER BY submitted_at_ms DESC LIMIT 50')

const columns = ref<ColumnMeta[]>([])
const rows = ref<Record<string, unknown>[]>([])
const totalCount = ref(0)
const loading = ref(false)
const error = ref<string | null>(null)
const hasExecuted = ref(false)

const PAGE_SIZE = 25
const currentPage = ref(0)

const totalPages = computed(() =>
  Math.max(1, Math.ceil(rows.value.length / PAGE_SIZE))
)

const paginatedRows = computed(() => {
  const start = currentPage.value * PAGE_SIZE
  return rows.value.slice(start, start + PAGE_SIZE)
})

function applyTemplate(template: QueryTemplate) {
  mode.value = template.mode
  if (template.mode === 'structured') {
    queryInput.value = template.content
  } else {
    rawSqlInput.value = template.content
  }
}

async function execute() {
  loading.value = true
  error.value = null
  hasExecuted.value = true
  currentPage.value = 0
  columns.value = []
  rows.value = []
  totalCount.value = 0

  try {
    if (mode.value === 'structured') {
      const parsed = JSON.parse(queryInput.value) as Query
      const request: QueryRequest = { query: parsed }
      const response = await executeQuery(request)
      columns.value = response.columns
      rows.value = parseRows(response.columns, response.rows)
      totalCount.value = response.totalCount
    } else {
      const response = await executeRawQuery({ sql: rawSqlInput.value })
      columns.value = response.columns
      rows.value = parseRows(response.columns, response.rows)
      totalCount.value = rows.value.length
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

function goToPage(page: number) {
  if (page < 0 || page >= totalPages.value) return
  currentPage.value = page
}

function formatCellValue(value: unknown): string {
  if (value === null || value === undefined) return '\u2014'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function handleKeydown(event: KeyboardEvent) {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
    execute()
  }
}
</script>

<template>
  <div class="space-y-4">
    <!-- Mode toggle and templates -->
    <div class="flex items-center gap-4 flex-wrap">
      <div class="flex rounded-lg border border-surface-border overflow-hidden">
        <button
          :class="[
            'px-3 py-1.5 text-sm font-medium transition-colors',
            mode === 'structured'
              ? 'bg-accent text-white'
              : 'bg-surface text-text-secondary hover:text-text hover:bg-surface-sunken',
          ]"
          @click="mode = 'structured'"
        >
          Structured Query
        </button>
        <button
          :class="[
            'px-3 py-1.5 text-sm font-medium transition-colors border-l border-surface-border',
            mode === 'raw'
              ? 'bg-accent text-white'
              : 'bg-surface text-text-secondary hover:text-text hover:bg-surface-sunken',
          ]"
          @click="mode = 'raw'"
        >
          Raw SQL
        </button>
      </div>

      <div class="flex items-center gap-2">
        <span class="text-xs text-text-secondary">Templates:</span>
        <button
          v-for="template in QUERY_TEMPLATES"
          :key="template.label"
          class="px-2.5 py-1 text-xs rounded border border-surface-border text-text-secondary
                 hover:text-text hover:bg-surface-sunken transition-colors"
          :title="template.description"
          @click="applyTemplate(template)"
        >
          {{ template.label }}
        </button>
      </div>
    </div>

    <!-- Query input -->
    <div>
      <div v-if="mode === 'structured'">
        <label class="block text-xs font-medium text-text-secondary mb-1">
          Query JSON
        </label>
        <textarea
          v-model="queryInput"
          rows="12"
          class="w-full font-mono text-[13px] bg-surface-sunken text-text border border-surface-border
                 rounded-lg px-3 py-2 resize-y focus:outline-none focus:ring-1 focus:ring-accent"
          placeholder='{"from": {"name": "jobs"}, "limit": 10}'
          spellcheck="false"
          @keydown="handleKeydown"
        />
      </div>
      <div v-else>
        <label class="block text-xs font-medium text-text-secondary mb-1">
          SQL Query (admin-only, SELECT only)
        </label>
        <textarea
          v-model="rawSqlInput"
          rows="6"
          class="w-full font-mono text-[13px] bg-surface-sunken text-text border border-surface-border
                 rounded-lg px-3 py-2 resize-y focus:outline-none focus:ring-1 focus:ring-accent"
          placeholder="SELECT * FROM jobs LIMIT 10"
          spellcheck="false"
          @keydown="handleKeydown"
        />
      </div>
    </div>

    <!-- Execute button -->
    <div class="flex items-center gap-3">
      <button
        class="px-4 py-2 text-sm font-medium text-white bg-accent rounded-lg
               hover:bg-accent/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        :disabled="loading"
        @click="execute"
      >
        <span v-if="loading" class="inline-flex items-center gap-1.5">
          <svg class="animate-spin h-3.5 w-3.5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Executing...
        </span>
        <span v-else>Execute</span>
      </button>
      <span class="text-xs text-text-muted">
        Ctrl+Enter / Cmd+Enter to run
      </span>
    </div>

    <!-- Error display -->
    <div
      v-if="error"
      class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <!-- Results -->
    <div v-if="hasExecuted && !error && !loading">
      <!-- Results header -->
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-text-secondary">
          {{ totalCount }} row{{ totalCount !== 1 ? 's' : '' }} returned
          <template v-if="columns.length > 0">
            &middot; {{ columns.length }} column{{ columns.length !== 1 ? 's' : '' }}
          </template>
        </span>
      </div>

      <!-- Results table -->
      <div v-if="rows.length > 0" class="overflow-x-auto border border-surface-border rounded-lg">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border bg-surface-sunken">
              <th
                v-for="col in columns"
                :key="col.name"
                class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary whitespace-nowrap"
              >
                {{ col.name }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(row, rowIdx) in paginatedRows"
              :key="rowIdx"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td
                v-for="col in columns"
                :key="col.name"
                class="px-3 py-2 text-[13px] font-mono whitespace-nowrap max-w-xs truncate"
                :title="formatCellValue(row[col.name])"
              >
                {{ formatCellValue(row[col.name]) }}
              </td>
            </tr>
          </tbody>
        </table>

        <!-- Pagination -->
        <div
          v-if="totalPages > 1"
          class="flex items-center justify-between px-3 py-2 text-xs text-text-secondary border-t border-surface-border"
        >
          <span>
            {{ currentPage * PAGE_SIZE + 1 }}–{{ Math.min((currentPage + 1) * PAGE_SIZE, rows.length) }}
            of {{ rows.length }}
          </span>
          <div class="flex items-center gap-1">
            <button
              :disabled="currentPage === 0"
              class="px-2 py-1 rounded hover:bg-surface-raised disabled:opacity-30 disabled:cursor-not-allowed"
              @click="goToPage(currentPage - 1)"
            >
              Prev
            </button>
            <span class="px-2 font-mono">{{ currentPage + 1 }} / {{ totalPages }}</span>
            <button
              :disabled="currentPage >= totalPages - 1"
              class="px-2 py-1 rounded hover:bg-surface-raised disabled:opacity-30 disabled:cursor-not-allowed"
              @click="goToPage(currentPage + 1)"
            >
              Next
            </button>
          </div>
        </div>
      </div>

      <!-- Empty results -->
      <div
        v-else
        class="flex items-center justify-center py-12 text-text-muted text-sm border border-surface-border rounded-lg"
      >
        Query returned no rows
      </div>
    </div>
  </div>
</template>
