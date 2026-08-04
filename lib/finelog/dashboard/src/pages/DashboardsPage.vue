<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

import DashboardEditor from '@/components/dashboard/DashboardEditor.vue'
import DashboardPanelView from '@/components/dashboard/DashboardPanelView.vue'
import { statsRpcCall } from '@/composables/useRpc'
import { BUILT_IN_DASHBOARDS } from '@/dashboards/examples'
import { deleteSavedDashboard, listSavedDashboards, saveDashboard } from '@/services/dashboards'
import type { DashboardDefinition, SavedDashboard } from '@/types/dashboard'
import { decodeArrowIpc, type ArrowResult } from '@/utils/arrow'
import { dashboardIntervalMs, expandDashboardSql } from '@/utils/dashboardSql'

interface QueryResponse {
  arrowIpc?: string
  rowCount?: string | number
}

interface PanelExecution {
  result: ArrowResult | null
  loading: boolean
  error: string | null
  rowCount: number
  elapsedMs: number | null
}

type SourceKind = 'builtin' | 'saved' | 'new'

const MINUTE_MS = 60_000

function cloneDefinition(definition: DashboardDefinition): DashboardDefinition {
  return structuredClone(definition)
}

function inputDateTime(timestampMs: number): string {
  const date = new Date(timestampMs)
  return new Date(timestampMs - date.getTimezoneOffset() * MINUTE_MS).toISOString().slice(0, 16)
}

const now = Date.now()
const saved = ref<SavedDashboard[]>([])
const selectedKey = ref(`builtin:${BUILT_IN_DASHBOARDS[0].id}`)
const sourceKind = ref<SourceKind>('builtin')
const current = ref<DashboardDefinition>(cloneDefinition(BUILT_IN_DASHBOARDS[0]))
const variableValues = ref<Record<string, string>>({})
const executions = ref<Record<string, PanelExecution>>({})
const fromInput = ref(inputDateTime(now - 30 * MINUTE_MS))
const toInput = ref(inputDateTime(now))
const editorOpen = ref(false)
const loadingSaved = ref(false)
const pageError = ref<string | null>(null)
const saving = ref(false)
const macroHelp = 'Use {{from_ms}}, {{to_ms}}, {{interval_ms}}, or a declared variable.'

const sourceLabel = computed(() => {
  if (sourceKind.value === 'builtin') return 'Built-in example'
  if (sourceKind.value === 'saved') return 'Shared dashboard'
  return 'Unsaved dashboard'
})

function resetRuntimeState() {
  variableValues.value = Object.fromEntries(
    current.value.variables.map((variable) => [variable.name, variable.default]),
  )
  executions.value = {}
  pageError.value = null
}

function selectBuiltIn(definition: DashboardDefinition) {
  selectedKey.value = `builtin:${definition.id}`
  sourceKind.value = 'builtin'
  current.value = cloneDefinition(definition)
  editorOpen.value = false
  resetRuntimeState()
}

function selectSaved(item: SavedDashboard) {
  selectedKey.value = `saved:${item.definition.id}`
  sourceKind.value = 'saved'
  current.value = cloneDefinition(item.definition)
  editorOpen.value = false
  resetRuntimeState()
}

function uniqueCopyId(sourceId: string): string {
  const occupied = new Set([
    ...BUILT_IN_DASHBOARDS.map((definition) => definition.id),
    ...saved.value.map((item) => item.definition.id),
  ])
  let candidate = `${sourceId}-copy`
  let suffix = 2
  while (occupied.has(candidate)) candidate = `${sourceId}-copy-${suffix++}`
  return candidate
}

function cloneCurrent() {
  const definition = cloneDefinition(current.value)
  definition.id = uniqueCopyId(definition.id)
  definition.title = `${definition.title} copy`
  selectedKey.value = 'new'
  sourceKind.value = 'new'
  current.value = definition
  editorOpen.value = true
  resetRuntimeState()
}

function newDashboard() {
  selectedKey.value = 'new'
  sourceKind.value = 'new'
  current.value = {
    version: 1,
    id: `dashboard-${Date.now()}`,
    title: 'New dashboard',
    description: '',
    variables: [],
    panels: [{
      id: 'panel-1',
      title: 'Panel 1',
      kind: 'table',
      width: 'full',
      description: '',
      sql: 'SELECT 1 AS value',
    }],
  }
  editorOpen.value = true
  resetRuntimeState()
}

function execution(panelId: string): PanelExecution {
  return executions.value[panelId] ?? {
    result: null,
    loading: false,
    error: null,
    rowCount: 0,
    elapsedMs: null,
  }
}

function ensureExecution(panelId: string): PanelExecution {
  if (!executions.value[panelId]) executions.value[panelId] = execution(panelId)
  return executions.value[panelId]
}

function timeRange(): { fromMs: number; toMs: number } {
  const fromMs = new Date(fromInput.value).getTime()
  const toMs = new Date(toInput.value).getTime()
  if (!Number.isFinite(fromMs) || !Number.isFinite(toMs) || toMs <= fromMs) {
    throw new Error('End time must be later than start time')
  }
  return { fromMs, toMs }
}

async function runPanel(panelId: string) {
  const panel = current.value.panels.find((candidate) => candidate.id === panelId)
  if (!panel) return
  const state = ensureExecution(panelId)
  state.loading = true
  state.error = null
  const started = performance.now()
  try {
    const { fromMs, toMs } = timeRange()
    const variables = Object.fromEntries(current.value.variables.map((variable) => [
      variable.name,
      variableValues.value[variable.name] ?? variable.default,
    ]))
    const sql = expandDashboardSql(panel.sql, {
      fromMs,
      toMs,
      intervalMs: dashboardIntervalMs(fromMs, toMs),
      variables,
    })
    const response = await statsRpcCall<QueryResponse>('Query', { sql })
    state.result = decodeArrowIpc(response.arrowIpc)
    state.rowCount = Number(response.rowCount ?? state.result.rows.length)
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error)
    state.result = null
    state.rowCount = 0
  } finally {
    state.elapsedMs = performance.now() - started
    state.loading = false
  }
}

async function runAll() {
  await Promise.all(current.value.panels.map((panel) => runPanel(panel.id)))
}

function setRange(durationMs: number) {
  const end = Date.now()
  fromInput.value = inputDateTime(end - durationMs)
  toInput.value = inputDateTime(end)
}

async function refreshSaved() {
  loadingSaved.value = true
  try {
    saved.value = await listSavedDashboards()
  } catch (error) {
    pageError.value = error instanceof Error ? error.message : String(error)
  } finally {
    loadingSaved.value = false
  }
}

async function saveCurrent() {
  saving.value = true
  pageError.value = null
  try {
    const item = await saveDashboard(current.value)
    await refreshSaved()
    sourceKind.value = 'saved'
    selectedKey.value = `saved:${item.definition.id}`
    current.value = cloneDefinition(item.definition)
    editorOpen.value = false
  } catch (error) {
    pageError.value = error instanceof Error ? error.message : String(error)
  } finally {
    saving.value = false
  }
}

async function removeCurrent() {
  if (sourceKind.value !== 'saved') return
  if (!window.confirm(`Delete shared dashboard “${current.value.title}”?`)) return
  try {
    await deleteSavedDashboard(current.value.id)
    await refreshSaved()
    selectBuiltIn(BUILT_IN_DASHBOARDS[0])
  } catch (error) {
    pageError.value = error instanceof Error ? error.message : String(error)
  }
}

onMounted(async () => {
  resetRuntimeState()
  await refreshSaved()
})
</script>

<template>
  <div class="grid lg:grid-cols-[250px_minmax(0,1fr)] gap-5">
    <aside class="space-y-4">
      <button class="w-full px-3 py-2 text-sm rounded bg-accent text-white hover:bg-accent-hover" @click="newDashboard">
        New dashboard
      </button>

      <section>
        <h2 class="px-2 mb-1 text-[11px] font-semibold uppercase tracking-wider text-text-muted">Examples</h2>
        <button
          v-for="definition in BUILT_IN_DASHBOARDS"
          :key="definition.id"
          class="w-full text-left px-2.5 py-2 text-sm rounded transition-colors"
          :class="selectedKey === `builtin:${definition.id}` ? 'bg-accent-subtle text-text' : 'hover:bg-surface-raised text-text-secondary'"
          @click="selectBuiltIn(definition)"
        >{{ definition.title }}</button>
      </section>

      <section>
        <h2 class="px-2 mb-1 text-[11px] font-semibold uppercase tracking-wider text-text-muted">Shared</h2>
        <div v-if="loadingSaved" class="px-2 py-2 text-xs text-text-muted">Loading…</div>
        <div v-else-if="saved.length === 0" class="px-2 py-2 text-xs text-text-muted">No saved dashboards.</div>
        <button
          v-for="item in saved"
          :key="item.definition.id"
          class="w-full text-left px-2.5 py-2 text-sm rounded transition-colors"
          :class="selectedKey === `saved:${item.definition.id}` ? 'bg-accent-subtle text-text' : 'hover:bg-surface-raised text-text-secondary'"
          @click="selectSaved(item)"
        >{{ item.definition.title }}</button>
      </section>
    </aside>

    <div class="min-w-0 space-y-4">
      <div
        v-if="pageError"
        class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg border border-status-danger-border rounded-lg"
      >{{ pageError }}</div>

      <header class="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div class="text-[11px] font-semibold uppercase tracking-wider text-accent">{{ sourceLabel }}</div>
          <h1 class="text-2xl font-semibold mt-0.5">{{ current.title }}</h1>
          <p v-if="current.description" class="text-sm text-text-secondary mt-1 max-w-3xl">{{ current.description }}</p>
        </div>
        <div class="flex gap-2">
          <button
            v-if="sourceKind === 'builtin'"
            class="px-3 py-1.5 text-sm rounded border border-surface-border hover:bg-surface-raised"
            @click="cloneCurrent"
          >Clone to edit</button>
          <template v-else>
            <button class="px-3 py-1.5 text-sm rounded border border-surface-border hover:bg-surface-raised" @click="editorOpen = !editorOpen">
              {{ editorOpen ? 'Close editor' : 'Edit' }}
            </button>
            <button
              class="px-3 py-1.5 text-sm rounded bg-accent text-white hover:bg-accent-hover disabled:opacity-50"
              :disabled="saving"
              @click="saveCurrent"
            >{{ saving ? 'Saving…' : 'Save' }}</button>
            <button
              v-if="sourceKind === 'saved'"
              class="px-3 py-1.5 text-sm rounded border border-status-danger-border text-status-danger hover:bg-status-danger-bg"
              @click="removeCurrent"
            >Delete</button>
          </template>
        </div>
      </header>

      <section class="bg-surface border border-surface-border rounded-lg p-3">
        <div class="flex flex-wrap items-end gap-3">
          <label class="text-xs text-text-secondary">
            From
            <input v-model="fromInput" type="datetime-local" class="block mt-1 text-xs font-mono bg-surface-sunken border border-surface-border rounded px-2 py-1.5">
          </label>
          <label class="text-xs text-text-secondary">
            To
            <input v-model="toInput" type="datetime-local" class="block mt-1 text-xs font-mono bg-surface-sunken border border-surface-border rounded px-2 py-1.5">
          </label>
          <div class="flex gap-1">
            <button class="px-2 py-1.5 text-xs rounded border border-surface-border hover:bg-surface-raised" @click="setRange(15 * MINUTE_MS)">15m</button>
            <button class="px-2 py-1.5 text-xs rounded border border-surface-border hover:bg-surface-raised" @click="setRange(60 * MINUTE_MS)">1h</button>
            <button class="px-2 py-1.5 text-xs rounded border border-surface-border hover:bg-surface-raised" @click="setRange(6 * 60 * MINUTE_MS)">6h</button>
            <button class="px-2 py-1.5 text-xs rounded border border-surface-border hover:bg-surface-raised" @click="setRange(24 * 60 * MINUTE_MS)">24h</button>
          </div>
          <label v-for="variable in current.variables" :key="variable.name" class="text-xs text-text-secondary min-w-[220px] flex-1">
            {{ variable.label }}
            <input
              v-model="variableValues[variable.name]"
              :placeholder="variable.default || variable.name"
              class="block mt-1 w-full text-xs font-mono bg-surface-sunken border border-surface-border rounded px-2 py-1.5"
            >
          </label>
          <button
            class="ml-auto px-3 py-1.5 text-sm rounded bg-accent text-white hover:bg-accent-hover disabled:opacity-50"
            :disabled="current.panels.some((panel) => execution(panel.id).loading)"
            @click="runAll"
          >Run all</button>
        </div>
      </section>

      <section v-if="editorOpen" class="bg-surface border border-accent-border rounded-lg p-4">
        <div class="flex items-center justify-between mb-4">
          <h2 class="font-semibold">Dashboard editor</h2>
          <span class="text-xs text-text-muted">{{ macroHelp }}</span>
        </div>
        <DashboardEditor :definition="current" :id-locked="sourceKind === 'saved'" @run="runPanel" />
      </section>

      <div class="grid md:grid-cols-2 gap-4 items-start">
        <DashboardPanelView
          v-for="panel in current.panels"
          :key="panel.id"
          :class="panel.width === 'full' ? 'md:col-span-2' : ''"
          :panel="panel"
          :result="execution(panel.id).result"
          :loading="execution(panel.id).loading"
          :error="execution(panel.id).error"
          :row-count="execution(panel.id).rowCount"
          :elapsed-ms="execution(panel.id).elapsedMs"
          @run="runPanel(panel.id)"
        />
      </div>
    </div>
  </div>
</template>
