<script setup lang="ts">
import type { DashboardDefinition, DashboardPanel, DashboardVariable } from '@/types/dashboard'

const props = defineProps<{
  definition: DashboardDefinition
}>()

const emit = defineEmits<{
  run: [panelId: string]
}>()

function addVariable() {
  const index = props.definition.variables.length + 1
  const variable: DashboardVariable = { name: `variable_${index}`, label: `Variable ${index}`, default: '' }
  props.definition.variables.push(variable)
}

function addPanel() {
  const index = props.definition.panels.length + 1
  const panel: DashboardPanel = {
    id: `panel-${index}`,
    title: `Panel ${index}`,
    description: '',
    kind: 'table',
    width: 'full',
    sql: 'SELECT 1 AS value',
  }
  props.definition.panels.push(panel)
}
</script>

<template>
  <div class="space-y-5">
    <section class="space-y-3">
      <label class="text-xs text-text-secondary">
        Title
        <input v-model="definition.title" class="mt-1 w-full text-sm bg-surface-sunken border border-surface-border rounded px-2.5 py-2">
      </label>
      <label class="block text-xs text-text-secondary">
        Description
        <input v-model="definition.description" class="mt-1 w-full text-sm bg-surface-sunken border border-surface-border rounded px-2.5 py-2">
      </label>
    </section>

    <section>
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-sm font-semibold">Variables</h3>
        <button class="text-xs text-accent hover:underline" @click="addVariable">+ Add variable</button>
      </div>
      <div v-if="definition.variables.length === 0" class="text-xs text-text-muted">No dashboard variables.</div>
      <div v-for="(variable, index) in definition.variables" :key="index" class="grid grid-cols-[1fr_1fr_1fr_auto] gap-2 mb-2">
        <input v-model="variable.name" placeholder="name" class="font-mono text-xs bg-surface-sunken border border-surface-border rounded px-2 py-1.5">
        <input v-model="variable.label" placeholder="Label" class="text-xs bg-surface-sunken border border-surface-border rounded px-2 py-1.5">
        <input v-model="variable.default" placeholder="Default" class="font-mono text-xs bg-surface-sunken border border-surface-border rounded px-2 py-1.5">
        <button class="px-2 text-text-muted hover:text-status-danger" title="Remove variable" @click="definition.variables.splice(index, 1)">×</button>
      </div>
    </section>

    <section>
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-sm font-semibold">Panels</h3>
        <button class="text-xs text-accent hover:underline" @click="addPanel">+ Add panel</button>
      </div>
      <div v-for="(panel, index) in definition.panels" :key="index" class="border border-surface-border rounded-lg p-3 mb-3 space-y-2">
        <div class="grid md:grid-cols-[1fr_2fr_0.8fr_0.8fr_auto] gap-2">
          <input v-model="panel.id" placeholder="panel-id" class="font-mono text-xs bg-surface-sunken border border-surface-border rounded px-2 py-1.5">
          <input v-model="panel.title" placeholder="Panel title" class="text-xs bg-surface-sunken border border-surface-border rounded px-2 py-1.5">
          <select v-model="panel.kind" class="text-xs bg-surface-sunken border border-surface-border rounded px-2 py-1.5">
            <option value="table">Table</option>
            <option value="stat">Stat</option>
            <option value="timeseries">Time series</option>
          </select>
          <select v-model="panel.width" class="text-xs bg-surface-sunken border border-surface-border rounded px-2 py-1.5">
            <option value="half">Half</option>
            <option value="full">Full</option>
          </select>
          <button class="px-2 text-text-muted hover:text-status-danger" title="Remove panel" @click="definition.panels.splice(index, 1)">×</button>
        </div>
        <input v-model="panel.description" placeholder="Description" class="w-full text-xs bg-surface-sunken border border-surface-border rounded px-2 py-1.5">
        <textarea
          v-model="panel.sql"
          spellcheck="false"
          class="w-full min-h-[150px] font-mono text-xs leading-relaxed bg-surface-sunken border border-surface-border rounded p-2.5 focus:outline-none focus:border-accent"
        />
        <div class="flex justify-end">
          <button class="px-2.5 py-1 text-xs rounded border border-surface-border hover:bg-surface-raised" @click="emit('run', panel.id)">Run panel</button>
        </div>
      </div>
    </section>
  </div>
</template>
