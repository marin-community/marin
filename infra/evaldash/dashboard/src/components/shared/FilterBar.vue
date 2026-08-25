<script setup lang="ts">
/**
 * A faceted filter strip: one labeled dropdown per facet, a "Clear all" affordance that appears once
 * anything is selected, and a live result count. Presentational only -- the parent supplies the facet
 * options (derived from the loaded data, so they reflect what is actually present) and owns the
 * filtering; this component just reads and writes the selection via v-model.
 */
import { computed } from 'vue'

export interface Facet {
  key: string
  label: string
  options: string[]
}

const props = defineProps<{
  facets: Facet[]
  modelValue: Record<string, string>
  resultCount: number
  totalCount: number
}>()

const emit = defineEmits<{ 'update:modelValue': [Record<string, string>] }>()

const activeCount = computed(() => Object.values(props.modelValue).filter(Boolean).length)

function setFacet(key: string, value: string) {
  emit('update:modelValue', { ...props.modelValue, [key]: value })
}

function clearAll() {
  emit('update:modelValue', {})
}
</script>

<template>
  <div class="flex flex-wrap items-end gap-3">
    <label v-for="facet in facets" :key="facet.key" class="flex flex-col text-xs text-text-secondary gap-1">
      {{ facet.label }}
      <select
        :value="modelValue[facet.key] ?? ''"
        class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[9rem]"
        @change="setFacet(facet.key, ($event.target as HTMLSelectElement).value)"
      >
        <option value="">All</option>
        <option v-for="opt in facet.options" :key="opt" :value="opt">{{ opt }}</option>
      </select>
    </label>

    <slot name="trailing" />

    <div class="flex items-center gap-3 ml-auto text-xs text-text-muted">
      <button
        v-if="activeCount > 0"
        class="px-2 py-1 rounded border border-surface-border hover:bg-surface-raised text-text-secondary"
        @click="clearAll"
      >Clear{{ activeCount > 1 ? ` (${activeCount})` : '' }}</button>
      <span class="tabular-nums whitespace-nowrap">{{ resultCount }} of {{ totalCount }}</span>
    </div>
  </div>
</template>
