<script setup lang="ts">
import { computed } from 'vue'
import SearchSelect from '@/components/shared/SearchSelect.vue'
import ModelName from '@/components/shared/ModelName.vue'

export interface Facet {
  key: string
  label: string
  options: string[]
  searchable?: boolean
}

const props = defineProps<{
  facets: Facet[]
  modelValue: Record<string, string>
  resultLabel: string
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
    <template v-for="facet in facets" :key="facet.key">
      <SearchSelect
        v-if="facet.searchable"
        :label="facet.label"
        :options="facet.options"
        :model-value="modelValue[facet.key] ?? ''"
        :class="facet.key === 'model' ? 'flex-1 basis-80' : 'w-56'"
        @update:model-value="setFacet(facet.key, $event)"
      >
        <template v-if="facet.key === 'model'" #option="{ option }"><ModelName :model="option" /></template>
      </SearchSelect>
      <label v-else class="flex flex-col text-xs text-text-secondary gap-1">
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
    </template>

    <slot name="trailing" />

    <div class="flex items-center gap-3 ml-auto text-xs text-text-muted">
      <button
        v-if="activeCount > 0"
        class="px-2 py-1 rounded border border-surface-border hover:bg-surface-raised text-text-secondary"
        @click="clearAll"
      >Clear{{ activeCount > 1 ? ` (${activeCount})` : '' }}</button>
      <span class="tabular-nums whitespace-nowrap">{{ resultLabel }}</span>
    </div>
  </div>
</template>
