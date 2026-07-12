<script setup lang="ts">
import { useId } from 'vue'

defineProps<{ sources: string[] }>()
const model = defineModel<string>({ required: true })
const listId = useId()
</script>

<template>
  <span class="flex items-center gap-1.5 text-sm text-text-secondary">
    source
    <input
      v-model="model"
      :list="listId"
      placeholder="type to filter source…"
      autocomplete="off"
      spellcheck="false"
      class="w-[280px] rounded border border-surface-border bg-surface px-2 py-1 text-[13px] text-text"
    />
    <datalist :id="listId">
      <option v-for="source in sources" :key="source" :value="source"></option>
    </datalist>
  </span>
</template>
