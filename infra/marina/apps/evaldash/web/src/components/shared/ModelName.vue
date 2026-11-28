<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ model: string }>()
const parts = computed(() => props.model.match(/^(.*)(@[a-f0-9]{12})$/))
</script>

<template>
  <span class="block min-w-0 font-sans [overflow-wrap:anywhere]" :title="model">
    <span>{{ parts?.[1] ?? model }}</span>
    <span v-if="parts" class="block font-mono text-xs font-normal text-text-muted">{{ parts[2] }}</span>
  </span>
</template>
