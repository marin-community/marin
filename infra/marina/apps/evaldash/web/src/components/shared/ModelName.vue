<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ model: string }>()
// The catalog identity is name@config-digest. Keep both parts visible and the exact ID searchable.
const parts = computed(() => props.model.match(/^(.*)(@[a-f0-9]{12})$/))
</script>

<template>
  <span class="block min-w-0 font-mono [overflow-wrap:anywhere]" :title="model">
    <span>{{ parts?.[1] ?? model }}</span>
    <span v-if="parts" class="block text-[10px] font-normal text-text-muted">{{ parts[2] }}</span>
  </span>
</template>
