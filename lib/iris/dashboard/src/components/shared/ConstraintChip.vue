<script setup lang="ts">
defineProps<{
  constraint: string
}>()
</script>

<template>
  <span
    class="inline-block rounded bg-surface-sunken px-2 py-0.5 font-mono text-xs text-text-secondary"
  >{{ constraint }}</span>
</template>
