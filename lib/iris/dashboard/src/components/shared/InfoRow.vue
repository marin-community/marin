<script setup lang="ts">
defineProps<{
  label: string
}>()
</script>

<template>
  <div class="flex items-baseline justify-between gap-4 text-sm">
    <span class="flex-shrink-0 text-text-secondary">{{ label }}</span>
    <span class="text-right text-text truncate">
      <slot />
    </span>
  </div>
</template>
