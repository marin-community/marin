<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>

<template>
  <div class="rounded-lg border border-surface-border bg-surface">
    <div class="px-4 pt-4 pb-2">
      <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">
        {{ title }}
      </h3>
    </div>
    <div class="px-4 pb-4 space-y-2">
      <slot />
    </div>
  </div>
</template>
