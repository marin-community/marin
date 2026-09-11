<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>

<template>
  <header class="border-b border-surface-border bg-surface">
    <div class="max-w-7xl mx-auto px-3 sm:px-6 py-3 sm:py-4 flex items-center justify-between gap-2 sm:gap-3">
      <h1 class="text-base sm:text-lg font-semibold text-text font-sans tracking-tight truncate min-w-0">
        {{ title }}
      </h1>
      <div class="flex items-center gap-2 sm:gap-3 shrink-0">
        <slot />
      </div>
    </div>
  </header>
</template>
