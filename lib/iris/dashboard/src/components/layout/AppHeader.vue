<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>

<template>
  <header class="border-b border-surface-border bg-surface">
    <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
      <h1 class="text-lg font-semibold text-text font-sans tracking-tight">
        {{ title }}
      </h1>
      <div class="flex items-center gap-3">
        <slot />
      </div>
    </div>
  </header>
</template>
