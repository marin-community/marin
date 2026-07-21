<script setup lang="ts">
defineProps<{ section: string; title: string }>()
</script>

<template>
  <div class="flex flex-wrap items-end justify-between gap-4">
    <div>
      <p class="mb-1 font-mono text-[11px] uppercase tracking-[0.18em] text-[#657169]">{{ section }}</p>
      <h2 class="text-3xl font-semibold tracking-tight">{{ title }}</h2>
      <slot />
    </div>
    <div v-if="$slots.actions" class="flex items-center gap-2">
      <slot name="actions" />
    </div>
  </div>
</template>
