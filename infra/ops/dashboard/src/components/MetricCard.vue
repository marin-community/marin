<script setup lang="ts">
defineProps<{ label: string; compact?: boolean }>()
</script>

<template>
  <div class="rounded-lg border border-[#d8dcd5] bg-white px-4 py-3 shadow-[0_1px_2px_rgba(20,35,27,.05)]">
    <p class="text-[11px] font-medium uppercase tracking-wide text-[#7a847e]">{{ label }}</p>
    <p class="mt-1 font-semibold" :class="compact ? 'text-sm leading-6' : 'text-2xl'"><slot /></p>
  </div>
</template>
