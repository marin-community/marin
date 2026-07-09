<script setup lang="ts">
export interface Stat {
  value: string
  label: string
}

defineProps<{ stats: Stat[] }>()
</script>

<template>
  <div class="flex flex-wrap gap-x-5 gap-y-2">
    <div v-for="stat in stats" :key="stat.label">
      <b class="block text-[17px]">{{ stat.value }}</b>
      <span class="text-xs text-text-muted">{{ stat.label }}</span>
    </div>
  </div>
</template>
