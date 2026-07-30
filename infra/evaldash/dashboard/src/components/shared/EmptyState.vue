<script setup lang="ts">
defineProps<{
  message: string
  icon?: string
}>()
</script>

<template>
  <div class="flex flex-col items-center justify-center py-16 text-center">
    <span
      v-if="icon"
      class="mb-3 text-3xl text-text-muted select-none"
      aria-hidden="true"
    >{{ icon }}</span>
    <p class="text-sm text-text-muted">{{ message }}</p>
  </div>
</template>
