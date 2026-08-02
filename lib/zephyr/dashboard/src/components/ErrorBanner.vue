<script setup lang="ts">
defineProps<{ message: string | null }>()
</script>

<template>
  <div v-if="message" class="rounded-lg border border-status-danger bg-status-danger-bg px-4 py-3 text-sm text-status-danger">
    {{ message }}
  </div>
</template>
