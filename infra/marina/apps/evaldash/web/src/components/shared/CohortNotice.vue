<script setup lang="ts">
defineProps<{ message: string }>()
</script>

<template>
  <p class="cohort-notice" role="note" aria-label="Cohort warning">{{ message }}</p>
</template>

<style scoped>
.cohort-notice {
  border-inline-start: 3px solid var(--c-status-warning);
  padding: .35rem .7rem;
  background: var(--c-status-warning-bg);
  color: var(--c-text);
  font-size: .8125rem;
  line-height: 1.4;
}
</style>
