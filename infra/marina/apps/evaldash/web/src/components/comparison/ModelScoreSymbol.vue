<script setup lang="ts">
defineProps<{ index: number }>()
</script>

<template>
  <span class="model-score-symbol" :data-symbol="index" aria-hidden="true"></span>
</template>

<style scoped>
.model-score-symbol {
  display: inline-block;
  flex: none;
  width: 0.46rem;
  height: 0.46rem;
  background: var(--c-accent);
}

.model-score-symbol[data-symbol='0'] {
  border-radius: 50%;
}

.model-score-symbol[data-symbol='1'] {
  transform: rotate(45deg);
  background: #9a4d08;
}

.model-score-symbol[data-symbol='2'] {
  background: #087561;
}

.model-score-symbol[data-symbol='3'] {
  clip-path: polygon(50% 0, 100% 100%, 0 100%);
  background: #864186;
}

@media (prefers-color-scheme: dark) {
  .model-score-symbol[data-symbol='1'] { background: #efad68; }
  .model-score-symbol[data-symbol='2'] { background: #62ccb4; }
  .model-score-symbol[data-symbol='3'] { background: #e4a4e4; }
}
</style>
