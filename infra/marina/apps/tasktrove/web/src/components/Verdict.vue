<script setup lang="ts">
// Whether shellsim runs it: yes, partial or no, as a pill.
defineProps<{ value: 'yes' | 'partial' | 'no' | undefined; prefix?: string }>()
</script>

<template>
  <span
    v-if="value"
    class="pill"
    :data-good="value === 'yes' ? 'true' : value === 'no' ? 'false' : null"
    >{{ prefix ?? '' }}{{ value }}</span
  >
</template>
