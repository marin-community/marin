<script setup lang="ts">
// A one-to-five score as five cells, with the number beside it.
defineProps<{ value: number; warn?: boolean }>()
</script>

<template>
  <span class="score" :data-warn="warn ? 'true' : null" :title="`${value} of 5`">
    <i v-for="n in 5" :key="n" :data-on="n <= value ? 'true' : 'false'" />
  </span>
</template>
