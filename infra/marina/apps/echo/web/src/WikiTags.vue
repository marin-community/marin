<script setup lang="ts">
defineProps<{ tags: string[] }>()
</script>

<template>
  <div v-if="tags.length" class="flex flex-wrap gap-1.5">
    <span
      v-for="tag in tags"
      :key="tag"
      class="rounded-md bg-mist px-2 py-1 font-mono text-[11px] text-moss"
    >
      {{ tag }}
    </span>
  </div>
</template>
