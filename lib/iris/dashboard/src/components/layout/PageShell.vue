<script setup lang="ts">
import { RouterLink } from 'vue-router'

defineProps<{
  title: string
  backTo?: string
  backLabel?: string
}>()
</script>

<template>
  <div class="max-w-7xl mx-auto px-6 py-6">
    <div class="mb-4 flex items-center gap-2 text-sm">
      <RouterLink
        v-if="backTo"
        :to="backTo"
        class="text-accent hover:underline"
      >
        &larr; {{ backLabel ?? 'Back' }}
      </RouterLink>
    </div>
    <div class="flex items-center gap-2 mb-6 min-w-0">
      <h2 class="text-lg sm:text-xl font-semibold text-text font-mono break-anywhere min-w-0">{{ title }}</h2>
      <slot name="title-suffix" />
    </div>
    <slot />
  </div>
</template>
