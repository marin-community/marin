<script setup lang="ts">
withDefaults(
  defineProps<{
    label?: string
    // Vertical padding of the centering wrapper. "lg" suits a full panel,
    // "sm" a nested section.
    size?: 'sm' | 'lg'
  }>(),
  {
    label: 'Loading…',
    size: 'lg',
  },
)
</script>

<template>
  <div
    role="status"
    :class="[
      'flex items-center justify-center text-text-muted text-sm',
      size === 'lg' ? 'py-12' : 'py-8',
    ]"
  >
    <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" aria-hidden="true">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
    {{ label }}
  </div>
</template>
