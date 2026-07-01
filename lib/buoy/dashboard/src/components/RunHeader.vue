<script setup lang="ts">
import type { Manifest } from '../types'

defineProps<{ manifest: Manifest }>()
defineEmits<{ refetch: [] }>()
</script>

<template>
  <div class="flex flex-wrap items-baseline gap-x-6 gap-y-2">
    <span class="break-all text-lg font-semibold text-accent">{{ manifest.display_name }}</span>
    <span class="rounded-full bg-surface-sunken px-2 py-0.5 text-xs font-semibold uppercase tracking-wide">
      {{ manifest.state }}
    </span>
    <span v-if="manifest.user" class="text-sm text-text-secondary">by <b class="text-text">{{ manifest.user }}</b></span>
    <span v-if="manifest.created_at" class="text-sm text-text-secondary">{{ manifest.created_at }}</span>
    <a :href="manifest.url" target="_blank" rel="noopener" class="text-sm text-accent underline">wandb ↗</a>
    <button
      class="rounded border border-surface-border bg-surface-raised px-2 py-0.5 text-xs text-accent hover:bg-accent-subtle"
      title="re-download everything including the profile artifact"
      @click="$emit('refetch')"
    >
      ↻ refetch
    </button>
  </div>
</template>
