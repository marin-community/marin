<script setup lang="ts">
import type { ProfilerType } from '@/composables/useProfileAction'

defineProps<{
  profiling: boolean
}>()

const emit = defineEmits<{
  profile: [type: ProfilerType]
}>()
</script>

<template>
  <div class="flex items-center gap-3">
    <span class="text-xs text-text-secondary font-medium uppercase tracking-wider">Profiling</span>
    <button
      class="px-3 py-1.5 text-xs border border-surface-border rounded hover:bg-surface-raised text-text-secondary disabled:opacity-50"
      :disabled="profiling"
      @click="emit('profile', 'cpu')"
    >
      {{ profiling ? '⏳' : 'CPU Profile' }}
    </button>
    <button
      class="px-3 py-1.5 text-xs border border-surface-border rounded hover:bg-surface-raised text-text-secondary disabled:opacity-50"
      :disabled="profiling"
      @click="emit('profile', 'memory')"
    >
      {{ profiling ? '⏳' : 'Memory Profile' }}
    </button>
    <button
      class="px-3 py-1.5 text-xs border border-surface-border rounded hover:bg-surface-raised text-text-secondary disabled:opacity-50"
      :disabled="profiling"
      @click="emit('profile', 'threads')"
    >
      {{ profiling ? '⏳' : 'Thread Dump' }}
    </button>
  </div>
</template>
