<script setup lang="ts">
import { useRouter } from 'vue-router'
import type { PolicyRejection } from '@/types/api'

defineProps<{ rejections: PolicyRejection[]; scope: string }>()
const router = useRouter()
</script>

<template>
  <details v-if="rejections.length" class="rounded border border-status-warning-border bg-status-warning-bg text-sm p-3">
    <summary class="cursor-pointer text-status-warning">
      {{ rejections.length }} run(s) excluded from {{ scope }}
    </summary>
    <ul class="mt-2 space-y-1 text-text-secondary">
      <li v-for="rejection in rejections" :key="rejection.run_id">
        <button class="font-mono text-accent hover:underline" @click="router.push(`/runs/${rejection.run_id}`)">
          {{ rejection.model }} · {{ rejection.benchmark }}
        </button>
        — {{ rejection.reasons.join('; ') }}
      </li>
    </ul>
  </details>
</template>
