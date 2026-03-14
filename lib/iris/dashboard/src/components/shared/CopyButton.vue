<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  /** The raw value to copy (e.g. "https://10.0.0.1:8080"). Protocol and port are stripped before copying. */
  value: string
  /** Tooltip text shown on hover. */
  title?: string
}>()

const copied = ref(false)

async function copy() {
  const ip = props.value.replace(/^https?:\/\//, '').replace(/:\d+$/, '')
  await navigator.clipboard.writeText(ip)
  copied.value = true
  setTimeout(() => { copied.value = false }, 1500)
}
</script>

<template>
  <button
    class="text-text-muted hover:text-text opacity-0 group-hover/addr:opacity-100 transition-opacity"
    :title="title ?? 'Copy IP'"
    @click="copy"
  >
    <svg v-if="copied" class="w-3.5 h-3.5 text-status-success" viewBox="0 0 20 20" fill="currentColor">
      <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
    </svg>
    <svg v-else class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
    </svg>
  </button>
</template>
