<script setup lang="ts">
import { computed, ref } from 'vue'
import type { ServerStatus, ServingInfo } from '../lib/types'

const props = defineProps<{
  info: ServingInfo | null
  status: ServerStatus
  model: string
}>()

const STATUS_LABELS: Record<ServerStatus, string> = {
  connecting: 'connecting…',
  ok: 'ready',
  loading: 'loading…',
  bad: 'unreachable',
}

const statusLabel = computed(() => STATUS_LABELS[props.status])
const dotClass = computed(() => {
  if (props.status === 'ok') return 'bg-status-success'
  if (props.status === 'bad') return 'bg-status-danger'
  return 'bg-text-muted animate-pulse'
})

const chips = computed(() => {
  if (!props.info) return []
  const info = props.info
  return [
    info.backend,
    info.tpu_type,
    `tp ${info.tensor_parallel_size}`,
    info.dtype,
    info.max_model_len ? `${info.max_model_len} ctx` : 'auto ctx',
  ]
})

const dark = ref(document.documentElement.classList.contains('dark'))

function toggleDark() {
  dark.value = !dark.value
  document.documentElement.classList.toggle('dark', dark.value)
  try {
    localStorage.setItem('marin-serve-dark', String(dark.value))
  } catch {}
}
</script>

<template>
  <header class="flex h-12 shrink-0 items-center gap-3 border-b border-surface-border px-4">
    <div class="whitespace-nowrap font-semibold tracking-tight">marin · quick serve</div>
    <div class="flex min-w-0 flex-1 items-center gap-2">
      <span class="truncate font-mono text-sm text-text-secondary" :title="model">{{ model }}</span>
      <span class="hidden items-center gap-1.5 lg:flex">
        <span
          v-for="chip in chips"
          :key="chip"
          class="whitespace-nowrap rounded bg-surface-sunken px-1.5 py-0.5 text-[11px] text-text-muted"
        >
          {{ chip }}
        </span>
      </span>
    </div>
    <div class="flex items-center gap-1.5 whitespace-nowrap text-xs text-text-muted">
      <span class="h-2 w-2 rounded-full" :class="dotClass"></span>
      <span>{{ statusLabel }}</span>
    </div>
    <button
      class="text-text-muted transition-colors hover:text-text"
      :title="dark ? 'Switch to light mode' : 'Switch to dark mode'"
      @click="toggleDark"
    >
      <svg v-if="dark" class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="4" />
        <path
          d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4m11.4-11.4 1.4-1.4"
        />
      </svg>
      <svg v-else class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />
      </svg>
    </button>
  </header>
</template>
