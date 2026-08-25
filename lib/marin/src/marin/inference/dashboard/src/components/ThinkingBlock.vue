<script setup lang="ts">
import { computed, ref, watch } from 'vue'

const props = defineProps<{
  thinking: string
  /** True while the model is still emitting reasoning (before visible content). */
  active: boolean
  seconds: number | null
}>()

const open = ref(props.active)
const body = ref<HTMLElement | null>(null)

// Follow the stream: expanded while reasoning, collapsed once the answer starts.
watch(
  () => props.active,
  (active) => {
    open.value = active
  },
)

// Keep the live reasoning scrolled to its tail.
watch(
  () => props.thinking,
  () => {
    if (props.active && body.value) body.value.scrollTop = body.value.scrollHeight
  },
  { flush: 'post' },
)

const label = computed(() => {
  if (props.active) return 'Thinking…'
  if (props.seconds === null) return 'Thought'
  const total = Math.max(1, Math.round(props.seconds))
  return total >= 60 ? `Thought for ${Math.floor(total / 60)}m ${total % 60}s` : `Thought for ${total}s`
})
</script>

<template>
  <div class="mb-2">
    <button
      class="flex items-center gap-1 text-xs text-text-muted transition-colors hover:text-text-secondary"
      :class="{ 'animate-pulse': active }"
      @click="open = !open"
    >
      <svg
        class="h-3 w-3 transition-transform"
        :class="{ 'rotate-90': open }"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2.5"
      >
        <path d="m9 6 6 6-6 6" />
      </svg>
      {{ label }}
    </button>
    <div
      v-show="open"
      ref="body"
      class="mt-1.5 max-h-72 overflow-y-auto whitespace-pre-wrap break-words border-l-2 border-surface-border pl-3 text-[0.85rem] leading-relaxed text-text-muted"
    >
      {{ thinking }}
    </div>
  </div>
</template>
