<script setup lang="ts">
/**
 * Collapsible display for a long shared prompt (MCQ few-shot context, or a completions prompt).
 * Defaults to showing just the tail — the last paragraph, kept between roughly 400 and 1500
 * characters — with a toggle to reveal the full text.
 */
import { computed, ref } from 'vue'

const props = defineProps<{ text: string }>()

const PREVIEW_MAX = 1500
const PREVIEW_MIN = 400

const expanded = ref(false)

const tail = computed(() => {
  const text = props.text
  if (text.length <= PREVIEW_MIN) return text
  const floor = text.slice(-PREVIEW_MIN)
  const boundary = text.lastIndexOf('\n\n')
  if (boundary < text.length - PREVIEW_MAX) return floor
  const candidate = text.slice(boundary + 2)
  return candidate.length >= PREVIEW_MIN ? candidate : floor
})

const truncated = computed(() => tail.value.length < props.text.length)
const shown = computed(() => (expanded.value ? props.text : tail.value))
</script>

<template>
  <div>
    <div class="rounded border border-surface-border bg-surface-sunken p-3 text-[13px] font-mono whitespace-pre-wrap">{{ shown }}</div>
    <button
      v-if="truncated"
      class="mt-1 text-xs text-accent hover:text-accent-hover hover:underline"
      @click="expanded = !expanded"
    >{{ expanded ? 'Collapse' : `Show full prompt (${text.length} chars)` }}</button>
  </div>
</template>
