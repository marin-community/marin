<script setup lang="ts">
/**
 * Renders a free-form generative sample: the prompt (as a chat transcript when the row carries
 * `prompt_messages`, else a collapsed text block), the generated output with code fences split
 * out, an extracted-vs-target answer strip, and the raw row JSON as a fallback.
 */
import { computed } from 'vue'
import type { SampleRow } from '@/types/api'
import PromptBlock from '@/components/samples/PromptBlock.vue'
import ChatTranscript from '@/components/samples/ChatTranscript.vue'
import FencedText from '@/components/samples/FencedText.vue'

const props = defineProps<{ sample: SampleRow }>()

const extractedClass = computed(() =>
  props.sample.correct
    ? 'border-status-success-border bg-status-success-bg'
    : 'border-status-danger-border bg-status-danger-bg',
)
const targetClass = computed(() =>
  props.sample.correct ? 'border-status-success-border bg-status-success-bg' : 'border-surface-border bg-surface-sunken',
)

const rawJson = computed(() => JSON.stringify(props.sample, null, 2))
</script>

<template>
  <div class="space-y-4">
    <div>
      <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1">Prompt</h3>
      <ChatTranscript v-if="sample.prompt_messages" :messages="sample.prompt_messages" />
      <PromptBlock v-else :text="sample.prompt_text ?? ''" />
    </div>

    <div>
      <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1">Model output</h3>
      <div class="rounded border border-surface-border bg-surface p-3">
        <FencedText :text="sample.output ?? ''" />
      </div>
    </div>

    <div class="grid grid-cols-2 gap-3">
      <div class="rounded border p-3" :class="extractedClass">
        <h4 class="text-[10px] font-semibold uppercase tracking-wider text-text-muted mb-1">Extracted</h4>
        <p class="text-sm font-mono whitespace-pre-wrap">{{ sample.extracted || '—' }}</p>
      </div>
      <div class="rounded border p-3" :class="targetClass">
        <h4 class="text-[10px] font-semibold uppercase tracking-wider text-text-muted mb-1">Target</h4>
        <p class="text-sm font-mono whitespace-pre-wrap">{{ sample.target_text || '—' }}</p>
      </div>
    </div>

    <details>
      <summary class="text-xs text-text-muted cursor-pointer hover:text-text-secondary">Raw sample JSON</summary>
      <pre class="mt-2 rounded border border-surface-border bg-surface-sunken p-3 text-[12px] font-mono overflow-auto max-h-96 whitespace-pre-wrap">{{ rawJson }}</pre>
    </details>
  </div>
</template>
