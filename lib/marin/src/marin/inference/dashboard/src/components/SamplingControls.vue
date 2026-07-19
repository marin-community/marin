<script setup lang="ts">
import type { SamplingParams } from '../lib/types'

defineProps<{
  params: SamplingParams
  /** Chat mode also edits the conversation's system prompt. */
  showSystem: boolean
}>()

const system = defineModel<string>('system', { default: '' })
</script>

<template>
  <div class="flex flex-wrap items-start gap-x-5 gap-y-3 border-b border-surface-border bg-surface-raised px-4 py-3">
    <label class="flex flex-col gap-1 text-xs text-text-muted">
      Temperature
      <input
        v-model.number="params.temperature"
        type="number"
        step="0.1"
        min="0"
        max="2"
        class="w-24 rounded-lg border border-surface-border bg-surface px-2 py-1.5 text-sm text-text"
      />
    </label>
    <label class="flex flex-col gap-1 text-xs text-text-muted">
      Max tokens
      <input
        v-model.number="params.maxTokens"
        type="number"
        min="1"
        step="1"
        class="w-28 rounded-lg border border-surface-border bg-surface px-2 py-1.5 text-sm text-text"
      />
    </label>
    <label class="flex flex-col gap-1 text-xs text-text-muted">
      Top-p
      <input
        v-model.number="params.topP"
        type="number"
        step="0.05"
        min="0"
        max="1"
        class="w-24 rounded-lg border border-surface-border bg-surface px-2 py-1.5 text-sm text-text"
      />
    </label>
    <label v-if="showSystem" class="flex min-w-60 flex-1 flex-col gap-1 text-xs text-text-muted">
      System prompt
      <textarea
        v-model="system"
        rows="2"
        placeholder="(none)"
        class="resize-y rounded-lg border border-surface-border bg-surface px-2 py-1.5 font-mono text-[0.8rem] text-text"
      ></textarea>
    </label>
  </div>
</template>
