<script setup lang="ts">
import { ThinkingMode, type ThinkingMode as ThinkingModeValue } from '../lib/chat_template'
import type { SamplingParams } from '../lib/types'

defineProps<{
  params: SamplingParams
  showChatControls: boolean
}>()

const system = defineModel<string>('system', { default: '' })
const thinkingMode = defineModel<ThinkingModeValue>('thinkingMode', { default: ThinkingMode.TemplateDefault })
const customInstructions = defineModel<string>('customInstructions', { default: '' })
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
    <label v-if="showChatControls" class="flex min-w-60 flex-1 flex-col gap-1 text-xs text-text-muted">
      System prompt
      <textarea
        v-model="system"
        rows="2"
        placeholder="(none)"
        class="resize-y rounded-lg border border-surface-border bg-surface px-2 py-1.5 font-mono text-[0.8rem] text-text"
      ></textarea>
      <span>Sent as a <code>role: "system"</code> message in the conversation.</span>
    </label>
    <label v-if="showChatControls" class="flex min-w-44 flex-col gap-1 text-xs text-text-muted">
      Thinking
      <select
        v-model="thinkingMode"
        class="rounded-lg border border-surface-border bg-surface px-2 py-1.5 text-[0.8rem] text-text"
      >
        <option :value="ThinkingMode.TemplateDefault">Template default</option>
        <option :value="ThinkingMode.Enabled">Enabled</option>
        <option :value="ThinkingMode.Disabled">Disabled</option>
      </select>
      <span>Passed to the active model template as <code>enable_thinking</code>. Templates may ignore it.</span>
    </label>
    <label v-if="showChatControls" class="flex min-w-72 flex-1 flex-col gap-1 text-xs text-text-muted">
      Custom template instructions
      <textarea
        v-model="customInstructions"
        rows="2"
        placeholder="Optional instructions for templates that support them"
        class="resize-y rounded-lg border border-surface-border bg-surface px-2 py-1.5 font-mono text-[0.8rem] text-text"
      ></textarea>
      <span>Passed to the active model template as <code>custom_instructions</code>.</span>
    </label>
  </div>
</template>
