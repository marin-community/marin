<script setup lang="ts">
/**
 * Renders a marin-serve dashboard conversation: the system prompt and template instructions, then
 * each turn with its reasoning, tool calls, tool results, and errors.
 */
import type { PlaygroundConversation, PlaygroundMessage } from '@/types/api'

defineProps<{ conversation: PlaygroundConversation }>()

const ROLE_BORDER: Record<string, string> = {
  system: 'border-l-text-muted',
  user: 'border-l-accent',
  assistant: 'border-l-status-success',
  tool: 'border-l-status-warning',
}

// Strings print as-is; other values print as indented JSON.
function displayValue(value: unknown): string {
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2)
}

function label(message: PlaygroundMessage): string {
  return message.role === 'tool' ? `tool result · ${message.name}` : message.role
}
</script>

<template>
  <div class="space-y-2">
    <div
      v-if="conversation.system"
      class="rounded border border-surface-border bg-surface-sunken border-l-4 p-3"
      :class="ROLE_BORDER.system"
    >
      <div class="text-[10px] font-semibold uppercase tracking-wider text-text-muted mb-1">system</div>
      <div class="text-[13px] whitespace-pre-wrap">{{ conversation.system }}</div>
    </div>
    <div
      v-if="conversation.customInstructions"
      class="rounded border border-surface-border bg-surface-sunken border-l-4 p-3"
      :class="ROLE_BORDER.system"
    >
      <div class="text-[10px] font-semibold uppercase tracking-wider text-text-muted mb-1">custom template instructions</div>
      <div class="text-[13px] whitespace-pre-wrap">{{ conversation.customInstructions }}</div>
    </div>
    <div
      v-for="(m, i) in conversation.messages"
      :key="i"
      class="rounded border border-surface-border bg-surface-sunken border-l-4 p-3 space-y-2"
      :class="ROLE_BORDER[m.role] ?? 'border-l-surface-border'"
    >
      <div class="text-[10px] font-semibold uppercase tracking-wider text-text-muted">{{ label(m) }}</div>
      <template v-if="m.role === 'tool'">
        <pre class="text-[12px] whitespace-pre-wrap font-mono">{{ displayValue(m.result) }}</pre>
      </template>
      <template v-else-if="m.role === 'assistant'">
        <details v-if="m.thinking" class="text-[12px] text-text-secondary">
          <summary class="cursor-pointer select-none">Reasoning</summary>
          <div class="mt-1 whitespace-pre-wrap">{{ m.thinking }}</div>
        </details>
        <div v-if="m.content" class="text-[13px] whitespace-pre-wrap">{{ m.content }}</div>
        <div v-for="call in m.toolCalls ?? []" :key="call.id" class="text-[12px]">
          <span class="text-text-muted">tool call · </span><span class="font-mono">{{ call.name }}</span>
          <pre class="mt-1 whitespace-pre-wrap font-mono text-text-secondary">{{ displayValue(call.arguments) }}</pre>
        </div>
        <div v-if="m.error" class="text-[12px] text-status-danger whitespace-pre-wrap">Error: {{ m.error }}</div>
        <div v-if="m.finishReason === 'length'" class="text-[11px] text-status-warning">Stopped at the token limit.</div>
      </template>
      <div v-else class="text-[13px] whitespace-pre-wrap">{{ m.content }}</div>
    </div>
  </div>
</template>
