<script setup lang="ts">
/**
 * Renders a chat-formatted prompt (parsed from a JSON `[{role, content}, ...]` string) as a
 * sequence of role-labeled turn cards, left-border colored per role.
 */
import type { ChatMessage } from '@/types/api'

defineProps<{ messages: ChatMessage[] }>()

const ROLE_BORDER: Record<string, string> = {
  system: 'border-l-text-muted',
  user: 'border-l-accent',
  assistant: 'border-l-status-success',
}

function borderClass(role: string): string {
  return ROLE_BORDER[role] ?? 'border-l-surface-border'
}
</script>

<template>
  <div class="space-y-2">
    <div
      v-for="(m, i) in messages"
      :key="i"
      class="rounded border border-surface-border bg-surface-sunken border-l-4 p-3"
      :class="borderClass(m.role)"
    >
      <div class="text-[10px] font-semibold uppercase tracking-wider text-text-muted mb-1">{{ m.role }}</div>
      <div class="text-[13px] whitespace-pre-wrap">{{ m.content }}</div>
    </div>
  </div>
</template>
