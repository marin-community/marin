<script setup lang="ts">
import type { Conversation } from '../lib/types'

defineProps<{
  conversations: Conversation[]
  activeId: string
  /** Below the md breakpoint the panel renders as an overlay drawer. */
  mobileOpen: boolean
}>()

const emit = defineEmits<{
  select: [id: string]
  new: []
  remove: [id: string]
  clear: []
}>()

function timeAgo(timestamp: number): string {
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000))
  if (seconds < 60) return 'just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

function confirmClear() {
  if (window.confirm('Delete all saved conversations?')) emit('clear')
}
</script>

<template>
  <aside
    class="w-64 shrink-0 flex-col border-r border-surface-border bg-surface-raised md:static md:z-auto md:flex md:shadow-none"
    :class="mobileOpen ? 'absolute inset-y-0 left-0 z-20 flex shadow-xl' : 'hidden'"
  >
    <div class="p-3">
      <button
        class="w-full rounded-lg bg-accent px-3 py-2 text-sm font-semibold text-surface transition-colors hover:bg-accent-hover"
        @click="emit('new')"
      >
        New chat
      </button>
    </div>
    <div class="flex-1 space-y-0.5 overflow-y-auto px-2 pb-2">
      <div
        v-for="conversation in conversations"
        :key="conversation.id"
        class="group flex cursor-pointer items-center gap-1 rounded-lg px-2.5 py-2 text-sm"
        :class="
          conversation.id === activeId ? 'bg-accent-subtle text-text' : 'text-text-secondary hover:bg-surface-sunken'
        "
        @click="emit('select', conversation.id)"
      >
        <div class="min-w-0 flex-1">
          <div class="truncate">{{ conversation.title || 'Untitled' }}</div>
          <div class="text-xs text-text-muted">{{ timeAgo(conversation.updatedAt) }}</div>
        </div>
        <button
          class="px-1 text-text-muted opacity-0 transition-opacity hover:text-status-danger group-hover:opacity-100"
          title="Delete conversation"
          @click.stop="emit('remove', conversation.id)"
        >
          ×
        </button>
      </div>
      <div v-if="!conversations.length" class="px-2.5 py-2 text-xs leading-relaxed text-text-muted">
        No saved chats yet. Conversations are stored in this browser.
      </div>
    </div>
    <div v-if="conversations.length" class="border-t border-surface-border p-3">
      <button class="text-xs text-text-muted transition-colors hover:text-status-danger" @click="confirmClear">
        Clear history
      </button>
    </div>
  </aside>
</template>
