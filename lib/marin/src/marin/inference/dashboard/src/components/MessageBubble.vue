<script setup lang="ts">
import { computed, ref } from 'vue'
import { renderMarkdown } from '../lib/markdown'
import type { ChatMessage } from '../lib/types'
import ThinkingBlock from './ThinkingBlock.vue'

const props = defineProps<{
  message: ChatMessage
  /** True while this message is the one currently being streamed. */
  streaming: boolean
}>()

const rendered = computed(() => renderMarkdown(props.message.content))
const thinkingActive = computed(() => props.streaming && !props.message.content)
const empty = computed(
  () => !props.streaming && !props.message.content && !props.message.thinking && !props.message.error,
)

const copied = ref(false)

async function copy() {
  await navigator.clipboard.writeText(props.message.content)
  copied.value = true
  setTimeout(() => (copied.value = false), 1200)
}
</script>

<template>
  <div v-if="message.role === 'user'" class="flex justify-end">
    <div
      class="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md border border-surface-border bg-accent-subtle px-4 py-2.5 text-[0.925rem] leading-relaxed"
    >
      {{ message.content }}
    </div>
  </div>

  <div v-else class="group flex justify-start">
    <div class="min-w-0 max-w-[92%]">
      <ThinkingBlock
        v-if="message.thinking"
        :thinking="message.thinking"
        :active="thinkingActive"
        :seconds="message.thinkingSeconds"
      />
      <div v-if="message.content" class="markdown-body text-[0.925rem] leading-relaxed" v-html="rendered"></div>
      <div v-if="streaming && !message.content && !message.thinking" class="flex gap-1 py-2">
        <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-text-muted"></span>
        <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-text-muted [animation-delay:150ms]"></span>
        <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-text-muted [animation-delay:300ms]"></span>
      </div>
      <div v-if="empty" class="text-sm italic text-text-muted">(no output)</div>
      <div v-if="message.error" class="mt-1 whitespace-pre-wrap break-words font-mono text-sm text-status-danger">
        {{ message.error }}
      </div>
      <button
        v-if="!streaming && message.content"
        class="mt-1 text-xs text-text-muted opacity-0 transition-opacity hover:text-text-secondary group-hover:opacity-100"
        @click="copy"
      >
        {{ copied ? 'copied' : 'copy' }}
      </button>
    </div>
  </div>
</template>
