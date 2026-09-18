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

const rendered = computed(() =>
  props.message.role === 'assistant' ? renderMarkdown(props.message.content) : '',
)
const thinkingActive = computed(
  () => props.message.role === 'assistant' && props.streaming && !props.message.content,
)
const empty = computed(
  () =>
    props.message.role === 'assistant' &&
    !props.streaming &&
    !props.message.content &&
    !props.message.thinking &&
    !props.message.toolCalls?.length &&
    !props.message.error,
)

const copied = ref(false)

async function copy() {
  if (props.message.role !== 'assistant') return
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

  <div v-else-if="message.role === 'tool'" class="flex justify-start">
    <details class="max-w-[92%] rounded-lg border border-surface-border bg-surface-sunken px-3 py-2 text-xs">
      <summary class="cursor-pointer font-mono text-text-secondary">
        {{ message.name }} result
      </summary>
      <pre class="mt-2 overflow-x-auto whitespace-pre-wrap break-words text-text-muted">{{
        JSON.stringify(message.result, null, 2)
      }}</pre>
    </details>
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
      <div v-if="message.toolCalls?.length" class="mt-2 space-y-1.5">
        <details
          v-for="call in message.toolCalls"
          :key="call.id"
          class="rounded-lg border border-surface-border bg-surface-sunken px-3 py-2 text-xs"
        >
          <summary class="cursor-pointer font-mono text-text-secondary">{{ call.name }}</summary>
          <pre class="mt-2 overflow-x-auto whitespace-pre-wrap break-words text-text-muted">{{
            JSON.stringify(call.arguments, null, 2)
          }}</pre>
        </details>
      </div>
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
