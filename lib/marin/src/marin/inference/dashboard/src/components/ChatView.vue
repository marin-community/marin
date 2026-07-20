<script setup lang="ts">
import { nextTick, onUnmounted, ref, watch } from 'vue'
import { streamSse } from '../lib/api'
import { CHAT_EXAMPLES } from '../lib/examples'
import { splitThinking } from '../lib/thinking'
import type { Conversation, SamplingParams } from '../lib/types'
import MessageBubble from './MessageBubble.vue'

const props = defineProps<{
  conversation: Conversation
  params: SamplingParams
  model: string
  hasChatTemplate: boolean
}>()

const emit = defineEmits<{ persist: [] }>()

const draft = ref('')
const busy = ref(false)
const scroller = ref<HTMLElement | null>(null)
const composer = ref<HTMLTextAreaElement | null>(null)
let abort: AbortController | null = null

watch(
  () => props.conversation.id,
  () => {
    stopStreaming()
    draft.value = ''
  },
)
onUnmounted(stopStreaming)

function stopStreaming() {
  abort?.abort()
  abort = null
}

function resizeComposer() {
  const el = composer.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 200)}px`
}

function onKeydown(event: KeyboardEvent) {
  if (event.isComposing) return
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    send()
  }
}

// Follow the stream unless the user scrolled up to read something.
watch(
  () => props.conversation.messages.map((m) => m.content.length + m.thinking.length).join(','),
  async () => {
    const el = scroller.value
    if (!el) return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 150
    if (!nearBottom) return
    await nextTick()
    el.scrollTo({ top: el.scrollHeight })
  },
)

async function send(text?: string) {
  const content = (text ?? draft.value).trim()
  if (!content || busy.value) return
  draft.value = ''
  await nextTick()
  resizeComposer()

  const conversation = props.conversation
  if (!conversation.title) conversation.title = content.slice(0, 80)
  conversation.messages.push({ role: 'user', content, thinking: '', thinkingSeconds: null, error: null })

  const request: { role: string; content: string }[] = []
  if (conversation.system.trim()) request.push({ role: 'system', content: conversation.system.trim() })
  // Send visible content only: feeding thinking segments back confuses models.
  for (const message of conversation.messages) request.push({ role: message.role, content: message.content })

  conversation.messages.push({ role: 'assistant', content: '', thinking: '', thinkingSeconds: null, error: null })
  // Mutate through the reactive proxy (not the pushed object) so streaming deltas re-render.
  const reply = conversation.messages[conversation.messages.length - 1]
  conversation.updatedAt = Date.now()
  emit('persist')

  busy.value = true
  abort = new AbortController()
  let rawContent = ''
  let reasoningStream = ''
  let thinkingStartedAt: number | null = null
  try {
    await streamSse(
      'v1/chat/completions',
      {
        model: props.model,
        messages: request,
        stream: true,
        temperature: props.params.temperature,
        max_tokens: props.params.maxTokens,
        top_p: props.params.topP,
      },
      abort.signal,
      (data) => {
        const delta = data.choices?.[0]?.delta
        if (!delta) return
        // Reasoning arrives either as a dedicated delta field (vLLM reasoning
        // parsers) or as raw <think> tags inside content (untouched templates).
        const reasoning = delta.reasoning_content ?? delta.reasoning
        if (reasoning) reasoningStream += reasoning
        if (delta.content) rawContent += delta.content
        const split = splitThinking(rawContent)
        reply.thinking = reasoningStream + split.thinking
        reply.content = split.visible
        if (reply.thinking && thinkingStartedAt === null) thinkingStartedAt = performance.now()
        if (thinkingStartedAt !== null && reply.thinkingSeconds === null && reply.content) {
          reply.thinkingSeconds = (performance.now() - thinkingStartedAt) / 1000
        }
      },
    )
  } catch (error) {
    if (!(error instanceof DOMException && error.name === 'AbortError')) reply.error = String(error)
  } finally {
    if (thinkingStartedAt !== null && reply.thinkingSeconds === null) {
      reply.thinkingSeconds = (performance.now() - thinkingStartedAt) / 1000
    }
    busy.value = false
    abort = null
    conversation.updatedAt = Date.now()
    emit('persist')
  }
}
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col">
    <div ref="scroller" class="min-h-0 flex-1 overflow-y-auto">
      <div v-if="!conversation.messages.length" class="flex h-full items-center justify-center px-6">
        <div class="w-full max-w-lg">
          <div class="mb-1 text-center font-mono text-sm text-text-secondary">{{ model || '…' }}</div>
          <div class="mb-5 text-center text-sm text-text-muted">
            Send a message to start. Conversations stay in this browser.
          </div>
          <div class="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <button
              v-for="example in CHAT_EXAMPLES"
              :key="example"
              class="rounded-xl border border-surface-border bg-surface-raised px-4 py-3 text-left text-sm text-text-secondary transition-colors hover:border-accent hover:text-text"
              @click="send(example)"
            >
              {{ example }}
            </button>
          </div>
          <div v-if="!hasChatTemplate" class="mt-4 text-center text-xs text-text-muted">
            This model reports no chat template — chat requests may fail; try completion mode.
          </div>
        </div>
      </div>
      <div v-else class="mx-auto max-w-3xl space-y-4 px-4 py-5 md:px-6">
        <MessageBubble
          v-for="(message, index) in conversation.messages"
          :key="index"
          :message="message"
          :streaming="busy && index === conversation.messages.length - 1"
        />
      </div>
    </div>

    <div class="border-t border-surface-border px-4 py-3">
      <div class="mx-auto flex max-w-3xl items-end gap-2">
        <textarea
          ref="composer"
          v-model="draft"
          rows="1"
          placeholder="Message… (Enter to send, Shift+Enter for a newline)"
          class="max-h-50 min-h-10 flex-1 resize-none rounded-xl border border-surface-border bg-surface-raised px-3.5 py-2.5 text-[0.925rem] leading-relaxed text-text outline-none transition-colors focus:border-accent"
          @input="resizeComposer"
          @keydown="onKeydown"
        ></textarea>
        <button
          v-if="busy"
          class="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-surface-border text-text-secondary transition-colors hover:border-status-danger hover:text-status-danger"
          title="Stop generating"
          @click="stopStreaming"
        >
          <svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2" /></svg>
        </button>
        <button
          v-else
          class="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent text-surface transition-colors hover:bg-accent-hover disabled:opacity-40"
          :disabled="!draft.trim()"
          title="Send"
          @click="send()"
        >
          <svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
            <path d="M12 19V5m-6 6 6-6 6 6" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>
