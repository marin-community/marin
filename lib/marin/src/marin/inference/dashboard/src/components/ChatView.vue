<script setup lang="ts">
import { nextTick, onUnmounted, ref, watch } from 'vue'
import { invokeTool, requestCompletion } from '../lib/api'
import { CHAT_EXAMPLES } from '../lib/examples'
import { splitThinking } from '../lib/thinking'
import { executableToolCalls, inlineToolCalls, mergeToolCallDeltas } from '../lib/tool_calls'
import type { ChatMessage, Conversation, SamplingParams, ToolCall, ToolDefinition } from '../lib/types'
import MessageBubble from './MessageBubble.vue'

const MAX_TOOL_ROUNDS = 8

const props = defineProps<{
  conversation: Conversation
  params: SamplingParams
  model: string
  hasChatTemplate: boolean
  streaming: boolean
  tools: ToolDefinition[]
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

  conversation.updatedAt = Date.now()
  emit('persist')

  busy.value = true
  abort = new AbortController()
  let reply: ChatMessage | null = null
  try {
    for (let round = 0; round < MAX_TOOL_ROUNDS; round += 1) {
      const request = modelMessages(conversation)
      reply = { role: 'assistant', content: '', thinking: '', thinkingSeconds: null, error: null, toolCalls: [] }
      conversation.messages.push(reply)
      // Mutate through the reactive proxy so streaming deltas re-render.
      reply = conversation.messages[conversation.messages.length - 1]
      emit('persist')

      await complete(reply, request, abort.signal)
      const calls = reply.toolCalls ?? []
      conversation.updatedAt = Date.now()
      emit('persist')
      if (!calls.length) break

      for (const call of calls) {
        const result = await callTool(call, abort.signal)
        conversation.messages.push({
          role: 'tool',
          name: call.function.name,
          toolCallId: call.id,
          content: result,
          thinking: '',
          thinkingSeconds: null,
          error: null,
        })
        conversation.updatedAt = Date.now()
        emit('persist')
      }

      if (round === MAX_TOOL_ROUNDS - 1) {
        reply.error = `Stopped after ${MAX_TOOL_ROUNDS} consecutive tool rounds.`
      }
    }
  } catch (error) {
    if (!(error instanceof DOMException && error.name === 'AbortError')) {
      if (!reply) {
        reply = { role: 'assistant', content: '', thinking: '', thinkingSeconds: null, error: null }
        conversation.messages.push(reply)
      }
      reply.error = String(error)
    }
  } finally {
    busy.value = false
    abort = null
    conversation.updatedAt = Date.now()
    emit('persist')
  }
}

type ModelMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | null
  name?: string
  tool_call_id?: string
  tool_calls?: ToolCall[]
}

function modelMessages(conversation: Conversation): ModelMessage[] {
  const request: ModelMessage[] = []
  if (conversation.system.trim()) request.push({ role: 'system', content: conversation.system.trim() })
  for (const message of conversation.messages) {
    if (message.role === 'assistant') {
      request.push({
        role: 'assistant',
        content: message.content || null,
        ...(message.toolCalls?.length ? { tool_calls: message.toolCalls } : {}),
      })
    } else if (message.role === 'tool') {
      request.push({
        role: 'tool',
        content: message.content,
        name: message.name,
        tool_call_id: message.toolCallId,
      })
    } else {
      request.push({ role: 'user', content: message.content })
    }
  }
  return request
}

async function complete(reply: ChatMessage, messages: ModelMessage[], signal: AbortSignal) {
  let rawContent = ''
  let reasoningStream = ''
  let structuredCalls: ToolCall[] = []
  let fallbackCalls: ToolCall[] = []
  let thinkingStartedAt: number | null = null

  const body: Record<string, unknown> = {
    model: props.model,
    messages,
    stream: props.streaming,
    temperature: props.params.temperature,
    max_tokens: props.params.maxTokens,
    top_p: props.params.topP,
  }
  if (props.tools.length) body.tools = props.tools

  await requestCompletion('v1/chat/completions', body, props.streaming, signal, (data) => {
    const delta = data.choices?.[0]?.delta ?? data.choices?.[0]?.message
    if (!delta) return
    const reasoning = delta.reasoning_content ?? delta.reasoning
    if (reasoning) reasoningStream += reasoning
    if (delta.content) rawContent += delta.content
    if (Array.isArray(delta.tool_calls)) structuredCalls = mergeToolCallDeltas(structuredCalls, delta.tool_calls)

    const split = splitThinking(rawContent)
    const inline = inlineToolCalls(split.visible)
    fallbackCalls = inline.calls
    reply.thinking = reasoningStream + split.thinking
    reply.content = inline.visible
    reply.toolCalls = executableToolCalls(structuredCalls.length ? structuredCalls : fallbackCalls)
    if (reply.thinking && thinkingStartedAt === null) thinkingStartedAt = performance.now()
    if (thinkingStartedAt !== null && reply.thinkingSeconds === null && (reply.content || reply.toolCalls.length)) {
      reply.thinkingSeconds = (performance.now() - thinkingStartedAt) / 1000
    }
  })

  reply.toolCalls = executableToolCalls(structuredCalls.length ? structuredCalls : fallbackCalls)
  if (thinkingStartedAt !== null && reply.thinkingSeconds === null) {
    reply.thinkingSeconds = (performance.now() - thinkingStartedAt) / 1000
  }
}

async function callTool(call: ToolCall, signal: AbortSignal): Promise<string> {
  let arguments_: unknown
  try {
    arguments_ = JSON.parse(call.function.arguments || '{}')
  } catch (error) {
    return JSON.stringify({ error: 'model returned invalid JSON tool arguments', details: String(error) })
  }
  if (!arguments_ || typeof arguments_ !== 'object' || Array.isArray(arguments_)) {
    return JSON.stringify({ error: 'model returned non-object tool arguments' })
  }
  try {
    return await invokeTool(call.function.name, arguments_ as Record<string, unknown>, signal)
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    return JSON.stringify({ error: 'tool request failed', details: String(error) })
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
