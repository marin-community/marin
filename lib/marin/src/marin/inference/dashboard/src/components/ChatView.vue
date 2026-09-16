<script setup lang="ts">
import { nextTick, onUnmounted, ref, watch } from 'vue'
import { invokeTool, requestCompletion } from '../lib/api'
import { CHAT_EXAMPLES } from '../lib/examples'
import type { ChatExample } from '../lib/examples'
import { splitThinking } from '../lib/thinking'
import { executableToolCalls, inlineToolCalls, mergeToolCallDeltas } from '../lib/tool_calls'
import type { ChatMessage, Conversation, SamplingParams, ToolCall } from '../lib/types'
import MessageBubble from './MessageBubble.vue'

const MAX_TOOL_ROUNDS = 8
const ABORT_ERROR_NAME = 'AbortError'

const props = defineProps<{
  conversation: Conversation
  params: SamplingParams
  model: string
  hasChatTemplate: boolean
  streaming: boolean
}>()

const emit = defineEmits<{ persist: [] }>()

const draft = ref('')
const busy = ref(false)
const showTools = ref(false)
const scroller = ref<HTMLElement | null>(null)
const composer = ref<HTMLTextAreaElement | null>(null)
let abort: AbortController | null = null

watch(
  () => props.conversation.id,
  () => {
    stopStreaming()
    draft.value = ''
    showTools.value = Boolean(props.conversation.pythonTools)
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

async function useExample(example: ChatExample) {
  if (example.pythonTools !== undefined) {
    props.conversation.pythonTools = example.pythonTools
    showTools.value = true
  }
  await send(example.prompt)
}

async function send(text?: string) {
  const content = (text ?? draft.value).trim()
  if (!content || busy.value) return
  draft.value = ''
  await nextTick()
  resizeComposer()

  const conversation = props.conversation
  const pythonTools = conversation.pythonTools.trim()
  if (!conversation.title) conversation.title = content.slice(0, 80)
  conversation.messages.push({ role: 'user', content, thinking: '', thinkingSeconds: null, error: null })

  conversation.updatedAt = Date.now()
  emit('persist')

  busy.value = true
  abort = new AbortController()
  let reply: ChatMessage | null = null
  try {
    for (let round = 0; round < MAX_TOOL_ROUNDS; round += 1) {
      const request = modelMessages(conversation, pythonTools)
      reply = { role: 'assistant', content: '', thinking: '', thinkingSeconds: null, error: null, toolCalls: [] }
      conversation.messages.push(reply)
      // Mutate through the reactive proxy so streaming deltas re-render.
      reply = conversation.messages[conversation.messages.length - 1]
      emit('persist')

      await complete(reply, request, Boolean(pythonTools), abort.signal)
      const calls = reply.toolCalls ?? []
      conversation.updatedAt = Date.now()
      emit('persist')
      if (!calls.length) break

      for (const call of calls) {
        const result = await callTool(call, pythonTools, abort.signal)
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
    if (error instanceof DOMException && error.name === ABORT_ERROR_NAME) {
      appendCancelledToolResults(conversation, reply)
    } else {
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

function appendCancelledToolResults(conversation: Conversation, reply: ChatMessage | null) {
  const completed = new Set(
    conversation.messages.filter((message) => message.role === 'tool').map((message) => message.toolCallId),
  )
  for (const call of reply?.toolCalls ?? []) {
    if (completed.has(call.id)) continue
    conversation.messages.push({
      role: 'tool',
      name: call.function.name,
      toolCallId: call.id,
      content: JSON.stringify({ error: 'tool call cancelled' }),
      thinking: '',
      thinkingSeconds: null,
      error: null,
    })
  }
}

type ModelMessage = {
  role: 'system' | 'user' | 'assistant'
  content: string
}

function modelMessages(conversation: Conversation, pythonTools: string): ModelMessage[] {
  const request: ModelMessage[] = []
  const system = [conversation.system.trim(), pythonTools ? pythonToolInstructions(pythonTools) : '']
    .filter(Boolean)
    .join('\n\n')
  if (system) request.push({ role: 'system', content: system })
  for (let index = 0; index < conversation.messages.length; index += 1) {
    const message = conversation.messages[index]
    if (message.role === 'assistant') {
      request.push({
        role: 'assistant',
        content: [message.content, ...(message.toolCalls ?? []).map(pythonToolCallMessage)].filter(Boolean).join('\n'),
      })
    } else if (message.role === 'tool') {
      const results = [pythonToolResultMessage(message)]
      while (conversation.messages[index + 1]?.role === 'tool') {
        index += 1
        results.push(pythonToolResultMessage(conversation.messages[index]))
      }
      request.push({
        role: 'user',
        content: results.join('\n'),
      })
    } else {
      request.push({ role: 'user', content: message.content })
    }
  }
  return request
}

function pythonToolInstructions(source: string): string {
  const cdataSource = source.split(']]>').join(']]]]><![CDATA[>')
  return `The user provided executable Python functions inside this XML block:
<python_tools><![CDATA[
${cdataSource}
]]></python_tools>
Call a function only by emitting this exact XML form with JSON arguments:
<tool_call>{"name":"function_name","arguments":{"parameter":"value"}}</tool_call>
The application will execute the function and return a <tool_result> XML element in the next user message. Use that
result to answer the user or make another call. Do not invent functions outside the block.`
}

function pythonToolCallMessage(call: ToolCall): string {
  let arguments_: unknown = {}
  try {
    const parsed = JSON.parse(call.function.arguments)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) arguments_ = parsed
  } catch {
    // The matching tool result tells the model that its arguments were invalid.
  }
  return `<tool_call>${JSON.stringify({ name: call.function.name, arguments: arguments_ })}</tool_call>`
}

function pythonToolResultMessage(message: ChatMessage): string {
  let result: unknown = message.content
  try {
    result = JSON.parse(message.content)
  } catch {
    // Tool endpoints normally return JSON; preserve unexpected output as a string.
  }
  const payload = JSON.stringify({ name: message.name, result }).split(']]>').join(']]]]><![CDATA[>')
  return `<tool_result><![CDATA[${payload}]]></tool_result>`
}

async function complete(reply: ChatMessage, messages: ModelMessage[], toolsEnabled: boolean, signal: AbortSignal) {
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
  await requestCompletion('v1/chat/completions', body, props.streaming, signal, (data) => {
    const delta = data.choices?.[0]?.delta ?? data.choices?.[0]?.message
    if (!delta) return
    const reasoning = delta.reasoning_content ?? delta.reasoning
    if (reasoning) reasoningStream += reasoning
    if (delta.content) rawContent += delta.content
    if (toolsEnabled && Array.isArray(delta.tool_calls)) {
      structuredCalls = mergeToolCallDeltas(structuredCalls, delta.tool_calls)
    }

    const split = splitThinking(rawContent)
    reply.thinking = reasoningStream + split.thinking
    if (toolsEnabled) {
      const inline = inlineToolCalls(split.visible)
      fallbackCalls = inline.calls
      reply.content = inline.visible
      reply.toolCalls = executableToolCalls(structuredCalls.length ? structuredCalls : fallbackCalls)
    } else {
      reply.content = split.visible
      reply.toolCalls = []
    }
    if (reply.thinking && thinkingStartedAt === null) thinkingStartedAt = performance.now()
    if (thinkingStartedAt !== null && reply.thinkingSeconds === null && (reply.content || reply.toolCalls.length)) {
      reply.thinkingSeconds = (performance.now() - thinkingStartedAt) / 1000
    }
  })

  if (toolsEnabled) reply.toolCalls = executableToolCalls(structuredCalls.length ? structuredCalls : fallbackCalls)
  if (thinkingStartedAt !== null && reply.thinkingSeconds === null) {
    reply.thinkingSeconds = (performance.now() - thinkingStartedAt) / 1000
  }
}

async function callTool(call: ToolCall, source: string, signal: AbortSignal): Promise<string> {
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
    return await invokeTool(call.function.name, source, arguments_ as Record<string, unknown>, signal)
  } catch (error) {
    if (error instanceof DOMException && error.name === ABORT_ERROR_NAME) throw error
    return JSON.stringify({ error: 'tool request failed', details: String(error) })
  }
}
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col">
    <div ref="scroller" class="min-h-0 flex-1 overflow-y-auto">
      <div v-if="!conversation.messages.length" class="flex h-full items-center justify-center px-6">
        <div class="w-full max-w-2xl">
          <div class="mb-1 text-center font-mono text-sm text-text-secondary">{{ model || '…' }}</div>
          <div class="mb-5 text-center text-sm text-text-muted">
            Send a message to start. Conversations stay in this browser.
          </div>
          <div class="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <button
              v-for="example in CHAT_EXAMPLES"
              :key="example.label"
              class="rounded-xl border border-surface-border bg-surface-raised px-4 py-3 text-left text-sm text-text-secondary transition-colors hover:border-accent hover:text-text"
              @click="useExample(example)"
            >
              <span>{{ example.label }}</span>
              <span v-if="example.pythonTools" class="mt-1 block font-mono text-[0.68rem] uppercase tracking-wide text-accent">
                Typed Python tool
              </span>
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
      <div class="mx-auto max-w-3xl">
        <button
          class="mb-2 flex items-center gap-2 text-xs font-medium text-text-muted transition-colors hover:text-text-secondary"
          :class="{ 'text-accent': conversation.pythonTools.trim() }"
          @click="showTools = !showTools"
        >
          <span>{{ showTools ? '▾' : '▸' }}</span>
          <span>Python tools</span>
          <span v-if="conversation.pythonTools.trim()" class="rounded bg-accent/10 px-1.5 py-0.5 text-[0.65rem] uppercase tracking-wide">
            configured
          </span>
        </button>
        <div v-if="showTools" class="mb-3">
          <textarea
            v-model="conversation.pythonTools"
            rows="7"
            :disabled="busy"
            spellcheck="false"
            placeholder="def lookup(value: str) -> dict[str, str]:&#10;    &quot;&quot;&quot;Describe the tool.&quot;&quot;&quot;&#10;    return {&quot;value&quot;: value}"
            class="w-full resize-y rounded-xl border border-surface-border bg-surface-sunken px-3 py-2 font-mono text-xs leading-relaxed text-text outline-none transition-colors focus:border-accent disabled:opacity-60"
            @input="emit('persist')"
          ></textarea>
          <p class="mt-1 text-[0.7rem] text-text-muted">
            Define typed top-level functions. They are sent inside &lt;python_tools&gt; XML and execute on this Iris task.
          </p>
        </div>
        <div class="flex items-end gap-2">
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
  </div>
</template>
