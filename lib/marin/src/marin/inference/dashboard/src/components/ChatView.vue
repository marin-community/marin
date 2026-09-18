<script setup lang="ts">
import { computed, nextTick, onUnmounted, ref, watch } from 'vue'
import {
  fetchToolDefinitions,
  invokeShell,
  invokeTool,
  isAbortError,
  requestCompletion,
} from '../lib/api'
import { chatTemplateRequestFields } from '../lib/chat_template'
import { CHAT_EXAMPLES } from '../lib/examples'
import type { ChatExample } from '../lib/examples'
import { modelMessages } from '../lib/python_tools'
import type { ModelMessage, ToolDefinition } from '../lib/python_tools'
import { plainTextChat } from '../lib/plain_text_chat'
import {
  BASH_TOOL_DEFINITION,
  BASH_TOOL_NAME,
  bashCommand,
  parseWorkspaceFiles,
} from '../lib/shell_workspace'
import { splitThinking } from '../lib/thinking'
import {
  appendToolCallDelta,
  createToolCallAccumulator,
  finalizeToolCalls,
  inlineToolCalls,
} from '../lib/tool_calls'
import { newId } from '../lib/storage'
import type {
  AssistantMessage,
  ChatTemplateProtocol,
  Conversation,
  SamplingParams,
  ToolCall,
  ToolMessage,
} from '../lib/types'
import MessageBubble from './MessageBubble.vue'
import ShellWorkspacePanel from './ShellWorkspacePanel.vue'

const MAX_TOOL_ROUNDS = 8

const props = defineProps<{
  conversation: Conversation
  params: SamplingParams
  model: string
  hasChatTemplate: boolean
  chatTemplateProtocol: ChatTemplateProtocol | null
  streaming: boolean
}>()

const emit = defineEmits<{ persist: [] }>()

const draft = ref('')
const busy = ref(false)
const showTools = ref(false)
const showWorkspace = ref(false)
const showRawChat = ref(false)
const rawChat = computed(() => plainTextChat(props.conversation))
const scroller = ref<HTMLElement | null>(null)
const composer = ref<HTMLTextAreaElement | null>(null)
let abort: AbortController | null = null

watch(
  () => props.conversation.id,
  () => {
    stopStreaming()
    draft.value = ''
    showTools.value = Boolean(props.conversation.pythonTools)
    showWorkspace.value = Boolean(props.conversation.shellWorkspace)
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
  () =>
    props.conversation.messages
      .map((message) => {
        if (message.role === 'tool') return JSON.stringify(message.result).length
        return message.content.length + (message.role === 'assistant' ? message.thinking.length : 0)
      })
      .join(','),
  async () => {
    const el = scroller.value
    if (!el) return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 150
    if (!nearBottom) return
    await nextTick()
    el.scrollTo({ top: el.scrollHeight })
  },
)

async function applyExample(example: ChatExample) {
  if (example.pythonTools !== undefined) {
    props.conversation.pythonTools = example.pythonTools
    showTools.value = true
  }
  if (example.workspaceFiles !== undefined) {
    props.conversation.shellWorkspace = {
      filesJson: JSON.stringify(example.workspaceFiles, null, 2),
      history: [],
      repositoryUrl: '',
    }
    showWorkspace.value = true
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
  conversation.messages.push({ role: 'user', content })
  persistConversation(conversation)

  busy.value = true
  abort = new AbortController()
  const signal = abort.signal
  try {
    await runToolExchange(conversation, pythonTools, signal)
  } finally {
    busy.value = false
    abort = null
    persistConversation(conversation)
  }
}

async function runToolExchange(conversation: Conversation, pythonTools: string, signal: AbortSignal) {
  let reply: AssistantMessage | null = null
  try {
    const tools = pythonTools ? await fetchToolDefinitions(pythonTools, signal) : []
    let workspaceFiles: Record<string, string> | null = null
    if (conversation.shellWorkspace) {
      workspaceFiles = parseWorkspaceFiles(conversation.shellWorkspace.filesJson)
      if (tools.some((tool) => tool.function.name === BASH_TOOL_NAME)) {
        throw new Error(`Python tool name ${JSON.stringify(BASH_TOOL_NAME)} is reserved for the shell workspace`)
      }
      tools.push(BASH_TOOL_DEFINITION)
    }
    const templateFields = chatTemplateRequestFields(
      conversation.thinkingMode,
      conversation.customInstructions,
      tools,
    )
    for (let round = 0; round < MAX_TOOL_ROUNDS; round += 1) {
      const request = modelMessages(conversation)
      reply = appendAssistantReply(conversation)

      await complete(reply, request, tools, templateFields, signal)
      const calls = reply.toolCalls ?? []
      persistConversation(conversation)
      if (!calls.length) break

      await executeToolCalls(conversation, calls, pythonTools, workspaceFiles, signal)

      if (round === MAX_TOOL_ROUNDS - 1) {
        reply.error = `Stopped after ${MAX_TOOL_ROUNDS} consecutive tool rounds.`
      }
    }
  } catch (error) {
    if (isAbortError(error)) {
      appendCancelledToolResults(conversation, reply)
    } else {
      reply ??= appendAssistantReply(conversation)
      reply.error = String(error)
    }
  }
}

function appendAssistantReply(conversation: Conversation): AssistantMessage {
  conversation.messages.push({
    role: 'assistant',
    content: '',
    thinking: '',
    rawContent: '',
    rawReasoning: '',
    thinkingSeconds: null,
    error: null,
    toolCalls: [],
  })
  // Return the reactive proxy so streaming deltas re-render.
  const reply = conversation.messages[conversation.messages.length - 1]
  if (reply.role !== 'assistant') throw new Error('Expected an assistant reply')
  persistConversation(conversation)
  return reply
}

async function executeToolCalls(
  conversation: Conversation,
  calls: ToolCall[],
  pythonTools: string,
  workspaceFiles: Record<string, string> | null,
  signal: AbortSignal,
) {
  for (const call of calls) {
    let result: unknown
    try {
      if (call.name === BASH_TOOL_NAME) {
        const workspace = conversation.shellWorkspace
        if (!workspace || !workspaceFiles) throw new Error('Shell workspace is not enabled')
        const command = bashCommand(call.arguments)
        const shellResult = await invokeShell(workspaceFiles, workspace.history, command, signal)
        result = shellResult
        if (shellResult.stop_reason === null) workspace.history.push(command)
      } else {
        result = await invokeTool(call.name, pythonTools, call.arguments, signal)
      }
    } catch (error) {
      if (isAbortError(error)) throw error
      result = { error: String(error) }
    }
    conversation.messages.push(toolResultMessage(call, result))
    persistConversation(conversation)
  }
}

function persistConversation(conversation: Conversation) {
  conversation.updatedAt = Date.now()
  emit('persist')
}

function appendCancelledToolResults(conversation: Conversation, reply: AssistantMessage | null) {
  const completed = new Set(
    conversation.messages.filter((message) => message.role === 'tool').map((message) => message.toolCallId),
  )
  for (const call of reply?.toolCalls ?? []) {
    if (completed.has(call.id)) continue
    conversation.messages.push(toolResultMessage(call, { error: 'tool call cancelled' }))
  }
}

function toolResultMessage(call: ToolCall, result: unknown): ToolMessage {
  return {
    role: 'tool',
    name: call.name,
    toolCallId: call.id,
    result,
  }
}

async function complete(
  reply: AssistantMessage,
  messages: ModelMessage[],
  tools: ToolDefinition[],
  templateFields: ReturnType<typeof chatTemplateRequestFields>,
  signal: AbortSignal,
) {
  let rawContent = ''
  let reasoningStream = ''
  let thinkingStartedAt: number | null = null
  const structuredCalls = createToolCallAccumulator()

  const body: Record<string, unknown> = {
    model: props.model,
    messages,
    stream: props.streaming,
    temperature: props.params.temperature,
    max_tokens: props.params.maxTokens,
    top_p: props.params.topP,
    ...templateFields,
  }
  if (tools.length) {
    // Permit model-native call generation without requiring vLLM auto-tool parsing.
    // The response handling below also accepts structured calls when a server emits them.
    body.tool_choice = null
  }
  await requestCompletion('v1/chat/completions', body, props.streaming, signal, (data) => {
    const delta = data.choices?.[0]?.delta ?? data.choices?.[0]?.message
    if (!delta) return
    const reasoning = delta.reasoning_content ?? delta.reasoning
    if (reasoning) reasoningStream += reasoning
    if (delta.content) rawContent += delta.content
    reply.rawContent = rawContent
    reply.rawReasoning = reasoningStream
    if (tools.length && delta.tool_calls !== undefined) appendToolCallDelta(structuredCalls, delta.tool_calls)

    const split = splitThinking(rawContent, props.chatTemplateProtocol)
    const reasoningSplit = splitThinking(reasoningStream, props.chatTemplateProtocol)
    const streamedThinking = [reasoningSplit.thinking, reasoningSplit.visible].filter(Boolean).join('\n')
    reply.thinking = [streamedThinking, split.thinking].filter(Boolean).join('\n')
    if (tools.length) {
      const inline = inlineToolCalls(split.visible, props.chatTemplateProtocol, newId)
      reply.content = inline.visible
      reply.toolCalls = structuredCalls.calls.size ? [] : inline.calls
    } else {
      reply.content = split.visible
      reply.toolCalls = []
    }
    if (reply.thinking && thinkingStartedAt === null) thinkingStartedAt = performance.now()
    if (thinkingStartedAt !== null && reply.thinkingSeconds === null && (reply.content || reply.toolCalls.length)) {
      reply.thinkingSeconds = (performance.now() - thinkingStartedAt) / 1000
    }
  })

  if (thinkingStartedAt !== null && reply.thinkingSeconds === null) {
    reply.thinkingSeconds = (performance.now() - thinkingStartedAt) / 1000
  }
  if (tools.length) {
    const split = splitThinking(rawContent, props.chatTemplateProtocol)
    const inline = inlineToolCalls(split.visible, props.chatTemplateProtocol, newId)
    reply.content = inline.visible
    reply.toolCalls = structuredCalls.calls.size ? finalizeToolCalls(structuredCalls, newId) : inline.calls
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
              @click="applyExample(example)"
            >
              <span>{{ example.label }}</span>
              <span v-if="example.pythonTools" class="mt-1 block font-mono text-[0.68rem] uppercase tracking-wide text-accent">
                Typed Python tool
              </span>
              <span v-if="example.workspaceFiles" class="mt-1 block font-mono text-[0.68rem] uppercase tracking-wide text-accent">
                ShellSim workspace
              </span>
            </button>
          </div>
          <div v-if="!hasChatTemplate" class="mt-4 text-center text-xs text-text-muted">
            This model reports no chat template — chat requests may fail; try completion mode.
          </div>
        </div>
      </div>
      <pre
        v-else-if="showRawChat"
        class="mx-auto max-w-3xl whitespace-pre-wrap break-words px-4 py-5 font-mono text-xs leading-relaxed text-text-secondary md:px-6"
      >{{ rawChat }}</pre>
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
        <div class="mb-2 flex items-center justify-between gap-4">
          <div class="flex items-center gap-4">
            <button
              class="flex items-center gap-2 text-xs font-medium text-text-muted transition-colors hover:text-text-secondary"
              :class="{ 'text-accent': conversation.pythonTools.trim() }"
              @click="showTools = !showTools"
            >
              <span>{{ showTools ? '▾' : '▸' }}</span>
              <span>Python tools</span>
              <span v-if="conversation.pythonTools.trim()" class="rounded bg-accent/10 px-1.5 py-0.5 text-[0.65rem] uppercase tracking-wide">
                configured
              </span>
            </button>
            <button
              class="flex items-center gap-2 text-xs font-medium text-text-muted transition-colors hover:text-text-secondary"
              :class="{ 'text-accent': conversation.shellWorkspace }"
              @click="showWorkspace = !showWorkspace"
            >
              <span>{{ showWorkspace ? '▾' : '▸' }}</span>
              <span>Shell workspace</span>
              <span v-if="conversation.shellWorkspace" class="rounded bg-accent/10 px-1.5 py-0.5 text-[0.65rem] uppercase tracking-wide">
                enabled
              </span>
            </button>
          </div>
          <label
            class="flex cursor-pointer items-center gap-2 text-xs font-medium text-text-muted"
            title="Show the whole chat as a plain-text transcript"
          >
            <input v-model="showRawChat" type="checkbox" class="accent-accent" />
            Raw chat
          </label>
        </div>
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
            Define synchronous typed functions. Calls run in an isolated, resource-bounded ShellSim environment.
          </p>
        </div>
        <ShellWorkspacePanel
          v-if="showWorkspace"
          v-model:workspace="conversation.shellWorkspace"
          :disabled="busy"
          @close="showWorkspace = false"
          @persist="persistConversation(conversation)"
        />
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
