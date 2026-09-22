import type { ThinkingMode } from './chat_template'
import type { AssistantMessage, ChatMessage, Conversation, ToolCall } from './types'

const SHARED_CHAT_PREFIX = '#chat='
const SHARED_CHAT_VERSION = 1
const BINARY_CHUNK_SIZE = 0x8000

interface SharedChat {
  version: typeof SHARED_CHAT_VERSION
  title: string
  model: string
  messages: ChatMessage[]
}

export function isSharedChatHash(hash: string): boolean {
  return hash.startsWith(SHARED_CHAT_PREFIX)
}

function base64UrlEncode(text: string): string {
  const bytes = new TextEncoder().encode(text)
  let binary = ''
  for (let start = 0; start < bytes.length; start += BINARY_CHUNK_SIZE) {
    binary += String.fromCharCode(...bytes.subarray(start, start + BINARY_CHUNK_SIZE))
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function base64UrlDecode(encoded: string): string {
  const base64 = encoded.replace(/-/g, '+').replace(/_/g, '/')
  const binary = atob(base64.padEnd(Math.ceil(base64.length / 4) * 4, '='))
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0))
  return new TextDecoder().decode(bytes)
}

function sharedMessages(messages: ChatMessage[]): ChatMessage[] {
  return messages.map((message) => {
    if (message.role === 'user') return { role: 'user', content: message.content }
    if (message.role === 'tool') {
      return {
        role: 'tool',
        name: message.name,
        toolCallId: message.toolCallId,
        result: message.result,
      }
    }
    return {
      role: 'assistant',
      content: message.content,
      thinking: message.thinking,
      thinkingSeconds: message.thinkingSeconds,
      error: message.error,
      toolCalls: message.toolCalls,
    }
  })
}

/** Build a link containing a visible, non-executable snapshot of a conversation. */
export function sharedChatUrl(currentUrl: string, conversation: Conversation): string {
  const payload: SharedChat = {
    version: SHARED_CHAT_VERSION,
    title: conversation.title,
    model: conversation.model,
    messages: sharedMessages(conversation.messages),
  }
  const url = new URL(currentUrl)
  url.hash = `chat=${base64UrlEncode(JSON.stringify(payload))}`
  return url.toString()
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parsedToolCall(value: unknown): ToolCall | null {
  if (!isRecord(value)) return null
  if (typeof value.id !== 'string' || typeof value.name !== 'string' || !isRecord(value.arguments)) return null
  return { id: value.id, name: value.name, arguments: value.arguments }
}

function parsedChatMessage(value: unknown): ChatMessage | null {
  if (!isRecord(value)) return null
  if (value.role === 'user' && typeof value.content === 'string') {
    return { role: 'user', content: value.content }
  }
  if (
    value.role === 'tool' &&
    typeof value.name === 'string' &&
    typeof value.toolCallId === 'string' &&
    'result' in value
  ) {
    return { role: 'tool', name: value.name, toolCallId: value.toolCallId, result: value.result }
  }
  if (
    value.role !== 'assistant' ||
    typeof value.content !== 'string' ||
    typeof value.thinking !== 'string' ||
    (value.thinkingSeconds !== null && typeof value.thinkingSeconds !== 'number') ||
    (value.error !== null && typeof value.error !== 'string')
  ) {
    return null
  }

  let calls: ToolCall[] | undefined
  if (value.toolCalls !== undefined) {
    if (!Array.isArray(value.toolCalls)) return null
    calls = []
    for (const candidate of value.toolCalls) {
      const call = parsedToolCall(candidate)
      if (!call) return null
      calls.push(call)
    }
  }
  const message: AssistantMessage = {
    role: 'assistant',
    content: value.content,
    thinking: value.thinking,
    thinkingSeconds: value.thinkingSeconds,
    error: value.error,
  }
  if (calls) message.toolCalls = calls
  return message
}

/** Decode an imported chat fragment into a new local conversation. */
export function sharedConversationFromHash(
  hash: string,
  conversationId: string,
  timestamp: number,
): Conversation | null {
  if (!isSharedChatHash(hash)) return null
  try {
    const value: unknown = JSON.parse(base64UrlDecode(hash.slice(SHARED_CHAT_PREFIX.length)))
    if (!isRecord(value)) return null
    if (
      value.version !== SHARED_CHAT_VERSION ||
      typeof value.title !== 'string' ||
      typeof value.model !== 'string' ||
      !Array.isArray(value.messages)
    ) {
      return null
    }
    const messages: ChatMessage[] = []
    for (const candidate of value.messages) {
      const message = parsedChatMessage(candidate)
      if (!message) return null
      messages.push(message)
    }
    return {
      id: conversationId,
      title: value.title,
      model: value.model,
      system: '',
      pythonTools: '',
      shellWorkspace: null,
      thinkingMode: 'default' satisfies ThinkingMode,
      customInstructions: '',
      createdAt: timestamp,
      updatedAt: timestamp,
      messages,
    }
  } catch {
    return null
  }
}
