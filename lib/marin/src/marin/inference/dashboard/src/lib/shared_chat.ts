import type { ThinkingMode } from './chat_template'
import type { AssistantMessage, ChatMessage, Conversation } from './types'

const SHARED_CHAT_PREFIX = '#chat='
const SHARED_CHAT_VERSION = 1
const SHARED_CHAT_ID_PATTERN = /^[A-Za-z0-9_-]{8,64}$/

type SharedChatMessage =
  | { role: 'user'; content: string }
  | {
      role: 'assistant'
      content: string
      thinking: string
      thinkingSeconds: number | null
      error: string | null
    }

export interface SharedChatSnapshot {
  version: typeof SHARED_CHAT_VERSION
  title: string
  model: string
  messages: SharedChatMessage[]
}

export function isSharedChatHash(hash: string): boolean {
  return hash.startsWith(SHARED_CHAT_PREFIX)
}

function sharedMessages(messages: ChatMessage[]): SharedChatMessage[] {
  const shared: SharedChatMessage[] = []
  for (const message of messages) {
    if (message.role === 'user') {
      shared.push({ role: 'user', content: message.content })
    } else if (message.role === 'assistant' && (message.content || message.error)) {
      shared.push({
        role: 'assistant',
        content: message.content,
        thinking: '',
        thinkingSeconds: null,
        error: message.error,
      })
    }
  }
  return shared
}

/** Build the bounded, non-executable snapshot stored by the dashboard server. */
export function sharedChatSnapshot(conversation: Conversation): SharedChatSnapshot {
  return {
    version: SHARED_CHAT_VERSION,
    title: conversation.title,
    model: conversation.model,
    messages: sharedMessages(conversation.messages),
  }
}

/** Build a short link to a snapshot stored by the dashboard server. */
export function sharedChatUrl(currentUrl: string, shareId: string): string {
  if (!SHARED_CHAT_ID_PATTERN.test(shareId)) throw new Error('Invalid shared chat ID')
  const url = new URL(currentUrl)
  url.hash = `${SHARED_CHAT_PREFIX}${shareId}`
  return url.toString()
}

export function sharedChatIdFromHash(hash: string): string | null {
  if (!isSharedChatHash(hash)) return null
  const shareId = hash.slice(SHARED_CHAT_PREFIX.length)
  return SHARED_CHAT_ID_PATTERN.test(shareId) ? shareId : null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parsedChatMessage(value: unknown): ChatMessage | null {
  if (!isRecord(value)) return null
  if (value.role === 'user' && typeof value.content === 'string') {
    return { role: 'user', content: value.content }
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

  const message: AssistantMessage = {
    role: 'assistant',
    content: value.content,
    thinking: value.thinking,
    thinkingSeconds: value.thinkingSeconds,
    error: value.error,
  }
  return message
}

/** Validate a server snapshot and import it as a new local conversation. */
export function conversationFromSharedChat(
  value: unknown,
  conversationId: string,
  timestamp: number,
): Conversation | null {
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
}
