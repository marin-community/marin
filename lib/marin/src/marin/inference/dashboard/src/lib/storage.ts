import type { Conversation, SamplingParams } from './types'

const CONVERSATIONS_KEY = 'marin-serve:conversations:v1'
const PARAMS_KEY = 'marin-serve:params:v1'
const MAX_CONVERSATIONS = 100

export const DEFAULT_PARAMS: SamplingParams = { temperature: 0.7, maxTokens: 1024, topP: 1.0 }

export function newId(): string {
  return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(CONVERSATIONS_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function saveConversations(conversations: Conversation[]): void {
  const kept = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_CONVERSATIONS)
  try {
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(kept))
  } catch (error) {
    // Quota exhaustion must not break the chat itself; history just stops persisting.
    console.warn('failed to persist conversations', error)
  }
}

export function loadParams(): SamplingParams {
  try {
    const raw = localStorage.getItem(PARAMS_KEY)
    return raw ? { ...DEFAULT_PARAMS, ...JSON.parse(raw) } : { ...DEFAULT_PARAMS }
  } catch {
    return { ...DEFAULT_PARAMS }
  }
}

export function saveParams(params: SamplingParams): void {
  try {
    localStorage.setItem(PARAMS_KEY, JSON.stringify(params))
  } catch (error) {
    console.warn('failed to persist params', error)
  }
}
