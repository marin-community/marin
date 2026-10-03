import { newConversation } from './storage'
import type { Comparison } from './comparison'

const COMPARISON_KEY = 'marin-serve:comparison:v1'

export function newComparison(leftModel: string, rightModel: string): Comparison {
  return { left: newConversation(leftModel), right: newConversation(rightModel), votes: {} }
}

export function loadComparison(): Comparison | null {
  try {
    const raw = localStorage.getItem(COMPARISON_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!parsed.left || !parsed.right || !Array.isArray(parsed.left.messages) || !Array.isArray(parsed.right.messages)) {
      return null
    }
    return { ...parsed, votes: parsed.votes ?? {} } as Comparison
  } catch {
    return null
  }
}

export function saveComparison(comparison: Comparison): void {
  try {
    localStorage.setItem(COMPARISON_KEY, JSON.stringify(comparison))
  } catch (error) {
    console.warn('failed to persist comparison', error)
  }
}
