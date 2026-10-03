import type { Conversation } from './types'

export interface Comparison {
  left: Conversation
  right: Conversation
  votes: Record<number, PreferenceWinner>
}

export type PreferenceWinner = 'left' | 'right' | 'tie' | 'both_bad'

export interface ComparisonTurn {
  index: number
  prompt: string
}

function turns(conversation: Conversation): { prompt: string; answered: boolean }[] {
  const result: { prompt: string; answered: boolean }[] = []
  for (const message of conversation.messages) {
    if (message.role === 'user') result.push({ prompt: message.content, answered: false })
    if (message.role === 'assistant' && result.length) result[result.length - 1].answered = true
  }
  return result
}

/** Return turns for which both models produced a response to the same prompt. */
export function comparisonTurns(left: Conversation, right: Conversation): ComparisonTurn[] {
  const leftTurns = turns(left)
  const rightTurns = turns(right)
  return leftTurns.flatMap((turn, index) =>
    turn.answered && rightTurns[index]?.answered && turn.prompt === rightTurns[index].prompt
      ? [{ index, prompt: turn.prompt }]
      : [],
  )
}

/** Accept a Marin serve dashboard URL on this browser origin. */
export function comparisonBaseUrl(input: string, currentUrl: string): string {
  const url = new URL(input.trim())
  if (url.origin !== new URL(currentUrl).origin) throw new Error('Use a dashboard URL on this Iris origin')
  url.hash = ''
  url.search = ''
  url.pathname = url.pathname.replace(/\/(dashboard|v1)\/?$/, '/')
  if (!url.pathname.endsWith('/')) url.pathname += '/'
  return url.toString()
}
