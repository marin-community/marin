/**
 * Classifies a lm-eval sample row as multiple-choice (loglikelihood over fixed choices) or
 * generative (free-form completion), and extracts the fields each rendering needs.
 *
 * MCQ rows carry one `arguments` entry per choice (`[context, continuation]`, context identical
 * across entries) and one `responses` entry per choice holding a `[loglikelihood, is_greedy]`
 * pair, optionally wrapped in a singleton list. Generative rows carry a single `arguments` entry
 * (`[prompt, gen_kwargs]`) and a single generated string in `responses`; when the prompt string
 * itself JSON-parses to a list of `{role, content}` messages it is a chat-templated prompt.
 * Rows that match neither shape fall back to `{ kind: 'raw' }` so callers can dump raw JSON.
 */
import type { SampleRow } from '@/types/api'

export interface ChatMessage {
  role: string
  content: string
}

export interface McqChoice {
  index: number
  label: string
  continuation: string
  logLikelihood: number | null
  isGreedy: boolean | null
}

export interface McqSampleData {
  kind: 'mcq'
  context: string
  choices: McqChoice[]
  pickIndex: number | null
  goldIndex: number | null
}

export interface GenerativeSampleData {
  kind: 'generative'
  prompt: string
  chatMessages: ChatMessage[] | null
  outputText: string
}

export interface RawSampleData {
  kind: 'raw'
}

export type DetectedSample = McqSampleData | GenerativeSampleData | RawSampleData

function indexToLetter(i: number): string {
  return String.fromCharCode(65 + i)
}

function isNumberBooleanPair(value: unknown): value is [number, boolean] {
  return Array.isArray(value) && value.length === 2 && typeof value[0] === 'number' && typeof value[1] === 'boolean'
}

/** Unwraps a `responses` entry to its `[loglikelihood, is_greedy]` pair, tolerating a singleton-list wrapper. */
function extractLLPair(entry: unknown): [number, boolean] | null {
  if (isNumberBooleanPair(entry)) return entry
  if (Array.isArray(entry) && entry.length > 0 && isNumberBooleanPair(entry[0])) return entry[0]
  return null
}

function extractLabels(doc: unknown): string[] | null {
  if (!doc || typeof doc !== 'object') return null
  const choices = (doc as Record<string, unknown>).choices
  if (!choices || typeof choices !== 'object') return null
  const label = (choices as Record<string, unknown>).label
  return Array.isArray(label) && label.every((l) => typeof l === 'string') ? (label as string[]) : null
}

/** The argmax-loglikelihood choice index, or null when no choice has a usable loglikelihood. */
export function argmaxChoice(choices: McqChoice[]): number | null {
  let best: number | null = null
  let bestLL = -Infinity
  for (const choice of choices) {
    if (choice.logLikelihood !== null && choice.logLikelihood > bestLL) {
      bestLL = choice.logLikelihood
      best = choice.index
    }
  }
  return best
}

/** Resolves `target` (an index, a label like "B", or the choice text) to a choice index. */
export function resolveTargetIndex(target: unknown, choices: McqChoice[]): number | null {
  if (typeof target === 'number' && Number.isInteger(target)) {
    return target >= 0 && target < choices.length ? target : null
  }
  if (typeof target === 'string') {
    const trimmed = target.trim()
    const byLabel = choices.find((c) => c.label === trimmed)
    if (byLabel) return byLabel.index
    const byText = choices.find((c) => c.continuation.trim() === trimmed)
    if (byText) return byText.index
    if (/^\d+$/.test(trimmed)) {
      const asIndex = Number(trimmed)
      return asIndex >= 0 && asIndex < choices.length ? asIndex : null
    }
  }
  return null
}

function tryBuildMcq(row: SampleRow): McqSampleData | null {
  const args = row.arguments
  if (!Array.isArray(args) || args.length <= 1) return null
  const responses = row.responses
  if (!Array.isArray(responses) || responses.length !== args.length) return null
  const pairs = responses.map(extractLLPair)
  if (pairs.some((p) => p === null)) return null

  const labels = extractLabels(row.doc)
  const context = Array.isArray(args[0]) && typeof args[0][0] === 'string' ? args[0][0] : ''
  const choices: McqChoice[] = args.map((entry, i) => {
    const continuation = Array.isArray(entry) && typeof entry[1] === 'string' ? entry[1] : ''
    const pair = pairs[i]
    return {
      index: i,
      label: labels?.[i] ?? indexToLetter(i),
      continuation,
      logLikelihood: pair ? pair[0] : null,
      isGreedy: pair ? pair[1] : null,
    }
  })

  return {
    kind: 'mcq',
    context,
    choices,
    pickIndex: argmaxChoice(choices),
    goldIndex: resolveTargetIndex(row.target, choices),
  }
}

/** Parses a prompt string as chat messages iff it is a JSON array of `{role, content}` objects. */
function tryParseChatMessages(text: string): ChatMessage[] | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return null
  }
  if (!Array.isArray(parsed) || parsed.length === 0) return null
  const messages: ChatMessage[] = []
  for (const item of parsed) {
    if (
      !item ||
      typeof item !== 'object' ||
      typeof (item as Record<string, unknown>).role !== 'string' ||
      typeof (item as Record<string, unknown>).content !== 'string'
    ) {
      return null
    }
    messages.push({ role: (item as Record<string, string>).role, content: (item as Record<string, string>).content })
  }
  return messages
}

function extractGenerativeOutput(responses: unknown): string {
  if (!Array.isArray(responses) || responses.length === 0) return ''
  const first = responses[0]
  if (typeof first === 'string') return first
  if (Array.isArray(first) && typeof first[0] === 'string') return first[0]
  return ''
}

function tryBuildGenerative(row: SampleRow): GenerativeSampleData | null {
  const args = row.arguments
  if (!Array.isArray(args) || args.length === 0) return null
  const first = args[0]
  const prompt = Array.isArray(first) ? first[0] : first
  if (typeof prompt !== 'string') return null
  return {
    kind: 'generative',
    prompt,
    chatMessages: tryParseChatMessages(prompt),
    outputText: extractGenerativeOutput(row.responses),
  }
}

export function detectSample(row: SampleRow): DetectedSample {
  return tryBuildMcq(row) ?? tryBuildGenerative(row) ?? { kind: 'raw' }
}

/** A short "answer" summary for a row, used by the samples table: picked choice or filtered response. */
export function answerSummary(row: SampleRow): string {
  const detected = detectSample(row)
  if (detected.kind === 'mcq') {
    if (detected.pickIndex === null) return '—'
    const choice = detected.choices[detected.pickIndex]
    return `${choice.label}: ${choice.continuation.trim()}`
  }
  const filtered = row.filtered_responses
  if (filtered === null || filtered === undefined) return ''
  if (typeof filtered === 'string') return filtered
  if (Array.isArray(filtered) && filtered.length === 1 && typeof filtered[0] === 'string') return filtered[0]
  return JSON.stringify(filtered)
}
