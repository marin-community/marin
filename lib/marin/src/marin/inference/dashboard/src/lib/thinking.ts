import type { ChatTemplateProtocol } from './types'

type ThinkingDelimiters = readonly [open: string, close: string]

const KNOWN_THINKING_DELIMITERS: readonly ThinkingDelimiters[] = Object.freeze([
  // Browser fallback for missing or stale /info metadata. Keep aligned with
  // chat_template_protocol.py; the Python service and TypeScript bundle cannot share a runtime constant.
  Object.freeze(['<|start_think|>', '<|end_think|>'] as const),
  Object.freeze(['<think>', '</think>'] as const),
  Object.freeze(['<THINK>', '</THINK>'] as const),
])

export interface ThinkingSplit {
  thinking: string
  visible: string
  /** True while an opened thinking segment has not been closed yet. */
  inThinking: boolean
}

function selectThinkingDelimiters(raw: string, protocol: ChatTemplateProtocol | null): ThinkingDelimiters | null {
  const configured = protocol?.thinking_start && protocol.thinking_end
    ? ([protocol.thinking_start, protocol.thinking_end] as const)
    : null
  const candidates = configured
    ? [configured, ...KNOWN_THINKING_DELIMITERS.filter(([open, close]) => open !== configured[0] || close !== configured[1])]
    : KNOWN_THINKING_DELIMITERS
  const trimmed = raw.trimStart()

  for (const delimiters of candidates) {
    const [open, close] = delimiters
    if (raw.includes(open) || raw.includes(close) || open.startsWith(trimmed)) return delimiters
  }
  return configured
}

/** Longest suffix of `text` that is a strict prefix of an unfinished tag, so a
 * streaming cut mid-`</think>` is neither shown nor treated as content. */
function trailingPartialTag(text: string, tags: string[]): string {
  const maxTagLength = Math.max(...tags.map((tag) => tag.length))
  const max = Math.min(text.length, maxTagLength - 1)
  for (let len = max; len > 0; len--) {
    const suffix = text.slice(text.length - len)
    if (tags.some((tag) => tag.startsWith(suffix))) return suffix
  }
  return ''
}

/** Split accumulated model output into a thinking segment and visible text.
 *
 * Delimiters prefer the served model's active chat-template metadata, with
 * known formats as a fallback. The template may open the block inside the
 * prompt, leaving only a bare closing delimiter in generated content. Re-run
 * on the full accumulated text after each delta.
 */
export function splitThinking(raw: string, protocol: ChatTemplateProtocol | null): ThinkingSplit {
  const delimiters = selectThinkingDelimiters(raw, protocol)
  if (!delimiters) return { thinking: '', visible: raw, inThinking: false }
  const [open, close] = delimiters

  const held = trailingPartialTag(raw, [open, close])
  const text = held ? raw.slice(0, raw.length - held.length) : raw

  const lead = text.length - text.trimStart().length
  const hasOpeningTag = text.startsWith(open, lead)
  const closeIndex = text.indexOf(close)

  if (closeIndex !== -1 && !hasOpeningTag) {
    // Bare closing tag: everything before it was reasoning.
    return {
      thinking: text.slice(0, closeIndex).trim(),
      visible: text.slice(closeIndex + close.length).replace(/^\s+/, ''),
      inThinking: false,
    }
  }
  if (hasOpeningTag) {
    const after = text.slice(lead + open.length)
    const end = after.indexOf(close)
    if (end === -1) {
      return { thinking: after.replace(/^\s+/, ''), visible: '', inThinking: true }
    }
    return {
      thinking: after.slice(0, end).trim(),
      visible: after.slice(end + close.length).replace(/^\s+/, ''),
      inThinking: false,
    }
  }
  return { thinking: '', visible: text, inThinking: false }
}
