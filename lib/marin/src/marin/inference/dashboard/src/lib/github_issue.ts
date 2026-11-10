import type { ThinkingMode } from './chat_template'
import type { Conversation, SamplingParams, ServingInfo } from './types'

const GITHUB_NEW_ISSUE_URL = 'https://github.com/marin-community/marin/issues/new'
// GitHub rejects new-issue URLs much longer than ~8 KB.
const MAX_ISSUE_URL_LENGTH = 8000
export const ISSUE_TRANSCRIPT_FILENAME = 'model-feedback-transcript.txt'
const TITLE_PREFIX = '[model feedback]'
const MAX_TITLE_SUMMARY_LENGTH = 60
const OMITTED_CONVERSATION =
  'The reporter omitted the conversation. <!-- Optionally paste a minimal, non-sensitive reproduction here. -->'

const THINKING_LABELS: Record<ThinkingMode, string> = {
  default: 'template default',
  enabled: 'enabled',
  disabled: 'disabled',
}

export interface IssueDraft {
  url: string
  /** Complete transcript to attach, set only when the URL carries a truncated copy. */
  fullTranscript: string | null
}

/**
 * Build a prefilled GitHub new-issue URL for a conversation the model handled poorly.
 *
 * `chatTranscript` is the conversation's plain-text rendering (see `plainTextChat`), or `null` when the
 * reporter omits the conversation. Omitting it also drops the conversation title and custom
 * template instructions, which are conversation content. Included instructions are prepended
 * to the transcript so URL truncation covers them.
 *
 * The dashboard URL is never included: it carries the Iris capability token.
 */
export function issueDraft(
  conversation: Conversation,
  chatTranscript: string | null,
  info: ServingInfo | null,
  params: SamplingParams,
  maxUrlLength = MAX_ISSUE_URL_LENGTH,
): IssueDraft {
  const included = chatTranscript !== null
  const title = issueTitle(conversation, included)
  const setup = setupSection(conversation, info, params, included)
  const urlFor = (body: string) => {
    const url = new URL(GITHUB_NEW_ISSUE_URL)
    url.searchParams.set('title', title)
    url.searchParams.set('body', body)
    return url.toString()
  }

  if (chatTranscript === null) {
    return { url: withinLimit(urlFor(issueBody(setup, OMITTED_CONVERSATION)), maxUrlLength), fullTranscript: null }
  }

  const instructions = conversation.customInstructions.trim()
  const transcript = instructions
    ? `[custom template instructions]\n${instructions}\n\n${chatTranscript}`
    : chatTranscript

  const fullUrl = urlFor(issueBody(setup, conversationSection(transcript, false)))
  if (fullUrl.length <= maxUrlLength) return { url: fullUrl, fullTranscript: null }

  // Code points, so a cut never leaves a lone surrogate that URL encoding rejects.
  const chars = Array.from(transcript)
  const truncatedUrl = (kept: number) => urlFor(issueBody(setup, conversationSection(elideMiddle(chars, kept), true)))
  let low = 0
  let high = chars.length - 1
  while (low < high) {
    const mid = Math.ceil((low + high) / 2)
    if (truncatedUrl(mid).length <= maxUrlLength) low = mid
    else high = mid - 1
  }
  return { url: withinLimit(truncatedUrl(low), maxUrlLength), fullTranscript: transcript }
}

function withinLimit(url: string, maxUrlLength: number): string {
  if (url.length > maxUrlLength) throw new Error('Issue metadata alone exceeds the GitHub URL limit')
  return url
}

function issueTitle(conversation: Conversation, includeConversation: boolean): string {
  const model = conversation.model || 'unknown model'
  const summary = includeConversation ? Array.from(conversation.title.replace(/\s+/g, ' ').trim()) : []
  if (!summary.length) return `${TITLE_PREFIX} ${model}`
  const clipped = summary.length > MAX_TITLE_SUMMARY_LENGTH
    ? `${summary.slice(0, MAX_TITLE_SUMMARY_LENGTH).join('').trimEnd()}…`
    : summary.join('')
  return `${TITLE_PREFIX} ${model}: ${clipped}`
}

function setupSection(
  conversation: Conversation,
  info: ServingInfo | null,
  params: SamplingParams,
  includeConversation: boolean,
): string {
  const lines = [`- Model: \`${conversation.model || info?.model || 'unknown'}\``]
  if (info) {
    const context = info.max_model_len ? `${info.max_model_len} tokens` : 'auto'
    lines.push(`- Serving: ${info.backend}, ${info.dtype}, tensor parallel ${info.tensor_parallel_size}, context ${context}`)
  }
  lines.push(
    `- Sampling when reported: temperature ${params.temperature}, top_p ${params.topP}, max_tokens ${params.maxTokens}`,
    `- Thinking: ${THINKING_LABELS[conversation.thinkingMode]}`,
  )
  if (includeConversation && conversation.customInstructions.trim()) {
    lines.push('- Custom template instructions: set (first section of the transcript)')
  }
  if (conversation.pythonTools.trim()) lines.push('- Python tools: configured (source not included)')
  if (conversation.shellWorkspace) lines.push('- Shell workspace: enabled (files not included)')
  if (conversation.webSearch) lines.push('- Web search: enabled')
  return lines.join('\n')
}

function issueBody(setup: string, conversation: string): string {
  return [
    // An HTML comment is visible while editing but not rendered in the submitted issue.
    '<!-- This repository is public. Remove private or sensitive content before submitting. -->',
    '## What went wrong\n\n<!-- Describe what the model did poorly and what a good response would look like. -->',
    `## Setup\n\n${setup}`,
    `## Conversation\n\n${conversation}`,
  ].join('\n\n')
}

function conversationSection(transcript: string, truncated: boolean): string {
  const fence = '`'.repeat(Math.max(3, longestBacktickRun(transcript) + 1))
  const sections = [`${fence}text\n${transcript}\n${fence}`]
  if (truncated) {
    sections.push(
      'The transcript above was shortened to fit the GitHub URL limit. The dashboard downloaded the full ' +
        `transcript as \`${ISSUE_TRANSCRIPT_FILENAME}\`; drag it into this issue to attach it.`,
    )
  }
  return sections.join('\n\n')
}

function elideMiddle(chars: string[], kept: number): string {
  const head = Math.ceil(kept / 2)
  const tail = kept - head
  const omitted = chars.length - kept
  return `${chars.slice(0, head).join('')}\n\n[… ${omitted} characters omitted …]\n\n${chars.slice(chars.length - tail).join('')}`
}

function longestBacktickRun(text: string): number {
  let longest = 0
  for (const run of text.match(/`+/g) ?? []) longest = Math.max(longest, run.length)
  return longest
}
